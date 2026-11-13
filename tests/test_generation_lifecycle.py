"""Real lifecycle/HTTP adapter tests using controllable Ollama streams and isolated data."""
import asyncio
import json
import time
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.database import SessionLocal
from backend.main import app, GenerationResponse, _generation_events
from backend.models import BibleSection, Conversation, Fact, GenerationRequest, Message, World
from backend.schemas import BibleSectionEditItem, CandidateProposalItem, ChatMessageRequest, CreativePermissionSchema, PostTurnExtractionSchema, TurnAnalysisSchema, WorldCreate
from backend.services.bible_service import bible_service
from backend.services.generation_service import generation_service
from backend.services.llm_service import llm_service
from backend.services.world_service import world_service
from backend.services.entity_service import entity_service

TEXT = 'Planet X is called Pallene.'


class GateStream(httpx.AsyncByteStream):
    def __init__(self, parts, started, released, closed):
        self.parts, self.started, self.released, self.closed = parts, started, released, closed

    async def __aiter__(self):
        self.started.set()
        try:
            for part in self.parts:
                yield (json.dumps(part) + '\n').encode()
            await self.released.wait()
            yield b'{"message":{"content":""},"done":true}\n'
        finally:
            self.closed.set()

    async def aclose(self):
        self.closed.set()


@pytest.fixture
def world():
    with SessionLocal() as db:
        world = world_service.create_world(db, WorldCreate(name='Lifecycle test'))
        conv = db.query(Conversation).filter(Conversation.world_id == world.id).one()
        data = (world.id, conv.id)
    yield data
    with SessionLocal() as db:
        obj = db.get(World, data[0])
        if obj:
            world_service.delete_world(db, obj)


@pytest.fixture
def runtime(monkeypatch):
    class Runtime:
        block = None
        fail = None
        calls = 0
        extraction = PostTurnExtractionSchema(proposals=[CandidateProposalItem(
            proposed_title='Planet X', proposed_content=TEXT, source_quote=TEXT,
        )])
        def __init__(self):
            self.started, self.release, self.closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def handler(self, request):
            self.calls += 1
            payload = json.loads(request.content)
            title = payload.get('format', {}).get('title')
            phase = 'analyzing' if title == 'TurnAnalysisSchema' else 'postprocessing' if title else 'generating'
            if self.fail == phase:
                raise httpx.ReadTimeout('stalled runtime', request=request)
            if title == 'TurnAnalysisSchema':
                value = TurnAnalysisSchema(primary_intent='CANON_DECLARATION', requires_factual_retrieval=False,
                    creative_permission=CreativePermissionSchema(level='IMPLEMENTATION_SCOPED')).model_dump_json()
            elif title:
                value = self.extraction.model_dump_json()
            else:
                value = 'Recorded Pallene.'
            if self.block == phase:
                parts = [] if title else [{'message': {'content': value}, 'done': False}]
                return httpx.Response(200, stream=GateStream(parts, self.started, self.release, self.closed))
            return httpx.Response(200, json={'message': {'content': value}, 'done': True})
    value = Runtime()
    monkeypatch.setattr(llm_service, '_async_client', lambda: httpx.AsyncClient(
        base_url='http://ollama.test', transport=httpx.MockTransport(value.handler),
    ))
    return value


async def start(world, content=TEXT, request_id=None):
    payload = ChatMessageRequest(content=content, request_id=request_id or uuid4())
    rid, wid, cached = await asyncio.to_thread(generation_service.reserve, world[1], payload)
    assert cached is None
    return generation_service.launch(rid, world[1], wid, payload), payload


async def finish(control):
    await asyncio.wait_for(control.done.wait(), 5)
    await control.task
    return control.terminal


def count(model, world):
    with SessionLocal() as db:
        column = model.world_id if hasattr(model, 'world_id') else model.conversation_id
        identity = world[0] if hasattr(model, 'world_id') else world[1]
        return db.query(model).filter(column == identity).count()


@pytest.mark.asyncio
async def test_success_events_atomic_persistence_and_terminal_replay(world, runtime):
    control, payload = await start(world)
    terminal = await finish(control)
    events = []
    while not control.events.empty():
        events.append(control.events.get_nowait())
    types = [event['type'] for event in events]
    assert types.index('answer_completed') < types.index('response_saved') < types.index('postprocessing_started')
    assert terminal['type'] == 'complete'
    assert count(Message, world) == 2 and count(Fact, world) == 1
    replay = generation_service.reserve(world[1], payload)[2]
    assert replay == terminal
    assert runtime.calls == 3
    assert not generation_service.controls


@pytest.mark.asyncio
@pytest.mark.parametrize('phase,saved', [('analyzing', False), ('generating', False), ('postprocessing', True)])
async def test_stop_closes_active_ollama_http_before_acknowledgement(world, runtime, phase, saved):
    runtime.block = phase
    control, _ = await start(world)
    await asyncio.wait_for(runtime.started.wait(), 3)
    if saved:
        assert count(Message, world) == 2
    await asyncio.to_thread(generation_service.cancel, world[1], control.request_id)
    terminal = await finish(control)
    assert runtime.closed.is_set(), 'The runtime connection must close before terminal cancellation'
    assert terminal['type'] == 'cancelled' and terminal['answer_saved'] is saved
    assert count(Message, world) == (2 if saved else 0)
    assert count(Fact, world) == 0
    generation_service.cancel(world[1], control.request_id)
    assert generation_service.status(world[1], control.request_id)['terminal'] == terminal


@pytest.mark.asyncio
async def test_cancel_before_submit_creates_no_work(world, runtime):
    rid = str(uuid4())
    generation_service.cancel(world[1], rid)
    generation_service.cancel(world[1], rid)
    payload = ChatMessageRequest(content=TEXT, request_id=rid)
    cached = generation_service.reserve(world[1], payload)[2]
    assert cached['type'] == 'cancelled'
    assert runtime.calls == 0 and count(Message, world) == 0


@pytest.mark.asyncio
async def test_duplicate_same_conversation_rejected_but_another_can_run(world, runtime):
    runtime.block = 'generating'
    first, payload = await start(world)
    await asyncio.wait_for(runtime.started.wait(), 3)
    with pytest.raises(HTTPException) as duplicate:
        generation_service.reserve(world[1], payload)
    assert duplicate.value.status_code == 409
    with pytest.raises(HTTPException) as overlapping:
        generation_service.reserve(world[1], ChatMessageRequest(content='Different request'))
    assert overlapping.value.status_code == 409
    with SessionLocal() as db:
        other = Conversation(world_id=world[0], title='Independent thread')
        db.add(other); db.commit(); other_id = other.id
    second, _ = await start((world[0], other_id))
    # Both have entered independent pipelines rather than being rejected or globally locked.
    await asyncio.sleep(0.1)
    assert len(generation_service.controls) == 2
    await asyncio.to_thread(generation_service.cancel, world[1], first.request_id)
    await asyncio.to_thread(generation_service.cancel, other_id, second.request_id)
    await finish(first); await finish(second)


@pytest.mark.asyncio
async def test_world_bible_updates_rollback_as_one_batch(world, runtime, monkeypatch):
    with SessionLocal() as db:
        db.add(Fact(world_id=world[0], title='Prior valid information', content='Planet Z is Atlas.')); db.commit()
    runtime.extraction = PostTurnExtractionSchema(
        bible_section_updates=[BibleSectionEditItem(section_slug='cosmology-physics', section_title='Cosmology', source_quote=TEXT, updated_markdown=TEXT)],
        proposals=[CandidateProposalItem(proposed_title='Planet X', proposed_content=TEXT, source_quote=TEXT)],
    )
    original = entity_service._project
    def fails_after_flush(*args, **kwargs):
        original(*args, **kwargs) # Real property/Fact/section writes have been flushed.
        raise RuntimeError('Injected persistence failure')
    monkeypatch.setattr(entity_service, '_project', fails_after_flush)
    control, _ = await start(world)
    terminal = await finish(control)
    assert terminal['type'] == 'error' and terminal['answer_saved']
    assert count(Fact, world) == 1 and count(Message, world) == 2
    with SessionLocal() as db:
        assert all(not section.markdown_content for section in db.query(BibleSection).filter(BibleSection.world_id == world[0]))
        assert db.query(Fact).filter(Fact.world_id == world[0]).one().content == 'Planet Z is Atlas.'


@pytest.mark.asyncio
async def test_runtime_timeout_and_overall_deadline_release_request(world, runtime, monkeypatch):
    runtime.fail = 'generating'
    control, _ = await start(world)
    terminal = await finish(control)
    assert terminal['type'] == 'error' and terminal['error_code'] == 'LOCAL_MODEL_TIMEOUT'
    assert count(Message, world) == 0
    runtime.fail = None; runtime.block = 'generating'
    monkeypatch.setenv('GENERATION_DEADLINE_SECONDS', '1')
    control, _ = await start(world)
    await asyncio.wait_for(runtime.started.wait(), 3)
    terminal = await finish(control)
    assert terminal['error_code'] == 'REQUEST_DEADLINE'
    assert runtime.closed.is_set()
    runtime.block = None
    control, _ = await start(world)
    assert (await finish(control))['type'] == 'complete'


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['generating', 'postprocessing'])
async def test_browser_disconnect_cancels_underlying_request(world, runtime, phase):
    runtime.block = phase
    control, _ = await start(world)
    connected = asyncio.Event()
    async def receive():
        await connected.wait()
        return {'type': 'http.disconnect'}
    async def send(message):
        pass
    response = GenerationResponse(_generation_events(control), control)
    task = asyncio.create_task(response({'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}}, receive, send))
    await asyncio.wait_for(runtime.started.wait(), 3)
    connected.set()
    await asyncio.wait_for(task, 3)
    assert (await finish(control))['type'] == 'cancelled'
    assert runtime.closed.is_set() and count(Fact, world) == 0


def test_http_duplicate_replay_and_cancellation_endpoints(world, runtime):
    client = TestClient(app)
    rid = str(uuid4())
    response = client.post(f'/api/conversations/{world[1]}/stream', json={'content': TEXT, 'request_id': rid})
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[-1]['type'] == 'complete'
    calls = runtime.calls
    replay = client.post(f'/api/conversations/{world[1]}/stream', json={'content': TEXT, 'request_id': rid})
    assert json.loads(replay.text[6:]) == events[-1]
    assert runtime.calls == calls
    assert client.post(f'/api/conversations/{world[1]}/requests/{rid}/cancel').json()['state'] == 'complete'
    assert client.post(f'/api/conversations/{world[1]}/stream', json={'content': 'Changed', 'request_id': rid}).status_code == 409


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_overwrite_the_terminal_journal(world, runtime):
    runtime.block = 'generating'
    control, _ = await start(world)
    await asyncio.wait_for(runtime.started.wait(), 3)
    await asyncio.gather(*(asyncio.to_thread(generation_service.cancel, world[1], control.request_id) for _ in range(10)))
    terminal = await finish(control)
    assert terminal['type'] == 'cancelled'
    assert generation_service.status(world[1], control.request_id)['state'] == 'cancelled'
    assert not generation_service.controls


@pytest.mark.asyncio
async def test_cancellation_after_atomic_commit_begins_observes_completed_commit(world, runtime, monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()
    original = entity_service._project
    def pause(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(3)
        return result
    monkeypatch.setattr(entity_service, '_project', pause)
    control, _ = await start(world)
    assert await asyncio.to_thread(entered.wait, 3)
    cancel = asyncio.create_task(asyncio.to_thread(generation_service.cancel, world[1], control.request_id))
    await asyncio.sleep(0.05)
    release.set()
    await cancel
    assert (await finish(control))['type'] == 'complete'
    assert count(Fact, world) == 1
    assert generation_service.status(world[1], control.request_id)['state'] == 'complete'


@pytest.mark.asyncio
@pytest.mark.parametrize('saved_answer', [False, True])
async def test_restart_recovery_releases_stale_requests_without_deleting_information(world, runtime, saved_answer):
    rid = str(uuid4())
    with SessionLocal() as db:
        journal = GenerationRequest(id=rid, world_id=world[0], conversation_id=world[1], active_conversation=world[1], state='postprocessing')
        db.add(journal)
        if saved_answer:
            user = Message(conversation_id=world[1], role='USER', content=TEXT)
            assistant = Message(conversation_id=world[1], role='ASSISTANT', content='Recorded Pallene.',
                                metadata_json=json.dumps({'request_id': rid, 'request_state': 'answer_saved'}))
            db.add_all([user, assistant]); db.flush()
            assistant_id = assistant.id
            journal.terminal_json = json.dumps({'type': 'response_saved', 'answer_saved': True,
                'user_message_id': user.id, 'assistant_message': {'id': assistant.id,
                'content': assistant.content, 'metadata': json.loads(assistant.metadata_json)}})
        db.add(Fact(world_id=world[0], title='Existing information', content='Keep this.'))
        db.commit()
    generation_service.recover_interrupted()
    status = generation_service.status(world[1], rid)
    assert status['state'] == 'error' and status['terminal']['error_code'] == 'SERVER_RESTARTED'
    assert count(Fact, world) == 1
    if saved_answer:
        assert status['terminal']['answer_saved']
        assert status['terminal']['assistant_message']['metadata']['request_state'] == 'error'
        with SessionLocal() as db:
            assert json.loads(db.get(Message, assistant_id).metadata_json)['request_state'] == 'error'
    control, _ = await start(world)
    assert (await finish(control))['type'] == 'complete'
