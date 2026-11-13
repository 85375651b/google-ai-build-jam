"""Request journal, cancellation ownership, and transaction-safe chat lifecycle."""
import asyncio
import hashlib
import json
import logging
import threading
import time
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import SessionLocal
from backend.models import BibleSection, Conversation, GenerationRequest, Message
from backend.services.agent_orchestrator import agent_orchestrator
from backend.services.author_intent import extraction_eligible
from backend.services.context_assembler import context_assembler
from backend.services.generation_control import GenerationCancelled, GenerationControl, current_generation
from backend.services.llm_service import LocalModelError, llm_service
from backend.services.world_service import world_service
from backend.services.persistence_reporting import PendingReplyFilter, pending_reply

logger = logging.getLogger(__name__)
TERMINAL = {'complete', 'cancelled', 'error'}


class ControlledSession(Session):
    """Existing helper commits become flushes only inside the owned atomic unit."""
    def __init__(self, *args, control, **kwargs):
        super().__init__(*args, **kwargs)
        self.control = control
        self.defer_commits = False

    def commit(self):
        self.control.check()
        if self.defer_commits:
            self.flush()
        else:
            super().commit()

    def flush(self, *args, **kwargs):
        self.control.check()
        return super().flush(*args, **kwargs)


class GenerationService:
    def __init__(self):
        self.controls = {}
        self.lock = threading.RLock()

    def reserve(self, conversation_id, payload):
        request_id = str(payload.request_id or uuid4())
        fingerprint = hashlib.sha256(json.dumps(
            [payload.content, payload.focused_subject_id], ensure_ascii=False,
        ).encode()).hexdigest()
        with SessionLocal() as db:
            conv = db.get(Conversation, conversation_id)
            if conv is None:
                raise HTTPException(404, 'Conversation not found.')
            previous = db.get(GenerationRequest, request_id)
            if previous is not None:
                return self._existing(previous, conversation_id, fingerprint)
            row = GenerationRequest(
                id=request_id, conversation_id=conversation_id, world_id=conv.world_id,
                active_conversation=conversation_id, fingerprint=fingerprint, state='accepted',
            )
            db.add(row)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                previous = db.get(GenerationRequest, request_id)
                if previous is not None:
                    return self._existing(previous, conversation_id, fingerprint)
                raise HTTPException(409, {'error_code': 'CONVERSATION_BUSY', 'message': 'This conversation already has an active request.'})
            return request_id, conv.world_id, None

    @staticmethod
    def _existing(row, conversation_id, fingerprint):
        if row.conversation_id != conversation_id or row.fingerprint not in (None, fingerprint):
            raise HTTPException(409, {'error_code': 'REQUEST_ID_CONFLICT', 'message': 'Request ID was already used for different content or conversation.'})
        if row.state not in TERMINAL:
            raise HTTPException(409, {'error_code': 'REQUEST_ACTIVE', 'message': 'This request is already being processed.', 'request_id': row.id})
        return row.id, row.world_id, json.loads(row.terminal_json)

    def launch(self, request_id, conversation_id, world_id, payload, streaming=True):
        control = GenerationControl(request_id, conversation_id, world_id)
        with self.lock:
            self.controls[request_id] = control
        control.task = asyncio.create_task(self._drive(control, payload, streaming))
        return control

    async def _drive(self, control, payload, streaming):
        settings.reload()
        deadline = time.monotonic() + settings.generation_deadline_seconds
        worker = asyncio.create_task(asyncio.to_thread(self._pipeline, control, payload, streaming))
        try:
            while not worker.done():
                await asyncio.wait({worker}, timeout=0.25)
                if worker.done():
                    break
                if time.monotonic() >= deadline:
                    await asyncio.to_thread(control.cancel, 'The overall request deadline was reached.', 'REQUEST_DEADLINE')
                elif not control.cancelled.is_set():
                    requested = await asyncio.to_thread(self._cancel_requested, control.request_id)
                    if requested:
                        await asyncio.to_thread(control.cancel)
            await worker
        except asyncio.CancelledError:
            await asyncio.to_thread(control.cancel, 'Server shutting down.')
            await asyncio.shield(worker)
            raise
        except Exception:
            logger.exception('Unable to finalize request journal %s', control.request_id)
            control.finish(dict(type='error', request_id=control.request_id,
                                conversation_id=control.conversation_id, world_id=control.world_id,
                                error_code='REQUEST_STATE_UNAVAILABLE',
                                message='The server could not finalize request state. Reload the conversation before retrying.'))
        finally:
            # Network task cleanup must finish before cancellation is acknowledged.
            await control.drain_io()
            control.done.set()
            with self.lock:
                self.controls.pop(control.request_id, None)

    @staticmethod
    def _cancel_requested(request_id):
        with SessionLocal() as db:
            row = db.get(GenerationRequest, request_id)
            return row is None or row.cancel_requested

    @staticmethod
    def _phase(db, control, state, event=None, **data):
        control.check()
        row = db.get(GenerationRequest, control.request_id)
        if row.cancel_requested:
            control.cancel()
            control.check()
        row.state = state
        db.commit()
        control.state = state
        control.emit(event or state, stage=state, **data)

    @staticmethod
    def _answer(user, assistant, meta):
        return {
            'answer_saved': True, 'user_message_id': user.id,
            'assistant_message': {
                'id': assistant.id, 'role': 'ASSISTANT', 'content': assistant.content,
                'interaction_mode': assistant.interaction_mode, 'permission_level': assistant.permission_level,
                'metadata': dict(meta), 'created_at': assistant.created_at.isoformat(),
            },
        }

    @staticmethod
    def _commit(db, control):
        # Cancellation cannot interrupt a database commit midway. Once entered,
        # this short boundary either commits all its valid writes or rolls back.
        with control.lock:
            control.check()
            Session.commit(db)

    def _pipeline(self, control, payload, streaming):
        token = current_generation.set(control)
        with SessionLocal() as base:
            bind = base.get_bind()
        db = ControlledSession(bind=bind, expire_on_commit=False, control=control)
        try:
            control.check()
            conv = db.get(Conversation, control.conversation_id)
            world = world_service.get_world(db, control.world_id)
            from backend.models import Entity
            initial_entity_versions = dict(db.query(Entity.id, Entity.version_number).filter(Entity.world_id == world.id).all())
            self._phase(db, control, 'accepted', 'request_started',
                        client_idle_seconds=settings.generation_client_idle_seconds,
                        deadline_seconds=settings.generation_deadline_seconds)
            self._phase(db, control, 'analyzing', 'progress', message='Analyzing your message…')
            analysis = agent_orchestrator.analyze_turn(db, world, payload.content, conversation=conv, persist_expertise=False)
            control.check()
            self._phase(db, control, 'retrieving', 'progress', message='Consulting reference sources…')
            knowledge = agent_orchestrator.retrieve_knowledge(db, analysis, user_message=payload.content)
            control.emit('turn_meta', primary_intent=analysis.primary_intent,
                         permission_level=analysis.creative_permission.level,
                         naming_mode=analysis.creative_permission.naming_mode,
                         active_domains=analysis.active_domains, book_coverage=knowledge['book']['coverage_status'])
            system, contents, _ = context_assembler.assemble_consultation(
                db, world, analysis, knowledge['book'], knowledge['online'], conv, payload.content,
                focused_subject_id=payload.focused_subject_id or conv.focused_subject_id,
            )
            self._phase(db, control, 'generating', 'generation_started')
            chunks = []
            filtered = PendingReplyFilter() if extraction_eligible(analysis) else None
            if filtered and streaming:
                control.emit('delta', text='World Bible changes are pending backend validation.\n')
            if streaming:
                for chunk in llm_service.stream_text(contents=contents, system_instruction=system, temperature=0.4):
                    control.check()
                    chunks.append(chunk)
                    visible = filtered.feed(chunk) if filtered else chunk
                    if visible:
                        control.emit('delta', text=visible)
            else:
                chunks.append(llm_service.generate_text(contents=contents, system_instruction=system, temperature=0.4))
            control.check()
            sanitized = agent_orchestrator.sanitize_response_and_verify_citations(''.join(chunks), knowledge['book'], knowledge['online'])
            raw_reply = sanitized['content']
            if filtered:
                sanitized['content'] = pending_reply(raw_reply)
            control.emit('answer_completed', text=sanitized['content'])
            self._phase(db, control, 'saving', 'saving_started')
            db.defer_commits = True
            user = Message(conversation_id=conv.id, role='USER', content=payload.content.strip(),
                           interaction_mode=analysis.primary_intent, permission_level=analysis.creative_permission.level)
            meta = {
                'request_id': control.request_id, 'request_state': 'answer_saved',
                'primary_intent': analysis.primary_intent, 'secondary_intents': analysis.secondary_intents,
                'permission_level': analysis.creative_permission.level, 'naming_mode': analysis.creative_permission.naming_mode,
                'active_domains': analysis.active_domains, 'book_coverage': knowledge['book']['coverage_status'],
                'online_research_status': knowledge['online']['status'], 'citations': sanitized['citations'],
            }
            assistant = Message(conversation_id=conv.id, role='ASSISTANT', content=sanitized['content'],
                                interaction_mode=analysis.primary_intent, permission_level=analysis.creative_permission.level,
                                metadata_json=json.dumps(meta))
            db.add_all([user, assistant])
            if analysis.expertise_signals:
                world_service.apply_expertise_signals(db, world.id, analysis.expertise_signals)
            db.flush()
            saved_answer = self._answer(user, assistant, meta)
            journal = db.get(GenerationRequest, control.request_id)
            journal.terminal_json = json.dumps(dict(saved_answer, type='response_saved', request_id=control.request_id,
                                                    conversation_id=control.conversation_id, world_id=control.world_id))
            self._commit(db, control)
            db.defer_commits = False
            control.answer = saved_answer
            control.emit('response_saved', **control.answer)
            needs_summary = db.query(Message).filter(Message.conversation_id == conv.id).count() > 12
            needs_canon = extraction_eligible(analysis)
            summary = None
            prepared = None
            if needs_summary or needs_canon:
                self._phase(db, control, 'postprocessing', 'postprocessing_started', world_bible=needs_canon,
                            message='Updating World Bible…' if needs_canon else 'Summarizing conversation…')
                if needs_summary:
                    summary = agent_orchestrator.update_rolling_summary(db, conv, persist=False)
                if needs_canon:
                    from types import SimpleNamespace
                    extraction_reply = SimpleNamespace(id=assistant.id, content=raw_reply)
                    prepared = agent_orchestrator.prepare_post_turn_extraction(db, world, conv, user, extraction_reply, analysis)
                    if prepared is not None:
                        prepared['entity_versions'] = initial_entity_versions
            control.check()
            self._phase(db, control, 'committing', 'progress', message='Finishing saved updates…')
            # Release read snapshots. No inference occurs while the write lock is held.
            user_id, assistant_id = user.id, assistant.id
            db.rollback()
            db.defer_commits = True
            with control.lock:
                control.check()
                if db.bind.dialect.name == 'sqlite':
                    db.execute(text('BEGIN IMMEDIATE'))
                journal = db.get(GenerationRequest, control.request_id)
                if journal.cancel_requested:
                    control.cancel()
                    control.check()
                conv = db.get(Conversation, control.conversation_id)
                world = world_service.get_world(db, control.world_id)
                user, assistant = db.get(Message, user_id), db.get(Message, assistant_id)
                if prepared is not None:
                    actual = {s.id: s.version_number for s in db.query(BibleSection).filter(BibleSection.world_id == world.id)}
                    if actual != prepared['section_versions']:
                        raise LocalModelError('The World Bible changed while this request was processing. Retry against the latest version.', 409, 'WORLD_CHANGED')
                    updates = agent_orchestrator.extract_post_turn_proposals(
                        db, world, conv, user, extraction_reply, analysis, prepared=prepared, manage_summary=False,
                    )
                else:
                    updates = {'staged_proposals': [], 'updated_sections': [], 'detected_contradictions': []}
                if needs_summary:
                    conv.rolling_summary = summary
                meta['request_state'] = 'complete'
                if updates.get('canon_result'):
                    meta['canon_result'] = updates['canon_result']
                    assistant.content = sanitized['content'] + '\n\n' + updates['canon_result']['message']
                elif needs_canon:
                    assistant.content = sanitized['content'] + ('\n\nThe changes remain proposals; no fictional canon was changed.'
                        if updates['staged_proposals'] else '\n\nWorld Bible updates committed.' if updates['updated_sections']
                        else '\n\nNo World Bible change was confirmed.')
                if updates['updated_sections']:
                    meta['updated_bible_sections'] = updates['updated_sections']
                if updates['detected_contradictions']:
                    meta['detected_contradictions'] = updates['detected_contradictions']
                assistant.metadata_json = json.dumps(meta)
                event = dict(self._answer(user, assistant, meta),
                             type='complete', request_id=control.request_id, conversation_id=control.conversation_id,
                             world_id=control.world_id, updated_bible_sections=updates['updated_sections'],
                             detected_contradictions=updates['detected_contradictions'],
                             staged_proposals=[{'id': p.id, 'proposed_title': p.proposed_title,
                                                'proposed_content': p.proposed_content,
                                                'proposed_provenance': p.proposed_provenance} for p in updates['staged_proposals']])
                row = db.get(GenerationRequest, control.request_id)
                row.state, row.active_conversation, row.terminal_json = 'complete', None, json.dumps(event)
                Session.commit(db)
                control.finish(event)
            # Terminal delivery is handled after I/O cleanup by the async stream.
        except GenerationCancelled:
            db.rollback()
            self._fail(control, 'error' if control.error_code else 'cancelled', control.reason, control.error_code)
        except Exception as exc:
            db.rollback()
            logger.exception('Generation %s failed in %s', control.request_id, control.state)
            self._fail(control, 'error', exc.message if isinstance(exc, LocalModelError) else 'The request failed while saving or processing its response. No incomplete World Bible update was committed.',
                       exc.error_code if isinstance(exc, LocalModelError) else 'REQUEST_FAILED')
        finally:
            db.close()
            current_generation.reset(token)

    @staticmethod
    def _fail(control, state, message, error_code=None):
        with SessionLocal() as db:
            from backend.models import CanonOperationBatch
            pending = getattr(control, 'pending_canon_batch', None)
            outcome = None
            if pending:
                outcome = {'status': 'clarification_required' if error_code in {'CLARIFICATION_REQUIRED', 'AMBIGUOUS_ENTITY'} else state,
                    'message':message, 'operations':[{'index':i, 'state':state} for i in range(len(json.loads(pending['operations_json'])))]}
                prior = db.query(CanonOperationBatch).filter_by(world_id=control.world_id, idempotency_key=pending['idempotency_key']).first()
                if prior is None:
                    db.add(CanonOperationBatch(**pending, fingerprint=hashlib.sha256(pending['operations_json'].encode()).hexdigest(),
                        status=outcome['status'], result_json=json.dumps(outcome)))
            event = dict(control.answer or {}, type=state, request_id=control.request_id,
                         conversation_id=control.conversation_id, world_id=control.world_id,
                         message=message, error_code=error_code, answer_saved=control.answer is not None)
            if control.answer:
                assistant = db.get(Message, control.answer['assistant_message']['id'])
                if assistant:
                    meta = json.loads(assistant.metadata_json)
                    meta['request_state'] = state
                    if outcome:
                        meta['canon_result'] = outcome
                    assistant.metadata_json = json.dumps(meta)
                    if state == 'error':
                        assistant.content = 'I could not commit the World Bible changes. Existing canon was preserved. ' + message
                        event['assistant_message'] = dict(event['assistant_message'], content=assistant.content)
                    event['assistant_message'] = dict(event['assistant_message'], metadata=meta)
            row = db.get(GenerationRequest, control.request_id)
            row.state, row.active_conversation, row.terminal_json = state, None, json.dumps(event)
            db.commit()
            control.finish(event)

    def cancel(self, conversation_id, request_id):
        # Signal live transport first; do not wait for a database lock to cancel I/O.
        with self.lock:
            control = self.controls.get(request_id)
        if control is not None and control.conversation_id == conversation_id:
            control.cancel()
        with SessionLocal() as db:
            conv = db.get(Conversation, conversation_id)
            if conv is None:
                raise HTTPException(404, 'Conversation not found.')
            row = db.get(GenerationRequest, request_id)
            if row is not None and row.conversation_id != conversation_id:
                raise HTTPException(404, 'Request not found in this conversation.')
            if row is None:
                # Stop may arrive before the submit POST. This tombstone prevents a late start.
                event = dict(type='cancelled', request_id=request_id, conversation_id=conversation_id,
                             world_id=conv.world_id, answer_saved=False, message='Cancelled before generation started.')
                row = GenerationRequest(id=request_id, conversation_id=conversation_id, world_id=conv.world_id,
                                        state='cancelled', cancel_requested=True, terminal_json=json.dumps(event))
                db.add(row)
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    return self.cancel(conversation_id, request_id)
            elif row.state not in TERMINAL:
                db.query(GenerationRequest).filter(
                    GenerationRequest.id == request_id, GenerationRequest.state.notin_(TERMINAL),
                ).update({'cancel_requested': True, 'state': 'cancelling'}, synchronize_session=False)
                db.commit()
        return control

    @staticmethod
    def status(conversation_id, request_id):
        with SessionLocal() as db:
            row = db.get(GenerationRequest, request_id)
            if row is None or row.conversation_id != conversation_id:
                raise HTTPException(404, 'Request not found.')
            return {'request_id': row.id, 'conversation_id': row.conversation_id, 'world_id': row.world_id,
                    'state': row.state, 'terminal': json.loads(row.terminal_json) if row.state in TERMINAL and row.terminal_json else None}

    @staticmethod
    def recover_interrupted():
        with SessionLocal() as db:
            for row in db.query(GenerationRequest).filter(GenerationRequest.active_conversation.is_not(None)):
                saved = json.loads(row.terminal_json) if row.terminal_json else {}
                event = dict(saved, type='error', request_id=row.id, conversation_id=row.conversation_id, world_id=row.world_id,
                             error_code='SERVER_RESTARTED', message='The server restarted before this request finished. Reload the conversation before retrying.')
                if saved.get('assistant_message'):
                    assistant = db.get(Message, saved['assistant_message']['id'])
                    if assistant:
                        meta = json.loads(assistant.metadata_json)
                        meta['request_state'] = 'error'
                        assistant.metadata_json = json.dumps(meta)
                        event['assistant_message']['metadata'] = meta
                row.state, row.active_conversation, row.terminal_json = 'error', None, json.dumps(event)
            db.commit()

    async def shutdown(self):
        with self.lock:
            controls = list(self.controls.values())
        for control in controls:
            await asyncio.to_thread(control.cancel, 'Server shutting down.')
        if controls:
            await asyncio.gather(*(control.task for control in controls), return_exceptions=True)


generation_service = GenerationService()
