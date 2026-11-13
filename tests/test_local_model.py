"""Test local HTTP generation and the streamed World Bible update workflow."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend import config
from backend.config import settings
from backend.main import app
from backend.schemas import BibleSectionEditItem, CreativePermissionSchema, PostTurnExtractionSchema, TurnAnalysisSchema
from backend.services.llm_service import LocalModelError, llm_service


@pytest.fixture
def mock_ollama(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENV_FILE_PATH", tmp_path / ".env")
    monkeypatch.setenv("LOCAL_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("LOCAL_CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("LOCAL_MAX_OUTPUT_TOKENS", "4096")
    settings.reload()
    def install(handler):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(llm_service, "_client", lambda timeout=None: httpx.Client(
            base_url=settings.ollama_base_url, transport=transport,
        ))
        monkeypatch.setattr(llm_service, "_async_client", lambda: httpx.AsyncClient(
            base_url=settings.ollama_base_url, transport=transport,
        ))
    return install


def reply(text):
    return httpx.Response(200, json={"message": {"content": text}, "done": True})


def stream(events):
    return httpx.Response(200, content="\n".join(json.dumps(event) for event in events))


class Extracted(BaseModel):
    name: str
    count: int


def test_local_requests_and_schema_validation(mock_ollama):
    requests = []
    def handler(request):
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        assert "authorization" not in request.headers and "x-api-key" not in request.headers
        payload = json.loads(request.content)
        requests.append(payload)
        return reply('{"name":"islands","count":3}' if "format" in payload else "Local answer")
    mock_ollama(handler)
    assert llm_service.generate_text("Question", system_instruction="Respect canon", max_output_tokens=1000) == "Local answer"
    assert llm_service.generate_structured("Extract", Extracted).count == 3
    assert all(p["model"] == "gpt-oss:20b" and p["think"] == "low" for p in requests)
    assert requests[0]["messages"][0] == {"role": "system", "content": "Respect canon"}
    assert requests[0]["options"]["num_predict"] == 1000
    assert requests[1]["format"] == Extracted.model_json_schema()


@pytest.mark.parametrize("text", ["invalid JSON", '{"name":"islands","count":"invalid"}'])
def test_invalid_structured_output(mock_ollama, text):
    mock_ollama(lambda request: reply(text))
    with pytest.raises(LocalModelError) as error:
        llm_service.generate_structured("Extract", Extracted)
    assert error.value.error_code == "LOCAL_STRUCTURED_OUTPUT_INVALID"


def test_stream_excludes_reasoning(mock_ollama):
    mock_ollama(lambda request: stream([
        {"message": {"thinking": "internal reasoning"}, "done": False},
        {"message": {"content": "Hello "}, "done": False},
        {"message": {"content": "author"}, "done": True},
    ]))
    assert list(llm_service.stream_text("Hello")) == ["Hello ", "author"]


@pytest.mark.parametrize("events", [
    [{"message": {"content": "partial"}, "done": False}],
    [{"error": "out of memory"}],
    [{"message": {"content": ""}, "done": True}],
    [{"message": {"content": "partial"}, "done": True, "done_reason": "length"}],
])
def test_interrupted_empty_and_truncated_streams_fail(mock_ollama, events):
    mock_ollama(lambda request: stream(events))
    with pytest.raises(LocalModelError):
        list(llm_service.stream_text("Hello"))


def test_unavailable_model_runtime_timeout_and_context(mock_ollama):
    mock_ollama(lambda request: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LocalModelError, match="ollama pull") as error:
        llm_service.generate_text("Hello")
    assert error.value.error_code == "LOCAL_MODEL_NOT_FOUND"
    def unavailable(request):
        raise httpx.ConnectError("refused", request=request)
    mock_ollama(unavailable)
    assert llm_service.get_status()["ready"] is False
    with pytest.raises(LocalModelError, match="ollama serve"):
        llm_service.generate_text("Hello")
    def timeout(request):
        raise httpx.ReadTimeout("slow model", request=request)
    mock_ollama(timeout)
    with pytest.raises(LocalModelError) as error:
        llm_service.generate_text("Hello")
    assert error.value.status_code == 504
    with pytest.raises(LocalModelError) as error:
        llm_service.generate_text("canon " * 20000)
    assert error.value.error_code == "LOCAL_CONTEXT_LIMIT"


def test_status_model_selection_and_persistence(mock_ollama, monkeypatch):
    config.ENV_FILE_PATH.write_text("TAVILY_API_KEY='preserve-this-value'\n")
    mock_ollama(lambda request: httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}]}))
    assert llm_service.get_status()["ready"] is True
    client = TestClient(app)
    assert client.put("/api/system/model", json={"local_model": "other:7b"}).status_code == 200
    assert "TAVILY_API_KEY='preserve-this-value'" in config.ENV_FILE_PATH.read_text()
    monkeypatch.delenv("LOCAL_MODEL")
    settings.reload()
    assert settings.local_model == "other:7b"
    assert llm_service.get_status()["ready"] is False
    assert client.put("/api/system/model", json={"local_model": " "}).status_code == 422


def test_full_streamed_canon_update(mock_ollama):
    analysis = TurnAnalysisSchema(
        primary_intent="CANON_DECLARATION", creative_permission=CreativePermissionSchema(level="IMPLEMENTATION_SCOPED"),
        active_domains=["geography_geology"], requires_factual_retrieval=False, search_queries=["islands"],
    )
    extraction = PostTurnExtractionSchema(bible_section_updates=[BibleSectionEditItem(
        section_slug="geography-geology", section_title="Geography and Geology",
        updated_markdown="The world has exactly three islands.",
        source_quote="My world has three islands.",
    )])
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "format" in payload:
            return reply(analysis.model_dump_json() if payload["format"]["title"] == "TurnAnalysisSchema" else extraction.model_dump_json())
        return stream([{"message": {"content": "Recorded three islands."}, "done": True}])
    mock_ollama(handler)
    client = TestClient(app)
    world = client.post("/api/worlds", json={"name": "Local test", "seed_default_subjects": True}).json()
    try:
        conv = client.get(f"/api/worlds/{world['id']}/conversations").json()[0]
        response = client.post(f"/api/conversations/{conv['id']}/stream", json={"content": "My world has three islands. Add that to the World Bible."})
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        event_types = [event["type"] for event in events]
        assert event_types.index("answer_completed") < event_types.index("postprocessing_started") < event_types.index("complete")
        assert "delta" in event_types and "response_saved" in event_types
        assert len(requests) == 3
        assert all(payload["model"] == "gpt-oss:20b" for payload in requests)
        sections = client.get(f"/api/worlds/{world['id']}/bible/sections").json()
        assert any("three islands" in section["markdown_content"] for section in sections)
    finally:
        client.delete(f"/api/worlds/{world['id']}")


def test_stream_failure_does_not_save_partial_messages(mock_ollama):
    mock_ollama(lambda request: httpx.Response(404, json={"error": "model not found"}))
    client = TestClient(app)
    world = client.post("/api/worlds", json={"name": "Unavailable model", "seed_default_subjects": True}).json()
    try:
        conv = client.get(f"/api/worlds/{world['id']}/conversations").json()[0]
        response = client.post(f"/api/conversations/{conv['id']}/stream", json={"content": "Hello"})
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1]["error_code"] == "LOCAL_MODEL_NOT_FOUND"
        assert not any(event["type"] == "complete" for event in events)
        assert client.get(f"/api/conversations/{conv['id']}/messages").json() == []
    finally:
        client.delete(f"/api/worlds/{world['id']}")


def test_runtime_metrics_exclude_content(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger='backend.services.llm_service'):
        llm_service._record_metrics({'done': True, 'prompt_eval_count': 17,
            'prompt_eval_duration': 123, 'message': {'content': 'private manuscript'}})
    assert '"prompt_eval_count": 17' in caplog.text
    assert 'private manuscript' not in caplog.text
    assert '"eval_count": null' in caplog.text


def test_optional_classifier_falls_back_without_changing_schema(mock_ollama, monkeypatch):
    monkeypatch.setenv('LOCAL_CLASSIFICATION_MODEL', 'llama3.2:1b')
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return reply('invalid' if payload['model'] == 'llama3.2:1b' else '{"name":"ok","count":1}')
    mock_ollama(handler)
    assert llm_service.generate_structured('Extract', Extracted, task='classification').count == 1
    assert [p['model'] for p in requests] == ['llama3.2:1b', 'gpt-oss:20b']
    assert requests[0]['messages'] == requests[1]['messages']
    assert requests[0]['format'] == requests[1]['format']
