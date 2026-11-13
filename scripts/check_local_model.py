"""Smoke test a running local model without changing the author's worlds."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    with tempfile.TemporaryDirectory(prefix="worldbuilding-local-smoke-") as directory:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(directory) / 'smoke.db'}"
        os.environ["TAVILY_API_KEY"] = ""
        from backend.database import engine
        from backend.main import app
        from backend.services.llm_service import llm_service
        from fastapi.testclient import TestClient

        status = llm_service.get_status()
        if not status["ready"]:
            raise RuntimeError(status["message"])
        successful_schemas = set()
        original_structured = llm_service.generate_structured

        def record_structured(*args, **kwargs):
            started = time.monotonic()
            result = original_structured(*args, **kwargs)
            successful_schemas.add(type(result).__name__)
            print(f"Validated {type(result).__name__} in {time.monotonic() - started:.1f}s", flush=True)
            return result

        llm_service.generate_structured = record_structured
        try:
            with TestClient(app) as client:
                world_response = client.post("/api/worlds", json={"name": "Local inference check", "seed_default_subjects": True})
                world_response.raise_for_status()
                world = world_response.json()
                conv = client.get(f"/api/worlds/{world['id']}/conversations").json()[0]
                print("Running analysis, consultation, and extraction with the local model…", flush=True)
                response = client.post(f"/api/conversations/{conv['id']}/stream", json={
                    "content": "My world has exactly three inhabited islands. Record this author-established fact in the Geography and Geology section of the World Bible. Keep your reply under 50 words; add no invented details."
                })
                response.raise_for_status()
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                errors = [event for event in events if event["type"] == "error"]
                if errors:
                    raise RuntimeError(errors[-1]["message"])
                if not any(event["type"] == "delta" for event in events) or not any(event["type"] == "complete" for event in events):
                    raise RuntimeError("Stream did not complete")
                sections = client.get(f"/api/worlds/{world['id']}/bible/sections").json()
                geography = next(section for section in sections if section["slug"] == "geography-geology")
                text = geography["markdown_content"].lower()
                if "island" not in text or not ("three" in text or "3" in text):
                    raise RuntimeError("The real model did not save the island count to the World Bible")
                if not {"TurnAnalysisSchema", "PostTurnExtractionSchema"}.issubset(successful_schemas):
                    raise RuntimeError("Structured generation fell back to heuristics; inspect model compatibility")
                print("PASS: streamed reply completed and the World Bible contains the author's three islands.")
        finally:
            llm_service.generate_structured = original_structured
            engine.dispose()


if __name__ == "__main__":
    main()
