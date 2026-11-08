"""End-to-end integration and behavioral tests for Step 3 & Step 4 MVP."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import CreativePermissionSchema, TurnAnalysisSchema
from backend.services.agent_orchestrator import agent_orchestrator
from backend.services.gemini_service import gemini_service

client = TestClient(app)


def test_system_status_and_ingested_reference_book():
    """Verify system status endpoint and automatic ingestion of The Planet Construction Kit."""
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert "has_valid_gemini_key" in data
    assert len(data["reference_documents"]) >= 1
    book = data["reference_documents"][0]
    assert book["title"] == "The Planet Construction Kit"
    assert book["ingestion_status"] == "INDEXED"
    assert book["total_chunks"] > 300


def test_project_isolation_bible_sync_proposals_and_exports():
    """Verify separate worlds do not share canon, direct Bible editing, proposals, and exports."""
    # Create World A and World B
    w1 = client.post(
        "/api/worlds",
        json={"name": "Thalassia", "description": "Archipelago republic", "seed_default_subjects": True},
    ).json()
    w2 = client.post(
        "/api/worlds",
        json={"name": "Varnic Plateau", "description": "Arid highland", "seed_default_subjects": True},
    ).json()

    # Edit a World Bible section directly in World A
    secs1 = client.get(f"/api/worlds/{w1['id']}/bible/sections").json()
    pol_sec = next(s for s in secs1 if "Politics" in s["title"])
    upd_res = client.put(
        f"/api/worlds/{w1['id']}/bible/sections/{pol_sec['id']}",
        json={
            "title": pol_sec["title"],
            "markdown_content": "Only citizens who complete five years of naval service may vote.",
            "expected_version_number": pol_sec["version_number"],
        },
    )
    assert upd_res.status_code == 200

    # Verify fact was created with AUTHOR_CREATED provenance in World A, and NOT in World B
    facts1 = client.get(f"/api/worlds/{w1['id']}/facts").json()
    facts2 = client.get(f"/api/worlds/{w2['id']}/facts").json()
    assert len(facts1) == 1
    assert facts1[0]["provenance_type"] == "AUTHOR_CREATED"
    assert len(facts2) == 0

    # Decision 3(b): Deleting paragraph in World Bible prompts for orphaned fact resolution
    del_para_res = client.put(
        f"/api/worlds/{w1['id']}/bible/sections/{pol_sec['id']}",
        json={
            "title": pol_sec["title"],
            "markdown_content": "",
        },
    ).json()
    assert len(del_para_res["orphaned_facts_requiring_confirmation"]) == 1

    # Restore via revision history
    rev_res = client.post(
        f"/api/worlds/{w1['id']}/bible/sections/{pol_sec['id']}/revert/2"
    )
    assert rev_res.status_code == 200
    assert "five years of naval service" in rev_res.json()["markdown_content"]

    # Verify Markdown, PDF, and DOCX exports
    md_exp = client.get(f"/api/worlds/{w1['id']}/bible/export?fmt=md")
    assert md_exp.status_code == 200
    assert "five years of naval service" in md_exp.text

    pdf_exp = client.get(f"/api/worlds/{w1['id']}/bible/export?fmt=pdf")
    assert pdf_exp.status_code == 200
    assert pdf_exp.content.startswith(b"%PDF")

    docx_exp = client.get(f"/api/worlds/{w1['id']}/bible/export?fmt=docx")
    assert docx_exp.status_code == 200
    assert len(docx_exp.content) > 500


def test_creative_permission_guard_and_citation_verification(monkeypatch):
    """Verify Guard 1 (Permission Downgrade on non-brainstorm prompts) & Guard 5 (No Silent Canonization)."""
    w = client.post(
        "/api/worlds",
        json={"name": "Keth", "description": "Test World", "seed_default_subjects": True},
    ).json()
    convs = client.get(f"/api/worlds/{w['id']}/conversations").json()
    cid = convs[0]["id"]

    # Mock Gemini structured & text calls to verify orchestrator guards deterministically
    def fake_structured(prompt, response_schema, **kwargs):
        if response_schema.__name__ == "TurnAnalysisSchema":
            # Simulate an over-eager model trying to set BOUNDED_BRAINSTORMING on a factual question
            return TurnAnalysisSchema(
                primary_intent="QUESTION_EXPLANATION",
                secondary_intents=[],
                ambiguity_status="CLEAR",
                raw_user_phrasing="How did medieval taxation work?",
                inferred_domain_concepts=["medieval taxation", "feudal aids", "corvee"],
                creative_permission=CreativePermissionSchema(
                    level="BOUNDED_BRAINSTORMING",
                    allowed_targets=["taxation"],
                ),
                active_domains=["economics", "history"],
                requires_factual_retrieval=True,
                search_queries=["medieval taxation tribute corvee"],
            )
        raise AssertionError("Pass 3 should be blocked by Guard 5 on QUESTION_EXPLANATION")

    monkeypatch.setattr(gemini_service, "generate_structured", fake_structured)
    monkeypatch.setattr(
        gemini_service,
        "generate_text",
        lambda contents, system_instruction, **kw: (
            "In pre-modern states, rulers collected taxes in kind or via labor [Book-p112].\n\n"
            "What would you like to explore next?"
        ),
    )

    res = client.post(
        f"/api/conversations/{cid}/messages",
        json={"content": "How did medieval taxation work?"},
    )
    assert res.status_code == 200
    body = res.json()
    # Guard 1 downgraded BOUNDED_BRAINSTORMING -> IMPLEMENTATION_SCOPED because user didn't ask for options
    assert body["assistant_message"]["permission_level"] == "IMPLEMENTATION_SCOPED"
    # Guard 4 stripped generic closing hook "What would you like to explore next?"
    assert "What would you like to explore next?" not in body["assistant_message"]["content"]
    # Guard 5 ensured zero fictional canon proposals were created from a factual research question
    assert body["staged_proposals"] == []
