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

    client.delete(f"/api/worlds/{w1['id']}")
    client.delete(f"/api/worlds/{w2['id']}")


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

    client.delete(f"/api/worlds/{w['id']}")


def test_brain_dump_automatically_edits_world_bible_sections(monkeypatch):
    """Verify pasting a brain dump extracts details across domains and directly edits World Bible sections."""
    from backend.schemas import (
        BibleSectionEditItem,
        CandidateProposalItem,
        PostTurnExtractionSchema,
    )

    w = client.post(
        "/api/worlds",
        json={"name": "Ashfall", "description": "Caldera world", "seed_default_subjects": True},
    ).json()
    convs = client.get(f"/api/worlds/{w['id']}/conversations").json()
    cid = convs[0]["id"]

    def fake_structured(prompt, response_schema, **kwargs):
        if response_schema.__name__ == "TurnAnalysisSchema":
            return TurnAnalysisSchema(
                primary_intent="CANON_DECLARATION",
                secondary_intents=["WORLD_BIBLE_MANAGEMENT"],
                ambiguity_status="CLEAR",
                raw_user_phrasing="Vaelor is built inside a volcanic caldera and priests control the grain silos",
                inferred_domain_concepts=["caldera settlement", "temple granary"],
                creative_permission=CreativePermissionSchema(level="ANALYTICAL_ADVISORY"),
                active_domains=["geography_geology", "economics"],
                requires_factual_retrieval=False,
                search_queries=["caldera"],
            )
        return PostTurnExtractionSchema(
            bible_section_updates=[
                BibleSectionEditItem(
                    section_slug="geography-geology",
                    section_title="Geography and Geology",
                    updated_markdown="### Vaelor Caldera\n\nThe capital city of Vaelor is built inside an active volcanic caldera.",
                    summary_of_changes="Added Vaelor caldera geography from brain dump",
                    provenance="AUTHOR_CREATED",
                ),
                BibleSectionEditItem(
                    section_slug="economics-trade",
                    section_title="Economics and Trade",
                    updated_markdown="### Temple Granaries & Currency\n\nThe priesthood controls all central grain silos and issues obsidian coins.",
                    summary_of_changes="Added temple granaries and obsidian currency from brain dump",
                    provenance="AUTHOR_CREATED",
                ),
            ],
            proposals=[
                CandidateProposalItem(
                    proposal_type="NEW_FACT",
                    proposed_title="Vaelor Caldera Location",
                    proposed_content="The capital city of Vaelor is built inside a volcanic caldera.",
                    proposed_provenance="AUTHOR_CREATED",
                    subject_slug="geography-geology",
                    auto_apply_to_bible=True,
                )
            ],
        )

    monkeypatch.setattr(gemini_service, "generate_structured", fake_structured)
    monkeypatch.setattr(
        gemini_service,
        "generate_text",
        lambda contents, system_instruction, **kw: "I have extracted your brain dump and updated the Geography and Economics sections of your World Bible.",
    )

    res = client.post(
        f"/api/conversations/{cid}/messages",
        json={"content": "Brain dump: Vaelor is built inside a volcanic caldera and priests control the grain silos and use obsidian coins."},
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body["updated_bible_sections"]) == 2

    # Confirm the World Bible sections were directly edited and persisted
    secs = client.get(f"/api/worlds/{w['id']}/bible/sections").json()
    geo_sec = next(s for s in secs if s["slug"] == "geography-geology")
    econ_sec = next(s for s in secs if s["slug"] == "economics-trade")
    assert "Vaelor is built inside an active volcanic caldera" in geo_sec["markdown_content"]
    assert "obsidian coins" in econ_sec["markdown_content"]

    client.delete(f"/api/worlds/{w['id']}")


def test_rolling_summary_dependencies_contradictions_and_multi_thread(monkeypatch):
    """Verify rolling conversation summary (>12 messages), Pass 3 dependency/contradiction wiring, and thread management."""
    from backend.database import SessionLocal
    from backend.models import Conversation, Message
    from backend.schemas import (
        CandidateProposalItem,
        DetectedContradictionItem,
        PostTurnExtractionSchema,
    )
    from backend.services.context_assembler import context_assembler

    w = client.post(
        "/api/worlds",
        json={"name": "TestRollingAndConsistency", "description": "Test World", "seed_default_subjects": True},
    ).json()

    try:
        # 1. Verify Multiple Conversation Threads (Create, Rename, Switch, Archive)
        convs_init = client.get(f"/api/worlds/{w['id']}/conversations").json()
        assert len(convs_init) == 1
        c1_id = convs_init[0]["id"]

        c2 = client.post(
            f"/api/worlds/{w['id']}/conversations",
            json={"title": "Linguistics Consultation"},
        ).json()
        assert c2["title"] == "Linguistics Consultation"

        rename_res = client.put(
            f"/api/conversations/{c2['id']}",
            json={"title": "Highland Phonology Thread"},
        ).json()
        assert rename_res["title"] == "Highland Phonology Thread"

        convs_two = client.get(f"/api/worlds/{w['id']}/conversations").json()
        assert len(convs_two) == 2

        arch_res = client.put(
            f"/api/conversations/{c2['id']}",
            json={"is_archived": True},
        ).json()
        assert arch_res["is_archived"] is True
        convs_after_arch = client.get(f"/api/worlds/{w['id']}/conversations").json()
        assert len(convs_after_arch) == 1
        assert convs_after_arch[0]["id"] == c1_id

        # 2. Seed an initial established canon fact
        base_fact = client.post(
            f"/api/worlds/{w['id']}/facts",
            json={
                "title": "Perpetual Acid Smog",
                "content": "The planetary surface is covered in perpetual corrosive acid smog.",
                "canon_status": "ESTABLISHED_CANON",
                "provenance_type": "AUTHOR_CREATED",
            },
        ).json()

        # Seed 13 historical messages in c1_id so total messages exceed 12 turns
        db = SessionLocal()
        try:
            for idx in range(13):
                db.add(
                    Message(
                        conversation_id=c1_id,
                        role="USER" if idx % 2 == 0 else "ASSISTANT",
                        content=f"Historical turn #{idx + 1} discussing sealed dome habitats.",
                    )
                )
            db.commit()
        finally:
            db.close()

        # 3. Run a turn that triggers Pass 3 dependency inference, contradiction wiring, and rolling summary generation
        def fake_structured(prompt, response_schema, **kwargs):
            if response_schema.__name__ == "TurnAnalysisSchema":
                return TurnAnalysisSchema(
                    primary_intent="CANON_DECLARATION",
                    secondary_intents=[],
                    ambiguity_status="CLEAR",
                    raw_user_phrasing="Trade convoys depend on Perpetual Acid Smog filters, and the sky is crystal clear.",
                    inferred_domain_concepts=["trade convoys"],
                    creative_permission=CreativePermissionSchema(level="ANALYTICAL_ADVISORY"),
                    active_domains=["economics", "geography_geology"],
                    requires_factual_retrieval=False,
                    search_queries=["acid smog"],
                )
            return PostTurnExtractionSchema(
                bible_section_updates=[],
                proposals=[
                    CandidateProposalItem(
                        proposal_type="NEW_FACT",
                        proposed_title="Filtered Convoy Network",
                        proposed_content="Overland trade convoys require heavy filtration units because of the Perpetual Acid Smog.",
                        proposed_provenance="AUTHOR_CREATED",
                        subject_slug="economics-trade",
                        auto_apply_to_bible=True,
                        depends_on_fact_ids=[base_fact["id"][:8]],
                    )
                ],
                detected_contradictions=[
                    DetectedContradictionItem(
                        conflicting_fact_id=base_fact["id"][:8],
                        new_fact_title="Filtered Convoy Network",
                        description="Claim of crystal clear sky contradicts Perpetual Acid Smog.",
                    )
                ],
            )

        monkeypatch.setattr(gemini_service, "generate_structured", fake_structured)
        monkeypatch.setattr(
            gemini_service,
            "generate_text",
            lambda contents, system_instruction, **kw: (
                "Condensed summary: Author established sealed dome habitats and Perpetual Acid Smog."
                if "Condense" in (system_instruction or "")
                else "Noted your trade convoy filtration rule and flagged the sky contradiction."
            ),
        )

        msg_res = client.post(
            f"/api/conversations/{c1_id}/messages",
            json={"content": "Trade convoys depend on Perpetual Acid Smog filters, and the sky is crystal clear."},
        )
        assert msg_res.status_code == 200

        # Verify Rolling Summary was generated and persisted on the conversation
        convs_updated = client.get(f"/api/worlds/{w['id']}/conversations").json()
        assert "Condensed summary:" in convs_updated[0]["rolling_summary"]

        # Verify ContextAssembler includes rolling_summary in both system prompt and conversation contents
        db = SessionLocal()
        try:
            conv_obj = db.query(Conversation).filter(Conversation.id == c1_id).first()
            built_contents = context_assembler.build_conversation_contents(
                db, conv_obj, "Follow-up question"
            )
            assert "Rolling Summary of Earlier Conversation Turns" in built_contents[0].parts[0].text
        finally:
            db.close()

        # Verify DEPENDS_ON and CONTRADICTS relationships were wired in Consistency Report
        cons_report = client.get(f"/api/worlds/{w['id']}/consistency").json()
        rel_types = {r["relationship_type"] for r in cons_report["relationships"]}
        assert "DEPENDS_ON" in rel_types
        assert "CONTRADICTS" in rel_types
        assert len(cons_report["flagged_facts"]) >= 1
    finally:
        client.delete(f"/api/worlds/{w['id']}")


def test_step5_clarification_scoped_brainstorming_and_proposal_approval(monkeypatch):
    """Verify Step 5 Tests 3, 4, 5: Clarification on vague prompt, scoped brainstorming, and proposal approval/rejection."""
    from backend.schemas import CandidateProposalItem, PostTurnExtractionSchema

    w = client.post(
        "/api/worlds",
        json={"name": "Step5_ClarifyAndScope", "description": "Test World", "seed_default_subjects": True},
    ).json()
    try:
        convs = client.get(f"/api/worlds/{w['id']}/conversations").json()
        cid = convs[0]["id"]

        # Turn 1: Vague prompt ("I need help figuring out religion in my world") -> ANALYTICAL_ADVISORY + clarifying questions
        def fake_structured_vague(prompt, response_schema, **kwargs):
            if response_schema.__name__ == "TurnAnalysisSchema":
                return TurnAnalysisSchema(
                    primary_intent="IMPLEMENTATION_ASSISTANCE",
                    secondary_intents=[],
                    ambiguity_status="NEEDS_CLARIFICATION",
                    raw_user_phrasing="I need help figuring out religion in my world.",
                    inferred_domain_concepts=["religious organization", "cosmology", "ritual authority"],
                    creative_permission=CreativePermissionSchema(level="ANALYTICAL_ADVISORY"),
                    active_domains=["religion_mythology"],
                    requires_factual_retrieval=True,
                    search_queries=["religious organization priesthood state"],
                )
            return PostTurnExtractionSchema(bible_section_updates=[], proposals=[])

        monkeypatch.setattr(gemini_service, "generate_structured", fake_structured_vague)
        monkeypatch.setattr(
            gemini_service,
            "generate_text",
            lambda contents, system_instruction, **kw: (
                "To help structure your religion, could you clarify whether deities are empirically active "
                "in your setting, and whether religious authority is centralized or local?"
            ),
        )

        r1 = client.post(
            f"/api/conversations/{cid}/messages",
            json={"content": "I need help figuring out religion in my world."},
        ).json()
        assert r1["assistant_message"]["permission_level"] == "ANALYTICAL_ADVISORY"
        assert client.get(f"/api/worlds/{w['id']}/facts").json() == []

        # Turn 2: Explicit opt-in brainstorming ("Give me 3 options for how the priesthood collects tithes")
        # Verify unapproved AI options are staged as PENDING proposals rather than auto-committed to canon
        def fake_structured_brainstorm(prompt, response_schema, **kwargs):
            if response_schema.__name__ == "TurnAnalysisSchema":
                return TurnAnalysisSchema(
                    primary_intent="CREATIVE_GENERATION",
                    secondary_intents=[],
                    ambiguity_status="CLEAR",
                    raw_user_phrasing="Give me three options for how the priesthood collects tithes.",
                    inferred_domain_concepts=["tithe collection", "temple economy"],
                    creative_permission=CreativePermissionSchema(
                        level="BOUNDED_BRAINSTORMING",
                        explicit_authorization_quote="Give me three options",
                        allowed_targets=["priesthood tithe collection"],
                        requested_option_count=3,
                    ),
                    active_domains=["religion_mythology", "economics"],
                    requires_factual_retrieval=True,
                    search_queries=["temple tithe collection grain"],
                )
            return PostTurnExtractionSchema(
                bible_section_updates=[],
                proposals=[
                    CandidateProposalItem(
                        proposal_type="NEW_FACT",
                        proposed_title="Option 1: Harvest Water-Lock Tithe",
                        proposed_content="Temples collect a 10% water-ration tithe at irrigation sluices.",
                        proposed_provenance="AI_GENERATED_AUTHOR_APPROVED",
                        subject_slug="religion-mythology",
                        auto_apply_to_bible=False,
                    ),
                    CandidateProposalItem(
                        proposal_type="NEW_FACT",
                        proposed_title="Option 2: Coinage Mint Monopoly",
                        proposed_content="Priests mint all silver coinage and retain a seigniorage fee.",
                        proposed_provenance="AI_GENERATED_AUTHOR_APPROVED",
                        subject_slug="economics-trade",
                        auto_apply_to_bible=False,
                    ),
                ],
            )

        monkeypatch.setattr(gemini_service, "generate_structured", fake_structured_brainstorm)
        monkeypatch.setattr(
            gemini_service,
            "generate_text",
            lambda contents, system_instruction, **kw: "Here are three scoped options for temple tithes...",
        )

        r2 = client.post(
            f"/api/conversations/{cid}/messages",
            json={"content": "Give me three options for how the priesthood collects tithes."},
        ).json()
        assert r2["assistant_message"]["permission_level"] == "BOUNDED_BRAINSTORMING"
        assert len(r2["staged_proposals"]) == 2
        # Nothing became ESTABLISHED_CANON yet without author approval
        assert client.get(f"/api/worlds/{w['id']}/facts?canon_status=ESTABLISHED_CANON").json() == []

        # Approve Option 1 and Reject Option 2
        p1_id = r2["staged_proposals"][0]["id"]
        p2_id = r2["staged_proposals"][1]["id"]

        app_res = client.post(
            f"/api/worlds/{w['id']}/proposals/{p1_id}/resolve",
            json={"action": "APPROVE"},
        )
        assert app_res.status_code == 200
        rej_res = client.post(
            f"/api/worlds/{w['id']}/proposals/{p2_id}/resolve",
            json={"action": "REJECT"},
        )
        assert rej_res.status_code == 200

        canon_facts = client.get(
            f"/api/worlds/{w['id']}/facts?canon_status=ESTABLISHED_CANON"
        ).json()
        assert len(canon_facts) == 1
        assert canon_facts[0]["provenance_type"] == "AI_GENERATED_AUTHOR_APPROVED"

        rejected_facts = client.get(
            f"/api/worlds/{w['id']}/facts?canon_status=REJECTED"
        ).json()
        assert len(rejected_facts) == 1
        assert "Coinage Mint Monopoly" in rejected_facts[0]["title"]
    finally:
        client.delete(f"/api/worlds/{w['id']}")


def test_step5_manual_bible_override_precedence_and_non_destructive_impact():
    """Verify Step 5 Tests 6, 8, 9: Manual World Bible edit precedence, dependency review propagation, and source retrieval."""
    from backend.database import SessionLocal
    from backend.models import World
    from backend.schemas import CreativePermissionSchema, TurnAnalysisSchema
    from backend.services.book_ingestion_service import book_ingestion_service
    from backend.services.context_assembler import context_assembler

    w = client.post(
        "/api/worlds",
        json={"name": "Step5_PrecedenceWorld", "description": "Initial description", "seed_default_subjects": True},
    ).json()
    try:
        # 1. Verify world rename persistence
        ren = client.put(
            f"/api/worlds/{w['id']}",
            json={"name": "Step5_RenamedWorld", "description": "Updated description"},
        ).json()
        assert ren["name"] == "Step5_RenamedWorld"

        # 2. Create an underlying fact and a dependent AI-approved fact, then modify the underlying fact
        f_base = client.post(
            f"/api/worlds/{w['id']}/facts",
            json={
                "title": "Tidal Locked Orbit",
                "content": "The planet is tidally locked to its red dwarf primary.",
                "canon_status": "ESTABLISHED_CANON",
                "provenance_type": "AUTHOR_CREATED",
            },
        ).json()
        f_dep = client.post(
            f"/api/worlds/{w['id']}/facts",
            json={
                "title": "Terminator Zone Agriculture",
                "content": "All agriculture is restricted to the permanent twilight terminator belt.",
                "canon_status": "ESTABLISHED_CANON",
                "provenance_type": "AI_GENERATED_AUTHOR_APPROVED",
            },
        ).json()

        client.post(
            f"/api/worlds/{w['id']}/relationships",
            json={
                "source_fact_id": f_dep["id"],
                "target_fact_id": f_base["id"],
                "relationship_type": "DEPENDS_ON",
                "description": "Terminator agriculture depends on tidal locking.",
            },
        )

        # Modify f_base: verify f_dep is flagged NEEDS_REVIEW without rewriting f_dep's content
        upd = client.put(
            f"/api/worlds/{w['id']}/facts/{f_base['id']}",
            json={
                "content": "The planet rotates once every 36 standard hours (not tidally locked).",
                "change_reason": "Author changed planetary rotation.",
            },
        ).json()
        assert len(upd["impacted_facts_flagged_for_review"]) == 1
        assert upd["impacted_facts_flagged_for_review"][0]["id"] == f_dep["id"]

        cons = client.get(f"/api/worlds/{w['id']}/consistency").json()
        assert any(ff["id"] == f_dep["id"] and ff["consistency_status"] == "NEEDS_REVIEW" for ff in cons["flagged_facts"])

        # Resolve the consistency flag via the new endpoint
        res_flag = client.post(
            f"/api/worlds/{w['id']}/consistency/facts/{f_dep['id']}/resolve"
        )
        assert res_flag.status_code == 200
        assert res_flag.json()["consistency_status"] == "VALID"

        # 3. Verify manual World Bible edit is immediately reflected in the assembled system prompt
        secs = client.get(f"/api/worlds/{w['id']}/bible/sections").json()
        clim_sec = next(s for s in secs if s["slug"] == "climate-ecology")
        client.put(
            f"/api/worlds/{w['id']}/bible/sections/{clim_sec['id']}",
            json={
                "title": clim_sec["title"],
                "markdown_content": "Monsoon cycles last nine full months across the southern hemisphere.",
                "expected_version_number": clim_sec["version_number"],
            },
        )

        db = SessionLocal()
        try:
            world_obj = db.query(World).filter(World.id == w["id"]).first()
            book_res = book_ingestion_service.retrieve_passages(
                db=db,
                queries=["planet atmosphere climate monsoon"],
                active_domains=["geography_geology"],
                top_k=3,
            )
            assert len(book_res["passages"]) >= 1
            assert book_res["passages"][0]["book_page_start"] > 0

            sys_prompt = context_assembler.assemble_consultant_system_prompt(
                db=db,
                world=world_obj,
                turn_analysis=TurnAnalysisSchema(
                    primary_intent="IMPLEMENTATION_ASSISTANCE",
                    creative_permission=CreativePermissionSchema(level="IMPLEMENTATION_SCOPED"),
                    active_domains=["geography_geology"],
                    search_queries=["climate monsoon"],
                ),
                book_retrieval=book_res,
                online_research={"status": "SKIPPED", "sources": []},
            )
            assert "Monsoon cycles last nine full months" in sys_prompt
            assert "Provenance: AUTHOR_CREATED" in sys_prompt
        finally:
            db.close()
    finally:
        client.delete(f"/api/worlds/{w['id']}")


def test_step6_full_project_backup_and_restore_archive():
    """Verify Step 6 Full Project Backup (.json) and Restore (Import) preserving Bible, revisions, facts, and conversations."""
    w = client.post(
        "/api/worlds",
        json={
            "name": "Step6_ArchiveWorld",
            "description": "Original world before backup",
            "genre_notes": "Low-magic maritime",
            "seed_default_subjects": True,
        },
    ).json()
    restored_id = None
    try:
        # 1. Populate a World Bible section and create a fact + conversation thread
        secs = client.get(f"/api/worlds/{w['id']}/bible/sections").json()
        hist_sec = next(s for s in secs if s["slug"] == "history-chronology")
        client.put(
            f"/api/worlds/{w['id']}/bible/sections/{hist_sec['id']}",
            json={
                "title": hist_sec["title"],
                "markdown_content": "### The Concord of Sails\n\nSigned in Year 412 by the seven harbor-syndicates.",
                "expected_version_number": hist_sec["version_number"],
            },
        )
        client.post(
            f"/api/worlds/{w['id']}/conversations",
            json={"title": "Maritime Trade Thread"},
        )

        # 2. Export the full project bundle (.json)
        exp_res = client.get(f"/api/worlds/{w['id']}/export-project")
        assert exp_res.status_code == 200
        bundle = exp_res.json()
        assert bundle["schema_version"] == "1.0"
        assert bundle["world"]["name"] == "Step6_ArchiveWorld"
        assert any("Concord of Sails" in s["markdown_content"] for s in bundle["bible_sections"])
        assert len(bundle["facts"]) >= 1
        assert len(bundle["conversations"]) == 2

        # 3. Import the bundle while the original world still exists (verifying UUID remapping)
        bundle["world"]["name"] = "Step6_RestoredWorld"
        imp_res = client.post("/api/worlds/import-project", json=bundle)
        assert imp_res.status_code == 200
        restored = imp_res.json()
        restored_id = restored["id"]
        assert restored_id != w["id"]
        assert restored["name"] == "Step6_RestoredWorld"

        # 4. Verify restored sections, revisions, facts, and conversation threads
        rest_secs = client.get(f"/api/worlds/{restored_id}/bible/sections").json()
        rest_hist = next(s for s in rest_secs if s["slug"] == "history-chronology")
        assert "Signed in Year 412 by the seven harbor-syndicates." in rest_hist["markdown_content"]

        rest_revs = client.get(
            f"/api/worlds/{restored_id}/bible/sections/{rest_hist['id']}/revisions"
        ).json()
        assert len(rest_revs) >= 1

        rest_facts = client.get(f"/api/worlds/{restored_id}/facts").json()
        assert len(rest_facts) >= 1
        assert any("Signed in Year 412" in f["content"] for f in rest_facts)

        rest_convs = client.get(f"/api/worlds/{restored_id}/conversations").json()
        assert len(rest_convs) == 2
        assert any(c["title"] == "Maritime Trade Thread" for c in rest_convs)
    finally:
        client.delete(f"/api/worlds/{w['id']}")
        if restored_id:
            client.delete(f"/api/worlds/{restored_id}")


def test_step7_context_aware_naming_intelligence_tests_a_to_j(monkeypatch):
    """Verify Step 7 Naming Tests A-J: Major vs. Minor naming, linguistic registry, anti-collision, rejected names, and non-naming isolation."""
    from backend.database import SessionLocal
    from backend.models import World
    from backend.services.context_assembler import context_assembler

    w = client.post(
        "/api/worlds",
        json={
            "name": "Step7_NamingWorld",
            "description": "World with distinct highland and maritime cultures",
            "seed_default_subjects": True,
        },
    ).json()
    try:
        # 1. Seed established linguistic conventions and existing proper nouns in the World Bible (Tests B, C, F)
        secs = client.get(f"/api/worlds/{w['id']}/bible/sections").json()
        ling_sec = next(s for s in secs if s["slug"] == "languages-linguistics")
        client.put(
            f"/api/worlds/{w['id']}/bible/sections/{ling_sec['id']}",
            json={
                "title": ling_sec["title"],
                "markdown_content": (
                    "Highland Varakhi uses uvular stops (q, kh) and compound toponyms ending in -khor (fortress) "
                    "such as Qarakhor and Zalkhor. Coastal Thalassian uses open vowels and liquid consonants (l, r, n) "
                    "such as Orelia and Lirion."
                ),
                "expected_version_number": ling_sec["version_number"],
            },
        )

        # Record a rejected name (Test H)
        prop_res = client.post(
            f"/api/worlds/{w['id']}/facts",
            json={
                "title": "Kael'thas Shadowblade",
                "content": "Rejected cliché fantasy name with gratuitous apostrophe.",
                "canon_status": "REJECTED",
                "provenance_type": "AI_PROPOSED_UNVERIFIED",
            },
        )
        assert prop_res.status_code == 200

        db = SessionLocal()
        try:
            world_obj = db.query(World).filter(World.id == w["id"]).first()

            # Mock Pass 1 to return baseline TurnAnalysisSchema so we verify AgentOrchestrator's naming classification
            monkeypatch.setattr(
                gemini_service,
                "generate_structured",
                lambda prompt, response_schema, **kw: TurnAnalysisSchema(
                    primary_intent="CREATIVE_GENERATION",
                    creative_permission=CreativePermissionSchema(level="IMPLEMENTATION_SCOPED"),
                    active_domains=["geography_geology"],
                    search_queries=["mountain pass"],
                ),
            )

            # Test A & E: Major Context-Driven Naming Request
            major_analysis = agent_orchestrator.analyze_turn(
                db,
                world_obj,
                "Help me name the ancient Highland Varakhi capital fortress guarding the northern pass.",
            )
            assert major_analysis.creative_permission.allow_invented_proper_nouns is True
            assert major_analysis.creative_permission.naming_mode == "MAJOR_CONTEXTUAL"
            assert major_analysis.creative_permission.level == "BOUNDED_BRAINSTORMING"
            assert "linguistics" in major_analysis.active_domains
            assert any("onomastics" in q or "toponymy" in q for q in major_analysis.search_queries)

            major_prompt = context_assembler.assemble_consultant_system_prompt(
                db=db,
                world=world_obj,
                turn_analysis=major_analysis,
                book_retrieval={"coverage_status": "LOW", "passages": []},
                online_research={"status": "SKIPPED", "sources": []},
            )
            assert '<layer_9_context_aware_naming_intelligence mode="MAJOR_CONTEXTUAL">' in major_prompt
            assert "MODE: MAJOR CONTEXT-DRIVEN NAMING" in major_prompt
            assert "Qarakhor" in major_prompt
            assert "Orelia" in major_prompt
            assert "Kael'thas Shadowblade" in major_prompt

            # Test D: Minor / Quick Efficient Naming Request
            minor_analysis = agent_orchestrator.analyze_turn(
                db,
                world_obj,
                "I just need a quick name for a minor tavern and a background dock guard in a Coastal Thalassian port.",
            )
            assert minor_analysis.creative_permission.allow_invented_proper_nouns is True
            assert minor_analysis.creative_permission.naming_mode == "MINOR_QUICK"
            assert minor_analysis.creative_permission.requested_option_count >= 6

            minor_prompt = context_assembler.assemble_consultant_system_prompt(
                db=db,
                world=world_obj,
                turn_analysis=minor_analysis,
                book_retrieval={"coverage_status": "LOW", "passages": []},
                online_research={"status": "SKIPPED", "sources": []},
            )
            assert '<layer_9_context_aware_naming_intelligence mode="MINOR_QUICK">' in minor_prompt
            assert "ZERO INTERROGATION OVERHEAD" in minor_prompt

            # Verify contrastive phrasing containing both "major" and "minor"
            contrast_minor = agent_orchestrator.analyze_turn(
                db,
                world_obj,
                "Help me name this character—he is not a major character, just a minor guard at the gate.",
            )
            assert contrast_minor.creative_permission.naming_mode == "MINOR_QUICK"

            contrast_major = agent_orchestrator.analyze_turn(
                db,
                world_obj,
                "I need a name for a major religious order, not some minor local sect.",
            )
            assert contrast_major.creative_permission.naming_mode == "MAJOR_CONTEXTUAL"

            # Test I: Non-Naming Turn Isolation (No unsolicited proper nouns)
            non_naming_analysis = agent_orchestrator.analyze_turn(
                db,
                world_obj,
                "How does rain shadow affect grain yields on the leeward side of a mountain range?",
            )
            assert non_naming_analysis.creative_permission.allow_invented_proper_nouns is False
            assert non_naming_analysis.creative_permission.naming_mode == "NONE"

            non_naming_prompt = context_assembler.assemble_consultant_system_prompt(
                db=db,
                world=world_obj,
                turn_analysis=non_naming_analysis,
                book_retrieval={"coverage_status": "HIGH", "passages": []},
                online_research={"status": "SKIPPED", "sources": []},
            )
            assert "<layer_9_context_aware_naming_intelligence" not in non_naming_prompt
        finally:
            db.close()
    finally:
        client.delete(f"/api/worlds/{w['id']}")






