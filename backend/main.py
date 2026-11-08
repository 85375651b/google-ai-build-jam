"""FastAPI application entrypoint exposing all REST and SSE endpoints for the Worldbuilding Agent."""

import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from backend.config import PROJECT_ROOT, settings
from backend.database import get_db, init_db
from backend.models import (
    AuthorExpertiseProfile,
    BibleSection,
    BibleSectionRevision,
    CanonProposal,
    Conversation,
    Fact,
    Message,
    ReferenceDocument,
    ResearchSource,
    Subject,
    World,
)
from backend.prompts.domain_cartridges import DOMAIN_CARTRIDGES
from backend.schemas import (
    BibleSectionCreate,
    BibleSectionUpdate,
    ChatMessageRequest,
    ConversationCreate,
    ExpertiseOverrideRequest,
    FactCreate,
    FactRelationshipCreate,
    FactUpdate,
    ModelConfigUpdateRequest,
    OrphanedFactResolutionRequest,
    ProposalActionRequest,
    SubjectCreate,
    SubjectMergeRequest,
    SubjectUpdate,
    WorldCreate,
    WorldUpdate,
)
from backend.services.agent_orchestrator import agent_orchestrator
from backend.services.bible_service import bible_service
from backend.services.book_ingestion_service import book_ingestion_service
from backend.services.consistency_service import consistency_service
from backend.services.context_assembler import context_assembler
from backend.services.gemini_service import GeminiServiceError, gemini_service
from backend.services.world_service import world_service

logger = logging.getLogger(__name__)

app = FastAPI(title="Worldbuilding Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    db_gen = get_db()
    db = next(db_gen)
    try:
        book_ingestion_service.ensure_primary_book_ingested(db, generate_embeddings=False)
    except Exception as exc:
        logger.warning("Startup book check skipped: %s", exc)
    finally:
        db.close()


# ============================================================================
# 1. System, Config & Knowledge Sources Endpoints
# ============================================================================

@app.get("/api/system/status")
def get_system_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    settings.reload()
    docs = db.query(ReferenceDocument).order_by(ReferenceDocument.created_at.desc()).all()
    online_sources = (
        db.query(ResearchSource).order_by(ResearchSource.retrieved_at.desc()).limit(50).all()
    )
    return {
        "has_valid_gemini_key": settings.has_valid_gemini_key,
        "has_valid_tavily_key": settings.has_valid_tavily_key,
        "gemini_model": settings.gemini_model,
        "gemini_embedding_model": settings.gemini_embedding_model,
        "reference_documents": [
            {
                "id": d.id,
                "title": d.title,
                "author": d.author,
                "file_path": Path(d.file_path).name,
                "total_pages": d.total_pages,
                "total_chunks": d.total_chunks,
                "embeddings_count": d.embeddings_count,
                "ingestion_status": d.ingestion_status,
                "error_message": d.error_message,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ],
        "online_sources": [
            {
                "id": s.id,
                "url": s.url,
                "title": s.title,
                "domain": s.domain,
                "credibility_tier": s.source_credibility_tier,
                "excerpt": s.extracted_excerpt,
                "retrieved_at": s.retrieved_at.isoformat(),
            }
            for s in online_sources
        ],
    }


@app.put("/api/system/model")
def update_model_config(payload: ModelConfigUpdateRequest) -> Dict[str, Any]:
    settings.update_model(payload.gemini_model, payload.gemini_embedding_model)
    return {
        "gemini_model": settings.gemini_model,
        "gemini_embedding_model": settings.gemini_embedding_model,
    }


@app.post("/api/sources/upload")
def upload_reference_book(
    file: UploadFile = File(...),
    title: Optional[str] = Query(None),
    author: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    uploads_dir = PROJECT_ROOT / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "reference_book.pdf").name
    dest_path = uploads_dir / safe_name
    with open(dest_path, "wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    try:
        doc = book_ingestion_service.ingest_document(
            db=db,
            file_path=dest_path,
            title=title or dest_path.stem,
            author=author or "Uploaded Reference",
            generate_embeddings=False,
        )
        return {
            "id": doc.id,
            "title": doc.title,
            "total_pages": doc.total_pages,
            "total_chunks": doc.total_chunks,
            "ingestion_status": doc.ingestion_status,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Book ingestion failed: {exc}") from exc


@app.post("/api/sources/{doc_id}/embed")
def generate_document_embeddings(
    doc_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    if not settings.has_valid_gemini_key:
        raise HTTPException(
            status_code=401,
            detail="GEMINI_API_KEY is not configured in .env. Please add your key first.",
        )
    count = book_ingestion_service.generate_missing_embeddings(db, doc_id, max_chunks=400)
    return {"embedded_chunks": count}


# ============================================================================
# 2. World / Project Management Endpoints
# ============================================================================

@app.get("/api/worlds")
def list_worlds(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    worlds = world_service.list_worlds(db)
    return [
        {
            "id": w.id,
            "name": w.name,
            "description": w.description,
            "genre_notes": w.genre_notes,
            "allow_unrequested_proper_nouns": w.allow_unrequested_proper_nouns,
            "brainstorm_option_count_default": w.brainstorm_option_count_default,
            "auto_stage_canon_proposals": w.auto_stage_canon_proposals,
            "created_at": w.created_at.isoformat(),
            "updated_at": w.updated_at.isoformat(),
        }
        for w in worlds
    ]


@app.post("/api/worlds")
def create_world(
    payload: WorldCreate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    w = world_service.create_world(db, payload)
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "genre_notes": w.genre_notes,
        "allow_unrequested_proper_nouns": w.allow_unrequested_proper_nouns,
        "brainstorm_option_count_default": w.brainstorm_option_count_default,
        "auto_stage_canon_proposals": w.auto_stage_canon_proposals,
        "created_at": w.created_at.isoformat(),
        "updated_at": w.updated_at.isoformat(),
    }


@app.get("/api/worlds/{world_id}")
def get_world_details(
    world_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    w = world_service.get_world(db, world_id)
    if not w:
        raise HTTPException(status_code=404, detail="World project not found.")
    subjects = world_service.list_subjects(db, world_id)
    profiles = (
        db.query(AuthorExpertiseProfile)
        .filter(AuthorExpertiseProfile.world_id == world_id)
        .all()
    )
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "genre_notes": w.genre_notes,
        "allow_unrequested_proper_nouns": w.allow_unrequested_proper_nouns,
        "brainstorm_option_count_default": w.brainstorm_option_count_default,
        "auto_stage_canon_proposals": w.auto_stage_canon_proposals,
        "subjects": [
            {
                "id": s.id,
                "name": s.name,
                "slug": s.slug,
                "description": s.description,
                "parent_subject_id": s.parent_subject_id,
                "domain_key": s.domain_key,
            }
            for s in subjects
        ],
        "expertise_profiles": [
            {
                "domain_key": p.domain_key,
                "domain_name": DOMAIN_CARTRIDGES.get(p.domain_key, {}).get("name", p.domain_key),
                "expertise_level": p.expertise_level,
                "confidence_score": round(p.confidence_score, 2),
                "is_explicit_override": p.is_explicit_override,
                "evidence_notes": p.evidence_notes,
            }
            for p in profiles
        ],
    }


@app.put("/api/worlds/{world_id}")
def update_world(
    world_id: str, payload: WorldUpdate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    w = world_service.get_world(db, world_id)
    if not w:
        raise HTTPException(status_code=404, detail="World project not found.")
    w = world_service.update_world(db, w, payload)
    return {"id": w.id, "name": w.name, "description": w.description, "genre_notes": w.genre_notes}


@app.delete("/api/worlds/{world_id}")
def delete_world(world_id: str, db: Session = Depends(get_db)) -> Dict[str, str]:
    w = world_service.get_world(db, world_id)
    if not w:
        raise HTTPException(status_code=404, detail="World project not found.")
    world_service.delete_world(db, w)
    return {"status": "deleted"}


@app.post("/api/worlds/{world_id}/subjects")
def create_subject(
    world_id: str, payload: SubjectCreate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    s = world_service.create_subject(db, world_id, payload)
    return {"id": s.id, "name": s.name, "slug": s.slug, "description": s.description}


@app.put("/api/worlds/{world_id}/expertise")
def override_expertise(
    world_id: str, payload: ExpertiseOverrideRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    p = world_service.set_explicit_expertise_override(
        db, world_id, payload.domain_key, payload.expertise_level
    )
    return {
        "domain_key": p.domain_key,
        "expertise_level": p.expertise_level,
        "is_explicit_override": p.is_explicit_override,
    }


# ============================================================================
# 3. World Bible, Atomic Facts, Revisions, Proposals & Export Endpoints
# ============================================================================

@app.get("/api/worlds/{world_id}/bible/sections")
def list_bible_sections(
    world_id: str, db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    sections = bible_service.list_sections(db, world_id)
    return [
        {
            "id": s.id,
            "title": s.title,
            "slug": s.slug,
            "markdown_content": s.markdown_content,
            "primary_subject_id": s.primary_subject_id,
            "parent_section_id": s.parent_section_id,
            "dominant_provenance": s.dominant_provenance,
            "version_number": s.version_number,
            "updated_at": s.updated_at.isoformat(),
        }
        for s in sections
    ]


@app.post("/api/worlds/{world_id}/bible/sections")
def create_bible_section(
    world_id: str, payload: BibleSectionCreate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    s = bible_service.create_section(db, world_id, payload)
    return {
        "id": s.id,
        "title": s.title,
        "slug": s.slug,
        "markdown_content": s.markdown_content,
        "dominant_provenance": s.dominant_provenance,
        "version_number": s.version_number,
    }


@app.put("/api/worlds/{world_id}/bible/sections/{section_id}")
def update_bible_section(
    world_id: str,
    section_id: str,
    payload: BibleSectionUpdate,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    sec = bible_service.get_section(db, world_id, section_id)
    if not sec:
        raise HTTPException(status_code=404, detail="World Bible section not found.")
    try:
        updated_sec, orphaned = bible_service.update_section_directly(db, sec, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "section": {
            "id": updated_sec.id,
            "title": updated_sec.title,
            "slug": updated_sec.slug,
            "markdown_content": updated_sec.markdown_content,
            "dominant_provenance": updated_sec.dominant_provenance,
            "version_number": updated_sec.version_number,
            "updated_at": updated_sec.updated_at.isoformat(),
        },
        "orphaned_facts_requiring_confirmation": [
            {
                "id": f.id,
                "title": f.title,
                "content": f.content,
                "provenance_type": f.provenance_type,
            }
            for f in orphaned
        ],
    }


@app.delete("/api/worlds/{world_id}/bible/sections/{section_id}")
def delete_bible_section(
    world_id: str, section_id: str, db: Session = Depends(get_db)
) -> Dict[str, str]:
    sec = bible_service.get_section(db, world_id, section_id)
    if not sec:
        raise HTTPException(status_code=404, detail="Section not found.")
    bible_service.delete_section(db, sec)
    return {"status": "deleted"}


@app.get("/api/worlds/{world_id}/bible/sections/{section_id}/revisions")
def list_section_revisions(
    world_id: str, section_id: str, db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    sec = bible_service.get_section(db, world_id, section_id)
    if not sec:
        raise HTTPException(status_code=404, detail="Section not found.")
    revs = (
        db.query(BibleSectionRevision)
        .filter(BibleSectionRevision.section_id == section_id)
        .order_by(BibleSectionRevision.version_number.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "version_number": r.version_number,
            "previous_markdown": r.previous_markdown,
            "new_markdown": r.new_markdown,
            "edit_origin": r.edit_origin,
            "author_note": r.author_note,
            "created_at": r.created_at.isoformat(),
        }
        for r in revs
    ]


@app.post("/api/worlds/{world_id}/bible/sections/{section_id}/revert/{version_number}")
def revert_bible_section(
    world_id: str,
    section_id: str,
    version_number: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    sec = bible_service.get_section(db, world_id, section_id)
    if not sec:
        raise HTTPException(status_code=404, detail="Section not found.")
    reverted = bible_service.revert_section_to_version(db, sec, version_number)
    return {
        "id": reverted.id,
        "title": reverted.title,
        "markdown_content": reverted.markdown_content,
        "version_number": reverted.version_number,
    }


@app.post("/api/worlds/{world_id}/bible/orphaned-facts/resolve")
def resolve_orphaned_facts(
    world_id: str,
    payload: OrphanedFactResolutionRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    resolved = bible_service.resolve_orphaned_facts(db, world_id, payload)
    return {"resolved_count": len(resolved), "action": payload.action}


@app.get("/api/worlds/{world_id}/facts")
def list_world_facts(
    world_id: str,
    canon_status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    facts = bible_service.list_facts(db, world_id, canon_status=canon_status)
    return [
        {
            "id": f.id,
            "title": f.title,
            "content": f.content,
            "canon_status": f.canon_status,
            "provenance_type": f.provenance_type,
            "authority_tier": f.authority_tier,
            "consistency_status": f.consistency_status,
            "consistency_notes": f.consistency_notes,
            "version_number": f.version_number,
            "updated_at": f.updated_at.isoformat(),
        }
        for f in facts
    ]


@app.post("/api/worlds/{world_id}/facts")
def create_world_fact(
    world_id: str, payload: FactCreate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    f = bible_service.create_fact(db, world_id, payload)
    return {
        "id": f.id,
        "title": f.title,
        "content": f.content,
        "canon_status": f.canon_status,
        "provenance_type": f.provenance_type,
    }


@app.put("/api/worlds/{world_id}/facts/{fact_id}")
def update_world_fact(
    world_id: str, fact_id: str, payload: FactUpdate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    f = db.query(Fact).filter(Fact.id == fact_id, Fact.world_id == world_id).first()
    if not f:
        raise HTTPException(status_code=404, detail="Fact not found.")
    updated, impacted = bible_service.update_fact(db, f, payload)
    return {
        "fact": {
            "id": updated.id,
            "title": updated.title,
            "content": updated.content,
            "canon_status": updated.canon_status,
            "provenance_type": updated.provenance_type,
        },
        "impacted_facts_flagged_for_review": [
            {"id": imp.id, "title": imp.title, "consistency_notes": imp.consistency_notes}
            for imp in impacted
        ],
    }


@app.post("/api/worlds/{world_id}/relationships")
def create_fact_relationship(
    world_id: str, payload: FactRelationshipCreate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    rel = consistency_service.create_relationship(
        db=db,
        world_id=world_id,
        source_fact_id=payload.source_fact_id,
        target_fact_id=payload.target_fact_id,
        relationship_type=payload.relationship_type,
        description=payload.description or "",
        created_by="AUTHOR",
    )
    return {"id": rel.id, "relationship_type": rel.relationship_type}


@app.get("/api/worlds/{world_id}/consistency")
def get_consistency_report(
    world_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    return consistency_service.get_world_consistency_report(db, world_id)


@app.get("/api/worlds/{world_id}/proposals")
def list_canon_proposals(
    world_id: str,
    status: Optional[str] = Query("PENDING"),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    proposals = bible_service.list_proposals(db, world_id, status=status)
    return [
        {
            "id": p.id,
            "proposal_type": p.proposal_type,
            "proposed_title": p.proposed_title,
            "proposed_content": p.proposed_content,
            "proposed_provenance": p.proposed_provenance,
            "target_subject_id": p.target_subject_id,
            "target_section_id": p.target_section_id,
            "status": p.status,
            "created_at": p.created_at.isoformat(),
        }
        for p in proposals
    ]


@app.post("/api/worlds/{world_id}/proposals/{proposal_id}/resolve")
def resolve_canon_proposal(
    world_id: str,
    proposal_id: str,
    payload: ProposalActionRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    prop = (
        db.query(CanonProposal)
        .filter(CanonProposal.id == proposal_id, CanonProposal.world_id == world_id)
        .first()
    )
    if not prop:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    return bible_service.resolve_proposal(db, prop, payload)


@app.get("/api/worlds/{world_id}/bible/export")
def export_world_bible(
    world_id: str,
    fmt: str = Query("md", description="md | pdf | docx"),
    db: Session = Depends(get_db),
) -> Response:
    w = world_service.get_world(db, world_id)
    if not w:
        raise HTTPException(status_code=404, detail="World not found.")
    fmt_lower = fmt.lower()
    if fmt_lower == "pdf":
        pdf_bytes = bible_service.export_world_bible_pdf(db, w)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{w.name}_World_Bible.pdf"'},
        )
    if fmt_lower == "docx":
        docx_bytes = bible_service.export_world_bible_docx(db, w)
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{w.name}_World_Bible.docx"'},
        )
    md_text = bible_service.export_world_bible_markdown(db, w)
    return Response(
        content=md_text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{w.name}_World_Bible.md"'},
    )


# ============================================================================
# 4. Conversational Agent & Streaming Chat Endpoints
# ============================================================================

@app.get("/api/worlds/{world_id}/conversations")
def list_conversations(
    world_id: str, db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    convs = (
        db.query(Conversation)
        .filter(Conversation.world_id == world_id, Conversation.is_archived.is_(False))
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return [
        {
            "id": c.id,
            "title": c.title,
            "focused_subject_id": c.focused_subject_id,
            "updated_at": c.updated_at.isoformat(),
        }
        for c in convs
    ]


@app.post("/api/worlds/{world_id}/conversations")
def create_conversation(
    world_id: str, payload: ConversationCreate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    c = Conversation(
        world_id=world_id,
        title=(payload.title or "New Consultation").strip(),
        focused_subject_id=payload.focused_subject_id,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return {"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()}


@app.get("/api/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: str, db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    out: List[Dict[str, Any]] = []
    for m in msgs:
        meta = {}
        if m.metadata_json:
            try:
                meta = json.loads(m.metadata_json)
            except Exception:
                meta = {}
        out.append(
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "interaction_mode": m.interaction_mode,
                "permission_level": m.permission_level,
                "metadata": meta,
                "created_at": m.created_at.isoformat(),
            }
        )
    return out


@app.post("/api/conversations/{conversation_id}/messages")
def send_message_sync(
    conversation_id: str,
    payload: ChatMessageRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    world = world_service.get_world(db, conv.world_id)
    if not world:
        raise HTTPException(status_code=404, detail="World not found.")

    try:
        # 1. Pass 1: Analyze turn
        analysis = agent_orchestrator.analyze_turn(db, world, payload.content)
        # 2. Retrieve book & online research
        knowledge = agent_orchestrator.retrieve_knowledge(db, analysis)
        # 3. Assemble 8-layer context
        sys_prompt = context_assembler.assemble_consultant_system_prompt(
            db=db,
            world=world,
            turn_analysis=analysis,
            book_retrieval=knowledge["book"],
            online_research=knowledge["online"],
            focused_subject_id=payload.focused_subject_id or conv.focused_subject_id,
        )
        contents = context_assembler.build_conversation_contents(
            db, conv, payload.content
        )
        # 4. Generate consultant response
        raw_reply = gemini_service.generate_text(
            contents=contents,
            system_instruction=sys_prompt,
            temperature=0.4,
        )
        sanitized = agent_orchestrator.sanitize_response_and_verify_citations(
            raw_reply, knowledge["book"], knowledge["online"]
        )

        # 5. Persist messages ONLY after successful generation
        user_msg = Message(
            conversation_id=conv.id,
            role="USER",
            content=payload.content.strip(),
            interaction_mode=analysis.primary_intent,
            permission_level=analysis.creative_permission.level,
        )
        db.add(user_msg)
        db.flush()

        meta_dict = {
            "primary_intent": analysis.primary_intent,
            "secondary_intents": analysis.secondary_intents,
            "permission_level": analysis.creative_permission.level,
            "active_domains": analysis.active_domains,
            "book_coverage": knowledge["book"]["coverage_status"],
            "online_research_status": knowledge["online"]["status"],
            "citations": sanitized["citations"],
        }
        assistant_msg = Message(
            conversation_id=conv.id,
            role="ASSISTANT",
            content=sanitized["content"],
            interaction_mode=analysis.primary_intent,
            permission_level=analysis.creative_permission.level,
            metadata_json=json.dumps(meta_dict),
        )
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)

        # 6. Pass 3: Controlled Canon Proposal Extraction
        proposals = agent_orchestrator.extract_post_turn_proposals(
            db, world, conv, user_msg, assistant_msg, analysis
        )

        return {
            "user_message_id": user_msg.id,
            "assistant_message": {
                "id": assistant_msg.id,
                "role": "ASSISTANT",
                "content": assistant_msg.content,
                "interaction_mode": assistant_msg.interaction_mode,
                "permission_level": assistant_msg.permission_level,
                "metadata": meta_dict,
                "created_at": assistant_msg.created_at.isoformat(),
            },
            "staged_proposals": [
                {
                    "id": p.id,
                    "proposed_title": p.proposed_title,
                    "proposed_content": p.proposed_content,
                    "proposed_provenance": p.proposed_provenance,
                }
                for p in proposals
            ],
        }
    except GeminiServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc


@app.post("/api/conversations/{conversation_id}/stream")
def stream_message(
    conversation_id: str,
    payload: ChatMessageRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    world = world_service.get_world(db, conv.world_id)
    if not world:
        raise HTTPException(status_code=404, detail="World not found.")

    if not settings.has_valid_gemini_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "MISSING_API_KEY",
                "message": "GEMINI_API_KEY is missing in .env. Please paste your Gemini API key into .env.",
            },
        )

    def event_generator():
        try:
            analysis = agent_orchestrator.analyze_turn(db, world, payload.content)
            knowledge = agent_orchestrator.retrieve_knowledge(db, analysis)
            yield "data: " + json.dumps(
                {
                    "type": "turn_meta",
                    "primary_intent": analysis.primary_intent,
                    "permission_level": analysis.creative_permission.level,
                    "active_domains": analysis.active_domains,
                    "book_coverage": knowledge["book"]["coverage_status"],
                }
            ) + "\n\n"

            sys_prompt = context_assembler.assemble_consultant_system_prompt(
                db=db,
                world=world,
                turn_analysis=analysis,
                book_retrieval=knowledge["book"],
                online_research=knowledge["online"],
                focused_subject_id=payload.focused_subject_id or conv.focused_subject_id,
            )
            contents = context_assembler.build_conversation_contents(
                db, conv, payload.content
            )

            accumulated: List[str] = []
            for chunk_str in gemini_service.stream_text(
                contents=contents,
                system_instruction=sys_prompt,
                temperature=0.4,
            ):
                accumulated.append(chunk_str)
                yield "data: " + json.dumps({"type": "delta", "text": chunk_str}) + "\n\n"

            full_raw = "".join(accumulated)
            sanitized = agent_orchestrator.sanitize_response_and_verify_citations(
                full_raw, knowledge["book"], knowledge["online"]
            )

            user_msg = Message(
                conversation_id=conv.id,
                role="USER",
                content=payload.content.strip(),
                interaction_mode=analysis.primary_intent,
                permission_level=analysis.creative_permission.level,
            )
            db.add(user_msg)
            db.flush()

            meta_dict = {
                "primary_intent": analysis.primary_intent,
                "secondary_intents": analysis.secondary_intents,
                "permission_level": analysis.creative_permission.level,
                "active_domains": analysis.active_domains,
                "book_coverage": knowledge["book"]["coverage_status"],
                "online_research_status": knowledge["online"]["status"],
                "citations": sanitized["citations"],
            }
            assistant_msg = Message(
                conversation_id=conv.id,
                role="ASSISTANT",
                content=sanitized["content"],
                interaction_mode=analysis.primary_intent,
                permission_level=analysis.creative_permission.level,
                metadata_json=json.dumps(meta_dict),
            )
            db.add(assistant_msg)
            db.commit()
            db.refresh(assistant_msg)

            proposals = agent_orchestrator.extract_post_turn_proposals(
                db, world, conv, user_msg, assistant_msg, analysis
            )

            yield "data: " + json.dumps(
                {
                    "type": "complete",
                    "assistant_message": {
                        "id": assistant_msg.id,
                        "role": "ASSISTANT",
                        "content": assistant_msg.content,
                        "interaction_mode": assistant_msg.interaction_mode,
                        "permission_level": assistant_msg.permission_level,
                        "metadata": meta_dict,
                        "created_at": assistant_msg.created_at.isoformat(),
                    },
                    "staged_proposals": [
                        {
                            "id": p.id,
                            "proposed_title": p.proposed_title,
                            "proposed_content": p.proposed_content,
                            "proposed_provenance": p.proposed_provenance,
                        }
                        for p in proposals
                    ],
                }
            ) + "\n\n"
        except GeminiServiceError as exc:
            yield "data: " + json.dumps(
                {"type": "error", "error_code": exc.error_code, "message": exc.message}
            ) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps(
                {"type": "error", "error_code": "SERVER_ERROR", "message": str(exc)}
            ) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
