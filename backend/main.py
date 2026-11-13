"""FastAPI application entrypoint exposing all REST and SSE endpoints for the Worldbuilding Agent."""

from contextlib import asynccontextmanager
import asyncio
from uuid import UUID
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse
from sqlalchemy import text
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
    ConversationUpdate,
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
from backend.services.llm_service import LocalModelError, llm_service
from backend.services.world_service import world_service
from backend.services.generation_service import generation_service
from backend.services.entity_service import entity_service
from backend.schemas import CanonBatchRequest, EntityOperation
from backend.models import Entity, CanonOperationBatch, NamingPolicy, CanonSyncIssue

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    generation_service.recover_interrupted()
    db_gen = get_db()
    db = next(db_gen)
    try:
        book_ingestion_service.ensure_primary_book_ingested(db)
    except Exception as exc:
        logger.warning("Startup book check skipped: %s", exc)
    finally:
        db.close()
    try:
        yield
    finally:
        await generation_service.shutdown()


app = FastAPI(title="Worldbuilding Agent API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(LocalModelError)
async def local_operation_error(_request, error):
    return JSONResponse(status_code=error.status_code, content={'detail': {'error_code': error.error_code, 'message':error.message}})


@app.get('/api/worlds/{world_id}/entities')
def list_entities(world_id: str, db: Session = Depends(get_db)):
    return [entity_service.snapshot(db, entity) for entity in db.query(Entity).filter_by(world_id=world_id)]


@app.get('/api/worlds/{world_id}/entities/{entity_id}')
def get_entity(world_id: str, entity_id: str, db: Session = Depends(get_db)):
    return entity_service.snapshot(db, entity_service.resolve(db, world_id, entity_id))


@app.post('/api/worlds/{world_id}/entity-operations')
def apply_entity_operations(world_id: str, payload: CanonBatchRequest, db: Session = Depends(get_db)):
    try:
        return entity_service.execute(db, world_id, payload.operations, payload.idempotency_key, manual=True)
    except LocalModelError:
        raise
    except Exception:
        raise HTTPException(500, 'The changes could not be saved. Existing World Bible entries are unchanged.')


@app.get('/api/worlds/{world_id}/entity-operation-batches')
def list_operation_batches(world_id: str, db: Session = Depends(get_db)):
    return [{'id': batch.id, 'status': batch.status, 'result': json.loads(batch.result_json)}
            for batch in db.query(CanonOperationBatch).filter_by(world_id=world_id)]


@app.post('/api/worlds/{world_id}/entity-operation-batches/{batch_id}/approve')
def approve_entity_batch(world_id: str, batch_id: str, payload: CanonBatchRequest, db: Session = Depends(get_db)):
    batch = db.get(CanonOperationBatch, batch_id)
    if batch is None or batch.world_id != world_id or batch.status not in {'proposed', 'approved'}:
        raise HTTPException(409, 'This batch is not an unapproved proposal.')
    if batch.status == 'approved':
        return json.loads(batch.result_json)
    stored = json.loads(batch.operations_json)
    if len(stored) != len(payload.operations):
        raise HTTPException(409, 'Approve the complete proposal batch.')
    for old, new in zip(stored, payload.operations):
        new_value = new.model_dump()
        if old != new_value:
            raise HTTPException(409, 'Proposal contents changed; submit an explicit author edit instead.')
    operations = [op.model_copy(update={'decision_origin': 'AUTHOR', 'action': 'set_property' if op.action == 'propose' else op.action,
                    'source_quote': f'Author approved batch {batch_id}'}) for op in payload.operations]
    result = entity_service.execute(db, world_id, operations, payload.idempotency_key, manual='APPROVED', approval_batch_id=batch_id)
    return result


@app.get('/api/worlds/{world_id}/canon-sync-issues')
def list_canon_sync_issues(world_id: str, db: Session = Depends(get_db)):
    return [{'id': row.id, 'entity_id': row.entity_id, 'section_id': row.section_id, 'status': row.status, 'detail': row.detail}
            for row in db.query(CanonSyncIssue).filter_by(world_id=world_id)]


@app.put('/api/worlds/{world_id}/naming-policies/{entity_type}')
def update_naming_policy(world_id: str, entity_type: str, payload: Dict[str, bool], db: Session = Depends(get_db)):
    from backend.services.entity_service import TYPES
    if entity_type not in TYPES or set(payload) - {'canonical_unique', 'aliases_unique'}:
        raise HTTPException(422, 'Unsupported naming policy.')
    try:
        db.execute(text('BEGIN IMMEDIATE'))
        if db.get(World, world_id) is None:
            raise HTTPException(404, 'World not found.')
        policy = db.query(NamingPolicy).filter_by(world_id=world_id, entity_type=entity_type).first()
        if policy is None:
            policy = NamingPolicy(world_id=world_id, entity_type=entity_type,
                canonical_unique=entity_type not in {'character','family','species'}, aliases_unique=False)
            db.add(policy)
        for name, value in payload.items():
            setattr(policy, name, value)
        db.flush()
        entities = {e.id: e for e in db.query(Entity).filter_by(world_id=world_id, entity_type=entity_type)}
        entity_service.validate(db, world_id, entities)
        db.commit()
        return {'entity_type': entity_type, 'canonical_unique': policy.canonical_unique, 'aliases_unique': policy.aliases_unique}
    except Exception:
        db.rollback(); raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "llm_provider": "ollama",
        "local_model": settings.local_model,
        "local_model_status": llm_service.get_status(),
        "has_valid_tavily_key": settings.has_valid_tavily_key,
        "retrieval_method": "fts5-bm25",
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
    settings.update_model(payload.local_model)
    return {"local_model": settings.local_model, "llm_provider": "ollama"}


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
    raise HTTPException(
        status_code=409,
        detail="Book retrieval uses local SQLite full-text search; vector embeddings are not required.",
    )


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


@app.post("/api/worlds/{world_id}/consistency/facts/{fact_id}/resolve")
def resolve_consistency_flag(
    world_id: str, fact_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    fact = consistency_service.resolve_fact_consistency_flag(db, world_id, fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found.")
    return {
        "id": fact.id,
        "consistency_status": fact.consistency_status,
        "consistency_notes": fact.consistency_notes,
    }


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
    fmt: str = Query("md", description="md | pdf | docx | json"),
    db: Session = Depends(get_db),
) -> Response:
    w = world_service.get_world(db, world_id)
    if not w:
        raise HTTPException(status_code=404, detail="World not found.")
    fmt_lower = fmt.lower()
    if fmt_lower == "json":
        bundle = world_service.export_project_bundle(db, w)
        return Response(
            content=json.dumps(bundle, indent=2),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{w.name}_Project_Backup.json"'},
        )
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


@app.get("/api/worlds/{world_id}/export-project")
def export_world_project_bundle(
    world_id: str, db: Session = Depends(get_db)
) -> Response:
    w = world_service.get_world(db, world_id)
    if not w:
        raise HTTPException(status_code=404, detail="World not found.")
    bundle = world_service.export_project_bundle(db, w)
    return Response(
        content=json.dumps(bundle, indent=2),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{w.name}_Project_Backup.json"'},
    )


@app.post("/api/worlds/import-project")
def import_world_project_bundle(
    bundle: Dict[str, Any], db: Session = Depends(get_db)
) -> Dict[str, Any]:
    try:
        w = world_service.import_project_bundle(db, bundle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "genre_notes": w.genre_notes,
        "created_at": w.created_at.isoformat(),
    }


# ============================================================================
# 4. Conversational Agent & Streaming Chat Endpoints
# ============================================================================

@app.get("/api/worlds/{world_id}/conversations")
def list_conversations(
    world_id: str,
    include_archived: bool = Query(False),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    query = db.query(Conversation).filter(Conversation.world_id == world_id)
    if not include_archived:
        query = query.filter(Conversation.is_archived.is_(False))
    convs = query.order_by(Conversation.updated_at.desc()).all()
    return [
        {
            "id": c.id,
            "title": c.title,
            "focused_subject_id": c.focused_subject_id,
            "rolling_summary": c.rolling_summary or "",
            "is_archived": c.is_archived,
            "created_at": c.created_at.isoformat(),
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
    return {
        "id": c.id,
        "title": c.title,
        "focused_subject_id": c.focused_subject_id,
        "rolling_summary": c.rolling_summary or "",
        "is_archived": c.is_archived,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }


@app.put("/api/conversations/{conversation_id}")
def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if payload.title is not None and payload.title.strip():
        conv.title = payload.title.strip()
    if payload.focused_subject_id is not None:
        conv.focused_subject_id = payload.focused_subject_id
    if payload.is_archived is not None:
        conv.is_archived = payload.is_archived
    db.commit()
    db.refresh(conv)

    # If the last active thread in the world was archived, automatically create a fresh default thread
    replacement_conv = None
    if conv.is_archived:
        remaining = (
            db.query(Conversation)
            .filter(
                Conversation.world_id == conv.world_id,
                Conversation.is_archived.is_(False),
            )
            .count()
        )
        if remaining == 0:
            replacement_conv = Conversation(
                world_id=conv.world_id,
                title="New Consultation",
            )
            db.add(replacement_conv)
            db.commit()
            db.refresh(replacement_conv)

    return {
        "id": conv.id,
        "title": conv.title,
        "focused_subject_id": conv.focused_subject_id,
        "rolling_summary": conv.rolling_summary or "",
        "is_archived": conv.is_archived,
        "updated_at": conv.updated_at.isoformat(),
        "replacement_conversation_id": replacement_conv.id if replacement_conv else None,
    }


@app.delete("/api/conversations/{conversation_id}")
def archive_or_delete_conversation(
    conversation_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    conv.is_archived = True
    db.commit()

    remaining = (
        db.query(Conversation)
        .filter(
            Conversation.world_id == conv.world_id,
            Conversation.is_archived.is_(False),
        )
        .first()
    )
    if not remaining:
        remaining = Conversation(
            world_id=conv.world_id,
            title="New Consultation",
        )
        db.add(remaining)
        db.commit()
        db.refresh(remaining)

    return {
        "status": "archived",
        "active_conversation_id": remaining.id,
    }


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



class GenerationResponse(StreamingResponse):
    """Framework disconnects cancel the owned operation, not just the SSE reader."""
    def __init__(self, content, control=None):
        super().__init__(content, media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
        })
        self.control = control

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self.control is not None and not self.control.done.is_set():
                await asyncio.to_thread(self.control.cancel, "Browser connection closed.")


def _sse(event):
    return "data: " + json.dumps(event) + "\n\n"


async def _generation_events(control):
    pending = asyncio.create_task(control.events.get())
    finished = asyncio.create_task(control.done.wait())
    try:
        while True:
            ready, _ = await asyncio.wait({pending, finished},
                                          timeout=settings.generation_heartbeat_seconds,
                                          return_when=asyncio.FIRST_COMPLETED)
            if pending in ready:
                yield _sse(pending.result())
                pending = asyncio.create_task(control.events.get())
            elif finished in ready:
                # A queued event may have been picked up in the same loop tick.
                if pending.done():
                    yield _sse(pending.result())
                while not control.events.empty():
                    yield _sse(control.events.get_nowait())
                yield _sse(control.terminal)
                return
            else:
                yield _sse(dict(type="heartbeat", request_id=control.request_id,
                                conversation_id=control.conversation_id, world_id=control.world_id,
                                stage=control.state))
    finally:
        pending.cancel()
        finished.cancel()
        await asyncio.gather(pending, finished, return_exceptions=True)


async def _replay_event(event):
    yield _sse(event)


@app.post("/api/conversations/{conversation_id}/stream")
async def stream_message(conversation_id: str, payload: ChatMessageRequest) -> StreamingResponse:
    request_id, world_id, terminal = await asyncio.to_thread(generation_service.reserve, conversation_id, payload)
    if terminal is not None:
        return GenerationResponse(_replay_event(terminal))
    control = generation_service.launch(request_id, conversation_id, world_id, payload)
    return GenerationResponse(_generation_events(control), control)


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message_sync(conversation_id: str, payload: ChatMessageRequest, request: Request) -> Dict[str, Any]:
    request_id, world_id, terminal = await asyncio.to_thread(generation_service.reserve, conversation_id, payload)
    if terminal is None:
        control = generation_service.launch(request_id, conversation_id, world_id, payload, streaming=False)
        try:
            while not control.done.is_set():
                if await request.is_disconnected():
                    await asyncio.to_thread(control.cancel, "Browser connection closed.")
                await asyncio.wait({control.task}, timeout=0.25)
            terminal = control.terminal
        except asyncio.CancelledError:
            await asyncio.to_thread(control.cancel, "Browser connection closed.")
            raise
    if terminal["type"] != "complete":
        raise HTTPException(409 if terminal["type"] == "cancelled" else 502, detail=terminal)
    return terminal


@app.post("/api/conversations/{conversation_id}/requests/{request_id}/cancel")
async def cancel_generation(conversation_id: str, request_id: UUID):
    control = await asyncio.to_thread(generation_service.cancel, conversation_id, str(request_id))
    if control is not None:
        try:
            await asyncio.wait_for(control.done.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            # Cancellation has been requested; completion is still observable via SSE/status.
            pass
    return await asyncio.to_thread(generation_service.status, conversation_id, str(request_id))


@app.get("/api/conversations/{conversation_id}/requests/{request_id}")
def get_generation_status(conversation_id: str, request_id: UUID):
    return generation_service.status(conversation_id, str(request_id))
