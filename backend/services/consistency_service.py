"""Cross-domain consistency, fact indexing, and non-destructive dependency propagation."""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import or_, text
from sqlalchemy.orm import Session
from backend.models import Fact, FactRelationship, FactRevision, FactSubject, Subject

logger = logging.getLogger(__name__)


class ConsistencyService:
    """Manages FTS5 fact indexing, cross-domain fact relationships, and impact flagging."""

    def sync_fact_fts(self, db: Session, fact: Fact) -> None:
        """Synchronize a Fact record with the SQLite FTS5 virtual table."""
        db.execute(
            text("DELETE FROM facts_fts WHERE fact_id = :fid"),
            {"fid": fact.id},
        )
        if fact.canon_status in {"ESTABLISHED_CANON", "UNRESOLVED_POSSIBILITY", "HYPOTHETICAL"}:
            db.execute(
                text(
                    """
                    INSERT INTO facts_fts (fact_id, world_id, title, content)
                    VALUES (:fid, :wid, :title, :content)
                    """
                ),
                {
                    "fid": fact.id,
                    "wid": fact.world_id,
                    "title": fact.title,
                    "content": fact.content,
                },
            )

    def remove_fact_fts(self, db: Session, fact_id: str) -> None:
        """Remove a fact from the FTS5 index."""
        db.execute(
            text("DELETE FROM facts_fts WHERE fact_id = :fid"),
            {"fid": fact_id},
        )

    def create_relationship(
        self,
        db: Session,
        world_id: str,
        source_fact_id: str,
        target_fact_id: str,
        relationship_type: str = "DEPENDS_ON",
        description: str = "",
        created_by: str = "AUTHOR",
    ) -> FactRelationship:
        """Create a cross-domain link or contradiction edge between two facts."""
        existing = (
            db.query(FactRelationship)
            .filter(
                FactRelationship.world_id == world_id,
                FactRelationship.source_fact_id == source_fact_id,
                FactRelationship.target_fact_id == target_fact_id,
                FactRelationship.relationship_type == relationship_type,
            )
            .first()
        )
        if existing:
            existing.description = description or existing.description
            if relationship_type == "CONTRADICTS":
                existing.is_resolved = False
            db.commit()
            db.refresh(existing)
            return existing

        is_resolved = relationship_type != "CONTRADICTS"
        rel = FactRelationship(
            world_id=world_id,
            source_fact_id=source_fact_id,
            target_fact_id=target_fact_id,
            relationship_type=relationship_type,
            description=description,
            is_resolved=is_resolved,
            created_by=created_by,
        )
        db.add(rel)

        # If this is a contradiction, mark both facts neutrally without rewriting either
        if relationship_type == "CONTRADICTS":
            for fid in (source_fact_id, target_fact_id):
                f_obj = db.query(Fact).filter(Fact.id == fid).first()
                if f_obj:
                    f_obj.consistency_status = "CONTRADICTED"
                    f_obj.consistency_notes = description or "Flagged for potential contradiction."

        db.commit()
        db.refresh(rel)
        return rel

    def propagate_fact_change_impact(
        self, db: Session, modified_fact: Fact, change_summary: str = ""
    ) -> List[Fact]:
        """Non-destructively flag dependent facts as NEEDS_REVIEW when an underlying fact changes.

        Per Step 1 Section 6 & Step 3 Section 9:
        Changing an established fact must NEVER automatically rewrite other established facts.
        Instead, dependent facts (especially AI-approved ones) are marked NEEDS_REVIEW.
        """
        impacted_facts: List[Fact] = []

        # Find facts where source_fact DEPENDS_ON modified_fact, or modified_fact INFLUENCES/CONSTRAINS target_fact
        incoming_deps = (
            db.query(FactRelationship)
            .filter(
                FactRelationship.world_id == modified_fact.world_id,
                FactRelationship.target_fact_id == modified_fact.id,
                FactRelationship.relationship_type.in_(["DEPENDS_ON", "ELABORATES"]),
            )
            .all()
        )
        outgoing_influences = (
            db.query(FactRelationship)
            .filter(
                FactRelationship.world_id == modified_fact.world_id,
                FactRelationship.source_fact_id == modified_fact.id,
                FactRelationship.relationship_type.in_(["INFLUENCES", "CONSTRAINS"]),
            )
            .all()
        )

        dependent_ids = {r.source_fact_id for r in incoming_deps} | {
            r.target_fact_id for r in outgoing_influences
        }
        dependent_ids.discard(modified_fact.id)

        if not dependent_ids:
            return []

        dependents = (
            db.query(Fact)
            .filter(
                Fact.id.in_(dependent_ids),
                Fact.canon_status == "ESTABLISHED_CANON",
            )
            .all()
        )
        for dep in dependents:
            dep.consistency_status = "NEEDS_REVIEW"
            origin_tag = (
                "AI-generated (author-approved)"
                if dep.provenance_type == "AI_GENERATED_AUTHOR_APPROVED"
                else "author-established"
            )
            note = (
                f"Marked for review ({origin_tag}): underlying fact '{modified_fact.title}' "
                f"was modified. {change_summary}".strip()
            )
            dep.consistency_notes = note
            impacted_facts.append(dep)

        db.commit()
        return impacted_facts

    def get_world_consistency_report(
        self, db: Session, world_id: str
    ) -> Dict[str, Any]:
        """Return all facts needing review, active contradictions, and cross-domain relationships."""
        flagged_facts = (
            db.query(Fact)
            .filter(
                Fact.world_id == world_id,
                Fact.canon_status.in_(["ESTABLISHED_CANON", "PENDING_RETIREMENT_CONFIRMATION"]),
                Fact.consistency_status.in_(["NEEDS_REVIEW", "CONTRADICTED"]),
            )
            .all()
        )
        pending_retirement = (
            db.query(Fact)
            .filter(
                Fact.world_id == world_id,
                Fact.canon_status == "PENDING_RETIREMENT_CONFIRMATION",
            )
            .all()
        )
        relationships = (
            db.query(FactRelationship)
            .filter(FactRelationship.world_id == world_id)
            .order_by(FactRelationship.created_at.desc())
            .all()
        )
        return {
            "flagged_facts": [
                {
                    "id": f.id,
                    "title": f.title,
                    "content": f.content,
                    "provenance_type": f.provenance_type,
                    "consistency_status": f.consistency_status,
                    "consistency_notes": f.consistency_notes,
                }
                for f in flagged_facts
            ],
            "pending_retirement_facts": [
                {
                    "id": f.id,
                    "title": f.title,
                    "content": f.content,
                    "provenance_type": f.provenance_type,
                }
                for f in pending_retirement
            ],
            "relationships": [
                {
                    "id": r.id,
                    "source_fact_id": r.source_fact_id,
                    "target_fact_id": r.target_fact_id,
                    "relationship_type": r.relationship_type,
                    "description": r.description,
                    "is_resolved": r.is_resolved,
                    "created_by": r.created_by,
                }
                for r in relationships
            ],
        }


consistency_service = ConsistencyService()
