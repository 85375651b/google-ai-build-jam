"""Service for managing the Hybrid World Bible, Atomic Facts, and Canon Proposals."""

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from backend.models import (
    BibleSection,
    BibleSectionFact,
    BibleSectionRevision,
    CanonProposal,
    Fact,
    FactRevision,
    FactSubject,
    Subject,
    World,
)
from backend.schemas import (
    BibleSectionCreate,
    BibleSectionUpdate,
    FactCreate,
    FactUpdate,
    OrphanedFactResolutionRequest,
    ProposalActionRequest,
)
from backend.services.consistency_service import consistency_service
from backend.services.world_service import slugify


def _paragraph_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def _split_markdown_paragraphs(md: str) -> List[str]:
    """Split Markdown prose into meaningful non-header content paragraphs/blocks."""
    blocks: List[str] = []
    for raw_block in (md or "").split("\n\n"):
        cleaned = raw_block.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        blocks.append(cleaned)
    return blocks


class BibleService:
    """Manages World Bible sections, direct editing sync, atomic facts, and canon proposals."""

    # -------------------------------------------------------------------------
    # World Bible Sections
    # -------------------------------------------------------------------------

    def list_sections(self, db: Session, world_id: str) -> List[BibleSection]:
        return (
            db.query(BibleSection)
            .filter(BibleSection.world_id == world_id)
            .order_by(BibleSection.display_order.asc(), BibleSection.title.asc())
            .all()
        )

    def get_section(
        self, db: Session, world_id: str, section_id: str
    ) -> Optional[BibleSection]:
        return (
            db.query(BibleSection)
            .filter(BibleSection.id == section_id, BibleSection.world_id == world_id)
            .first()
        )

    def create_section(
        self, db: Session, world_id: str, payload: BibleSectionCreate
    ) -> BibleSection:
        base_slug = slugify(payload.title)
        slug = base_slug
        counter = 2
        while (
            db.query(BibleSection)
            .filter(BibleSection.world_id == world_id, BibleSection.slug == slug)
            .first()
        ):
            slug = f"{base_slug}-{counter}"
            counter += 1

        max_order = (
            db.query(BibleSection).filter(BibleSection.world_id == world_id).count()
        )
        section = BibleSection(
            world_id=world_id,
            parent_section_id=payload.parent_section_id,
            primary_subject_id=payload.primary_subject_id,
            title=payload.title.strip(),
            slug=slug,
            markdown_content=payload.markdown_content or "",
            display_order=max_order,
            dominant_provenance="AUTHOR_CREATED",
            version_number=1,
        )
        db.add(section)
        db.flush()

        # Record initial revision
        rev = BibleSectionRevision(
            section_id=section.id,
            version_number=1,
            previous_markdown="",
            new_markdown=section.markdown_content,
            edit_origin="AUTHOR_EDITOR",
            author_note="Section created",
        )
        db.add(rev)

        if section.markdown_content.strip():
            self._sync_direct_markdown_to_facts(db, section)

        db.commit()
        db.refresh(section)
        return section

    def update_section_directly(
        self, db: Session, section: BibleSection, payload: BibleSectionUpdate
    ) -> Tuple[BibleSection, List[Fact]]:
        """Update a World Bible section directly from the Author Editor.

        Enforces Optimistic Concurrency Control (version_number), records a revision,
        synchronizes modified/new paragraphs into Tier-1 AUTHOR_CREATED facts, and
        implements Decision 3(b): returns any linked facts whose prose was deleted
        so the author can confirm whether to retire or keep them active.
        """
        from sqlalchemy import update
        expected = payload.expected_version_number if payload.expected_version_number is not None else section.version_number
        checked = db.execute(update(BibleSection).where(BibleSection.id == section.id,
            BibleSection.version_number == expected).values(version_number=expected).execution_options(synchronize_session=False))
        if checked.rowcount != 1:
            raise ValueError('Version conflict: reload the author-edited section before saving.')
        if (
            payload.expected_version_number is not None
            and payload.expected_version_number != section.version_number
        ):
            raise ValueError(
                f"Version conflict: section is at version {section.version_number}, "
                f"but editor submitted version {payload.expected_version_number}."
            )

        prev_md = section.markdown_content
        if payload.title is not None and payload.title.strip():
            section.title = payload.title.strip()
        if payload.primary_subject_id is not None:
            section.primary_subject_id = payload.primary_subject_id or None

        orphaned_facts: List[Fact] = []
        if payload.markdown_content is not None and payload.markdown_content != prev_md:
            section.markdown_content = payload.markdown_content
            section.version_number += 1
            section.has_pending_review = False

            rev = BibleSectionRevision(
                section_id=section.id,
                version_number=section.version_number,
                previous_markdown=prev_md,
                new_markdown=section.markdown_content,
                edit_origin="AUTHOR_EDITOR",
                author_note=payload.author_note or "Direct edit by author",
            )
            db.add(rev)

            orphaned_facts = self._sync_direct_markdown_to_facts(db, section)

        if payload.markdown_content is not None and payload.markdown_content != prev_md:
            from backend.services.entity_service import entity_service
            entity_service.sync_manual_section(db, section, prev_md)
        db.commit()
        db.refresh(section)
        return section, orphaned_facts

    def _sync_direct_markdown_to_facts(
        self, db: Session, section: BibleSection
    ) -> List[Fact]:
        """Synchronize direct author edits in a World Bible section with atomic Facts."""
        paragraphs = _split_markdown_paragraphs(section.markdown_content)
        existing_links = (
            db.query(BibleSectionFact)
            .filter(BibleSectionFact.section_id == section.id)
            .all()
        )
        linked_facts_by_id: Dict[str, Fact] = {}
        for link in existing_links:
            f_obj = db.query(Fact).filter(Fact.id == link.fact_id).first()
            if f_obj:
                linked_facts_by_id[f_obj.id] = f_obj

        matched_fact_ids = set()

        for idx, para in enumerate(paragraphs):
            clean_para = para.lstrip("-*• ").strip()
            if not clean_para:
                continue
            p_hash = _paragraph_hash(clean_para)

            # 1. Check exact content or anchor match
            matched_fact: Optional[Fact] = None
            for link in existing_links:
                f_obj = linked_facts_by_id.get(link.fact_id)
                if not f_obj:
                    continue
                if (
                    link.paragraph_anchor == p_hash
                    or f_obj.content.strip() == clean_para
                ):
                    matched_fact = f_obj
                    break

            # 2. If not exact match, check if an unmatched linked fact in the same position was edited
            if matched_fact is None and len(paragraphs) == len(existing_links):
                candidate_link = existing_links[idx]
                candidate_fact = linked_facts_by_id.get(candidate_link.fact_id)
                if candidate_fact and candidate_fact.id not in matched_fact_ids:
                    matched_fact = candidate_fact
                    prev_content = matched_fact.content
                    prev_status = matched_fact.canon_status
                    prev_prov = matched_fact.provenance_type

                    matched_fact.content = clean_para
                    matched_fact.title = self._derive_fact_title(section.title, clean_para)
                    matched_fact.provenance_type = "AUTHOR_CREATED"
                    matched_fact.authority_tier = 1
                    matched_fact.canon_status = "ESTABLISHED_CANON"
                    matched_fact.version_number += 1
                    candidate_link.paragraph_anchor = p_hash

                    f_rev = FactRevision(
                        fact_id=matched_fact.id,
                        version_number=matched_fact.version_number,
                        previous_content=prev_content,
                        new_content=clean_para,
                        previous_canon_status=prev_status,
                        new_canon_status="ESTABLISHED_CANON",
                        previous_provenance_type=prev_prov,
                        new_provenance_type="AUTHOR_CREATED",
                        changed_by="AUTHOR_DIRECT_EDIT",
                        change_reason=f"Direct edit in World Bible section '{section.title}'",
                    )
                    db.add(f_rev)
                    consistency_service.sync_fact_fts(db, matched_fact)
                    consistency_service.propagate_fact_change_impact(
                        db,
                        matched_fact,
                        change_summary=f"Updated directly in World Bible section '{section.title}'.",
                    )

            if matched_fact is not None:
                matched_fact_ids.add(matched_fact.id)
                if matched_fact.canon_status == "PENDING_RETIREMENT_CONFIRMATION":
                    matched_fact.canon_status = "ESTABLISHED_CANON"
                    consistency_service.sync_fact_fts(db, matched_fact)
            else:
                # Create a new Tier-1 AUTHOR_CREATED fact from the newly written paragraph
                new_fact = Fact(
                    world_id=section.world_id,
                    title=self._derive_fact_title(section.title, clean_para),
                    content=clean_para,
                    canon_status="ESTABLISHED_CANON",
                    provenance_type="AUTHOR_CREATED",
                    authority_tier=1,
                    version_number=1,
                )
                db.add(new_fact)
                db.flush()

                if section.primary_subject_id:
                    db.add(
                        FactSubject(
                            fact_id=new_fact.id,
                            subject_id=section.primary_subject_id,
                            is_primary=True,
                        )
                    )

                db.add(
                    BibleSectionFact(
                        section_id=section.id,
                        fact_id=new_fact.id,
                        paragraph_anchor=p_hash,
                    )
                )
                db.add(
                    FactRevision(
                        fact_id=new_fact.id,
                        version_number=1,
                        previous_content=None,
                        new_content=clean_para,
                        previous_canon_status=None,
                        new_canon_status="ESTABLISHED_CANON",
                        previous_provenance_type=None,
                        new_provenance_type="AUTHOR_CREATED",
                        changed_by="AUTHOR_DIRECT_EDIT",
                        change_reason=f"Created via direct edit in World Bible section '{section.title}'",
                    )
                )
                consistency_service.sync_fact_fts(db, new_fact)
                matched_fact_ids.add(new_fact.id)

        # Decision 3(b): Check if any previously linked ESTABLISHED_CANON facts were removed from prose
        orphaned_facts: List[Fact] = []
        for fid, f_obj in linked_facts_by_id.items():
            if fid not in matched_fact_ids and f_obj.canon_status == "ESTABLISHED_CANON":
                f_obj.canon_status = "PENDING_RETIREMENT_CONFIRMATION"
                orphaned_facts.append(f_obj)

        self._recompute_section_provenance(db, section)
        return orphaned_facts

    def resolve_orphaned_facts(
        self,
        db: Session,
        world_id: str,
        payload: OrphanedFactResolutionRequest,
    ) -> List[Fact]:
        """Handle Decision 3(b) author confirmation when paragraphs are deleted in the World Bible editor."""
        facts = (
            db.query(Fact)
            .filter(Fact.world_id == world_id, Fact.id.in_(payload.fact_ids))
            .all()
        )
        action = payload.action.upper()
        for f_obj in facts:
            prev_status = f_obj.canon_status
            if action == "RETIRE":
                f_obj.canon_status = "SUPERSEDED"
                consistency_service.remove_fact_fts(db, f_obj.id)
                consistency_service.propagate_fact_change_impact(
                    db,
                    f_obj,
                    change_summary="Fact retired after its paragraph was removed from the World Bible.",
                )
            else:
                f_obj.canon_status = "ESTABLISHED_CANON"
                consistency_service.sync_fact_fts(db, f_obj)

            f_obj.version_number += 1
            db.add(
                FactRevision(
                    fact_id=f_obj.id,
                    version_number=f_obj.version_number,
                    previous_content=f_obj.content,
                    new_content=f_obj.content,
                    previous_canon_status=prev_status,
                    new_canon_status=f_obj.canon_status,
                    previous_provenance_type=f_obj.provenance_type,
                    new_provenance_type=f_obj.provenance_type,
                    changed_by="AUTHOR_DIRECT_EDIT",
                    change_reason=f"Orphaned fact resolution: {action}",
                )
            )
        db.commit()
        return facts

    def delete_section(self, db: Session, section: BibleSection) -> None:
        db.delete(section)
        db.commit()

    def revert_section_to_version(
        self, db: Session, section: BibleSection, target_version: int
    ) -> BibleSection:
        """Revert a World Bible section to a previous revision."""
        target_rev = (
            db.query(BibleSectionRevision)
            .filter(
                BibleSectionRevision.section_id == section.id,
                BibleSectionRevision.version_number == target_version,
            )
            .first()
        )
        if not target_rev:
            raise ValueError(f"Revision {target_version} not found for this section.")

        prev_md = section.markdown_content
        section.markdown_content = target_rev.new_markdown
        section.version_number += 1

        rev = BibleSectionRevision(
            section_id=section.id,
            version_number=section.version_number,
            previous_markdown=prev_md,
            new_markdown=section.markdown_content,
            edit_origin="REVERT",
            author_note=f"Reverted to version {target_version}",
        )
        db.add(rev)
        self._sync_direct_markdown_to_facts(db, section)
        db.commit()
        db.refresh(section)
        return section

    def apply_agent_section_edit(
        self,
        db: Session,
        world_id: str,
        section_slug: str,
        section_title: str,
        updated_markdown: str,
        summary_of_changes: str = "Updated by Agent from Author Brain Dump / Instructions",
        provenance: str = "AUTHOR_CREATED",
        expected_version_number: Optional[int] = None,
    ) -> Optional[BibleSection]:
        """Apply a structured World Bible section update, fitting into default categories first."""
        clean_slug = slugify(section_slug or section_title)
        combined_label = f"{clean_slug} {section_title.lower()}"

        # Never create or populate non-worldbuilding sections (plot outlines, character sheets, themes)
        if any(
            non_wb in combined_label
            for non_wb in ("plot-structure", "characters-plot", "character-arc", "story-outline", "act-1", "themes")
        ):
            return None

        # Map common topical headings onto the 15 default World Bible categories first
        default_category_Rules = [
            (("climate", "smog", "ecology", "environment", "weather", "atmosphere", "setting", "premise"), "climate-ecology"),
            (("language", "slang", "linguistics", "terminology", "dialect", "naming", "vocab"), "languages-linguistics"),
            (("tech", "network", "collaboration", "darknet", "software", "server", "backdoor", "engineering", "platform", "information", "identifier", "digital", "online"), "technology-engineering"),
            (("politic", "governance", "organization", "faction", "territory", "government", "law"), "politics-governance"),
            (("culture", "societ", "daily-life", "social", "custom", "community"), "cultures-societies"),
            (("histor", "chronolog", "timeline", "origin", "founding"), "history-chronology"),
            (("econom", "trade", "currency", "commerce", "market", "finance"), "economics-trade"),
            (("geograph", "geolog", "terrain", "location", "map", "city", "region"), "geography-geology"),
            (("war", "militar", "weapon", "security", "conflict", "defense"), "warfare-military"),
            (("architect", "material", "building", "habitat", "infrastructure", "dress"), "architecture-material-culture"),
            (("religio", "myth", "ritual", "faith", "deity", "church"), "religion-mythology"),
            (("biolog", "evolution", "flora", "fauna", "species", "health"), "biology-evolution"),
            (("cosmolog", "physics", "magic", "fundamental"), "cosmology-physics"),
            (("philosoph", "ideolog", "ethics", "belief"), "philosophy-ideologies"),
            (("demograph", "population", "census", "migration"), "demographics-population"),
        ]

        all_secs = self.list_sections(db, world_id)
        sec_by_slug = {s.slug: s for s in all_secs}

        section = sec_by_slug.get(clean_slug)
        if not section:
            for keywords, target_default_slug in default_category_Rules:
                if any(kw in combined_label for kw in keywords) and target_default_slug in sec_by_slug:
                    section = sec_by_slug[target_default_slug]
                    break

        if not section:
            for s in all_secs:
                if s.title.lower() == section_title.strip().lower() or clean_slug in s.slug or s.slug in clean_slug:
                    section = s
                    break

        if not section:
            max_order = (
                db.query(BibleSection).filter(BibleSection.world_id == world_id).count()
            )
            section = BibleSection(
                world_id=world_id,
                title=section_title.strip() or "World Notes",
                slug=clean_slug,
                markdown_content=updated_markdown.strip(),
                display_order=max_order,
                dominant_provenance=(
                    "AUTHOR_CREATED"
                    if provenance == "AUTHOR_CREATED"
                    else "AI_GENERATED_AUTHOR_APPROVED"
                ),
                version_number=1,
            )
            db.add(section)
            db.flush()
            db.add(
                BibleSectionRevision(
                    section_id=section.id,
                    version_number=1,
                    previous_markdown="",
                    new_markdown=section.markdown_content,
                    edit_origin="AUTHOR_APPROVED_PROPOSAL",
                    author_note=summary_of_changes,
                )
            )
        else:
            prev_md = section.markdown_content or ""
            new_md = updated_markdown.strip()
            from sqlalchemy import update
            expected = expected_version_number if expected_version_number is not None else section.version_number
            checked = db.execute(update(BibleSection).where(BibleSection.id == section.id,
                BibleSection.version_number == expected).values(version_number=expected).execution_options(synchronize_session=False))
            if checked.rowcount != 1:
                raise ValueError('The section changed before the agent update could be applied.')
            from backend.models import EntityProperty
            for fact in db.query(Fact).join(EntityProperty, EntityProperty.fact_id == Fact.id).join(BibleSectionFact,
                    BibleSectionFact.fact_id == Fact.id).filter(BibleSectionFact.section_id == section.id):
                if fact.content in prev_md and fact.content not in new_md:
                    raise ValueError('A tracked entity property must be changed through a structured operation.')
            if new_md and new_md != prev_md.strip():
                section.markdown_content = new_md
                section.version_number += 1
                if not prev_md.strip():
                    section.dominant_provenance = (
                        "AUTHOR_CREATED"
                        if provenance == "AUTHOR_CREATED"
                        else "AI_GENERATED_AUTHOR_APPROVED"
                    )
                elif section.dominant_provenance != provenance:
                    section.dominant_provenance = "MIXED"
                db.add(
                    BibleSectionRevision(
                        section_id=section.id,
                        version_number=section.version_number,
                        previous_markdown=prev_md,
                        new_markdown=new_md,
                        edit_origin="AUTHOR_APPROVED_PROPOSAL",
                        author_note=summary_of_changes,
                    )
                )

        self._sync_direct_markdown_to_facts(db, section)
        db.commit()
        db.refresh(section)
        return section

    # -------------------------------------------------------------------------
    # Atomic Facts CRUD
    # -------------------------------------------------------------------------

    def list_facts(
        self,
        db: Session,
        world_id: str,
        canon_status: Optional[str] = None,
        subject_id: Optional[str] = None,
    ) -> List[Fact]:
        query = db.query(Fact).filter(Fact.world_id == world_id)
        if canon_status:
            query = query.filter(Fact.canon_status == canon_status)
        if subject_id:
            query = query.join(FactSubject).filter(FactSubject.subject_id == subject_id)
        return query.order_by(Fact.updated_at.desc()).all()

    def create_fact(
        self,
        db: Session,
        world_id: str,
        payload: FactCreate,
        source_message_id: Optional[str] = None,
        sync_to_bible_section: bool = True,
    ) -> Fact:
        existing = db.query(Fact).filter_by(world_id=world_id, content=payload.content.strip(),
            canon_status=payload.canon_status, provenance_type=payload.provenance_type).first()
        if existing:
            return existing
        tier = 1 if payload.provenance_type == "AUTHOR_CREATED" else 2
        fact = Fact(
            world_id=world_id,
            title=payload.title.strip(),
            content=payload.content.strip(),
            canon_status=payload.canon_status,
            provenance_type=payload.provenance_type,
            authority_tier=tier,
            source_message_id=source_message_id,
            version_number=1,
        )
        db.add(fact)
        db.flush()

        for idx, sid in enumerate(payload.subject_ids):
            db.add(
                FactSubject(
                    fact_id=fact.id,
                    subject_id=sid,
                    is_primary=(idx == 0),
                )
            )

        db.add(
            FactRevision(
                fact_id=fact.id,
                version_number=1,
                previous_content=None,
                new_content=fact.content,
                previous_canon_status=None,
                new_canon_status=fact.canon_status,
                previous_provenance_type=None,
                new_provenance_type=fact.provenance_type,
                changed_by="AUTHOR_DIRECT_EDIT",
                change_reason="Fact created",
            )
        )
        consistency_service.sync_fact_fts(db, fact)

        if sync_to_bible_section and fact.canon_status == "ESTABLISHED_CANON":
            primary_sid = payload.subject_ids[0] if payload.subject_ids else None
            self._append_fact_to_matching_bible_section(db, world_id, fact, primary_sid)

        db.commit()
        db.refresh(fact)
        return fact

    def update_fact(
        self, db: Session, fact: Fact, payload: FactUpdate
    ) -> Tuple[Fact, List[Fact]]:
        """Update an atomic fact, record a revision, and propagate non-destructive review flags."""
        prev_content = fact.content
        prev_status = fact.canon_status
        prev_prov = fact.provenance_type

        if payload.title is not None:
            fact.title = payload.title.strip()
        if payload.content is not None:
            fact.content = payload.content.strip()
            # Direct author edit promotes authority to AUTHOR_CREATED (Tier 1)
            fact.provenance_type = "AUTHOR_CREATED"
            fact.authority_tier = 1
        if payload.canon_status is not None:
            fact.canon_status = payload.canon_status

        fact.version_number += 1
        fact.consistency_status = "VALID"
        fact.consistency_notes = ""

        if payload.subject_ids is not None:
            db.query(FactSubject).filter(FactSubject.fact_id == fact.id).delete()
            for idx, sid in enumerate(payload.subject_ids):
                db.add(
                    FactSubject(
                        fact_id=fact.id,
                        subject_id=sid,
                        is_primary=(idx == 0),
                    )
                )

        db.add(
            FactRevision(
                fact_id=fact.id,
                version_number=fact.version_number,
                previous_content=prev_content,
                new_content=fact.content,
                previous_canon_status=prev_status,
                new_canon_status=fact.canon_status,
                previous_provenance_type=prev_prov,
                new_provenance_type=fact.provenance_type,
                changed_by="AUTHOR_DIRECT_EDIT",
                change_reason=payload.change_reason or "Author updated fact",
            )
        )
        consistency_service.sync_fact_fts(db, fact)

        # Also update the corresponding paragraph in any linked World Bible section
        if prev_content != fact.content:
            links = (
                db.query(BibleSectionFact)
                .filter(BibleSectionFact.fact_id == fact.id)
                .all()
            )
            for link in links:
                sec = (
                    db.query(BibleSection)
                    .filter(BibleSection.id == link.section_id)
                    .first()
                )
                if sec and prev_content in sec.markdown_content:
                    old_md = sec.markdown_content
                    sec.markdown_content = sec.markdown_content.replace(
                        prev_content, fact.content
                    )
                    sec.version_number += 1
                    link.paragraph_anchor = _paragraph_hash(fact.content)
                    db.add(
                        BibleSectionRevision(
                            section_id=sec.id,
                            version_number=sec.version_number,
                            previous_markdown=old_md,
                            new_markdown=sec.markdown_content,
                            edit_origin="AUTHOR_EDITOR",
                            author_note=f"Synced from updated fact '{fact.title}'",
                        )
                    )

        impacted = consistency_service.propagate_fact_change_impact(
            db,
            fact,
            change_summary=payload.change_reason or "Underlying established fact was modified.",
        )
        db.commit()
        db.refresh(fact)
        return fact, impacted

    # -------------------------------------------------------------------------
    # Controlled Canon Proposals (Step 3 Section 6.1)
    # -------------------------------------------------------------------------

    def list_proposals(
        self, db: Session, world_id: str, status: Optional[str] = "PENDING"
    ) -> List[CanonProposal]:
        query = db.query(CanonProposal).filter(CanonProposal.world_id == world_id)
        if status:
            query = query.filter(CanonProposal.status == status)
        return query.order_by(CanonProposal.created_at.desc()).all()

    def stage_proposal(
        self,
        db: Session,
        world_id: str,
        proposed_title: str,
        proposed_content: str,
        proposed_provenance: str = "AUTHOR_CREATED",
        proposal_type: str = "NEW_FACT",
        conversation_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        target_subject_id: Optional[str] = None,
        target_section_id: Optional[str] = None,
        target_fact_id: Optional[str] = None,
    ) -> CanonProposal:
        """Create a pending canon proposal awaiting explicit author approval."""
        # Avoid duplicate pending proposals with identical content
        existing = (
            db.query(CanonProposal)
            .filter(
                CanonProposal.world_id == world_id,
                CanonProposal.status == "PENDING",
                CanonProposal.proposed_content == proposed_content.strip(),
            )
            .first()
        )
        if existing:
            return existing

        proposal = CanonProposal(
            world_id=world_id,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            proposal_type=proposal_type,
            target_fact_id=target_fact_id,
            target_section_id=target_section_id,
            target_subject_id=target_subject_id,
            proposed_title=proposed_title.strip(),
            proposed_content=proposed_content.strip(),
            proposed_provenance=proposed_provenance,
            status="PENDING",
        )
        db.add(proposal)
        db.flush()
        # Keep existing proposal buttons, but snapshot structured targets now so
        # approving a stale naming suggestion cannot overwrite a later author edit.
        from backend.services.canon_language import compile_decisions
        from backend.schemas import AuthorDecisionSchema
        from backend.services.entity_service import entity_service, dump
        from backend.models import CanonOperationBatch
        import hashlib
        operations = compile_decisions([AuthorDecisionSchema(kind='DECLARATION', source_quote=proposal.proposed_content)], proposal.proposed_content)
        if operations:
            for operation in operations:
                entity = entity_service.resolve(db, world_id, ref=operation.entity_ref, required=False)
                operation.expected_version = entity.version_number if entity else None
                operation.entity_id = entity.id if entity else None
            encoded = dump([operation.model_dump() for operation in operations])
            db.add(CanonOperationBatch(world_id=world_id, idempotency_key=f'legacy-proposal:{proposal.id}',
                fingerprint=hashlib.sha256(encoded.encode()).hexdigest(), status='proposed', operations_json=encoded,
                result_json=dump({'status':'proposed', 'message':'Not yet canon.'})))
        db.commit()
        db.refresh(proposal)
        return proposal

    def resolve_proposal(
        self,
        db: Session,
        proposal: CanonProposal,
        payload: ProposalActionRequest,
    ) -> Dict[str, Any]:
        """Approve, edit-and-approve, or reject a staged canon proposal."""
        action = payload.action.upper()
        if proposal.status in {'APPROVED', 'EDITED_AND_APPROVED'}:
            return {'proposal_status': proposal.status, 'fact_id': proposal.target_fact_id}
        if proposal.status == 'REJECTED' and action != 'REJECT':
            raise ValueError('A rejected suggestion requires a new explicit author decision.')
        now = datetime.now(timezone.utc)

        final_title = (payload.edited_title or proposal.proposed_title).strip()
        final_content = (payload.edited_content or proposal.proposed_content).strip()

        if action == "REJECT":
            proposal.status = "REJECTED"
            proposal.resolved_at = now
            # Record as a REJECTED fact so the agent never re-introduces it (Step 2 Test I)
            rejected_fact = Fact(
                world_id=proposal.world_id,
                title=final_title,
                content=final_content,
                canon_status="REJECTED",
                provenance_type="AI_PROPOSED_UNVERIFIED",
                authority_tier=4,
                source_message_id=proposal.source_message_id,
            )
            db.add(rejected_fact)
            db.commit()
            return {"proposal_status": "REJECTED", "rejected_fact_id": rejected_fact.id}

        if action not in {"APPROVE", "EDIT_AND_APPROVE"}:
            raise ValueError(f"Unsupported proposal action: {action}")

        from backend.models import FactRelationship
        unresolved = db.query(FactRelationship).join(Fact, Fact.id == FactRelationship.source_fact_id).filter(
            Fact.source_message_id == proposal.source_message_id, Fact.world_id == proposal.world_id,
            FactRelationship.relationship_type == 'CONTRADICTS', FactRelationship.is_resolved == False).first()
        if proposal.source_message_id and unresolved:
            raise ValueError('Clarify or resolve the recorded contradiction before approving this proposal.')

        from backend.models import CanonOperationBatch, EntityProperty
        from backend.schemas import EntityOperation
        from backend.services.entity_service import entity_service
        structured = db.query(CanonOperationBatch).filter_by(world_id=proposal.world_id,
            idempotency_key=f'legacy-proposal:{proposal.id}').first()
        if structured and action == 'APPROVE':
            from sqlalchemy import update
            db.execute(update(World).where(World.id == proposal.world_id).values(name=World.name))
            import json
            operations = [EntityOperation.model_validate(item).model_copy(update={
                'source_quote': f'Author approved proposal {proposal.id}', 'decision_origin':'AUTHOR'})
                for item in json.loads(structured.operations_json)]
            result = entity_service.apply(db, proposal.world_id, operations, f'approved-proposal:{proposal.id}', manual='APPROVED')
            entity_id = result['entities'][0] if result['entities'] else None
            prop = db.query(EntityProperty).filter_by(entity_id=entity_id, property_key='canonical_name').first()
            proposal.status, proposal.resolved_at = 'APPROVED', now
            proposal.target_fact_id = prop.fact_id if prop else None
            structured.status = 'approved'; structured.result_json = json.dumps(result)
            db.commit()
            return {'proposal_status':'APPROVED', 'fact_id':proposal.target_fact_id, 'canon_result':result}
        if structured and action == 'EDIT_AND_APPROVE':
            raise ValueError('Edit the structured property explicitly so its current version can be confirmed.')

        # Determine provenance: if author edited the text or originally created it, mark accordingly
        provenance = proposal.proposed_provenance
        if provenance not in {"AUTHOR_CREATED", "AI_GENERATED_AUTHOR_APPROVED"}:
            provenance = "AUTHOR_CREATED"
        authority_tier = 1 if provenance == "AUTHOR_CREATED" else 2

        if proposal.target_fact_id:
            target_fact = (
                db.query(Fact)
                .filter(
                    Fact.id == proposal.target_fact_id,
                    Fact.world_id == proposal.world_id,
                )
                .first()
            )
        else:
            target_fact = None

        if target_fact:
            prev_content = target_fact.content
            prev_status = target_fact.canon_status
            prev_prov = target_fact.provenance_type

            target_fact.title = final_title
            target_fact.content = final_content
            target_fact.canon_status = "ESTABLISHED_CANON"
            target_fact.provenance_type = provenance
            target_fact.authority_tier = authority_tier
            target_fact.version_number += 1

            db.add(
                FactRevision(
                    fact_id=target_fact.id,
                    version_number=target_fact.version_number,
                    previous_content=prev_content,
                    new_content=final_content,
                    previous_canon_status=prev_status,
                    new_canon_status="ESTABLISHED_CANON",
                    previous_provenance_type=prev_prov,
                    new_provenance_type=provenance,
                    changed_by="AUTHOR_CHAT_CONFIRMATION",
                    change_reason=f"Approved proposal ({action})",
                )
            )
            consistency_service.sync_fact_fts(db, target_fact)
            consistency_service.propagate_fact_change_impact(
                db, target_fact, change_summary="Updated via approved proposal."
            )
            fact_obj = target_fact
        else:
            fact_obj = Fact(
                world_id=proposal.world_id,
                title=final_title,
                content=final_content,
                canon_status="ESTABLISHED_CANON",
                provenance_type=provenance,
                authority_tier=authority_tier,
                source_message_id=proposal.source_message_id,
                version_number=1,
            )
            db.add(fact_obj)
            db.flush()

            if proposal.target_subject_id:
                db.add(
                    FactSubject(
                        fact_id=fact_obj.id,
                        subject_id=proposal.target_subject_id,
                        is_primary=True,
                    )
                )

            db.add(
                FactRevision(
                    fact_id=fact_obj.id,
                    version_number=1,
                    previous_content=None,
                    new_content=final_content,
                    previous_canon_status=None,
                    new_canon_status="ESTABLISHED_CANON",
                    previous_provenance_type=None,
                    new_provenance_type=provenance,
                    changed_by="AUTHOR_CHAT_CONFIRMATION",
                    change_reason=f"Approved proposal ({action})",
                )
            )
            consistency_service.sync_fact_fts(db, fact_obj)
            consistency_service.infer_and_link_dependencies(db, proposal.world_id, fact_obj)

        section_id = payload.target_section_id or proposal.target_section_id
        updated_section = self._append_fact_to_matching_bible_section(
            db,
            proposal.world_id,
            fact_obj,
            proposal.target_subject_id,
            explicit_section_id=section_id,
        )

        proposal.status = "EDITED_AND_APPROVED" if action == "EDIT_AND_APPROVE" else "APPROVED"
        proposal.resolved_at = now
        db.commit()

        return {
            "proposal_status": proposal.status,
            "fact_id": fact_obj.id,
            "section_id": updated_section.id if updated_section else None,
            "provenance_type": fact_obj.provenance_type,
        }

    # -------------------------------------------------------------------------
    # Export & Internal Helpers
    # -------------------------------------------------------------------------

    def export_world_bible_markdown(self, db: Session, world: World) -> str:
        """Export the complete World Bible as a clean, standalone Markdown reference document."""
        sections = self.list_sections(db, world.id)
        lines: List[str] = [
            f"# {world.name} — World Bible",
            "",
        ]
        if world.description:
            lines.extend([f"*{world.description}*", ""])
        if world.genre_notes:
            lines.extend([f"**Setting & Tone Notes:** {world.genre_notes}", ""])

        non_empty_sections = [s for s in sections if (s.markdown_content or "").strip()]
        if non_empty_sections:
            lines.append("## Table of Contents")
            for idx, sec in enumerate(non_empty_sections, start=1):
                lines.append(f"{idx}. [{sec.title}](#{sec.slug})")
            lines.append("")

        for sec in sections:
            if not (sec.markdown_content or "").strip():
                continue
            lines.append(f"## {sec.title}")
            lines.append("")
            lines.append(sec.markdown_content.strip())
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def export_world_bible_docx(self, db: Session, world: World) -> bytes:
        """Export the complete World Bible as a standalone DOCX file."""
        import io
        from docx import Document

        doc = Document()
        doc.add_heading(f"{world.name} — World Bible", level=0)
        if world.description:
            doc.add_paragraph(world.description)
        if world.genre_notes:
            doc.add_paragraph(f"Setting & Tone Notes: {world.genre_notes}")

        sections = self.list_sections(db, world.id)
        for sec in sections:
            content = (sec.markdown_content or "").strip()
            if not content:
                continue
            doc.add_heading(sec.title, level=1)
            for block in content.split("\n\n"):
                b_clean = block.strip()
                if not b_clean:
                    continue
                if b_clean.startswith("### "):
                    doc.add_heading(b_clean[4:].strip(), level=3)
                elif b_clean.startswith("## "):
                    doc.add_heading(b_clean[3:].strip(), level=2)
                else:
                    doc.add_paragraph(b_clean)

        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    def export_world_bible_pdf(self, db: Session, world: World) -> bytes:
        """Export the complete World Bible as a standalone PDF file."""
        from fpdf import FPDF

        def _ascii_safe(txt: str) -> str:
            replacements = {
                "\u2014": "--",
                "\u2013": "-",
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2026": "...",
                "\u2022": "*",
            }
            for k, v in replacements.items():
                txt = txt.replace(k, v)
            return txt.encode("latin-1", errors="replace").decode("latin-1")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.multi_cell(0, 11, _ascii_safe(f"{world.name} -- World Bible"))
        pdf.ln(3)

        if world.description:
            pdf.set_font("Helvetica", "I", 11)
            pdf.multi_cell(0, 6, _ascii_safe(world.description))
            pdf.ln(4)

        sections = self.list_sections(db, world.id)
        for sec in sections:
            content = (sec.markdown_content or "").strip()
            if not content:
                continue
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 9, _ascii_safe(sec.title))
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 11)
            for block in content.split("\n\n"):
                b_clean = block.strip()
                if not b_clean:
                    continue
                pdf.multi_cell(0, 6, _ascii_safe(b_clean))
                pdf.ln(3)
            pdf.ln(3)

        return bytes(pdf.output())

    def _append_fact_to_matching_bible_section(
        self,
        db: Session,
        world_id: str,
        fact: Fact,
        subject_id: Optional[str] = None,
        explicit_section_id: Optional[str] = None,
    ) -> Optional[BibleSection]:
        """Append an approved canonical fact into its corresponding World Bible section."""
        section: Optional[BibleSection] = None
        if explicit_section_id:
            section = (
                db.query(BibleSection)
                .filter(
                    BibleSection.id == explicit_section_id,
                    BibleSection.world_id == world_id,
                )
                .first()
            )
        if not section and subject_id:
            section = (
                db.query(BibleSection)
                .filter(
                    BibleSection.world_id == world_id,
                    BibleSection.primary_subject_id == subject_id,
                )
                .first()
            )
        if not section:
            section = (
                db.query(BibleSection)
                .filter(BibleSection.world_id == world_id)
                .order_by(BibleSection.display_order.asc())
                .first()
            )
        if not section:
            return None

        prev_md = section.markdown_content or ""
        if fact.content not in prev_md:
            if prev_md.strip():
                section.markdown_content = f"{prev_md.rstrip()}\n\n{fact.content}"
            else:
                section.markdown_content = fact.content
            section.version_number += 1

            db.add(
                BibleSectionRevision(
                    section_id=section.id,
                    version_number=section.version_number,
                    previous_markdown=prev_md,
                    new_markdown=section.markdown_content,
                    edit_origin="AUTHOR_APPROVED_PROPOSAL",
                    author_note=f"Added approved fact '{fact.title}' ({fact.provenance_type})",
                )
            )

        existing_link = (
            db.query(BibleSectionFact)
            .filter(
                BibleSectionFact.section_id == section.id,
                BibleSectionFact.fact_id == fact.id,
            )
            .first()
        )
        if not existing_link:
            db.add(
                BibleSectionFact(
                    section_id=section.id,
                    fact_id=fact.id,
                    paragraph_anchor=_paragraph_hash(fact.content),
                )
            )

        self._recompute_section_provenance(db, section)
        return section

    def _recompute_section_provenance(
        self, db: Session, section: BibleSection
    ) -> None:
        links = (
            db.query(BibleSectionFact)
            .filter(BibleSectionFact.section_id == section.id)
            .all()
        )
        if not links:
            section.dominant_provenance = "AUTHOR_CREATED"
            return
        provs = set()
        for link in links:
            f_obj = db.query(Fact).filter(Fact.id == link.fact_id).first()
            if f_obj and f_obj.canon_status == "ESTABLISHED_CANON":
                provs.add(f_obj.provenance_type)
        if len(provs) > 1:
            section.dominant_provenance = "MIXED"
        elif "AI_GENERATED_AUTHOR_APPROVED" in provs:
            section.dominant_provenance = "AI_GENERATED_AUTHOR_APPROVED"
        else:
            section.dominant_provenance = "AUTHOR_CREATED"

    @staticmethod
    def _derive_fact_title(section_title: str, paragraph: str) -> str:
        first_sentence = paragraph.split(".")[0].strip()
        if 5 <= len(first_sentence) <= 90:
            return first_sentence
        return f"{section_title}: {paragraph[:65].strip()}..."


bible_service = BibleService()
