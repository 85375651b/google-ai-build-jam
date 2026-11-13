"""Service for managing Worlds, flexible Subjects, Adaptive Expertise Profiles, and Project Backup/Restore."""

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from backend.models import (
    AuthorExpertiseProfile,
    BibleSection,
    BibleSectionRevision,
    CanonProposal,
    Conversation,
    Fact,
    FactRelationship,
    FactRevision,
    FactSubject,
    Message,
    Subject,
    UnresolvedQuestion,
    World,
)
from backend.prompts.domain_cartridges import DEFAULT_WORLD_SUBJECTS, DOMAIN_CARTRIDGES
from backend.schemas import (
    ExpertiseSignalSchema,
    SubjectCreate,
    SubjectMergeRequest,
    SubjectUpdate,
    WorldCreate,
    WorldUpdate,
)
from backend.services.consistency_service import consistency_service


def slugify(text_str: str) -> str:
    """Create a URL/identifier safe slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", text_str.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", cleaned)
    return slug or "section"


class WorldService:
    """Handles CRUD for fictional-world projects, flexible subjects, and expertise profiles."""

    def create_world(self, db: Session, payload: WorldCreate) -> World:
        """Create a new independent world project and optionally seed initial subjects & Bible sections."""
        world = World(
            name=payload.name.strip(),
            description=(payload.description or "").strip(),
            genre_notes=(payload.genre_notes or "").strip(),
        )
        db.add(world)
        db.flush()

        # Initialize expertise profile rows for all 12 domain modules
        for domain_key in DOMAIN_CARTRIDGES:
            profile = AuthorExpertiseProfile(
                world_id=world.id,
                domain_key=domain_key,
                expertise_level="UNKNOWN",
                numeric_score=0.0,
                confidence_score=0.0,
                is_explicit_override=False,
            )
            db.add(profile)

        # Optionally seed the 15 initial worldbuilding subjects and matching World Bible sections
        if payload.seed_default_subjects:
            for idx, item in enumerate(DEFAULT_WORLD_SUBJECTS):
                subj = Subject(
                    world_id=world.id,
                    name=item["name"],
                    slug=item["slug"],
                    description=item["description"],
                    domain_key=item["domain_key"],
                    display_order=idx,
                )
                db.add(subj)
                db.flush()

                section = BibleSection(
                    world_id=world.id,
                    primary_subject_id=subj.id,
                    title=item["name"],
                    slug=item["slug"],
                    markdown_content="",
                    display_order=idx,
                    dominant_provenance="AUTHOR_CREATED",
                )
                db.add(section)

        # Create an initial default conversation thread
        conv = Conversation(
            world_id=world.id,
            title="Worldbuilding Consultation",
        )
        db.add(conv)
        db.commit()
        db.refresh(world)
        return world

    def list_worlds(self, db: Session) -> List[World]:
        """Return all worlds ordered by most recently updated."""
        return db.query(World).order_by(World.updated_at.desc()).all()

    def get_world(self, db: Session, world_id: str) -> Optional[World]:
        """Retrieve a single world by ID."""
        return db.query(World).filter(World.id == world_id).first()

    def update_world(self, db: Session, world: World, payload: WorldUpdate) -> World:
        """Update world metadata or persistent creative preferences."""
        if payload.name is not None:
            world.name = payload.name.strip()
        if payload.description is not None:
            world.description = payload.description.strip()
        if payload.genre_notes is not None:
            world.genre_notes = payload.genre_notes.strip()
        if payload.allow_unrequested_proper_nouns is not None:
            world.allow_unrequested_proper_nouns = payload.allow_unrequested_proper_nouns
        if payload.brainstorm_option_count_default is not None:
            world.brainstorm_option_count_default = max(
                1, min(10, payload.brainstorm_option_count_default)
            )
        if payload.auto_stage_canon_proposals is not None:
            world.auto_stage_canon_proposals = payload.auto_stage_canon_proposals
        db.commit()
        db.refresh(world)
        return world

    def delete_world(self, db: Session, world: World) -> None:
        """Delete a world and all of its dependent records."""
        db.delete(world)
        db.commit()

    # -------------------------------------------------------------------------
    # Subject / Category Management (Flexible Hierarchy)
    # -------------------------------------------------------------------------

    def list_subjects(
        self, db: Session, world_id: str, include_archived: bool = False
    ) -> List[Subject]:
        query = db.query(Subject).filter(Subject.world_id == world_id)
        if not include_archived:
            query = query.filter(Subject.is_archived.is_(False))
        return query.order_by(Subject.display_order.asc(), Subject.name.asc()).all()

    def create_subject(
        self, db: Session, world_id: str, payload: SubjectCreate
    ) -> Subject:
        base_slug = slugify(payload.name)
        slug = base_slug
        counter = 2
        while (
            db.query(Subject)
            .filter(Subject.world_id == world_id, Subject.slug == slug)
            .first()
        ):
            slug = f"{base_slug}-{counter}"
            counter += 1

        max_order = (
            db.query(Subject).filter(Subject.world_id == world_id).count()
        )
        subj = Subject(
            world_id=world_id,
            parent_subject_id=payload.parent_subject_id,
            name=payload.name.strip(),
            slug=slug,
            description=(payload.description or "").strip(),
            domain_key=payload.domain_key,
            display_order=max_order,
        )
        db.add(subj)
        db.commit()
        db.refresh(subj)
        return subj

    def update_subject(
        self, db: Session, subject: Subject, payload: SubjectUpdate
    ) -> Subject:
        if payload.name is not None:
            subject.name = payload.name.strip()
        if payload.description is not None:
            subject.description = payload.description.strip()
        if payload.parent_subject_id is not None:
            subject.parent_subject_id = payload.parent_subject_id or None
        db.commit()
        db.refresh(subject)
        return subject

    def merge_subjects(
        self, db: Session, world_id: str, payload: SubjectMergeRequest
    ) -> Subject:
        """Merge source_subject into target_subject and archive source_subject."""
        source = (
            db.query(Subject)
            .filter(Subject.id == payload.source_subject_id, Subject.world_id == world_id)
            .first()
        )
        target = (
            db.query(Subject)
            .filter(Subject.id == payload.target_subject_id, Subject.world_id == world_id)
            .first()
        )
        if not source or not target or source.id == target.id:
            raise ValueError("Invalid source or target subject for merge.")

        # Reassign fact links
        source_links = (
            db.query(FactSubject).filter(FactSubject.subject_id == source.id).all()
        )
        for link in source_links:
            existing = (
                db.query(FactSubject)
                .filter(
                    FactSubject.fact_id == link.fact_id,
                    FactSubject.subject_id == target.id,
                )
                .first()
            )
            if existing:
                db.delete(link)
            else:
                link.subject_id = target.id

        # Reassign BibleSection primary_subject_id
        db.query(BibleSection).filter(
            BibleSection.world_id == world_id,
            BibleSection.primary_subject_id == source.id,
        ).update({"primary_subject_id": target.id})

        source.is_archived = True
        db.commit()
        db.refresh(target)
        return target

    # -------------------------------------------------------------------------
    # Adaptive Expertise Inference & Overrides (Step 2 Section 4)
    # -------------------------------------------------------------------------

    def get_expertise_map(
        self, db: Session, world_id: str
    ) -> Dict[str, AuthorExpertiseProfile]:
        profiles = (
            db.query(AuthorExpertiseProfile)
            .filter(AuthorExpertiseProfile.world_id == world_id)
            .all()
        )
        return {p.domain_key: p for p in profiles}

    def apply_expertise_signals(
        self, db: Session, world_id: str, signals: List[ExpertiseSignalSchema]
    ) -> None:
        """Update subject-specific author expertise using asymmetric smoothing."""
        if not signals:
            return
        profile_map = self.get_expertise_map(db, world_id)

        level_to_target_score = {
            "BEGINNER": -0.65,
            "INTERMEDIATE": 0.0,
            "ADVANCED": 0.55,
            "EXPERT": 0.90,
        }

        for sig in signals:
            if sig.domain_key not in DOMAIN_CARTRIDGES:
                continue
            profile = profile_map.get(sig.domain_key)
            if not profile:
                profile = AuthorExpertiseProfile(
                    world_id=world_id,
                    domain_key=sig.domain_key,
                    expertise_level="UNKNOWN",
                    numeric_score=0.0,
                    confidence_score=0.0,
                )
                db.add(profile)
                profile_map[sig.domain_key] = profile

            if profile.is_explicit_override and not sig.is_explicit_self_report:
                # Respect explicit manual override unless the author explicitly self-reports a change
                continue

            target = level_to_target_score.get(
                sig.demonstrated_level.upper(), profile.numeric_score
            )
            if sig.is_explicit_self_report:
                # Immediate update when author explicitly states their expertise level
                profile.numeric_score = target
                profile.expertise_level = sig.demonstrated_level.upper()
                profile.confidence_score = min(1.0, profile.confidence_score + 0.5)
            else:
                # Asymmetric smoothing: fast upward recognition, damped downward adjustment
                alpha = 0.45 if target >= profile.numeric_score else 0.12
                profile.numeric_score = (1.0 - alpha) * profile.numeric_score + alpha * target
                profile.confidence_score = min(1.0, profile.confidence_score + 0.25)

                if profile.numeric_score <= -0.25:
                    profile.expertise_level = "BEGINNER"
                elif profile.numeric_score < 0.30:
                    profile.expertise_level = "INTERMEDIATE"
                elif profile.numeric_score < 0.70:
                    profile.expertise_level = "ADVANCED"
                else:
                    profile.expertise_level = "EXPERT"

            if sig.evidence_note:
                profile.evidence_notes = sig.evidence_note[:500]

        db.commit()

    def set_explicit_expertise_override(
        self, db: Session, world_id: str, domain_key: str, level: str
    ) -> AuthorExpertiseProfile:
        """Allow the author to explicitly set or correct their expertise in a domain."""
        level_upper = level.upper()
        if level_upper not in {"UNKNOWN", "BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"}:
            raise ValueError(f"Invalid expertise level: {level}")
        profile = (
            db.query(AuthorExpertiseProfile)
            .filter(
                AuthorExpertiseProfile.world_id == world_id,
                AuthorExpertiseProfile.domain_key == domain_key,
            )
            .first()
        )
        if not profile:
            profile = AuthorExpertiseProfile(
                world_id=world_id,
                domain_key=domain_key,
            )
            db.add(profile)

        score_map = {
            "UNKNOWN": 0.0,
            "BEGINNER": -0.65,
            "INTERMEDIATE": 0.0,
            "ADVANCED": 0.55,
            "EXPERT": 0.90,
        }
        profile.expertise_level = level_upper
        profile.numeric_score = score_map[level_upper]
        profile.confidence_score = 1.0
        profile.is_explicit_override = level_upper != "UNKNOWN"
        profile.evidence_notes = "Explicitly set by the author."
        db.commit()
        db.refresh(profile)
        return profile

    # -------------------------------------------------------------------------
    # Full Project Backup & Restore (Step 6 Section 5)
    # -------------------------------------------------------------------------

    def export_project_bundle(self, db: Session, world: World) -> Dict[str, Any]:
        """Serialize an entire world project (Bible, revisions, facts, relationships, conversations) into a portable JSON bundle."""
        subjects = (
            db.query(Subject)
            .filter(Subject.world_id == world.id)
            .order_by(Subject.display_order.asc())
            .all()
        )
        subj_id_to_slug = {s.id: s.slug for s in subjects}

        profiles = (
            db.query(AuthorExpertiseProfile)
            .filter(AuthorExpertiseProfile.world_id == world.id)
            .all()
        )
        sections = (
            db.query(BibleSection)
            .filter(BibleSection.world_id == world.id)
            .order_by(BibleSection.display_order.asc())
            .all()
        )
        facts = (
            db.query(Fact)
            .filter(Fact.world_id == world.id)
            .order_by(Fact.created_at.asc())
            .all()
        )
        relationships = (
            db.query(FactRelationship)
            .filter(FactRelationship.world_id == world.id)
            .all()
        )
        conversations = (
            db.query(Conversation)
            .filter(Conversation.world_id == world.id)
            .order_by(Conversation.created_at.asc())
            .all()
        )
        proposals = (
            db.query(CanonProposal)
            .filter(CanonProposal.world_id == world.id)
            .order_by(CanonProposal.created_at.asc())
            .all()
        )
        unresolved = (
            db.query(UnresolvedQuestion)
            .filter(UnresolvedQuestion.world_id == world.id)
            .all()
        )

        sections_payload = []
        for sec in sections:
            revs = (
                db.query(BibleSectionRevision)
                .filter(BibleSectionRevision.section_id == sec.id)
                .order_by(BibleSectionRevision.version_number.asc())
                .all()
            )
            sections_payload.append(
                {
                    "id": sec.id,
                    "title": sec.title,
                    "slug": sec.slug,
                    "markdown_content": sec.markdown_content or "",
                    "display_order": sec.display_order,
                    "dominant_provenance": sec.dominant_provenance,
                    "version_number": sec.version_number,
                    "primary_subject_slug": subj_id_to_slug.get(sec.primary_subject_id or "", ""),
                    "revisions": [
                        {
                            "version_number": r.version_number,
                            "previous_markdown": r.previous_markdown,
                            "new_markdown": r.new_markdown,
                            "edit_origin": r.edit_origin,
                            "author_note": r.author_note,
                        }
                        for r in revs
                    ],
                }
            )

        facts_payload = []
        for f in facts:
            f_slugs = [
                subj_id_to_slug[link.subject_id]
                for link in f.subject_links
                if link.subject_id in subj_id_to_slug
            ]
            facts_payload.append(
                {
                    "id": f.id,
                    "title": f.title,
                    "content": f.content,
                    "canon_status": f.canon_status,
                    "provenance_type": f.provenance_type,
                    "authority_tier": f.authority_tier,
                    "consistency_status": f.consistency_status,
                    "consistency_notes": f.consistency_notes,
                    "source_message_id": f.source_message_id,
                    "version_number": f.version_number,
                    "subject_slugs": f_slugs,
                }
            )

        conversations_payload = []
        for conv in conversations:
            msgs = (
                db.query(Message)
                .filter(Message.conversation_id == conv.id)
                .order_by(Message.created_at.asc())
                .all()
            )
            conversations_payload.append(
                {
                    "id": conv.id,
                    "title": conv.title,
                    "rolling_summary": conv.rolling_summary or "",
                    "is_archived": conv.is_archived,
                    "messages": [
                        {
                            "id": m.id,
                            "role": m.role,
                            "content": m.content,
                            "interaction_mode": m.interaction_mode,
                            "permission_level": m.permission_level,
                            "metadata_json": m.metadata_json,
                        }
                        for m in msgs
                    ],
                }
            )

        from backend.services.canon_archive import export_canon
        return {
            "structured_canon": export_canon(db, world.id),
            "schema_version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "world": {
                "name": world.name,
                "description": world.description or "",
                "genre_notes": world.genre_notes or "",
                "allow_unrequested_proper_nouns": world.allow_unrequested_proper_nouns,
                "brainstorm_option_count_default": world.brainstorm_option_count_default,
                "auto_stage_canon_proposals": world.auto_stage_canon_proposals,
            },
            "subjects": [
                {
                    "id": s.id,
                    "name": s.name,
                    "slug": s.slug,
                    "description": s.description or "",
                    "domain_key": s.domain_key,
                    "display_order": s.display_order,
                    "is_archived": s.is_archived,
                }
                for s in subjects
            ],
            "expertise_profiles": [
                {
                    "domain_key": p.domain_key,
                    "expertise_level": p.expertise_level,
                    "numeric_score": p.numeric_score,
                    "confidence_score": p.confidence_score,
                    "is_explicit_override": p.is_explicit_override,
                    "evidence_notes": p.evidence_notes or "",
                }
                for p in profiles
            ],
            "bible_sections": sections_payload,
            "facts": facts_payload,
            "fact_relationships": [
                {
                    "source_fact_id": r.source_fact_id,
                    "target_fact_id": r.target_fact_id,
                    "relationship_type": r.relationship_type,
                    "description": r.description or "",
                    "is_resolved": r.is_resolved,
                    "created_by": r.created_by,
                }
                for r in relationships
            ],
            "conversations": conversations_payload,
            "proposals": [
                {
                    "proposal_type": p.proposal_type,
                    "proposed_title": p.proposed_title,
                    "proposed_content": p.proposed_content,
                    "proposed_provenance": p.proposed_provenance,
                    "status": p.status,
                }
                for p in proposals
            ],
            "unresolved_questions": [
                {
                    "question_text": q.question_text,
                    "status": q.status,
                    "resolution_note": q.resolution_note,
                }
                for q in unresolved
            ],
        }

    def import_project_bundle(self, db: Session, bundle: Dict[str, Any]) -> World:
        try:
            return self._import_project_bundle(db, bundle)
        except Exception:
            db.rollback()
            raise

    def _import_project_bundle(self, db: Session, bundle: Dict[str, Any]) -> World:
        """Restore a complete world project from an exported JSON bundle without primary-key collisions."""
        world_data = bundle.get("world")
        if not isinstance(world_data, dict) or not (world_data.get("name") or "").strip():
            raise ValueError("Invalid project archive: missing 'world.name'.")

        world = World(
            name=world_data["name"].strip(),
            description=(world_data.get("description") or "").strip(),
            genre_notes=(world_data.get("genre_notes") or "").strip(),
            allow_unrequested_proper_nouns=bool(
                world_data.get("allow_unrequested_proper_nouns", False)
            ),
            brainstorm_option_count_default=int(
                world_data.get("brainstorm_option_count_default", 3)
            ),
            auto_stage_canon_proposals=bool(
                world_data.get("auto_stage_canon_proposals", True)
            ),
        )
        db.add(world)
        db.flush()

        # 1. Restore expertise profiles
        imported_profiles = {
            p["domain_key"]: p
            for p in bundle.get("expertise_profiles", [])
            if isinstance(p, dict) and p.get("domain_key")
        }
        for domain_key in DOMAIN_CARTRIDGES:
            p_data = imported_profiles.get(domain_key, {})
            db.add(
                AuthorExpertiseProfile(
                    world_id=world.id,
                    domain_key=domain_key,
                    expertise_level=p_data.get("expertise_level", "UNKNOWN"),
                    numeric_score=float(p_data.get("numeric_score", 0.0)),
                    confidence_score=float(p_data.get("confidence_score", 0.0)),
                    is_explicit_override=bool(p_data.get("is_explicit_override", False)),
                    evidence_notes=p_data.get("evidence_notes", ""),
                )
            )

        identity_mapping = {}
        # 2. Restore subjects
        slug_to_subj_id: Dict[str, str] = {}
        raw_subjects = bundle.get("subjects") or DEFAULT_WORLD_SUBJECTS
        for idx, s_item in enumerate(raw_subjects):
            s_slug = s_item.get("slug") or slugify(s_item.get("name", f"subject-{idx}"))
            if s_slug in slug_to_subj_id:
                continue
            subj = Subject(
                world_id=world.id,
                name=(s_item.get("name") or s_slug).strip(),
                slug=s_slug,
                description=s_item.get("description") or "",
                domain_key=s_item.get("domain_key"),
                display_order=int(s_item.get("display_order", idx)),
                is_archived=bool(s_item.get("is_archived", False)),
            )
            db.add(subj)
            db.flush()
            slug_to_subj_id[s_slug] = subj.id
            if s_item.get("id"):
                identity_mapping[s_item["id"]] = subj.id

        # 3. Restore Bible sections & revision history
        raw_sections = bundle.get("bible_sections", [])
        for idx, sec_item in enumerate(raw_sections):
            sec_slug = sec_item.get("slug") or slugify(sec_item.get("title", f"section-{idx}"))
            primary_sid = slug_to_subj_id.get(
                sec_item.get("primary_subject_slug") or sec_slug
            )
            sec = BibleSection(
                world_id=world.id,
                primary_subject_id=primary_sid,
                title=(sec_item.get("title") or sec_slug).strip(),
                slug=sec_slug,
                markdown_content=sec_item.get("markdown_content") or "",
                display_order=int(sec_item.get("display_order", idx)),
                dominant_provenance=sec_item.get("dominant_provenance", "AUTHOR_CREATED"),
                version_number=int(sec_item.get("version_number", 1)),
            )
            db.add(sec)
            db.flush()

            if sec_item.get("id"):
                identity_mapping[sec_item["id"]] = sec.id
            for rev_item in sec_item.get("revisions", []):
                db.add(
                    BibleSectionRevision(
                        section_id=sec.id,
                        version_number=int(rev_item.get("version_number", 1)),
                        previous_markdown=rev_item.get("previous_markdown"),
                        new_markdown=rev_item.get("new_markdown") or "",
                        edit_origin=rev_item.get("edit_origin", "IMPORTED"),
                        author_note=rev_item.get("author_note") or "Restored from project archive",
                    )
                )

        # 4. Restore atomic facts & FTS5 index
        old_fact_id_to_new: Dict[str, str] = {}
        for f_item in bundle.get("facts", []):
            new_f = Fact(
                world_id=world.id,
                title=(f_item.get("title") or "Fact").strip(),
                content=f_item.get("content") or "",
                canon_status=f_item.get("canon_status", "ESTABLISHED_CANON"),
                provenance_type=f_item.get("provenance_type", "AUTHOR_CREATED"),
                authority_tier=int(f_item.get("authority_tier", 1)),
                consistency_status=f_item.get("consistency_status", "VALID"),
                consistency_notes=f_item.get("consistency_notes") or "",
                version_number=int(f_item.get("version_number", 1)),
            )
            db.add(new_f)
            db.flush()
            if f_item.get("id"):
                old_fact_id_to_new[f_item["id"]] = new_f.id

            for s_idx, s_slug in enumerate(f_item.get("subject_slugs", [])):
                sid = slug_to_subj_id.get(s_slug)
                if sid:
                    db.add(
                        FactSubject(
                            fact_id=new_f.id,
                            subject_id=sid,
                            is_primary=(s_idx == 0),
                        )
                    )
            db.add(
                FactRevision(
                    fact_id=new_f.id,
                    version_number=new_f.version_number,
                    previous_content=None,
                    new_content=new_f.content,
                    previous_canon_status=None,
                    new_canon_status=new_f.canon_status,
                    previous_provenance_type=None,
                    new_provenance_type=new_f.provenance_type,
                    changed_by="AUTHOR_DIRECT_EDIT",
                    change_reason="Restored from project archive",
                )
            )
            consistency_service.sync_fact_fts(db, new_f)

        identity_mapping.update(old_fact_id_to_new)
        # 5. Restore cross-domain fact relationships
        for rel_item in bundle.get("fact_relationships", []):
            src_id = old_fact_id_to_new.get(rel_item.get("source_fact_id", ""))
            tgt_id = old_fact_id_to_new.get(rel_item.get("target_fact_id", ""))
            if src_id and tgt_id and src_id != tgt_id:
                db.add(
                    FactRelationship(
                        world_id=world.id,
                        source_fact_id=src_id,
                        target_fact_id=tgt_id,
                        relationship_type=rel_item.get("relationship_type", "DEPENDS_ON"),
                        description=rel_item.get("description") or "",
                        is_resolved=bool(rel_item.get("is_resolved", True)),
                        created_by=rel_item.get("created_by", "AUTHOR"),
                    )
                )

        # 6. Restore conversations & messages
        raw_convs = bundle.get("conversations", [])
        if not raw_convs:
            db.add(Conversation(world_id=world.id, title="Worldbuilding Consultation"))
        else:
            for c_item in raw_convs:
                conv = Conversation(
                    world_id=world.id,
                    title=(c_item.get("title") or "Consultation").strip(),
                    rolling_summary=c_item.get("rolling_summary") or "",
                    is_archived=bool(c_item.get("is_archived", False)),
                )
                db.add(conv)
                db.flush()
                for m_item in c_item.get("messages", []):
                    message = Message(conversation_id=conv.id, role=m_item.get("role", "USER"),
                        content=m_item.get("content") or "", interaction_mode=m_item.get("interaction_mode", "GENERAL"),
                        permission_level=m_item.get("permission_level", "FACTUAL_ONLY"), metadata_json=m_item.get("metadata_json") or "{}")
                    db.add(message); db.flush()
                    if m_item.get("id"):
                        identity_mapping[m_item["id"]] = message.id

        # 7. Restore proposals & unresolved questions
        for prop_item in bundle.get("proposals", []):
            db.add(
                CanonProposal(
                    world_id=world.id,
                    proposal_type=prop_item.get("proposal_type", "NEW_FACT"),
                    proposed_title=prop_item.get("proposed_title") or "Proposal",
                    proposed_content=prop_item.get("proposed_content") or "",
                    proposed_provenance=prop_item.get("proposed_provenance", "AUTHOR_CREATED"),
                    status=prop_item.get("status", "PENDING"),
                )
            )
        for uq_item in bundle.get("unresolved_questions", []):
            db.add(
                UnresolvedQuestion(
                    world_id=world.id,
                    question_text=uq_item.get("question_text") or "",
                    status=uq_item.get("status", "OPEN"),
                    resolution_note=uq_item.get("resolution_note"),
                )
            )

        from backend.services.canon_archive import import_canon
        for item in bundle.get('facts', []):
            restored_id = old_fact_id_to_new.get(item.get('id'))
            if restored_id and item.get('source_message_id') in identity_mapping:
                db.get(Fact, restored_id).source_message_id = identity_mapping[item['source_message_id']]
        import_canon(db, world, bundle.get('structured_canon'), identity_mapping)
        db.commit()
        db.refresh(world)
        return world


world_service = WorldService()
