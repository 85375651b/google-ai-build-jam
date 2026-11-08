"""Service for managing Worlds, flexible Subjects, and Adaptive Expertise Profiles."""

import re
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from backend.models import (
    AuthorExpertiseProfile,
    BibleSection,
    Conversation,
    FactSubject,
    Subject,
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


world_service = WorldService()
