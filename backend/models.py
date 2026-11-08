"""SQLAlchemy ORM models implementing the approved Step 1 World & Knowledge schema."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class World(Base):
    """Independent fictional-world project."""

    __tablename__ = "worlds"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")
    genre_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")
    # Persistent project preferences (Step 2 Section 3.3)
    allow_unrequested_proper_nouns: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    brainstorm_option_count_default: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3
    )
    auto_stage_canon_proposals: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    subjects: Mapped[List["Subject"]] = relationship(
        "Subject", back_populates="world", cascade="all, delete-orphan"
    )
    expertise_profiles: Mapped[List["AuthorExpertiseProfile"]] = relationship(
        "AuthorExpertiseProfile", back_populates="world", cascade="all, delete-orphan"
    )
    facts: Mapped[List["Fact"]] = relationship(
        "Fact", back_populates="world", cascade="all, delete-orphan"
    )
    bible_sections: Mapped[List["BibleSection"]] = relationship(
        "BibleSection", back_populates="world", cascade="all, delete-orphan"
    )
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="world", cascade="all, delete-orphan"
    )
    canon_proposals: Mapped[List["CanonProposal"]] = relationship(
        "CanonProposal", back_populates="world", cascade="all, delete-orphan"
    )
    unresolved_questions: Mapped[List["UnresolvedQuestion"]] = relationship(
        "UnresolvedQuestion", back_populates="world", cascade="all, delete-orphan"
    )


class Subject(Base):
    """Flexible per-world hierarchical category/subject."""

    __tablename__ = "subjects"
    __table_args__ = (UniqueConstraint("world_id", "slug", name="uq_world_subject_slug"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    world_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False
    )
    parent_subject_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    slug: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")
    domain_key: Mapped[Optional[str]] = mapped_column(
        String(80), nullable=True
    )  # Maps to 1 of the 12 domain cartridges where applicable
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    world: Mapped["World"] = relationship("World", back_populates="subjects")
    fact_links: Mapped[List["FactSubject"]] = relationship(
        "FactSubject", back_populates="subject", cascade="all, delete-orphan"
    )


class AuthorExpertiseProfile(Base):
    """Subject-specific expertise level inferred from conversation or explicitly overridden."""

    __tablename__ = "author_expertise_profiles"
    __table_args__ = (
        UniqueConstraint("world_id", "domain_key", name="uq_world_domain_expertise"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    world_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False
    )
    domain_key: Mapped[str] = mapped_column(String(80), nullable=False)
    expertise_level: Mapped[str] = mapped_column(
        String(30), nullable=False, default="UNKNOWN"
    )  # UNKNOWN, BEGINNER, INTERMEDIATE, ADVANCED, EXPERT
    numeric_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )  # Continuous score [-1.0, 1.0] for asymmetric smoothing
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_explicit_override: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    evidence_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    world: Mapped["World"] = relationship("World", back_populates="expertise_profiles")


class Fact(Base):
    """Atomic worldbuilding fact, decision, hypothetical, or rejected option."""

    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    world_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(250), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # ESTABLISHED_CANON, UNRESOLVED_POSSIBILITY, HYPOTHETICAL, SUPERSEDED, REJECTED, PENDING_RETIREMENT_CONFIRMATION
    canon_status: Mapped[str] = mapped_column(
        String(40), nullable=False, default="ESTABLISHED_CANON"
    )

    # AUTHOR_CREATED, AI_GENERATED_AUTHOR_APPROVED, AI_PROPOSED_UNVERIFIED, RESEARCH_NOTE
    provenance_type: Mapped[str] = mapped_column(
        String(40), nullable=False, default="AUTHOR_CREATED"
    )

    # 1 = Author Explicit Decision, 2 = Author-Approved AI, 3 = Grounded Research, 4 = Model Fallback
    authority_tier: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # VALID, NEEDS_REVIEW, CONTRADICTED
    consistency_status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="VALID"
    )
    consistency_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")

    source_message_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    superseded_by_fact_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="SET NULL"), nullable=True
    )
    embedding_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON-serialized float vector for semantic matching
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    world: Mapped["World"] = relationship("World", back_populates="facts")
    subject_links: Mapped[List["FactSubject"]] = relationship(
        "FactSubject", back_populates="fact", cascade="all, delete-orphan"
    )
    revisions: Mapped[List["FactRevision"]] = relationship(
        "FactRevision", back_populates="fact", cascade="all, delete-orphan"
    )
    section_links: Mapped[List["BibleSectionFact"]] = relationship(
        "BibleSectionFact", back_populates="fact", cascade="all, delete-orphan"
    )
    citations: Mapped[List["FactCitation"]] = relationship(
        "FactCitation", back_populates="fact", cascade="all, delete-orphan"
    )


class FactSubject(Base):
    """Many-to-many join between Facts and Subjects."""

    __tablename__ = "fact_subjects"

    fact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="CASCADE"), primary_key=True
    )
    subject_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True
    )
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    fact: Mapped["Fact"] = relationship("Fact", back_populates="subject_links")
    subject: Mapped["Subject"] = relationship("Subject", back_populates="fact_links")


class FactRevision(Base):
    """Immutable audit log of changes to a Fact."""

    __tablename__ = "fact_revisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    fact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_content: Mapped[str] = mapped_column(Text, nullable=False)
    previous_canon_status: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    new_canon_status: Mapped[str] = mapped_column(String(40), nullable=False)
    previous_provenance_type: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    new_provenance_type: Mapped[str] = mapped_column(String(40), nullable=False)
    changed_by: Mapped[str] = mapped_column(
        String(40), nullable=False
    )  # AUTHOR_DIRECT_EDIT, AUTHOR_CHAT_CONFIRMATION, SYSTEM_REVERT
    change_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    fact: Mapped["Fact"] = relationship("Fact", back_populates="revisions")


class FactRelationship(Base):
    """Directed edge between two facts across domains (dependency or contradiction)."""

    __tablename__ = "fact_relationships"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    world_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False
    )
    source_fact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="CASCADE"), nullable=False
    )
    target_fact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="CASCADE"), nullable=False
    )
    # DEPENDS_ON, INFLUENCES, CONSTRAINS, CONTRADICTS, ELABORATES, CROSS_REFERENCES
    relationship_type: Mapped[str] = mapped_column(String(40), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")
    is_resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[str] = mapped_column(
        String(40), nullable=False, default="AUTHOR"
    )  # AUTHOR, AI_CONSISTENCY_ENGINE
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class BibleSection(Base):
    """Hierarchical section of the World Bible document."""

    __tablename__ = "bible_sections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    world_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False
    )
    parent_section_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("bible_sections.id", ondelete="CASCADE"), nullable=True
    )
    primary_subject_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(200), nullable=False)
    markdown_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # AUTHOR_CREATED, MIXED, AI_GENERATED_AUTHOR_APPROVED
    dominant_provenance: Mapped[str] = mapped_column(
        String(40), nullable=False, default="AUTHOR_CREATED"
    )
    has_pending_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    world: Mapped["World"] = relationship("World", back_populates="bible_sections")
    fact_links: Mapped[List["BibleSectionFact"]] = relationship(
        "BibleSectionFact", back_populates="section", cascade="all, delete-orphan"
    )
    revisions: Mapped[List["BibleSectionRevision"]] = relationship(
        "BibleSectionRevision", back_populates="section", cascade="all, delete-orphan"
    )


class BibleSectionFact(Base):
    """Bidirectional link between a World Bible section and an atomic Fact."""

    __tablename__ = "bible_section_facts"

    section_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("bible_sections.id", ondelete="CASCADE"), primary_key=True
    )
    fact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="CASCADE"), primary_key=True
    )
    paragraph_anchor: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    section: Mapped["BibleSection"] = relationship("BibleSection", back_populates="fact_links")
    fact: Mapped["Fact"] = relationship("Fact", back_populates="section_links")


class BibleSectionRevision(Base):
    """Revision history entry for a World Bible section."""

    __tablename__ = "bible_section_revisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    section_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("bible_sections.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_markdown: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    # AUTHOR_EDITOR, AUTHOR_APPROVED_PROPOSAL, REVERT
    edit_origin: Mapped[str] = mapped_column(String(40), nullable=False)
    author_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    section: Mapped["BibleSection"] = relationship("BibleSection", back_populates="revisions")


class Conversation(Base):
    """Chat conversation thread within a World project."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    world_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New Consultation")
    focused_subject_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True
    )
    rolling_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    world: Mapped["World"] = relationship("World", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """Individual chat message turn with recorded interaction mode, permission state, and citations."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # USER, ASSISTANT, SYSTEM
    content: Mapped[str] = mapped_column(Text, nullable=False)
    interaction_mode: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    permission_level: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Stores verified citations, turn analysis, and coverage stats
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")


class CanonProposal(Base):
    """Staging gate record for proposed World Bible / Fact updates requiring author approval."""

    __tablename__ = "canon_proposals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    world_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True
    )
    source_message_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=True
    )
    # NEW_FACT, UPDATE_FACT, BIBLE_SECTION_PATCH, OPEN_QUESTION
    proposal_type: Mapped[str] = mapped_column(String(40), nullable=False, default="NEW_FACT")
    target_fact_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="SET NULL"), nullable=True
    )
    target_section_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("bible_sections.id", ondelete="SET NULL"), nullable=True
    )
    target_subject_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True
    )
    proposed_title: Mapped[str] = mapped_column(String(250), nullable=False)
    proposed_content: Mapped[str] = mapped_column(Text, nullable=False)
    # AUTHOR_CREATED, AI_GENERATED_AUTHOR_APPROVED, UNRESOLVED_POSSIBILITY
    proposed_provenance: Mapped[str] = mapped_column(
        String(40), nullable=False, default="AUTHOR_CREATED"
    )
    # PENDING, APPROVED, EDITED_AND_APPROVED, REJECTED
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    world: Mapped["World"] = relationship("World", back_populates="canon_proposals")


class UnresolvedQuestion(Base):
    """Open worldbuilding question or deferred decision."""

    __tablename__ = "unresolved_questions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    world_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("worlds.id", ondelete="CASCADE"), nullable=False
    )
    subject_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    context_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="OPEN"
    )  # OPEN, ANSWERED, DISMISSED
    resolved_by_fact_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    world: Mapped["World"] = relationship("World", back_populates="unresolved_questions")


class ReferenceDocument(Base):
    """Ingested reference book (e.g., The Planet Construction Kit) — separate from fictional canon."""

    __tablename__ = "reference_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(250), nullable=False)
    author: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embeddings_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # PENDING, PROCESSING, INDEXED, FAILED
    ingestion_status: Mapped[str] = mapped_column(String(30), nullable=False, default="PENDING")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    chunks: Mapped[List["ReferenceChunk"]] = relationship(
        "ReferenceChunk", back_populates="document", cascade="all, delete-orphan"
    )


class ReferenceChunk(Base):
    """Hierarchical parent/child chunk from an ingested reference book."""

    __tablename__ = "reference_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("reference_documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_chunk_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("reference_chunks.id", ondelete="CASCADE"), nullable=True
    )
    # PARENT_SECTION, CHILD_PASSAGE
    chunk_level: Mapped[str] = mapped_column(String(30), nullable=False, default="CHILD_PASSAGE")
    chapter_title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    pdf_page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    pdf_page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    book_page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    book_page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ocr_repaired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    embedding_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON-serialized float vector

    document: Mapped["ReferenceDocument"] = relationship(
        "ReferenceDocument", back_populates="chunks"
    )


class ResearchSource(Base):
    """Cached online research source retrieved via Tavily — separate from fictional canon."""

    __tablename__ = "research_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    domain: Mapped[str] = mapped_column(String(150), nullable=False)
    publisher_or_author: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    published_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # ACADEMIC_PEER_REVIEWED, INSTITUTIONAL_ENCYCLOPEDIC, GENERAL_REFERENCE
    source_credibility_tier: Mapped[str] = mapped_column(
        String(40), nullable=False, default="GENERAL_REFERENCE"
    )
    extracted_excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    consensus_status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="CONSENSUS"
    )  # CONSENSUS, DISPUTED, THEORETICAL
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class FactCitation(Base):
    """Citation linking a worldbuilding note/fact to real-world reference book or online sources."""

    __tablename__ = "fact_citations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    fact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("facts.id", ondelete="CASCADE"), nullable=False
    )
    # REFERENCE_BOOK, ONLINE_RESEARCH, MODEL_GENERAL_KNOWLEDGE
    citation_source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    reference_chunk_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("reference_chunks.id", ondelete="SET NULL"), nullable=True
    )
    research_source_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("research_sources.id", ondelete="SET NULL"), nullable=True
    )
    citation_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    fact: Mapped["Fact"] = relationship("Fact", back_populates="citations")
