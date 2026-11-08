"""Pydantic schemas for API requests, responses, and structured Gemini outputs."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Structured Gemini Output Schemas (Step 2 Pass 1 & Pass 3)
# ============================================================================

class ExpertiseSignalSchema(BaseModel):
    domain_key: str = Field(
        description="One of the 12 domain keys, e.g., linguistics, political_science, economics, geography_geology"
    )
    demonstrated_level: str = Field(
        description="BEGINNER, INTERMEDIATE, ADVANCED, or EXPERT based on the author's message"
    )
    signal_Delta: float = Field(
        default=0.0,
        description="Adjustment signal from -0.8 (explicit novice) to +0.8 (advanced technical mastery)",
    )
    is_explicit_self_report: bool = Field(
        default=False,
        description="True if the author explicitly stated their familiarity or asked to adjust explanation depth",
    )
    evidence_note: str = Field(
        default="",
        description="Brief note on terminology or concepts demonstrated by the author",
    )


class CreativePermissionSchema(BaseModel):
    level: str = Field(
        description="FACTUAL_ONLY | ANALYTICAL_ADVISORY | IMPLEMENTATION_SCOPED | BOUNDED_BRAINSTORMING | EXPLICIT_DRAFTING"
    )
    explicit_authorization_quote: str = Field(
        default="",
        description="Exact substring from the author's message that explicitly requested creative generation (empty if none)",
    )
    allowed_targets: List[str] = Field(
        default_factory=list,
        description="Specific elements the author explicitly asked the agent to generate or draft",
    )
    prohibited_adjacent_targets: List[str] = Field(
        default_factory=list,
        description="Adjacent worldbuilding elements the agent must NOT invent unprompted",
    )
    requested_option_count: int = Field(
        default=0,
        description="Number of options requested if brainstorming (0 if not brainstorming)",
    )
    allow_invented_proper_nouns: bool = Field(
        default=False,
        description="True ONLY if the author explicitly asked the agent to invent names",
    )
    naming_mode: str = Field(
        default="NONE",
        description="NONE | MAJOR_CONTEXTUAL | MINOR_QUICK — MAJOR_CONTEXTUAL for culturally/narratively significant entities (major cities, factions, key characters, institutions); MINOR_QUICK for minor/background characters, small villages, or quick naming requests",
    )


class TurnAnalysisSchema(BaseModel):
    primary_intent: str = Field(
        description="QUESTION_EXPLANATION | FACTUAL_RESEARCH | IMPLEMENTATION_ASSISTANCE | CREATIVE_GENERATION | EVALUATION_CRITICISM | WORLD_BIBLE_MANAGEMENT | CANON_DECLARATION | HYPOTHETICAL_EXPLORATION | SUGGESTION_DISPOSITION"
    )
    secondary_intents: List[str] = Field(default_factory=list)
    ambiguity_status: str = Field(
        default="CLEAR",
        description="CLEAR or NEEDS_CLARIFICATION if the request is too vague to execute without guessing",
    )
    raw_user_phrasing: str = Field(default="")
    inferred_domain_concepts: List[str] = Field(
        default_factory=list,
        description="Accurate technical terms mapped from imprecise or everyday user phrasing",
    )
    creative_permission: CreativePermissionSchema
    active_domains: List[str] = Field(
        default_factory=list,
        description="1 to 3 relevant subject domain keys from the 12 modules",
    )
    expertise_signals: List[ExpertiseSignalSchema] = Field(default_factory=list)
    requires_factual_retrieval: bool = Field(
        default=True,
        description="True if real-world reference book or online research is helpful for this turn",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="1 to 3 de-fictionalized real-world search queries stripping fictional proper nouns",
    )


class CandidateProposalItem(BaseModel):
    proposal_type: str = Field(
        default="NEW_FACT",
        description="NEW_FACT | UPDATE_FACT | BIBLE_SECTION_PATCH | OPEN_QUESTION",
    )
    proposed_title: str = Field(description="Concise title of the established fact or section update")
    proposed_content: str = Field(
        description="Exact canonical statement established by the author without unstated embellishment"
    )
    proposed_provenance: str = Field(
        default="AUTHOR_CREATED",
        description="AUTHOR_CREATED | AI_GENERATED_AUTHOR_APPROVED | UNRESOLVED_POSSIBILITY",
    )
    subject_slug: str = Field(
        default="general",
        description="Slug of the most relevant worldbuilding subject (e.g., politics-governance, economics-trade, geography-geology)",
    )
    auto_apply_to_bible: bool = Field(
        default=True,
        description="True when the author stated this fact (brain dump / declaration) or explicitly asked to update/edit the World Bible; False ONLY if it is an unapproved AI suggestion.",
    )
    depends_on_fact_ids: List[str] = Field(
        default_factory=list,
        description="IDs or 8-char ID prefixes of existing ESTABLISHED_CANON facts that this fact logically depends on or elaborates",
    )


class BibleSectionEditItem(BaseModel):
    section_slug: str = Field(
        description="Target section slug (e.g., cosmology-physics, geography-geology, climate-ecology, biology-evolution, history-chronology, languages-linguistics, cultures-societies, religion-mythology, politics-governance, economics-trade, technology-engineering, warfare-military, philosophy-ideologies, demographics-population, architecture-material-culture, or a custom slug)"
    )
    section_title: str = Field(
        description="Human-readable title of the World Bible section"
    )
    updated_markdown: str = Field(
        description="Complete, well-organized Markdown content for this section incorporating the newly extracted details alongside any existing content for this section. Use clean Markdown headings/bullet points/prose without inventing facts not stated or approved by the author."
    )
    summary_of_changes: str = Field(
        default="Organized author worldbuilding details into section",
        description="Short 1-sentence summary of what was added or edited in this section",
    )
    provenance: str = Field(
        default="AUTHOR_CREATED",
        description="AUTHOR_CREATED or AI_GENERATED_AUTHOR_APPROVED",
    )


class RejectedIdeaItem(BaseModel):
    title: str
    content: str
    reason: str = ""


class DetectedContradictionItem(BaseModel):
    conflicting_fact_id: str = Field(
        default="",
        description="ID or 8-char ID prefix of the existing ESTABLISHED_CANON fact being contradicted",
    )
    new_fact_title: str = Field(
        default="",
        description="Title of the newly stated fact or claim that conflicts with the existing fact",
    )
    description: str = Field(description="Neutral description of the internal inconsistency")


class PostTurnExtractionSchema(BaseModel):
    bible_section_updates: List[BibleSectionEditItem] = Field(
        default_factory=list,
        description="Direct World Bible section edits/organizations extracted from the author's brain dump, stated world facts, or explicit World Bible editing instructions.",
    )
    proposals: List[CandidateProposalItem] = Field(
        default_factory=list,
        description="Atomic facts extracted from the author's brain dump or statements.",
    )
    rejected_ideas: List[RejectedIdeaItem] = Field(
        default_factory=list,
        description="Any ideas or AI suggestions the author explicitly rejected in this turn",
    )
    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="Any explicit open worldbuilding questions the author noted as unresolved",
    )
    detected_contradictions: List[DetectedContradictionItem] = Field(
        default_factory=list,
        description="Any direct contradictions with existing ESTABLISHED_CANON facts",
    )


# ============================================================================
# API Request / Response Schemas
# ============================================================================

class WorldCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = ""
    genre_notes: Optional[str] = ""
    seed_default_subjects: bool = True


class WorldUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    genre_notes: Optional[str] = None
    allow_unrequested_proper_nouns: Optional[bool] = None
    brainstorm_option_count_default: Optional[int] = None
    auto_stage_canon_proposals: Optional[bool] = None


class SubjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    description: Optional[str] = ""
    parent_subject_id: Optional[str] = None
    domain_key: Optional[str] = None


class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parent_subject_id: Optional[str] = None


class SubjectMergeRequest(BaseModel):
    source_subject_id: str
    target_subject_id: str


class BibleSectionCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    markdown_content: str = ""
    primary_subject_id: Optional[str] = None
    parent_section_id: Optional[str] = None


class BibleSectionUpdate(BaseModel):
    title: Optional[str] = None
    markdown_content: Optional[str] = None
    primary_subject_id: Optional[str] = None
    expected_version_number: Optional[int] = None
    author_note: Optional[str] = None


class OrphanedFactResolutionRequest(BaseModel):
    fact_ids: List[str]
    action: str = Field(
        ...,
        description="RETIRE (marks SUPERSEDED) or KEEP_ACTIVE (keeps ESTABLISHED_CANON)",
    )


class ProposalActionRequest(BaseModel):
    action: str = Field(
        ..., description="APPROVE | EDIT_AND_APPROVE | REJECT"
    )
    edited_title: Optional[str] = None
    edited_content: Optional[str] = None
    target_section_id: Optional[str] = None


class FactCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=250)
    content: str = Field(..., min_length=1)
    subject_ids: List[str] = Field(default_factory=list)
    canon_status: str = "ESTABLISHED_CANON"
    provenance_type: str = "AUTHOR_CREATED"


class FactUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    canon_status: Optional[str] = None
    subject_ids: Optional[List[str]] = None
    change_reason: Optional[str] = None


class FactRelationshipCreate(BaseModel):
    source_fact_id: str
    target_fact_id: str
    relationship_type: str = "DEPENDS_ON"
    description: Optional[str] = ""


class ExpertiseOverrideRequest(BaseModel):
    domain_key: str
    expertise_level: str = Field(
        ..., description="BEGINNER | INTERMEDIATE | ADVANCED | EXPERT"
    )


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Consultation"
    focused_subject_id: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    focused_subject_id: Optional[str] = None
    is_archived: Optional[bool] = None


class ChatMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)
    focused_subject_id: Optional[str] = None


class ModelConfigUpdateRequest(BaseModel):
    gemini_model: str
    gemini_embedding_model: Optional[str] = None
