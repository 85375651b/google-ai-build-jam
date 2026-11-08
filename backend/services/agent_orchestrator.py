"""Central Agent Orchestrator enforcing permissions, RAG, Tavily research, and canon proposals."""

import json
import logging
import re
from typing import Any, Dict, Generator, List, Optional
from sqlalchemy.orm import Session
from backend.models import (
    CanonProposal,
    Conversation,
    Fact,
    Message,
    Subject,
    UnresolvedQuestion,
    World,
)
from backend.prompts.templates import (
    PASS_1_TURN_ANALYZER_SYSTEM_PROMPT,
    PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT,
)
from backend.schemas import (
    CreativePermissionSchema,
    PostTurnExtractionSchema,
    TurnAnalysisSchema,
)
from backend.services.bible_service import bible_service
from backend.services.book_ingestion_service import book_ingestion_service
from backend.services.consistency_service import consistency_service
from backend.services.context_assembler import context_assembler
from backend.services.gemini_service import gemini_service
from backend.services.research_service import research_service
from backend.services.world_service import world_service

logger = logging.getLogger(__name__)

EXPLICIT_BRAINSTORM_PATTERNS = re.compile(
    r"\b(give me|brainstorm|suggest|generate|propose|come up with|list|what are some|options|alternatives|possibilities|ideas for|ways to|three|two|four|five)\b",
    re.IGNORECASE,
)


class AgentOrchestrator:
    """Coordinates Pass 1 (Intent/Permission), RAG/Tavily retrieval, Pass 2 (Response), and Pass 3 (Proposals)."""

    def analyze_turn(
        self, db: Session, world: World, user_message: str
    ) -> TurnAnalysisSchema:
        """Execute Pass 1 Turn Analysis and apply Guard 1 (Deterministic Permission Downgrade)."""
        prompt = (
            f"World Name: {world.name}\n"
            f"Author Message: {user_message}\n"
            "Classify intent, creative permission scope, expertise signals, and search queries."
        )
        try:
            analysis = gemini_service.generate_structured(
                prompt=prompt,
                response_schema=TurnAnalysisSchema,
                system_instruction=PASS_1_TURN_ANALYZER_SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("Pass 1 structured analysis fallback triggered: %s", exc)
            analysis = self._heuristic_turn_analysis(user_message)

        # Guard 1: Deterministic Creative Permission Downgrade Check
        if analysis.creative_permission.level == "BOUNDED_BRAINSTORMING":
            if not EXPLICIT_BRAINSTORM_PATTERNS.search(user_message):
                analysis.creative_permission.level = "IMPLEMENTATION_SCOPED"
                analysis.creative_permission.allowed_targets = []
                analysis.creative_permission.requested_option_count = 0

        if not re.search(r"\b(name|names|naming|call it|title)\b", user_message, re.I):
            analysis.creative_permission.allow_invented_proper_nouns = False

        if not analysis.search_queries:
            analysis.search_queries = [user_message]

        # Apply expertise signals to database
        if analysis.expertise_signals:
            world_service.apply_expertise_signals(
                db, world.id, analysis.expertise_signals
            )

        return analysis

    def retrieve_knowledge(
        self, db: Session, analysis: TurnAnalysisSchema
    ) -> Dict[str, Any]:
        """Retrieve passages from The Planet Construction Kit and trigger Tavily when coverage is PARTIAL/LOW."""
        book_ingestion_service.ensure_primary_book_ingested(db, generate_embeddings=False)
        book_res = book_ingestion_service.retrieve_passages(
            db=db,
            queries=analysis.search_queries,
            active_domains=analysis.active_domains,
            top_k=5,
        )
        coverage = book_res.get("coverage_status", "LOW")
        if coverage in {"PARTIAL", "LOW"} and analysis.requires_factual_retrieval:
            web_res = research_service.search_online(
                db=db, queries=analysis.search_queries, max_results=3
            )
        else:
            web_res = {
                "status": "SKIPPED_HIGH_BOOK_COVERAGE",
                "message": "Primary reference book sufficiently covers the topic.",
                "sources": [],
            }
        return {"book": book_res, "online": web_res}

    def sanitize_response_and_verify_citations(
        self,
        raw_response: str,
        book_res: Dict[str, Any],
        online_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply Guard 3 (Citation Verification) and Guard 4 (Anti-Hook Sanitizer)."""
        cleaned = re.sub(
            r"\n+What (would you like to|aspect of your world|shall we) (explore|build|develop|work on|discuss) next\?.*$",
            "",
            raw_response.strip(),
            flags=re.IGNORECASE,
        ).strip()

        valid_citations: List[Dict[str, Any]] = []
        for p in book_res.get("passages", []):
            cid = p["citation_id"]
            if cid in cleaned or "Book" in cleaned:
                valid_citations.append(
                    {
                        "citation_id": cid,
                        "source_type": "REFERENCE_BOOK",
                        "label": p["citation_label"],
                        "chapter": p["chapter_title"],
                        "section": p["section_title"],
                        "book_page": p["book_page_start"],
                        "pdf_page": p["pdf_page_start"],
                        "excerpt": p["content"][:400],
                    }
                )
        for s in online_res.get("sources", []):
            cid = s["citation_id"]
            if cid in cleaned or "Web-" in cleaned:
                valid_citations.append(
                    {
                        "citation_id": cid,
                        "source_type": "ONLINE_RESEARCH",
                        "label": s["citation_label"],
                        "url": s["url"],
                        "domain": s["domain"],
                        "credibility_tier": s["credibility_tier"],
                        "excerpt": s["excerpt"][:400],
                    }
                )
        return {"content": cleaned, "citations": valid_citations}

    def extract_post_turn_proposals(
        self,
        db: Session,
        world: World,
        conversation: Conversation,
        user_msg: Message,
        assistant_msg: Message,
        analysis: TurnAnalysisSchema,
    ) -> List[CanonProposal]:
        """Execute Pass 3 Canon & Consistency Auditor with Guard 5 (No Silent Canonization Firewall)."""
        # Guard 5: Pure factual/research questions NEVER create fictional canon proposals
        if (
            analysis.primary_intent in {"QUESTION_EXPLANATION", "FACTUAL_RESEARCH"}
            and "CANON_DECLARATION" not in analysis.secondary_intents
            and "WORLD_BIBLE_MANAGEMENT" not in analysis.secondary_intents
        ):
            return []

        existing_facts = (
            db.query(Fact)
            .filter(Fact.world_id == world.id, Fact.canon_status == "ESTABLISHED_CANON")
            .limit(20)
            .all()
        )
        canon_summary = "\n".join(
            f"[{f.id}] {f.title}: {f.content}" for f in existing_facts
        )
        prompt = (
            f"Established World Canon:\n{canon_summary or 'None yet.'}\n\n"
            f"Author Message:\n{user_msg.content}\n\n"
            f"Consultant Response:\n{assistant_msg.content}\n\n"
            "Extract only explicitly author-established facts, explicit World Bible addition requests, "
            "rejected suggestions, and internal contradictions."
        )
        try:
            extracted = gemini_service.generate_structured(
                prompt=prompt,
                response_schema=PostTurnExtractionSchema,
                system_instruction=PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as exc:
            logger.debug("Post-turn extraction skipped: %s", exc)
            return []

        # Record rejected ideas immediately so the agent never re-introduces them
        for rej in extracted.rejected_ideas:
            db.add(
                Fact(
                    world_id=world.id,
                    title=rej.title[:240],
                    content=rej.content,
                    canon_status="REJECTED",
                    provenance_type="AI_PROPOSED_UNVERIFIED",
                    authority_tier=4,
                    source_message_id=user_msg.id,
                )
            )

        for oq in extracted.unresolved_questions:
            db.add(
                UnresolvedQuestion(
                    world_id=world.id,
                    question_text=oq,
                    status="OPEN",
                )
            )

        subjects = world_service.list_subjects(db, world.id)
        subj_by_slug = {s.slug: s.id for s in subjects}
        default_subj_id = subjects[0].id if subjects else None

        staged: List[CanonProposal] = []
        for item in extracted.proposals:
            target_sid = subj_by_slug.get(item.subject_slug, default_subj_id)
            prop = bible_service.stage_proposal(
                db=db,
                world_id=world.id,
                proposed_title=item.proposed_title,
                proposed_content=item.proposed_content,
                proposed_provenance=item.proposed_provenance,
                proposal_type=item.proposal_type,
                conversation_id=conversation.id,
                source_message_id=user_msg.id,
                target_subject_id=target_sid,
            )
            staged.append(prop)

        db.commit()
        return staged

    def _heuristic_turn_analysis(self, user_message: str) -> TurnAnalysisSchema:
        is_brainstorm = bool(EXPLICIT_BRAINSTORM_PATTERNS.search(user_message))
        return TurnAnalysisSchema(
            primary_intent="CREATIVE_GENERATION" if is_brainstorm else "IMPLEMENTATION_ASSISTANCE",
            secondary_intents=[],
            ambiguity_status="CLEAR",
            raw_user_phrasing=user_message,
            inferred_domain_concepts=[],
            creative_permission=CreativePermissionSchema(
                level="BOUNDED_BRAINSTORMING" if is_brainstorm else "IMPLEMENTATION_SCOPED",
                explicit_authorization_quote=user_message if is_brainstorm else "",
                allowed_targets=["requested worldbuilding options"] if is_brainstorm else [],
                prohibited_adjacent_targets=["unrequested lore or proper nouns"],
                requested_option_count=3 if is_brainstorm else 0,
                allow_invented_proper_nouns=False,
            ),
            active_domains=["political_science", "history"],
            expertise_signals=[],
            requires_factual_retrieval=True,
            search_queries=[user_message],
        )


agent_orchestrator = AgentOrchestrator()
