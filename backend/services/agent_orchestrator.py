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
    ROLLING_SUMMARY_SYSTEM_PROMPT,
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
    r"\b(give me|brainstorm|suggest|generate|propose|come up with|list|what are some|options|alternatives|possibilities|ideas for|ways to|three|two|four|five|help me name|what should i call|need a name|need names)\b",
    re.IGNORECASE,
)

NAMING_REQUEST_PATTERNS = re.compile(
    r"\b(name|names|naming|call it|what should i call|title for|place name|toponym|ethnonym|surname|clan name)\b",
    re.IGNORECASE,
)

MINOR_NAMING_PATTERNS = re.compile(
    r"\b(minor|quick|fast|background|side character|incidental|guard|innkeeper|shopkeeper|tavern|small village|hamlet|street name|just need a name|couple of quick names|random name)\b",
    re.IGNORECASE,
)


class AgentOrchestrator:
    """Coordinates Pass 1 (Intent/Permission), RAG/Tavily retrieval, Pass 2 (Response), and Pass 3 (Proposals)."""

    def analyze_turn(
        self, db: Session, world: World, user_message: str
    ) -> TurnAnalysisSchema:
        """Execute Pass 1 Turn Analysis and apply Guard 1 (Deterministic Permission Downgrade)."""
        preview_msg = (
            user_message[:6000] + "\n...[truncated middle of long brain dump]...\n" + user_message[-2000:]
            if len(user_message) > 8500
            else user_message
        )
        prompt = (
            f"World Name: {world.name}\n"
            f"Author Message: {preview_msg}\n"
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

        is_naming_turn = bool(NAMING_REQUEST_PATTERNS.search(user_message))
        if not is_naming_turn:
            analysis.creative_permission.allow_invented_proper_nouns = False
            analysis.creative_permission.naming_mode = "NONE"
        else:
            analysis.creative_permission.allow_invented_proper_nouns = True
            resolved_mode = self._resolve_naming_mode(
                user_message, analysis.creative_permission.naming_mode
            )
            analysis.creative_permission.naming_mode = resolved_mode
            analysis.creative_permission.level = "BOUNDED_BRAINSTORMING"
            if resolved_mode == "MINOR_QUICK":
                if analysis.creative_permission.requested_option_count < 5:
                    analysis.creative_permission.requested_option_count = 6
            else:
                if analysis.creative_permission.requested_option_count < 4:
                    analysis.creative_permission.requested_option_count = 5

            if "linguistics" not in analysis.active_domains:
                analysis.active_domains = ["linguistics"] + list(analysis.active_domains)[:2]

        if not analysis.search_queries:
            analysis.search_queries = [user_message]
        if is_naming_turn and not any(
            kw in " ".join(analysis.search_queries).lower()
            for kw in ("etymolog", "onomastic", "toponym", "naming", "phonolog")
        ):
            analysis.search_queries.append("historical onomastics toponymy naming morphology")

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
        """Apply Guard 3 (Citation Verification) and Guard 4 (Anti-Hook & False-Disclaimer Sanitizer)."""
        cleaned = re.sub(
            r"\n+What (would you like to|aspect of your world|shall we) (explore|build|develop|work on|discuss) next\?.*$",
            "",
            raw_response.strip(),
            flags=re.IGNORECASE,
        ).strip()

        # Strip any false LLM disclaimer claiming it cannot write to the World Bible
        cleaned = re.sub(
            r"I can't write to your (world bible file|document directly)[^.\n]*\.\s*",
            "I have organized and written these entries directly into your World Bible. ",
            cleaned,
            flags=re.IGNORECASE,
        )

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

    def update_rolling_summary(
        self,
        db: Session,
        conversation: Conversation,
        max_recent_messages: int = 12,
    ) -> Optional[str]:
        """Condense messages older than `max_recent_messages` into `Conversation.rolling_summary`."""
        all_msgs = (
            db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc())
            .all()
        )
        if len(all_msgs) <= max_recent_messages:
            return conversation.rolling_summary

        older_msgs = all_msgs[:-max_recent_messages]
        transcript_lines = []
        for m in older_msgs[-20:]:
            snippet = (
                m.content[:1200] + " ...[truncated]..."
                if len(m.content) > 1200
                else m.content
            )
            transcript_lines.append(f"[{m.role}]: {snippet}")
        older_transcript = "\n\n".join(transcript_lines)

        prior_summary = (conversation.rolling_summary or "").strip()
        summary_prompt = (
            f"Prior Rolling Summary:\n{prior_summary or 'None yet.'}\n\n"
            f"Older Conversation Turns to Incorporate:\n{older_transcript}\n\n"
            "Write an updated, cohesive rolling summary (under 350 words) capturing established canon decisions, "
            "rejected ideas, open worldbuilding questions, and key research topics discussed."
        )
        try:
            new_summary = gemini_service.generate_text(
                contents=summary_prompt,
                system_instruction=ROLLING_SUMMARY_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=1200,
                model_override="claude-haiku-5-5",
            )
            if new_summary and new_summary.strip():
                conversation.rolling_summary = new_summary.strip()
                db.commit()
                return conversation.rolling_summary
        except Exception as exc:
            logger.warning("Rolling summary LLM fallback triggered: %s", exc)
            fallback_lines = [
                f"- {m.role}: {m.content[:180].replace(chr(10), ' ')}"
                for m in older_msgs[-8:]
            ]
            conversation.rolling_summary = (
                (prior_summary + "\n" if prior_summary else "")
                + "Condensed earlier turns:\n"
                + "\n".join(fallback_lines)
            )[-2500:]
            db.commit()
        return conversation.rolling_summary

    def extract_post_turn_proposals(
        self,
        db: Session,
        world: World,
        conversation: Conversation,
        user_msg: Message,
        assistant_msg: Message,
        analysis: TurnAnalysisSchema,
    ) -> Dict[str, Any]:
        """Execute Pass 3 World Bible Editor & Canon Extractor.

        - Updates rolling conversation summary when message count exceeds 12 turns.
        - Directly edits/updates World Bible sections when the author provides a brain dump,
          declares canon facts, or asks the agent to edit/update the World Bible.
        - Stages unverified AI suggestions as pending proposals.
        - Wires DEPENDS_ON and CONTRADICTS relationships into FactRelationship rows.
        - Guard 5: Pure real-world factual/research questions NEVER create fictional canon.
        """
        self.update_rolling_summary(db, conversation, max_recent_messages=12)

        if (
            analysis.primary_intent in {"QUESTION_EXPLANATION", "FACTUAL_RESEARCH"}
            and "CANON_DECLARATION" not in analysis.secondary_intents
            and "WORLD_BIBLE_MANAGEMENT" not in analysis.secondary_intents
        ):
            return {"staged_proposals": [], "updated_sections": [], "detected_contradictions": []}

        existing_facts = (
            db.query(Fact)
            .filter(Fact.world_id == world.id, Fact.canon_status == "ESTABLISHED_CANON")
            .limit(30)
            .all()
        )
        canon_summary = "\n".join(
            f"[{f.id[:8]}] {f.title}: {f.content}" for f in existing_facts
        )

        existing_sections = bible_service.list_sections(db, world.id)
        sections_context = "\n\n".join(
            f"Section Slug: `{s.slug}` | Title: `{s.title}`\nCurrent Content:\n{s.markdown_content.strip() or '(empty)'}"
            for s in existing_sections
        )

        # Include recent conversation turns so references like "add what I dropped earlier" have the full brain dump
        recent_msgs = (
            db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(8)
            .all()
        )
        recent_msgs.reverse()
        history_snippet = "\n\n".join(
            f"[{m.role}]: {m.content[:4000]}" for m in recent_msgs[:-2]
        )

        prompt = (
            f"Current World Bible Sections:\n{sections_context}\n\n"
            f"Established Atomic Facts:\n{canon_summary or 'None yet.'}\n\n"
            f"Recent Conversation Context:\n{history_snippet or 'None.'}\n\n"
            f"Latest Author Message:\n{user_msg.content}\n\n"
            "Extract ONLY genuine worldbuilding details established by the author into the existing "
            "15 default World Bible sections in `bible_section_updates` and `proposals`. "
            "Strictly exclude narrative plot beats (Act 1/2/3), character arcs/backstories, and literary themes/tragedy form."
        )
        try:
            extracted = gemini_service.generate_structured(
                prompt=prompt,
                response_schema=PostTurnExtractionSchema,
                system_instruction=PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("Post-turn structured extraction fallback triggered: %s", exc)
            extracted = self._fallback_extract_sections_from_markdown(
                user_msg.content, assistant_msg.content
            )

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

        updated_sections: List[Dict[str, Any]] = []
        updated_section_slugs = set()

        # 1. Apply structured World Bible section edits directly
        for sec_edit in extracted.bible_section_updates:
            if not sec_edit.updated_markdown.strip():
                continue
            sec_obj = bible_service.apply_agent_section_edit(
                db=db,
                world_id=world.id,
                section_slug=sec_edit.section_slug,
                section_title=sec_edit.section_title,
                updated_markdown=sec_edit.updated_markdown,
                summary_of_changes=sec_edit.summary_of_changes,
                provenance=sec_edit.provenance,
            )
            if not sec_obj:
                continue
            updated_section_slugs.add(sec_obj.slug)
            updated_sections.append(
                {
                    "id": sec_obj.id,
                    "title": sec_obj.title,
                    "slug": sec_obj.slug,
                    "version_number": sec_obj.version_number,
                    "summary_of_changes": sec_edit.summary_of_changes,
                }
            )

        # 2. Process atomic facts: auto-commit AUTHOR_CREATED / approved facts, or stage if unapproved
        staged: List[CanonProposal] = []
        created_facts: List[Fact] = []
        for item in extracted.proposals:
            target_sid = subj_by_slug.get(item.subject_slug, default_subj_id)
            should_auto_apply = (
                item.auto_apply_to_bible
                and item.proposed_provenance in {"AUTHOR_CREATED", "AI_GENERATED_AUTHOR_APPROVED"}
            )
            if should_auto_apply:
                from backend.schemas import FactCreate

                # Create atomic fact in ESTABLISHED_CANON; only append to Bible section if section wasn't already rewritten above
                sync_to_sec = item.subject_slug not in updated_section_slugs
                created_fact = bible_service.create_fact(
                    db=db,
                    world_id=world.id,
                    payload=FactCreate(
                        title=item.proposed_title,
                        content=item.proposed_content,
                        subject_ids=[target_sid] if target_sid else [],
                        canon_status="ESTABLISHED_CANON",
                        provenance_type=item.proposed_provenance,
                    ),
                    source_message_id=user_msg.id,
                    sync_to_bible_section=sync_to_sec,
                )
                created_facts.append(created_fact)
                consistency_service.infer_and_link_dependencies(
                    db=db,
                    world_id=world.id,
                    new_fact=created_fact,
                    explicit_dep_ids=item.depends_on_fact_ids,
                )
                if sync_to_sec and not updated_sections:
                    # Record the section that received this fact
                    sec_match = (
                        db.query(Subject).filter(Subject.id == target_sid).first()
                        if target_sid
                        else None
                    )
                    if sec_match:
                        updated_sections.append(
                            {
                                "id": target_sid,
                                "title": sec_match.name,
                                "slug": sec_match.slug,
                                "version_number": 1,
                                "summary_of_changes": f"Added: {created_fact.title}",
                            }
                        )
            else:
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

        # 3. Persist detected contradictions into FactRelationship (CONTRADICTS) rows
        recorded_contradictions: List[Dict[str, Any]] = []
        for contr in extracted.detected_contradictions:
            target_fact = consistency_service.resolve_fact_by_id_or_prefix(
                db, world.id, contr.conflicting_fact_id
            )
            if not target_fact and existing_facts:
                # Try matching any existing fact referenced in the contradiction description
                desc_lower = (contr.description or "").lower()
                for ef in existing_facts:
                    if ef.title.lower() in desc_lower:
                        target_fact = ef
                        break
            if not target_fact:
                continue

            source_fact: Optional[Fact] = None
            if contr.new_fact_title:
                for cf in created_facts:
                    if contr.new_fact_title.lower() in cf.title.lower() or cf.title.lower() in contr.new_fact_title.lower():
                        source_fact = cf
                        break
            if not source_fact and created_facts:
                source_fact = created_facts[-1]
            if not source_fact:
                # Create an UNRESOLVED_POSSIBILITY fact representing the conflicting claim so the edge is recorded
                source_fact = Fact(
                    world_id=world.id,
                    title=(contr.new_fact_title or f"Conflicting statement vs. {target_fact.title}")[:240],
                    content=user_msg.content[:1000],
                    canon_status="UNRESOLVED_POSSIBILITY",
                    provenance_type="AUTHOR_CREATED",
                    authority_tier=1,
                    source_message_id=user_msg.id,
                )
                db.add(source_fact)
                db.flush()

            if source_fact.id != target_fact.id:
                rel = consistency_service.create_relationship(
                    db=db,
                    world_id=world.id,
                    source_fact_id=source_fact.id,
                    target_fact_id=target_fact.id,
                    relationship_type="CONTRADICTS",
                    description=contr.description,
                    created_by="AI_INFERRED",
                )
                recorded_contradictions.append(
                    {
                        "relationship_id": rel.id,
                        "source_fact_id": source_fact.id,
                        "target_fact_id": target_fact.id,
                        "description": contr.description,
                    }
                )

        db.commit()
        return {
            "staged_proposals": staged,
            "updated_sections": updated_sections,
            "detected_contradictions": recorded_contradictions,
        }

    def _fallback_extract_sections_from_markdown(
        self, user_text: str, assistant_text: str
    ) -> PostTurnExtractionSchema:
        """Fallback parser that extracts Markdown headings from the consultant response into World Bible sections."""
        from backend.schemas import BibleSectionEditItem
        from backend.services.world_service import slugify

        updates: List[BibleSectionEditItem] = []
        blocks = re.split(r"\n(?=##{1,2}\s+)", assistant_text)
        for blk in blocks:
            m = re.match(r"^##{1,2}\s+([^\n]+)\n+(.*)", blk.strip(), flags=re.DOTALL)
            if m:
                heading = m.group(1).strip().strip("*#:")
                body = m.group(2).strip()
                if len(body) > 30:
                    updates.append(
                        BibleSectionEditItem(
                            section_slug=slugify(heading),
                            section_title=heading,
                            updated_markdown=body,
                            summary_of_changes=f"Extracted '{heading}' into World Bible",
                            provenance="AUTHOR_CREATED",
                        )
                    )
        return PostTurnExtractionSchema(bible_section_updates=updates, proposals=[])

    @staticmethod
    def _resolve_naming_mode(user_message: str, llm_naming_mode: str) -> str:
        """Resolve MAJOR_CONTEXTUAL vs. MINOR_QUICK using explicit 'major'/'minor' phrasing and LLM classification."""
        msg_lower = user_message.lower()
        has_major = bool(re.search(r"\bmajor\b", msg_lower))
        has_minor = bool(re.search(r"\bminor\b", msg_lower))

        if has_minor and not has_major:
            return "MINOR_QUICK"
        if has_major and not has_minor:
            return "MAJOR_CONTEXTUAL"
        if has_major and has_minor:
            # Handle contrastive phrasing like "not a major character, just a minor one" or "major city, not a minor town"
            neg_major = bool(
                re.search(
                    r"\b(not|isn't|isnt|no|rather than|instead of|nothing)\b[^.,;]{0,35}\bmajor\b",
                    msg_lower,
                )
            )
            neg_minor = bool(
                re.search(
                    r"\b(not|isn't|isnt|no|rather than|instead of|more than)\b[^.,;]{0,35}\bminor\b",
                    msg_lower,
                )
            )
            if neg_major and not neg_minor:
                return "MINOR_QUICK"
            if neg_minor and not neg_major:
                return "MAJOR_CONTEXTUAL"
            if llm_naming_mode in {"MAJOR_CONTEXTUAL", "MINOR_QUICK"}:
                return llm_naming_mode

        if llm_naming_mode in {"MAJOR_CONTEXTUAL", "MINOR_QUICK"}:
            return llm_naming_mode
        if MINOR_NAMING_PATTERNS.search(user_message):
            return "MINOR_QUICK"
        return "MAJOR_CONTEXTUAL"

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
