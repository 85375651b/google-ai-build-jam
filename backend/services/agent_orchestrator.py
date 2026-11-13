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
    FactRevision,
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
from backend.services.author_intent import (
    author_evidence, clear_author_decisions, delegated_evidence, extraction_eligible,
    naming_fact_supported, reconcile_turn, section_naming_supported, suggestion_in_scope,
)
from backend.services.bible_service import bible_service
from backend.services.book_ingestion_service import book_ingestion_service
from backend.services.consistency_service import consistency_service
from backend.services.context_assembler import context_assembler
from backend.services.context_retrieval import context_retriever
from backend.services.entity_service import entity_service
from backend.services.llm_service import llm_service
from backend.services.generation_control import current_generation
from backend.services.research_service import research_service
from backend.services.world_service import world_service

logger = logging.getLogger(__name__)

MINOR_NAMING_PATTERNS = re.compile(
    r"\b(minor|quick|fast|background|side character|incidental|guard|innkeeper|shopkeeper|tavern|small village|hamlet|street name|just need a name|couple of quick names|random name)\b",
    re.IGNORECASE,
)


class AgentOrchestrator:
    """Coordinates Pass 1 (Intent/Permission), RAG/Tavily retrieval, Pass 2 (Response), and Pass 3 (Proposals)."""

    def analyze_turn(
        self, db: Session, world: World, user_message: str, conversation: Optional[Conversation] = None, *, persist_expertise: bool = True
    ) -> TurnAnalysisSchema:
        """Classify once, then reconcile author evidence and scoped creative authority."""
        preview_msg = (
            user_message[:6000] + "\n...[truncated middle of long brain dump]...\n" + user_message[-2000:]
            if len(user_message) > 8500
            else user_message
        )
        reference_context = ""
        if conversation is not None:
            recent = (
                db.query(Message).filter(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc()).limit(2).all()
            )
            reference_context = "\n".join(f"[{m.role}]: {m.content[:4000]}" for m in reversed(recent))
        prompt = (
            f"Reference context (resolve pronouns/options only; NOT new authorization): {reference_context}\n"
            f"World Name: {world.name}\n"
            f"Author Message: {preview_msg}\n"
            "Classify intent, creative permission scope, expertise signals, and search queries."
        )
        classifier_failed = False
        try:
            analysis = llm_service.generate_structured(
                prompt=prompt,
                response_schema=TurnAnalysisSchema,
                task="classification",
                system_instruction=PASS_1_TURN_ANALYZER_SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as exc:
            if current_generation.get() is not None:
                raise
            logger.warning("Pass 1 structured analysis fallback triggered: %s", exc)
            classifier_failed = True
            analysis = self._heuristic_turn_analysis(user_message)

        analysis = reconcile_turn(user_message, analysis, reference_context, classifier_failed=classifier_failed)
        perm = analysis.creative_permission
        is_naming_turn = perm.allow_invented_proper_nouns
        if is_naming_turn:
            perm.naming_mode = self._resolve_naming_mode(user_message, perm.naming_mode)
            if not perm.requested_option_count:
                perm.requested_option_count = 6 if perm.naming_mode == "MINOR_QUICK" else 5
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
        if persist_expertise and analysis.expertise_signals:
            world_service.apply_expertise_signals(
                db, world.id, analysis.expertise_signals
            )

        return analysis

    def retrieve_knowledge(
        self, db: Session, analysis: TurnAnalysisSchema, user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve passages from The Planet Construction Kit and trigger Tavily when coverage is PARTIAL/LOW."""
        raw = user_message if user_message is not None else analysis.raw_user_phrasing or ' '.join(
            decision.source_quote for decision in clear_author_decisions(analysis))
        clauses = re.split(r'[.!?\n]+', raw)
        explicit_research = any(re.search(
            r'^(?:please\s+)?(?:research|fact-check|verify|investigate)\b|'
            r'\b(?:can|could|would|will) you (?:please )?(?:research|verify|investigate)\b|'
            r'^(?:explain|check|assess|evaluate)\b.*\b(?:real|scientific|plausib\w*|origin|meaning|etymolog\w*)\b', clause.strip(), re.I)
            for clause in clauses)
        question = '?' in raw or any(re.match(r'\s*(?:what|how|why|where|when)\b', clause, re.I) for clause in clauses)
        canon_only = (bool(clear_author_decisions(analysis))
            and analysis.primary_intent in {'CANON_DECLARATION', 'WORLD_BIBLE_MANAGEMENT'}
            and analysis.creative_permission.authorization_kind == 'NONE' and not explicit_research and not question)
        # This is reference selection only: never change the classifier's intent,
        # author decisions, eligibility, or permission contract.
        if canon_only or (not analysis.requires_factual_retrieval and not explicit_research):
            return {'book': {'coverage_status': 'NOT_REQUIRED', 'passages': [],
                             'coverage_reason': 'This turn does not require factual support.'},
                    'online': {'status': 'SKIPPED_NOT_REQUIRED', 'sources': []}}
        book_ingestion_service.ensure_primary_book_ingested(db)
        book_res = book_ingestion_service.retrieve_passages(
            db=db,
            queries=analysis.search_queries,
            active_domains=analysis.active_domains,
            top_k=5,
        )
        coverage = book_res.get("coverage_status", "LOW")
        if coverage in {"PARTIAL", "LOW"} and (analysis.requires_factual_retrieval or explicit_research):
            web_res = research_service.search_online(
                db=db, queries=analysis.search_queries, max_results=3
            )
        else:
            web_res = {
                "status": "SKIPPED_HIGH_BOOK_COVERAGE",
                "message": "Primary reference book sufficiently covers the topic.",
                "sources": [],
            }
        book_res, web_res = context_retriever.select_research(book_res, web_res, analysis.search_queries)
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
            if re.search(r'(?<![\w-])' + re.escape(cid) + r'(?![\w-])', cleaned):
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
            if re.search(r'(?<![\w-])' + re.escape(cid) + r'(?![\w-])', cleaned):
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
        *, persist: bool = True,
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
            new_summary = llm_service.generate_text(
                contents=summary_prompt,
                system_instruction=ROLLING_SUMMARY_SYSTEM_PROMPT,
                task="summarization",
                temperature=0.2,
                max_output_tokens=1200,
            )
            if new_summary and new_summary.strip():
                if persist:
                    conversation.rolling_summary = new_summary.strip()
                    db.commit()
                return new_summary.strip()
        except Exception as exc:
            if current_generation.get() is not None:
                raise
            logger.warning("Rolling summary LLM fallback triggered: %s", exc)
            fallback_lines = [
                f"- {m.role}: {m.content[:180].replace(chr(10), ' ')}"
                for m in older_msgs[-8:]
            ]
            fallback_summary = (
                (prior_summary + "\n" if prior_summary else "")
                + "Condensed earlier turns:\n"
                + "\n".join(fallback_lines)
            )[-2500:]
            if persist:
                conversation.rolling_summary = fallback_summary
                db.commit()
            return fallback_summary
        return conversation.rolling_summary

    def prepare_post_turn_extraction(
        self, db: Session, world: World, conversation: Conversation,
        user_msg: Message, assistant_msg: Message, analysis: TurnAnalysisSchema,
    ):
        """Run the existing extraction policy/model without applying any canon edits."""
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


        # Broad intent labels never veto independently evidenced author decisions.
        analysis = reconcile_turn(user_msg.content, analysis, history_snippet)
        if not extraction_eligible(analysis):
            return None

        response_schema = context_assembler.extraction_response_schema(PostTurnExtractionSchema, analysis)
        prompt, retrieved, _ = context_assembler.assemble_extraction(
            db, world, conversation, user_msg, assistant_msg, analysis,
            PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT, response_schema,
        )
        from backend.models import Entity
        entity_versions = dict(db.query(Entity.id, Entity.version_number).filter(Entity.world_id == world.id).all())

        extraction_failed = False
        try:
            extracted = llm_service.generate_structured(
                prompt=prompt,
                response_schema=response_schema,
                system_instruction=PASS_3_POST_TURN_EXTRACTOR_SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as exc:
            if current_generation.get() is not None:
                raise
            logger.warning("Post-turn structured extraction fallback triggered: %s", exc)
            # Never promote assistant prose to AUTHOR_CREATED when extraction fails.
            extracted = PostTurnExtractionSchema()
            extraction_failed = True

        context_assembler.expand_extracted_section_edits(extracted, retrieved,
            eligible=lambda edit: author_evidence(edit.source_quote, analysis))
        return {"analysis": analysis, "extracted": extracted, "history_snippet": history_snippet,
                "section_versions": retrieved.section_versions, 'entity_versions': entity_versions, 'extraction_failed': extraction_failed}

    def extract_post_turn_proposals(
        self, db: Session, world: World, conversation: Conversation,
        user_msg: Message, assistant_msg: Message, analysis: TurnAnalysisSchema,
        *, prepared=None, manage_summary: bool = True,
    ) -> Dict[str, Any]:
        """Apply the existing author-intent safeguards; lifecycle caller owns commit boundaries."""
        if manage_summary:
            self.update_rolling_summary(db, conversation, max_recent_messages=12)
        if prepared is None:
            prepared = self.prepare_post_turn_extraction(db, world, conversation, user_msg, assistant_msg, analysis)
        if prepared is None:
            return {"staged_proposals": [], "updated_sections": [], "detected_contradictions": []}
        if prepared.get('extraction_failed'):
            return {"staged_proposals": [], "updated_sections": [], "detected_contradictions": []}
        analysis = prepared["analysis"]
        extracted = prepared["extracted"]
        history_snippet = prepared["history_snippet"]
        existing_facts = db.query(Fact).filter(Fact.world_id == world.id, Fact.canon_status == "ESTABLISHED_CANON").limit(30).all()
        existing_sections = bible_service.list_sections(db, world.id)

        # Clear supported author clauses are independently compiled so a model
        # cannot omit the related owner/prohibition in a multi-entity correction.
        operations = entity_service.compile_author_operations(analysis, user_msg.content)
        from backend.schemas import EntityOperation
        for item in extracted.proposals:
            binding = re.fullmatch(r"(Planet \S+) is (?:named |called )?([\w'’-]+(?: [\w'’-]+)*)[.]?", item.proposed_content.strip().rstrip('.'))
            if binding and (author_evidence(item.source_quote, analysis) and naming_fact_supported(item, user_msg.content, history_snippet)
                            or delegated_evidence(item, analysis, assistant_msg.content, history_snippet)):
                operations.append(EntityOperation(action='rename', entity_ref=binding[1], entity_type='planet', value=binding[2],
                    source_quote=item.source_quote, decision_origin=item.decision_origin, decision_target=item.decision_target))
        operations.extend(extracted.operations)
        unique_operations = {}
        for operation in operations:
            signature = (operation.action, (operation.entity_id or operation.entity_ref).casefold(), operation.property_key, json.dumps(operation.value, sort_keys=True))
            unique_operations.setdefault(signature, operation)
        operations = list(unique_operations.values())
        if extracted.detected_contradictions and not operations:
            # Contradictory free-form claims require confirmation, before any
            # section/canonical Fact write. Retain reviewable non-canon proposals.
            from backend.schemas import FactCreate
            staged, recorded = [], []
            for item in extracted.proposals:
                if not author_evidence(item.source_quote, analysis):
                    continue
                staged.append(bible_service.stage_proposal(db, world.id, item.proposed_title,
                    item.proposed_content, 'AUTHOR_CREATED', conversation_id=conversation.id, source_message_id=user_msg.id))
                pending_fact = bible_service.create_fact(db, world.id, FactCreate(title=item.proposed_title,
                    content=item.proposed_content, canon_status='UNRESOLVED_POSSIBILITY', provenance_type='AUTHOR_CREATED'),
                    source_message_id=user_msg.id, sync_to_bible_section=False)
                consistency_service.infer_and_link_dependencies(db, world.id, pending_fact, item.depends_on_fact_ids)
                for contradiction in extracted.detected_contradictions:
                    target = consistency_service.resolve_fact_by_id_or_prefix(db, world.id, contradiction.conflicting_fact_id)
                    if target:
                        relation = consistency_service.create_relationship(db, world.id, pending_fact.id, target.id,
                            'CONTRADICTS', contradiction.description, 'AI_CONSISTENCY_ENGINE')
                        recorded.append({'relationship_id': relation.id, 'description': contradiction.description})
            return {'staged_proposals': staged, 'updated_sections': [], 'detected_contradictions': recorded,
                    'canon_result': {'status': 'clarification_required', 'operations': [],
                        'message': 'A possible conflict was found. The changes remain proposals; existing canonical values were preserved. Which version should be established?'}}
        structured = None
        if operations:
            structured = entity_service.apply(db, world.id, operations, f'turn:{user_msg.id}',
                analysis=analysis, author_text=user_msg.content, assistant_text=assistant_msg.content,
                source_message_id=user_msg.id, expected_versions=prepared.get('entity_versions', {}), reference_context=history_snippet)
        structured_quotes = {op.source_quote for op in operations if op.decision_origin in {'AUTHOR', 'DELEGATED'}}

        # Record rejected ideas immediately so the agent never re-introduces them
        for rej in extracted.rejected_ideas:
            if not author_evidence(rej.source_quote, analysis) or not re.search(
                r"\b(?:reject|discard|anywhere|at all|ever again)\b", rej.source_quote, re.I
            ):
                continue
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
            if sec_edit.source_quote in structured_quotes:
                continue  # The targeted property projection already owns this clause.
            if not sec_edit.updated_markdown.strip() or not author_evidence(sec_edit.source_quote, analysis):
                continue
            if not section_naming_supported(
                sec_edit, "\n".join(s.markdown_content for s in existing_sections),
                user_msg.content, history_snippet,
            ):
                continue
            sec_obj = bible_service.apply_agent_section_edit(
                db=db,
                world_id=world.id,
                section_slug=sec_edit.section_slug,
                section_title=sec_edit.section_title,
                updated_markdown=sec_edit.updated_markdown,
                summary_of_changes=sec_edit.summary_of_changes,
                provenance="AUTHOR_CREATED",
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
            if item.source_quote in structured_quotes and re.match(r'^Planet\s+\S+\s+is\b', item.proposed_content):
                continue
            target_sid = subj_by_slug.get(item.subject_slug, default_subj_id)
            from_author = (
                item.decision_origin == "AUTHOR" and author_evidence(item.source_quote, analysis)
                and naming_fact_supported(item, user_msg.content, history_snippet)
            )
            delegated = item.decision_origin == "DELEGATED" and delegated_evidence(item, analysis, assistant_msg.content, history_snippet)
            can_suggest = suggestion_in_scope(item, analysis)
            if not (from_author or delegated or can_suggest):
                continue
            should_auto_apply = from_author or delegated
            # The model's auto-apply/provenance flags cannot approve its own suggestions.
            item.proposed_provenance = "AUTHOR_CREATED" if from_author else "AI_GENERATED_AUTHOR_APPROVED"

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
                if delegated:
                    revision = db.query(FactRevision).filter(FactRevision.fact_id == created_fact.id).first()
                    revision.changed_by = "AI_AUTHOR_DELEGATED"
                    revision.change_reason = (
                        f"Author delegation: {item.source_quote} | Scope: {item.decision_target}"
                    )
                    db.commit()
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
            if not clear_author_decisions(analysis):
                continue
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
            'canon_result': structured,
        }

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
        """Fail closed; evidence reconciliation rescues only clear author speech acts."""
        return TurnAnalysisSchema(
            primary_intent="IMPLEMENTATION_ASSISTANCE",
            ambiguity_status="NEEDS_CLARIFICATION",
            creative_permission=CreativePermissionSchema(level="ANALYTICAL_ADVISORY"),
            raw_user_phrasing=user_message,
            search_queries=[user_message],
        )


agent_orchestrator = AgentOrchestrator()
