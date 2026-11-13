"""Dynamic AI Context Assembler implementing the 8-Layer Modular Prompt Architecture."""

import json
import re
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from backend.models import (
    AuthorExpertiseProfile,
    BibleSection,
    Conversation,
    World,
)
from backend.prompts.domain_cartridges import (
    DOMAIN_CARTRIDGES,
    EXPERTISE_PEDAGOGICAL_GUIDANCE,
)
from backend.prompts.templates import (
    INTENT_EXECUTION_INSTRUCTIONS,
    NAMING_INTELLIGENCE_INSTRUCTIONS,
)
from backend.schemas import TurnAnalysisSchema
from backend.config import settings
from backend.services.author_intent import clear_author_decisions
from backend.services.context_retrieval import context_retriever, ContextBudgetError, estimate_tokens



class ContextAssembler:
    """Constructs token-budgeted, epistemologically segregated prompts for the local model."""

    def assemble_consultant_system_prompt(
        self,
        db: Session,
        world: World,
        turn_analysis: TurnAnalysisSchema,
        book_retrieval: Dict[str, Any],
        online_research: Dict[str, Any],
        focused_subject_id: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        latest_user_message: str = "",
        retrieved=None,
    ) -> str:
        """Build the 8-layer modular system instruction string for Pass 2."""
        perm = turn_analysis.creative_permission
        query = latest_user_message or turn_analysis.raw_user_phrasing or " ".join(
            [perm.explicit_authorization_quote] + [d.source_quote for d in turn_analysis.author_decisions] + turn_analysis.search_queries
        )
        retrieved = retrieved or context_retriever.retrieve(db, world.id, query, turn_analysis, conversation, focused_subject_id)
        # Legacy world preferences cannot override this turn's author authorization.
        allow_proper_nouns = perm.allow_invented_proper_nouns
        req_count = (
            perm.requested_option_count
            if perm.requested_option_count > 0
            else (
                world.brainstorm_option_count_default
                if perm.level == "BOUNDED_BRAINSTORMING"
                else 0
            )
        )

        layer_1_2 = f"""<layer_1_consultant_identity>
You are an expert Worldbuilding Consultant and World Bible Editor assisting an author in developing their fictional world "{world.name}".
GOVERNING PRINCIPLE: The author is the sole creative authority. Your job is to provide subject-matter knowledge, analyze systemic implications, and help implement and organize the author's ideas—never to take over the creative direction.
CRITICAL CAPABILITY — DIRECT WORLD BIBLE EDITING IS ENABLED:
- You ARE directly integrated with the application's built-in World Bible database. Whenever the author asks you to add, organize, edit, or update details in their World Bible (or pastes a brain dump), our backend automatically writes the extracted sections and facts directly into their World Bible right as your response completes.
- NEVER say "I can't write to your world bible file", "I don't have access to your document", or "Here is a paste-ready version since I can't edit it." Even if an earlier message in the chat transcript claimed you couldn't edit the World Bible directly, ignore that outdated statement—your direct World Bible editing integration is LIVE.
- When asked to add or update the World Bible (or when given a brain dump), extract ONLY worldbuilding details (filtering out narrative plot outlines, Act 1/2/3 beats, character arcs, and themes—only mentioning characters when they define a worldbuilding institution or technology, e.g., who founded a major network) and organize them into the existing 15 default World Bible categories first (such as Climate and Ecology, Technology and Engineering, Languages and Linguistics, Cultures and Societies, Politics and Governance, History and Chronology). Only create a new category if something genuinely cannot fit.
- Do NOT invent lore, factions, history, rituals, or proper nouns unless explicitly authorized in <layer_2_creative_permission_contract>.
- Do NOT push the author toward conventional fantasy or sci-fi tropes.
- Do NOT end your response with generic coaching filler like "What would you like to explore next?". Stop naturally once you have answered the author's turn.
</layer_1_consultant_identity>

<layer_2_creative_permission_contract>
Current Turn Permission Level: {perm.level}
Explicit Authorization Quote: "{perm.explicit_authorization_quote}"
Allowed Creative Scope: {perm.allowed_targets}
Prohibited Unrequested Scope: {perm.prohibited_adjacent_targets}
Invented Proper Nouns Permitted: {allow_proper_nouns}
Naming Mode: {getattr(perm, "naming_mode", "NONE")}
Requested Option Count: {req_count}
Authorization Kind: {perm.authorization_kind}
Author-established clauses: {[d.model_dump() for d in turn_analysis.author_decisions]}
Uncertain clauses (not canon): {turn_analysis.uncertain_statements}
Clarification Question: {turn_analysis.clarification_question}
- Recording author-established decisions is permitted even without creative permission.
- SUGGEST/DRAFT authorizes only the quoted task; generated options remain non-canon.
- DELEGATE authorizes only the quoted decision. State that decision as one plain canonical sentence
  identifying the target, e.g. "Planet X is named Pallene." Do not decide unrelated properties.
- Corrections and prohibitions NEVER authorize replacement names or unrelated invention.
- Latest explicit author instructions supersede earlier permissions for the same task.
- If intent or a referent is unclear, ask one concise clarifying question before deciding it.
  Preserve clear decisions elsewhere in the same message; do not guess an unnamed owner.
- Do not claim a World Bible write has succeeded before persistence is confirmed.
</layer_2_creative_permission_contract>"""

        intent_instr = INTENT_EXECUTION_INSTRUCTIONS.get(
            turn_analysis.primary_intent,
            INTENT_EXECUTION_INSTRUCTIONS["IMPLEMENTATION_ASSISTANCE"],
        )
        layer_3 = f"""<layer_3_turn_intent_and_bridging>
Primary Intent: {turn_analysis.primary_intent}
Secondary Intents: {turn_analysis.secondary_intents}
Ambiguity Status: {turn_analysis.ambiguity_status}
Inferred Domain Concepts: {turn_analysis.inferred_domain_concepts}
Execution Instruction: {intent_instr}
</layer_3_turn_intent_and_bridging>"""

        active_domains = turn_analysis.active_domains or ["political_science", "history"]
        profiles = (
            db.query(AuthorExpertiseProfile)
            .filter(
                AuthorExpertiseProfile.world_id == world.id,
                AuthorExpertiseProfile.domain_key.in_(active_domains),
            )
            .all()
        )
        prof_map = {p.domain_key: p for p in profiles}
        exp_lines: List[str] = []
        for d_key in active_domains[:3]:
            cart = DOMAIN_CARTRIDGES.get(d_key)
            d_name = cart["name"] if cart else d_key
            prof = prof_map.get(d_key)
            lvl = prof.expertise_level if prof else "UNKNOWN"
            guidance = EXPERTISE_PEDAGOGICAL_GUIDANCE.get(
                lvl, EXPERTISE_PEDAGOGICAL_GUIDANCE["UNKNOWN"]
            )
            exp_lines.append(
                f"- Domain: {d_name} ({d_key}) | Author Expertise Level: {lvl}\n"
                f"  Pedagogical Guidance: {guidance}"
            )
        layer_4 = (
            "<layer_4_adaptive_expertise_profile>\n"
            + "\n".join(exp_lines)
            + "\n</layer_4_adaptive_expertise_profile>"
        )

        book_passages = book_retrieval.get("passages", [])
        book_status = book_retrieval.get("coverage_status", "LOW")
        book_xml_blocks = [
            f'  <passage citation_id="{p["citation_id"]}" chapter="{p["chapter_title"]}" '
            f'section="{p["section_title"]}" book_page="{p["book_page_start"]}" '
            f'book_page_end="{p.get("book_page_end", p["book_page_start"])}" '
            f'pdf_page="{p.get("pdf_page_start", "unknown")}">\n'
            f'  {p["content"]}\n  </passage>'
            for p in book_passages
        ]
        book_xml_str = (
            "\n".join(book_xml_blocks)
            if book_xml_blocks
            else "  <!-- No relevant passages in primary reference book. -->"
        )
        book_citation_instruction = (f'Cite inline using [{book_passages[0]["citation_id"]}] and the other supplied passage IDs.'
            if book_passages else 'No book passages were retrieved for this turn; do not invent book citations.')

        web_sources = online_research.get("sources", [])
        web_xml_blocks = [
            f'  <web_source citation_id="{s["citation_id"]}" title="{s["title"]}" '
            f'domain="{s["domain"]}" url="{s["url"]}">\n  {s["excerpt"]}\n  </web_source>'
            for s in web_sources
        ]
        web_xml_str = (
            "\n".join(web_xml_blocks)
            if web_xml_blocks
            else "  <!-- No supplementary online sources retrieved. -->"
        )

        layer_5 = f"""<layer_5_grounded_research_and_source_policy>
SOURCE HIERARCHY:
0. The author's explicit decisions govern fictional canon absolutely.
1. Primary Real-World Reference: Mark Rosenfelder's "The Planet Construction Kit". {book_citation_instruction}
2. Supplementary Online Research: use only citation IDs present in the supplied web sources.
3. General model knowledge is an unverified fallback, never independently verified research.
Real-world sources explain reality; they NEVER override the author's fictional canon.

<real_world_reference_book coverage_status="{book_status}">
{book_xml_str}
</real_world_reference_book>

<supplementary_online_research status="{online_research.get('status', 'SKIPPED')}">
{web_xml_str}
</supplementary_online_research>
</layer_5_grounded_research_and_source_policy>"""

        canon_text = "\n\n".join(i.text for i in retrieved.items if
            i.kind == "fact" and i.details.get("status") == "ESTABLISHED_CANON") or "No relevant established canon retrieved."
        unresolved_text = "\n\n".join(i.text for i in retrieved.items if i.kind in {"proposal", "question"} or
            (i.kind == "fact" and i.details.get("status") not in {"ESTABLISHED_CANON", "REJECTED"})) or "None."
        rejected_text = "\n\n".join(i.text for i in retrieved.items if i.kind == "fact" and i.details.get("status") == "REJECTED") or "None."

        layer_6 = f"""<layer_6_world_canon_and_epistemic_state>
World Name: {world.name}
World Description: {world.description or "Not yet specified."}

<established_world_canon rule="TIER_1_IMMUTABLE_FICTIONAL_TRUTH">
{canon_text}
</established_world_canon>

<world_bible_passages rule="PRESERVE_QUALIFICATIONS_AND_NEGATIONS; HYPOTHETICAL_WORDING_IS_NOT_CANON; FACT_STATUS_LABELS_TAKE_PRECEDENCE">
{retrieved.render({'section'})}
</world_bible_passages>

<unresolved_and_hypothetical_context rule="NON_CANON_DO_NOT_TREAT_AS_ESTABLISHED">
{unresolved_text}
</unresolved_and_hypothetical_context>

<rejected_and_prohibited_ideas rule="EXPLICITLY_REJECTED_NEVER_REINTRODUCE">
{rejected_text}
</rejected_and_prohibited_ideas>

<related_fact_relationships>
{retrieved.render({'entity', 'structured_entity', 'relationship', 'notice'})}
</related_fact_relationships>
<historical_discussions rule="REFERENCE_ONLY_NOT_FRESH_AUTHORITY; AI_TEXT_IS_NOT_AUTHOR_EVIDENCE; LATEST_AUTHOR_CORRECTIONS_WIN">
{retrieved.render({'history'})}
</historical_discussions>
</layer_6_world_canon_and_epistemic_state>"""

        cart_blocks = []
        for d_key in active_domains[:3]:
            cart = DOMAIN_CARTRIDGES.get(d_key)
            if cart:
                cart_blocks.append(
                    f'  <domain_module key="{cart["key"]}" name="{cart["name"]}">\n'
                    f'    Analytical Lenses: {cart["analytical_lenses"]}\n'
                    f'    Cross-Domain Dependencies: {cart["cross_domain_dependencies"]}\n'
                    f"  </domain_module>"
                )
        layer_7 = (
            "<layer_7_domain_knowledge_cartridges>\n"
            + "\n".join(cart_blocks)
            + "\n</layer_7_domain_knowledge_cartridges>"
        )

        layers = [layer_1_2, layer_3, layer_4, layer_5, layer_6, layer_7]
        naming_mode = getattr(perm, "naming_mode", "NONE")
        if allow_proper_nouns or naming_mode in {"MAJOR_CONTEXTUAL", "MINOR_QUICK"}:
            mode_key = (
                naming_mode
                if naming_mode in {"MAJOR_CONTEXTUAL", "MINOR_QUICK"}
                else "MAJOR_CONTEXTUAL"
            )
            naming_instr = (
                "Make only the explicitly delegated decision, within its quoted scope; do not offer or establish unrelated names."
                if perm.authorization_kind == "DELEGATE"
                else NAMING_INTELLIGENCE_INSTRUCTIONS[mode_key]
            )
            registry_text = self._build_existing_names_and_linguistics_registry(
                retrieved
            )
            layer_9 = (
                f'<layer_9_context_aware_naming_intelligence mode="{mode_key}">\n'
                f"{naming_instr}\n\n"
                f"{registry_text}\n"
                "</layer_9_context_aware_naming_intelligence>"
            )
            layers.append(layer_9)

        return "\n\n".join(layers)

    def build_conversation_contents(
        self, db: Session, conversation: Conversation, latest_user_message: str,
        max_recent_messages: Optional[int] = None, *, retrieved=None,
    ) -> List[Dict[str, str]]:
        retrieved = retrieved or context_retriever.retrieve(db, conversation.world_id, latest_user_message, conversation=conversation)
        messages = retrieved.recent[-max_recent_messages:] if max_recent_messages else retrieved.recent
        contents = []
        if retrieved.summary:
            contents.append({"role": "user", "content":
                "[Rolling Summary of Earlier Conversation Turns — reference only, not new authority; hypotheses remain non-canon]\n" + retrieved.summary})
        for message in messages:
            if message == messages[-1] and message.role == "USER" and message.content.strip() == latest_user_message.strip():
                continue
            # Preserve complete recent messages: budget overflow is handled explicitly.
            contents.append({"role": "assistant" if message.role == "ASSISTANT" else "user", "content": message.content})
        contents.append({"role": "user", "content": latest_user_message})
        return contents

    @staticmethod
    def _build_existing_names_and_linguistics_registry(retrieved):
        return (
            "Relevant names, aliases, cultural/linguistic context, reservations and rejected names are in Layer 6. "
            "Their canon status and provenance remain authoritative. Naming-related retrieval is topic selection only, "
            "never creative permission. Check relevant reservations before suggesting names; do not infer missing owners.\n"
            + retrieved.render({'naming_registry'})
        )

    def assemble_consultation(self, db, world, analysis, book, online, conversation,
                              latest_user_message, focused_subject_id=None):
        retrieved = context_retriever.retrieve(db, world.id, latest_user_message, analysis, conversation, focused_subject_id)
        budget = self.context_budget('consultation')
        while True:
            system = self.assemble_consultant_system_prompt(db, world, analysis, book, online,
                focused_subject_id, conversation, latest_user_message, retrieved)
            contents = self.build_conversation_contents(db, conversation, latest_user_message, retrieved=retrieved)
            components = self.components(system, contents, retrieved, book, online)
            total = sum(estimate_tokens(value) for value in components.values())
            if total <= budget:
                diagnostic = context_retriever.record('consultation', retrieved, components, budget,
                    len(book.get('passages', [])), len(online.get('sources', [])))
                return system, contents, diagnostic
            if not self.reduce_optional(retrieved):
                context_retriever.record('consultation', retrieved, components, budget,
                    len(book.get('passages', [])), len(online.get('sources', [])))
                raise ContextBudgetError('consultation', total, budget)

    def assemble_extraction(self, db, world, conversation, user_msg, assistant_msg, analysis, system_instruction, schema):
        query = user_msg.content
        if analysis.creative_permission.authorization_kind == 'DELEGATE':
            query += '\n' + assistant_msg.content
        retrieved = context_retriever.retrieve(db, world.id, query, analysis, conversation,
            exclude_message_ids=(user_msg.id, assistant_msg.id), purpose='extraction')
        # Empty/append-only target windows let new canon use existing categories
        # without loading their unrelated contents or replacing those contents.
        section_targets = set()
        domain_sections = {'geography_geology': 'geography-geology', 'astronomy': 'cosmology-physics',
            'cosmology_physics': 'cosmology-physics', 'political_science': 'politics-governance',
            'history': 'history-chronology', 'economics': 'economics-trade', 'linguistics': 'languages-linguistics',
            'anthropology_sociology': 'cultures-societies', 'biology_ecology': 'biology-evolution', 'religious_studies': 'religion-mythology',
            'physics_cosmology': 'cosmology-physics', 'technology_engineering': 'technology-engineering',
            'military_warfare': 'warfare-military', 'architecture_material_culture': 'architecture-material-culture'}
        section_targets.update(domain_sections[d] for d in analysis.active_domains if d in domain_sections)
        target_terms = {
            'cosmology-physics': r'planet|star|moon|orbit|solar|cosmology|gravity',
            'geography-geology': r'island|islands|caldera|volcano|volcanic|mountain|coast|continent|river|geograph|geolog',
            'climate-ecology': r'climate|ecolog|weather|smog|rain|atmospher|forest|desert',
            'biology-evolution': r'biolog|evolution|species|genetic|organism',
            'history-chronology': r'histor|chronolog|founded|centur|years ago',
            'languages-linguistics': r'language|linguistic|phonolog|dialect|naming convention',
            'cultures-societies': r'cultur|societ|character|birthday|mother|father|her name|his name',
            'religion-mythology': r'religio|mytholog|priest|temple|worship|deit',
            'politics-governance': r'politic|govern|kingdom|republic|empire|capital|council|treaty|alliance',
            'economics-trade': r'econom|trade|currency|coin|grain|tax|market|silo',
            'technology-engineering': r'technolog|engineer|network|computer|machine',
            'warfare-military': r'war|military|army|weapon|soldier',
            'philosophy-ideologies': r'philosoph|ideolog',
            'demographics-population': r'demograph|population|birth rate|migration',
            'architecture-material-culture': r'architect|building|dome|habitat|material|sealed',
        }
        section_targets.update(slug for slug, pattern in target_terms.items() if re.search(r'\b(?:' + pattern + ')', query, re.I))
        if re.search(r'\b(?:planet|star|moon|orbit|solar|cosmology)\b', query, re.I):
            section_targets.add('cosmology-physics')
        if retrieved.category == 'political':
            section_targets.add('politics-governance')
        if retrieved.category == 'character':
            section_targets.add('cultures-societies')
        if re.search(r'\b(?:naming convention|language|phonolog|linguistic)', query, re.I):
            section_targets.add('languages-linguistics')
        for section in db.query(BibleSection).filter(BibleSection.world_id == world.id):
            if section.slug in section_targets and section.slug not in retrieved.section_windows:
                end = len(section.markdown_content)
                retrieved.section_windows[section.slug] = {'id': section.id, 'title': section.title,
                    'version': section.version_number, 'provenance': section.dominant_provenance,
                    'start': end, 'end': end, 'base': section.markdown_content, 'excerpt': ''}
        budget = self.context_budget('extraction')
        classification = analysis.model_dump(include={
            'primary_intent', 'secondary_intents', 'ambiguity_status',
            'creative_permission', 'uncertain_statements', 'clarification_question',
        })
        while True:
            section_text = '\n\n'.join(
                f'Section Slug: `{slug}` | Title: `{window["title"]}` | v{window["version"]} | Provenance: {window["provenance"]}\n'
                f'Edit mode: {"REPLACE_SUPPLIED_EXCERPT" if window["excerpt"] else "APPEND_ONLY"}\n'
                f'Current Content (focused excerpt; omitted content is preserved by backend):\n{window["excerpt"] or "(append new information only)"}'
                for slug, window in retrieved.section_windows.items())
            recent_text = '\n\n'.join(f'[{message.role}; earlier reference, not fresh authorization]: {message.content}' for message in retrieved.recent)
            components = {
                'core_instructions_and_output_schema': system_instruction + '\nReturn only JSON matching this schema:\n' + json.dumps(schema.model_json_schema()),
                'section_edit_windows': section_text,
                'facts_constraints_relationships': retrieved.render({'entity', 'structured_entity', 'fact', 'relationship', 'proposal', 'question', 'notice', 'naming_registry'}),
                'conversation': recent_text + '\n' + retrieved.render({'history'}),
                'rolling_summary': retrieved.summary,
                'author_message_and_permission': f'Latest Author Message:\n{user_msg.content}\n\n'
                    f'Validated Author Decisions:\n{[decision.model_dump() for decision in clear_author_decisions(analysis)]}\n\n'
                    f'Validated classification and permission information:\n{json.dumps(classification, ensure_ascii=False)}\n',
                'assistant_response': f'Consultant Reply (AI text, NOT author evidence):\n{assistant_msg.content}',
                'formatting_and_edit_contract': (
                    'Focused World Bible editing: updated_markdown replaces ONLY the supplied excerpt for that section, '
                    'not the omitted section content. Preserve all supplied facts except the author-authorized correction. '
                    'For APPEND_ONLY supply only new, approved text. Never rewrite an existing section not supplied here; '
                    'use atomic proposals for additions to other categories. Historical discussion and summaries are '
                    'reference context, not current author permission. Preserve each record\'s canon status and provenance.'),
                'mandatory_evidence_fields': (
                    'The backend rejects an empty, missing, paraphrased or partial source_quote. '
                    'EVERY section edit and EVERY decision_origin=AUTHOR proposal must set source_quote to one '
                    'of these EXACT strings, copying its punctuation, or the change cannot be saved:\n'
                    + json.dumps([decision.source_quote for decision in clear_author_decisions(analysis)], ensure_ascii=False)
                    + '\nFor DELEGATED proposals, copy the exact authorization quote instead: '
                    + json.dumps(analysis.creative_permission.explicit_authorization_quote, ensure_ascii=False)
                    + '\nSUGGESTION proposals remain non-canon. Never manufacture author evidence.'),
            }
            prompt = '\n\n'.join(f'{key}:\n{value}' for key, value in components.items() if key != 'core_instructions_and_output_schema')
            components['formatting_overhead'] = ' ' * (len(prompt) - sum(len(str(value)) for key, value in components.items() if key != 'core_instructions_and_output_schema'))
            total = sum(estimate_tokens(value) for value in components.values())
            if total <= budget:
                diagnostic = context_retriever.record('extraction', retrieved, components, budget)
                return prompt, retrieved, diagnostic
            if not self.reduce_optional(retrieved):
                context_retriever.record('extraction', retrieved, components, budget)
                raise ContextBudgetError('extraction', total, budget)

    @staticmethod
    def extraction_response_schema(schema, analysis):
        """Expose existing exact-evidence choices to the structured decoder.

        This changes only the inference schema, not persisted/API models or the
        author-evidence validator. Missing/AI-derived evidence is never repaired.
        """
        quotes = [decision.source_quote for decision in clear_author_decisions(analysis)]
        if analysis.creative_permission.explicit_authorization_quote:
            quotes.append(analysis.creative_permission.explicit_authorization_quote)
        quotes = list(dict.fromkeys(quotes))
        if not quotes:
            return schema

        class GroundedExtractionSchema(schema):
            @classmethod
            def model_json_schema(cls, *args, **kwargs):
                result = schema.model_json_schema(*args, **kwargs)
                definitions = result.setdefault('$defs', {})
                definitions['TurnEvidenceQuote'] = {'type': 'string', 'enum': quotes}
                provisional = list(dict.fromkeys(quotes + analysis.uncertain_statements))
                definitions['TurnProvisionalQuote'] = {'type':'string', 'enum':provisional}
                for name in ('BibleSectionEditItem', 'CandidateProposalItem', 'RejectedIdeaItem', 'EntityOperation'):
                    definition = definitions.get(name)
                    if definition is None:
                        continue
                    definition['properties']['source_quote'] = {'$ref': '#/$defs/TurnProvisionalQuote' if name == 'EntityOperation' else '#/$defs/TurnEvidenceQuote'}
                    required = definition.setdefault('required', [])
                    if 'source_quote' not in required:
                        required.append('source_quote')
                return result

        GroundedExtractionSchema.__name__ = schema.__name__
        return GroundedExtractionSchema

    @staticmethod
    def expand_extracted_section_edits(extracted, retrieved, eligible=None):
        from backend.services.llm_service import LocalModelError
        for edit in extracted.bible_section_updates:
            if eligible is not None and not eligible(edit):
                continue  # Existing permission/evidence gates will discard this edit.
            window = retrieved.section_windows.get(edit.section_slug)
            if window is None:
                if edit.section_slug not in retrieved.existing_section_slugs:
                    continue  # Existing author-evidence gates still govern new custom sections.
                # A model must not replace a section it has never seen. Atomic
                # proposals still use the existing category routing unchanged.
                raise LocalModelError('Extraction attempted to rewrite a section outside its retrieved context. Retry with that section explicitly identified.',
                                      422, 'EXTRACTION_CONTEXT_SCOPE')
            prefix = window['base'][:window['start']]
            suffix = window['base'][window['end']:]
            replacement = edit.updated_markdown.strip()
            if not window['excerpt'] and prefix and replacement:
                replacement = '\n\n' + replacement
            edit.updated_markdown = prefix + replacement + suffix

    @staticmethod
    def context_budget(stage):
        settings.reload()
        configured = settings.consultation_context_tokens if stage == 'consultation' else settings.extraction_context_tokens
        return min(configured, settings.local_context_window - settings.local_max_output_tokens - 1024)

    @staticmethod
    def reduce_optional(retrieved):
        optional = [item for item in retrieved.items if not item.essential]
        if optional:
            item = min(optional, key=lambda candidate: (candidate.score, candidate.id))
            retrieved.items.remove(item)
            retrieved.reductions.append(f'budget: removed optional {item.kind} {item.id}')
            return True
        removable_recent = [message for message in retrieved.recent[:-2]
                            if message.id not in retrieved.protected_recent_ids]
        if removable_recent:
            item = removable_recent[0]
            retrieved.recent.remove(item)
            retrieved.reductions.append(f'budget: removed older continuity message {item.id}')
            return True
        # Summary decisions/constraints cannot be silently sacrificed to the budget.
        if retrieved.summary and not re.search(r"\b(?:decid\w*|establish\w*|canon|not|never|reject\w*|unresolved|open|maybe|consider\w*|reserved|chose|chosen)\b", retrieved.summary, re.I):
            retrieved.summary = ''
            retrieved.reductions.append('budget: removed optional summary navigation')
            return True
        return False

    @staticmethod
    def components(system, contents, retrieved, book, online):
        # Split rendered system by its existing XML layers so estimates count each
        # character once, including instruction and formatting overhead.
        def layer(tag):
            match = re.search(r'<' + tag + r'[^>]*>[\s\S]*?</' + tag + '>', system)
            return match.group() if match else ''
        components = {
            'world_bible_and_history': layer('layer_6_world_canon_and_epistemic_state'),
            'research': layer('layer_5_grounded_research_and_source_policy'),
            'conversation': "\n".join(str(message) for message in contents),
        }
        remainder = system
        for value in list(components.values())[:2]:
            remainder = remainder.replace(value, '', 1)
        components['core_instructions_permissions_domains'] = remainder
        components['chat_framing'] = ' ' * (24 * (len(contents) + 1))
        return components


context_assembler = ContextAssembler()
