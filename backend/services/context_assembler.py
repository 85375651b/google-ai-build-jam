"""Dynamic AI Context Assembler implementing the 8-Layer Modular Prompt Architecture."""

import re
from typing import Any, Dict, List, Optional
from google.genai import types
from sqlalchemy.orm import Session
from backend.models import (
    AuthorExpertiseProfile,
    BibleSection,
    Conversation,
    Fact,
    Message,
    Subject,
    UnresolvedQuestion,
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

COMMON_NON_PROPER_WORDS = {
    "The", "And", "For", "With", "From", "Into", "Under", "Over", "Between", "Through",
    "During", "Before", "After", "Above", "Below", "When", "Where", "While", "Because",
    "Although", "However", "Therefore", "Furthermore", "Moreover", "Section", "World",
    "Bible", "History", "Chronology", "Politics", "Governance", "Economics", "Trade",
    "Geography", "Geology", "Climate", "Ecology", "Biology", "Evolution", "Languages",
    "Linguistics", "Cultures", "Societies", "Religion", "Mythology", "Technology",
    "Engineering", "Warfare", "Military", "Philosophy", "Ideologies", "Demographics",
    "Population", "Architecture", "Material", "Culture", "Cosmology", "Fundamental",
    "Physics", "Only", "Every", "All", "Some", "Many", "Most", "Other", "Such",
    "This", "That", "These", "Those", "Their", "There", "Here", "They", "Them",
}


class ContextAssembler:
    """Constructs token-budgeted, epistemologically segregated prompts for Gemini."""

    def assemble_consultant_system_prompt(
        self,
        db: Session,
        world: World,
        turn_analysis: TurnAnalysisSchema,
        book_retrieval: Dict[str, Any],
        online_research: Dict[str, Any],
        focused_subject_id: Optional[str] = None,
        conversation: Optional[Conversation] = None,
    ) -> str:
        """Build the 8-layer modular system instruction string for Pass 2."""
        perm = turn_analysis.creative_permission
        allow_proper_nouns = (
            perm.allow_invented_proper_nouns or world.allow_unrequested_proper_nouns
        )
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
            f'section="{p["section_title"]}" book_page="{p["book_page_start"]}">\n'
            f'  {p["content"]}\n  </passage>'
            for p in book_passages
        ]
        book_xml_str = (
            "\n".join(book_xml_blocks)
            if book_xml_blocks
            else "  <!-- No relevant passages in primary reference book. -->"
        )

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
1. Primary Real-World Reference: Mark Rosenfelder's "The Planet Construction Kit". Cite inline using [{book_passages[0]["citation_id"] if book_passages else "Book-pX"}].
2. Supplementary Online Research: Cite inline using [Web-1], etc.
3. Real-world sources explain reality; they NEVER override the author's fictional canon.

<real_world_reference_book coverage_status="{book_status}">
{book_xml_str}
</real_world_reference_book>

<supplementary_online_research status="{online_research.get('status', 'SKIPPED')}">
{web_xml_str}
</supplementary_online_research>
</layer_5_grounded_research_and_source_policy>"""

        canon_text = self._select_relevant_canon(db, world.id)
        unresolved_text = self._select_unresolved_and_hypotheticals(db, world.id)
        rejected_text = self._select_rejected_ideas(db, world.id)

        layer_6 = f"""<layer_6_world_canon_and_epistemic_state>
World Name: {world.name}
World Description: {world.description or "Not yet specified."}

<established_world_canon rule="TIER_1_IMMUTABLE_FICTIONAL_TRUTH">
{canon_text}
</established_world_canon>

<unresolved_and_hypothetical_context rule="NON_CANON_DO_NOT_TREAT_AS_ESTABLISHED">
{unresolved_text}
</unresolved_and_hypothetical_context>

<rejected_and_prohibited_ideas rule="EXPLICITLY_REJECTED_NEVER_REINTRODUCE">
{rejected_text}
</rejected_and_prohibited_ideas>
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
        if conversation and (conversation.rolling_summary or "").strip():
            layer_8 = (
                "<layer_8_rolling_conversation_summary>\n"
                "Condensed summary of earlier turns in this conversation thread (older than the 12-turn active window):\n"
                f"{conversation.rolling_summary.strip()}\n"
                "</layer_8_rolling_conversation_summary>"
            )
            layers.append(layer_8)

        naming_mode = getattr(perm, "naming_mode", "NONE")
        if allow_proper_nouns or naming_mode in {"MAJOR_CONTEXTUAL", "MINOR_QUICK"}:
            mode_key = (
                naming_mode
                if naming_mode in {"MAJOR_CONTEXTUAL", "MINOR_QUICK"}
                else "MAJOR_CONTEXTUAL"
            )
            naming_instr = NAMING_INTELLIGENCE_INSTRUCTIONS[mode_key]
            registry_text = self._build_existing_names_and_linguistics_registry(
                db, world.id
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
        self,
        db: Session,
        conversation: Conversation,
        latest_user_message: str,
        max_recent_messages: int = 12,
    ) -> List[types.Content]:
        past_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(max_recent_messages)
            .all()
        )
        past_messages.reverse()
        contents: List[types.Content] = []
        if (conversation.rolling_summary or "").strip():
            summary_block = (
                "[Rolling Summary of Earlier Conversation Turns]\n"
                f"{conversation.rolling_summary.strip()}"
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=summary_block)],
                )
            )
        for msg in past_messages:
            if (
                msg == past_messages[-1]
                and msg.role == "USER"
                and msg.content.strip() == latest_user_message.strip()
            ):
                continue
            gemini_role = "model" if msg.role == "ASSISTANT" else "user"
            hist_text = (
                msg.content[:15000] + "\n...[earlier brain dump summarized in World Bible]...\n" + msg.content[-5000:]
                if len(msg.content) > 22000
                else msg.content
            )
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=hist_text)],
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=latest_user_message)],
            )
        )
        return contents

    def _select_relevant_canon(
        self, db: Session, world_id: str, max_facts: int = 25
    ) -> str:
        facts = (
            db.query(Fact)
            .filter(Fact.world_id == world_id, Fact.canon_status == "ESTABLISHED_CANON")
            .order_by(Fact.updated_at.desc())
            .limit(max_facts)
            .all()
        )
        subjects = {
            s.id: s.name
            for s in db.query(Subject).filter(Subject.world_id == world_id).all()
        }
        lines = [
            f"- [Fact #{f.id[:8]} | Provenance: {f.provenance_type} | Consistency: {f.consistency_status} | Subject: "
            f"{', '.join(subjects.get(l.subject_id, 'General') for l in f.subject_links) or 'General'}] "
            f"{f.title}: {f.content}"
            + (f" (Note: {f.consistency_notes})" if f.consistency_notes else "")
            for f in facts
        ]
        sections = (
            db.query(BibleSection)
            .filter(BibleSection.world_id == world_id)
            .order_by(BibleSection.display_order.asc())
            .all()
        )
        sec_blocks = [
            f"### World Bible Section: {s.title} (v{s.version_number} | Provenance: {s.dominant_provenance})\n{s.markdown_content.strip()[:1500]}"
            for s in sections
            if (s.markdown_content or "").strip()
        ]
        combined = ""
        if lines:
            combined += "Established Atomic Facts:\n" + "\n".join(lines) + "\n\n"
        if sec_blocks:
            combined += "Current World Bible Sections:\n" + "\n\n".join(sec_blocks)
        return combined.strip() or "No fictional world canon has been established yet."

    def _select_unresolved_and_hypotheticals(
        self, db: Session, world_id: str
    ) -> str:
        hypotheticals = (
            db.query(Fact)
            .filter(
                Fact.world_id == world_id,
                Fact.canon_status.in_(["UNRESOLVED_POSSIBILITY", "HYPOTHETICAL"]),
            )
            .limit(10)
            .all()
        )
        open_qs = (
            db.query(UnresolvedQuestion)
            .filter(
                UnresolvedQuestion.world_id == world_id,
                UnresolvedQuestion.status == "OPEN",
            )
            .limit(10)
            .all()
        )
        lines = [f"- [{h.canon_status}] {h.title}: {h.content}" for h in hypotheticals]
        lines += [f"- [OPEN QUESTION] {q.question_text}" for q in open_qs]
        return "\n".join(lines) if lines else "None."

    def _select_rejected_ideas(self, db: Session, world_id: str) -> str:
        rejected = (
            db.query(Fact)
            .filter(Fact.world_id == world_id, Fact.canon_status == "REJECTED")
            .limit(15)
            .all()
        )
        return (
            "\n".join(f"- [REJECTED] {r.title}: {r.content}" for r in rejected)
            if rejected
            else "None."
        )

    def _build_existing_names_and_linguistics_registry(
        self, db: Session, world_id: str
    ) -> str:
        """Extract established linguistic conventions and an anti-collision registry of existing proper nouns."""
        sections = (
            db.query(BibleSection)
            .filter(BibleSection.world_id == world_id)
            .order_by(BibleSection.display_order.asc())
            .all()
        )
        facts = (
            db.query(Fact)
            .filter(Fact.world_id == world_id, Fact.canon_status == "ESTABLISHED_CANON")
            .all()
        )
        rejected = (
            db.query(Fact)
            .filter(Fact.world_id == world_id, Fact.canon_status == "REJECTED")
            .all()
        )

        ling_notes: List[str] = []
        proper_nouns: List[str] = []
        seen_lower = set()

        for sec in sections:
            md = (sec.markdown_content or "").strip()
            if not md:
                continue
            if sec.slug in {"languages-linguistics", "cultures-societies"}:
                ling_notes.append(f"[{sec.title}]: {md[:800]}")
            for tok in re.findall(r"\b[A-Z][a-zA-Z'-]{2,24}\b", md):
                if tok not in COMMON_NON_PROPER_WORDS and tok.lower() not in seen_lower:
                    seen_lower.add(tok.lower())
                    proper_nouns.append(tok)

        for f in facts:
            for tok in re.findall(r"\b[A-Z][a-zA-Z'-]{2,24}\b", f"{f.title} {f.content}"):
                if tok not in COMMON_NON_PROPER_WORDS and tok.lower() not in seen_lower:
                    seen_lower.add(tok.lower())
                    proper_nouns.append(tok)

        rejected_names = [f"{r.title} ({r.content[:100]})" for r in rejected]

        ling_block = (
            "\n".join(ling_notes)
            if ling_notes
            else "No explicit phonological or linguistic rules recorded yet; infer from cultural/geographic canon."
        )
        names_block = (
            ", ".join(proper_nouns[:60])
            if proper_nouns
            else "None established yet."
        )
        rej_block = (
            "; ".join(rejected_names[:20])
            if rejected_names
            else "None."
        )

        return (
            f"Established Linguistic & Cultural Naming Context:\n{ling_block}\n\n"
            f"Existing Canon Proper Nouns Registry (Maintain cultural family resemblance within the same culture, "
            f"keep separate cultures phonetically distinct, and NEVER duplicate these existing names for new entities):\n"
            f"{names_block}\n\n"
            f"Rejected Names / Prohibited Patterns (NEVER re-suggest):\n{rej_block}"
        )


context_assembler = ContextAssembler()
