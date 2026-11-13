"""Evidence-based turn policy. This module performs no inference or database writes.

The existing model classifies unrestricted natural language. Conservative clause rules
rescue common explicit decisions even when its broad intent label is wrong. Topic
keywords alone never authorize creation. Unknown language fails closed to clarification.
"""
import re
from typing import List

from backend.schemas import AuthorDecisionSchema, CandidateProposalItem, CreativePermissionSchema, TurnAnalysisSchema


def clauses(text: str) -> List[str]:
    sentences = [m.group().strip() for m in re.finditer(r"[^.!?;\n]+[.!?;]*", text) if m.group().strip()]
    return [part.strip(' ,') for sentence in sentences for part in re.split(
        r",?\s+(?:but|and)\s+(?=(?:maybe|perhaps|I wonder|I'm considering|help me)\b)", sentence, flags=re.I,
    ) if part.strip(' ,')]


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'")).strip()


def bare(text: str) -> str:
    return re.sub(r"^(?:actually|well|please|instead)[,:]?\s+", "", normalized(text), flags=re.I).rstrip('.!;')


_EXPLORATION = re.compile(
    r"^(?:maybe\b|perhaps\b|what if\b|suppose\b|if\b|i wonder\b|"
    r"i(?: am|'m) (?:considering|thinking about|not sure)\b|"
    r"i (?:think|haven't|have not) (?:that\b|chosen\b|decided\b)?|could (?!you\b)|would (?!you\b))", re.I,
)
_FACTUAL = re.compile(r"^(?:what (?:does|do|is|are)|where (?:does|do|did)|how (?:do|does|did)|explain|research|look up)\b", re.I)
_NEGATIVE = re.compile(r"^(?:don't|do not|never|stop|avoid|i (?:don't|do not) want|i forbid|you (?:can't|cannot|must not|should not))\b", re.I)
_REQUEST = re.compile(
    r"^(?:give me|suggest|brainstorm|generate|propose|invent|come up with|"
    r"help me (?:think of|come up with|find|name)|"
    r"i(?: just)? need (?:a |some )?(?:quick |fast )?names?\b|"
    r"what are some (?!real\b|historical\b|existing\b).*(?:names? for|options|ideas))\b", re.I,
)
_DELEGATE = re.compile(
    r"^(?:(?:you )?decide\b|choose\b|pick\b|select\b|settle on\b|"
    r"i trust your (?:judgment|judgement|choice)\b|"
    r"(?:you have|i give you) (?:permission|authority) to (?:choose|decide|name)\b)", re.I,
)
_NEGATED_CREATION = re.compile(r"\b(?:don't|do not|no|not)\s+(?:invent|suggest|generate|names?\b)", re.I)
_NON_CREATIVE_REQUEST = re.compile(
    r"^(?:give me|help me find)\s+(?:the |an? )?(?:meaning|origin|etymology|definition|explanation|facts|information|historical|existing|real)\b", re.I,
)
_FICTIONAL_ASSIGNMENT = re.compile(
    r"^(?:planet\s+\S+|(?:the|my|our)\s+(?:capital|empire|kingdom|city|world|government)|"
    r"(?:her|his|their|its)\s+(?:name|birthday|age)|(?:that's|that is|that name)\b)", re.I,
)


def permission_denied(text: str) -> bool:
    return bool(_NEGATIVE.search(text) or _EXPLORATION.search(text) or _NEGATED_CREATION.search(text) or _NON_CREATIVE_REQUEST.search(text))


def request_kind(clause: str) -> str:
    """Recognize an affirmative speech act, never a naked naming-topic match."""
    text = bare(clause)
    text = re.sub(r"^(?:can|could|would|will) you (?:please )?", "", text, flags=re.I)
    if permission_denied(text):
        return "NONE"
    if _DELEGATE.search(text):
        return "DELEGATE"
    if _REQUEST.search(text) and (
        not re.match(r"^(?:give me|suggest|generate|propose|help me find)\b", text, re.I)
        or re.search(r"\b(?:names?|options|ideas|alternatives|possibilities|fictional|invented|original)\b", text, re.I)
    ):
        return "SUGGEST"
    if re.match(r"^(?:write|draft|create|design)\b", text, re.I) and not re.search(r"\b(?:record|these|this|bible|existing|above)\b", text, re.I):
        return "DRAFT"
    return "NONE"


def naming_authorized(quote: str, original: CreativePermissionSchema) -> bool:
    """Distinguish a requested naming task from a naming-related aside."""
    text = re.sub(r"^(?:can|could|would|will) you (?:please )?", "", bare(quote), flags=re.I)
    qualifiers = r"(?:(?:\d+|one|two|three|four|five|six|seven|eight|a|an|the|some|possible|interesting|different|new|another|good|best|quick|fast|major|minor)\s+)*"
    if re.search(rf"\b(?:give me|suggest|generate|propose|think of|come up with|find|need|pick|choose|invent)\s+{qualifiers}names?\b", text, re.I):
        return True
    if re.match(r"^(?:help me )?name\b", text, re.I):
        return True
    if re.search(r"\b(?:should be called|what should .+ be called)\b", text, re.I):
        return True
    if re.match(r"^i trust your (?:judgment|judgement|choice) on .+\bname\b", text, re.I):
        return True
    if re.match(rf"^what are some {qualifiers}names? for\b", text, re.I):
        return True
    # Non-template speech acts use semantic evidence, not a topic-word override.
    return (
        request_kind(quote) == "NONE"
        and original.authorization_kind != "NONE"
        and original.explicit_authorization_quote == quote
        and original.allow_invented_proper_nouns
    )


def is_exploratory(clause: str) -> bool:
    text = re.sub(r"^(?:can|could|would|will) you (?:please )?", "", bare(clause), flags=re.I)
    return bool(_EXPLORATION.search(text) or re.search(
        r"\b(?:might|may|would|could|maybe|perhaps|possibly|potentially|either|I guess|I suppose)\b", text, re.I,
    )) and request_kind(clause) == "NONE"


def factual_question(clause: str) -> bool:
    text = re.sub(r"^(?:can|could|would|will) you (?:please )?", "", bare(clause), flags=re.I)
    return bool(_FACTUAL.search(text) or _NON_CREATIVE_REQUEST.search(text)) and request_kind(clause) == "NONE"


def clear_decision(clause: str):
    """Narrow grammatical fallback; semantic model evidence handles other phrasing."""
    text = bare(clause)
    if is_exploratory(clause) or factual_question(clause) or request_kind(clause) != "NONE" or clause.endswith('?'):
        return None
    if _NEGATIVE.search(text):
        # Not knowing something is not a prohibition; only explicit prohibitions qualify.
        return AuthorDecisionSchema(
            kind="PROHIBITION", source_quote=clause,
            record_in_bible=not bool(re.search(
                r"\b(?:invent|inventing)\b|^(?:don't|do not) name it\b", text, re.I,
            )),
        )
    if re.match(r"^(?:that's|that is) not what happened\b", text, re.I):
        return AuthorDecisionSchema(kind="CORRECTION", source_quote=clause, requires_clarification=True)
    if re.match(r"^the name is wrong\b", text, re.I):
        return AuthorDecisionSchema(kind="CORRECTION", source_quote=clause, requires_clarification=True)
    if re.match(r"^(?:change|rename|record|add|update|organize|replace)\b", text, re.I):
        return AuthorDecisionSchema(
            kind="CORRECTION", source_quote=clause,
            requires_clarification=bool(re.match(r"^(?:change|rename)\b", text, re.I) and not re.search(r"\bto\b", text, re.I)),
        )
    if re.search(r"\b(?:let's|let us) (?:use|make|call|keep|set|go with)\b", text, re.I):
        return AuthorDecisionSchema(
            kind="CORRECTION", source_quote=clause,
            requires_clarification=bool(re.search(r"\b(?:older|younger|bigger|smaller)\b", text, re.I)),
        )
    if re.match(r"^i (?:want|prefer|have decided|decided|choose)\b", text, re.I):
        # Vague wants are not decisions. A concrete assignment or explicit adoption is.
        if re.search(r"\b(?:to be (?:called|named)|let's use|let us use|go with)\b", text, re.I):
            return AuthorDecisionSchema(kind="DECLARATION", source_quote=clause)
        return None
    if re.match(r"^(?:i|we|you)\b", text, re.I) and not re.match(r"^my world\b", text, re.I):
        return None
    if re.search(r"\b(?:real-world|in reality|etymology|means|derived from|comes from)\b", text, re.I):
        return None
    if (
        _FICTIONAL_ASSIGNMENT.search(text)
        or re.match(r"^brain dump:", text, re.I)
        or re.search(r"\bbelongs to Planet\s+\S+", text, re.I)
    ) and re.search(r"\b(?:is|are|was|were|has|have|belongs to|isn't|is not|that's|that is)\b", text, re.I):
        return AuthorDecisionSchema(
            kind="CORRECTION" if re.search(r"\b(?:not|actually|belongs to)\b", clause, re.I) else "DECLARATION",
            source_quote=clause,
            requires_clarification=bool(re.search(r"\b(?:another|some other|have in mind)\b", text, re.I)),
        )
    return None


def latest_decisions(decisions: List[AuthorDecisionSchema]) -> List[AuthorDecisionSchema]:
    """Resolve direct naming reversals without conflating unrelated entity properties."""
    operations = []
    for decision in decisions:
        text = bare(decision.source_quote).rstrip('.')
        match = re.fullmatch(r"(Planet \S+)(?:'s name is| is (?:named|called)) ([\w'’-]+(?: [\w'’-]+)*)", text, re.I)
        if match and re.search(r"\b(?:and|because|which)\b", match[2], re.I):
            match = None
        if not match:
            match = re.fullmatch(r"((?i:Planet) \S+) (?i:is) ([A-Z][\w'’-]*(?: [A-Z][\w'’-]*)*)", text)
        negative = False
        if not match:
            match = re.fullmatch(r"(Planet \S+) is not (?:(?:named|called) )?([\w'’-]+)", text, re.I)
            negative = bool(match)
        if not match:
            match = re.fullmatch(r"(?:don't|do not) use ([\w'’-]+) for (Planet \S+)", text, re.I)
            if match:
                value, target = match.groups()
                operations.append((target.lower(), value.lower(), True))
                continue
        operations.append((match[1].lower(), match[2].lower(), negative) if match else None)
    kept = []
    for index, (decision, operation) in enumerate(zip(decisions, operations)):
        superseded = False
        if operation:
            target, value, negative = operation
            for later in operations[index + 1:]:
                if later and later[0] == target:
                    # A newer assignment replaces the old one. A prohibition only
                    # supersedes an assignment to the prohibited value, not other facts.
                    if (not negative and (not later[2] or later[1] == value)) or (negative and not later[2] and later[1] == value):
                        superseded = True
        if not superseded:
            kept.append(decision)
    return kept


def reconcile_turn(message: str, analysis: TurnAnalysisSchema, reference_context: str = "", *, classifier_failed: bool = False) -> TurnAnalysisSchema:
    """Independently reconcile decisions, authorization, and uncertainty.

    No prior-turn permission is inherited. All evidence must be complete clauses of
    this message; a quotation of a naming word is insufficient evidence.
    """
    analysis = analysis.model_copy(deep=True)
    original_permission = analysis.creative_permission.model_copy(deep=True)
    parts = clauses(message)
    decisions = []
    uncertain = []
    grants = []
    for part in parts:
        if is_exploratory(part):
            uncertain.append(part)
            continue
        kind = request_kind(part)
        if kind != "NONE":
            grants.append((part, kind))
            continue
        known = clear_decision(part)
        if known:
            decisions.append(known)
        elif (
            original_permission.authorization_kind != "NONE"
            and original_permission.explicit_authorization_quote == part
            and original_permission.allowed_targets
            and not factual_question(part)
            and not permission_denied(bare(part))
            and not part.startswith(('"', "'", '“', '‘'))
            and not re.search(r"\b(?:said|says|wrote|asked)\s*[:\"']", part, re.I)
            and not re.match(r"^(?:dialogue|example|quotation|quote)\s*:", part, re.I)
        ):
            # Semantic classification handles speech acts outside the fallback grammar.
            # Require a full author clause, not a selected keyword or reported speech.
            grants.append((part, original_permission.authorization_kind))
        elif not factual_question(part) and not part.endswith('?'):
            # Accept semantic decisions only when the model provides full, grounded evidence.
            decisions.extend(d for d in analysis.author_decisions if d.source_quote == part)
    # The model may mark a grammatically declarative clause as genuinely ambiguous.
    marked_uncertain = set(analysis.uncertain_statements)
    decisions = [d for d in decisions if d.source_quote not in marked_uncertain or clear_decision(d.source_quote)]
    # Model uncertainty can narrow a decision's application without discarding other clauses.
    for decision in decisions:
        for evidence in analysis.author_decisions:
            if evidence.source_quote == decision.source_quote and evidence.requires_clarification:
                decision.requires_clarification = True
    analysis.author_decisions = latest_decisions(list({d.source_quote: d for d in decisions}.values()))
    analysis.uncertain_statements = list(dict.fromkeys(uncertain + [q for q in marked_uncertain if q in parts and not clear_decision(q)]))

    perm = analysis.creative_permission
    original = perm.model_copy(deep=True)
    # Derive permission only from affirmative, scoped author speech acts. Later
    # prohibitions override an earlier grant for overlapping tasks, conservatively.
    if grants:
        quote, kind = grants[-1]
        later = parts[parts.index(quote) + 1:]
        if any(_NEGATIVE.search(bare(p)) for p in later):
            grants = []
    if not grants:
        analysis.creative_permission = CreativePermissionSchema(
            level="ANALYTICAL_ADVISORY" if analysis.ambiguity_status == "NEEDS_CLARIFICATION" else "IMPLEMENTATION_SCOPED",
            prohibited_adjacent_targets=original.prohibited_adjacent_targets,
        )
    else:
        quote, kind = grants[-1]
        perm.authorization_kind = kind
        perm.explicit_authorization_quote = quote
        # Never retain model-supplied, broader targets not grounded in this task.
        perm.allowed_targets = [quote]
        perm.resolved_target = ""
        target = original.resolved_target
        if target and normalized(target).lower() in normalized(quote + " " + reference_context).lower():
            perm.resolved_target = target
        elif match := re.search(r"\bPlanet\s+[A-Za-z0-9_-]+\b", quote, re.I):
            perm.resolved_target = match.group()
        perm.level = "EXPLICIT_DRAFTING" if kind == "DRAFT" else "BOUNDED_BRAINSTORMING"
        perm.allow_invented_proper_nouns = naming_authorized(quote, original)
        perm.naming_mode = original.naming_mode if perm.allow_invented_proper_nouns else "NONE"
        if perm.allow_invented_proper_nouns and perm.naming_mode not in {"MAJOR_CONTEXTUAL", "MINOR_QUICK"}:
            perm.naming_mode = "MAJOR_CONTEXTUAL"
        if kind == "DELEGATE":
            perm.requested_option_count = 1
        elif count := re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight)\b", quote, re.I):
            word = count.group().lower()
            perm.requested_option_count = int(word) if word.isdigit() else ["one", "two", "three", "four", "five", "six", "seven", "eight"].index(word) + 1
    eligible = clear_author_decisions(analysis)
    if eligible:
        if "CANON_DECLARATION" not in analysis.secondary_intents:
            analysis.secondary_intents.append("CANON_DECLARATION")
        if not any(factual_question(p) for p in parts):
            analysis.primary_intent = "CANON_DECLARATION"
    elif grants:
        analysis.primary_intent = "CREATIVE_GENERATION"
    elif analysis.uncertain_statements:
        analysis.primary_intent = "HYPOTHETICAL_EXPLORATION"
    elif any(d.kind == "PROHIBITION" for d in analysis.author_decisions):
        analysis.primary_intent = "SUGGESTION_DISPOSITION"
    elif all(factual_question(p) or p.endswith('?') for p in parts):
        analysis.primary_intent = "FACTUAL_RESEARCH"
    elif not eligible:
        analysis.ambiguity_status = "NEEDS_CLARIFICATION"
    if any(d.requires_clarification for d in analysis.author_decisions):
        analysis.ambiguity_status = "NEEDS_CLARIFICATION"
    elif (classifier_failed or analysis.ambiguity_status != "NEEDS_CLARIFICATION") and (
        (eligible and not analysis.uncertain_statements) or (grants and analysis.creative_permission.resolved_target)
    ):
        analysis.ambiguity_status = "CLEAR"
        analysis.clarification_question = ""
    if analysis.creative_permission.authorization_kind == "DELEGATE" and not analysis.creative_permission.resolved_target:
        analysis.ambiguity_status = "NEEDS_CLARIFICATION"
    if analysis.ambiguity_status == "NEEDS_CLARIFICATION" and not analysis.clarification_question:
        analysis.clarification_question = "What would you like to establish or change, and which entity does it apply to?"
    if not grants and analysis.primary_intent == "CREATIVE_GENERATION":
        analysis.primary_intent = "IMPLEMENTATION_ASSISTANCE"
    if not eligible:
        analysis.secondary_intents = [i for i in analysis.secondary_intents if i not in {"CANON_DECLARATION", "WORLD_BIBLE_MANAGEMENT"}]
    return analysis


def clear_author_decisions(analysis: TurnAnalysisSchema) -> List[AuthorDecisionSchema]:
    return [d for d in analysis.author_decisions if not d.requires_clarification and d.record_in_bible]


def extraction_eligible(analysis: TurnAnalysisSchema) -> bool:
    return (
        bool(clear_author_decisions(analysis))
        or analysis.creative_permission.authorization_kind in {"SUGGEST", "DRAFT"}
        or (analysis.creative_permission.authorization_kind == "DELEGATE"
            and bool(analysis.creative_permission.resolved_target)
            and analysis.ambiguity_status != "NEEDS_CLARIFICATION")
    )


def author_evidence(quote: str, analysis: TurnAnalysisSchema) -> bool:
    return bool(quote) and any(quote == d.source_quote for d in clear_author_decisions(analysis))


def naming_fact_supported(item, author_message: str, reference_context: str = "") -> bool:
    """Reject invented replacements disguised as author-authored naming facts.

    This is a permission check for explicit naming assertions, not a general-purpose
    semantic validator for arbitrary World Bible prose.
    """
    text = normalized(item.proposed_content).strip().rstrip('.')
    binding = re.fullmatch(r"(.+?)(?:'s name is| is (?:named|called)) ([\w'’-]+(?: [\w'’-]+)*)", text, re.I)
    if not binding:
        binding = re.fullmatch(r"(Planet \S+) is ([A-Z][\w'’-]*(?: [A-Z][\w'’-]*)*)", text)
    if not binding:
        return not bool(re.search(r"\bis (?:named|called)\b", text, re.I))
    target, value = binding.groups()
    targets = re.findall(r"\bPlanet\s+[\w-]+\b", author_message, re.I)
    if targets and re.match(r"^Planet\b", target, re.I) and normalized(target).lower() not in {t.lower() for t in targets}:
        return False
    evidence = author_message
    if re.search(r"\b(?:let's use|let us use|go with|I choose|I pick)\b", item.source_quote, re.I):
        evidence += " " + reference_context
    if not re.search(rf"(?<!\w){re.escape(value)}(?!\w)", normalized(evidence), re.I):
        return False
    # A negative assignment is never permission to establish that same assignment.
    quote = normalized(item.source_quote)
    if re.search(rf"{re.escape(target)}\s+is\s+not\s+(?:(?:named|called)\s+)?{re.escape(value)}\b", quote, re.I):
        return False
    if re.search(rf"(?:don't|do not|can't|cannot)\s+use\s+{re.escape(value)}\s+for\s+{re.escape(target)}\b", quote, re.I):
        return False
    return True


def section_naming_supported(edit, existing_markdown: str, author_message: str, reference_context: str) -> bool:
    """Apply the same naming permission gate to newly written Markdown assertions."""
    existing = normalized(existing_markdown).lower()
    for line in re.split(r"[.!?](?:\s+|$)|\n", edit.updated_markdown):
        statement = line.strip().lstrip('-* ').strip()
        if not re.match(r"^Planet\s+\S+\s+(?:is|'s name is)\b", statement, re.I):
            continue
        if normalized(statement).lower() in existing:
            continue
        item = CandidateProposalItem(
            proposed_title="Section naming assertion", proposed_content=statement, source_quote=edit.source_quote,
        )
        if not naming_fact_supported(item, author_message, reference_context):
            return False
    return True


def suggestion_in_scope(item, analysis: TurnAnalysisSchema) -> bool:
    perm = analysis.creative_permission
    if perm.authorization_kind not in {"SUGGEST", "DRAFT"}:
        return False
    targets = re.findall(r"\bPlanet\s+[\w-]+\b", " ".join(perm.allowed_targets), re.I)
    proposed = re.findall(r"\bPlanet\s+[\w-]+\b", item.proposed_content, re.I)
    if targets and not {t.lower() for t in proposed}.issubset({t.lower() for t in targets}):
        return False
    if not perm.allow_invented_proper_nouns and re.search(r"\bis (?:named|called)\b", item.proposed_content, re.I):
        return False
    return True


def delegated_evidence(item, analysis: TurnAnalysisSchema, assistant_text: str, reference_context: str = "") -> bool:
    perm = analysis.creative_permission
    if perm.authorization_kind != "DELEGATE" or analysis.ambiguity_status == "NEEDS_CLARIFICATION" or item.source_quote != perm.explicit_authorization_quote:
        return False
    target = normalized(item.decision_target).lower()
    if not target or target != normalized(perm.resolved_target).lower():
        return False
    text = normalized(item.proposed_content).rstrip('.')
    if target not in text.lower() or text.lower() not in normalized(assistant_text).lower():
        return False
    if perm.allow_invented_proper_nouns or re.search(r"\b(?:named|called|name is)\b", text, re.I):
        # A delegated name must be a single name assignment, not additional lore.
        pattern = rf"(?:the )?{re.escape(normalized(item.decision_target))}(?:'s name is| is (?:named|called)) ([\w'’-]+(?: [\w'’-]+)*)"
        match = re.fullmatch(pattern, text, re.I)
        return (
            bool(match)
            and not re.search(r"\b(?:and|because|with|which|who|has|is)\b", match.group(1), re.I)
            and (perm.allow_invented_proper_nouns or match.group(1).lower() in normalized(reference_context).lower())
        )
    return True
