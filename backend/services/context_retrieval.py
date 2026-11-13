"""Read-only, lexical and relationship-aware context selection over existing data.

No permissions or canon decisions are inferred here. Topic detection affects retrieval
only. Essential records are never character-truncated or silently budget-dropped.
"""
from collections import Counter
from dataclasses import dataclass, field
import json
import logging
import math
import re

from sqlalchemy.orm import selectinload

from backend.config import settings
from backend.prompts.domain_cartridges import DEFAULT_WORLD_SUBJECTS
from backend.models import (BibleSection, CanonProposal, Conversation, Fact, FactRelationship,
                            Message, ReferenceChunk, ResearchSource, Subject, UnresolvedQuestion)
from backend.services.generation_control import current_generation
from backend.services.llm_service import LocalModelError

logger = logging.getLogger(__name__)
# Uvicorn's default logging config does not install a handler for application
# INFO records. Give these metadata-only diagnostics an explicit local handler.
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s %(name)s %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
STOP = set("a an the and or but of to in on at is are was were be been being for from by with as it its this that these those i me my we our you your he she her his they their them what which who how why can could would should do does did not no don't can't have has had about help think through please actually called call name names naming decide use let want make give some possible interesting question remember decided discussing discussion".split())
GENERIC = set("planet planets character characters kingdom republic country city system star capital mother father world bible section history geography politics governance languages linguistics cultures societies cosmology fundamental physics climate ecology biology evolution economics trade religion mythology technology engineering warfare military philosophy ideologies demographics population architecture material culture".split())
GENERIC.update(word.casefold() for subject in DEFAULT_WORLD_SUBJECTS for word in re.findall(r'\w+', subject['name']))
CATEGORY_NAMES = {subject['name'].casefold() for subject in DEFAULT_WORLD_SUBJECTS}
NEGATIVE = re.compile(r"\b(?:not|never|cannot|can't|don't|mustn't|reserved?|reservation|prohibit\w*|reject\w*|ruled out|must remain|must stay|unchange\w*|off limits|belongs to)\b", re.I)
TOPICS = {
    'naming': set('name names naming named alias aliases reservation reserved phonology phonological language linguistic linguistics culture cultural surname mother chose choose called'.split()),
    'political': set('political politics kingdom republic government treaty treaties alliance alliances conflict war leader leaders trade economy economic geography coast coastal northern institution institutions'.split()),
    'character': set('character biography personality motivation motivations mother father relationship relationships backstory birthday age older family daughter son'.split()),
    'planetary': set('planet planets star stars moon moons orbit orbital system solar radiation luminosity atmosphere climate weather gravity mass temperature heat light'.split()),
}


def tokens(value):
    return set(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", (value or '').casefold(), re.UNICODE)) - STOP


def contains(text, phrase):
    return bool(phrase and re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text, re.I))


def estimate_tokens(value):
    """Conservative UTF-8 estimate, not a model-specific tokenizer."""
    return math.ceil(len(str(value).encode('utf-8')) / 3)


def normalized(value):
    return re.sub(r'\W+', ' ', value.casefold()).strip()


def markdown_blocks(value):
    """Whole Markdown blocks with offsets and their enclosing heading path."""
    headings = []
    blocks = []
    for match in re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)', value or ''):
        offset = match.start()
        body_start = offset
        body = ''
        def flush():
            if body.strip():
                start = body_start + len(body) - len(body.lstrip())
                text = body.strip()
                blocks.append((start, start + len(text), ' / '.join(h for _, h in headings), text))
        for line in match.group().splitlines(keepends=True):
            header = re.match(r'^\s{0,3}(#{1,6})\s+(.+)', line)
            if header:
                flush()
                body = ''
                level = len(header[1])
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, line.strip()))
                blocks.append((offset, offset + len(line.rstrip()), ' / '.join(h for _, h in headings), line.rstrip()))
            else:
                if not body:
                    body_start = offset
                body += line
            offset += len(line)
        flush()
    return blocks


def contextual_indices(blocks, indices):
    """Preserve adjacent qualifications/negations without loading unrelated prose."""
    selected = set(indices)
    for index in list(selected):
        for following in range(index + 1, min(index + 3, len(blocks))):
            text = blocks[following][3]
            if text.startswith('#') or not re.match(r'^(?:However|But\b|Except|Unless|Only\b|Instead|This\b|That\b|These\b|Those\b|It\b|Its\b|They\b|Their\b|Neither|Nor\b|Despite|Otherwise)', text, re.I):
                break
            selected.add(following)
        if index and re.search(r'\b(?:hypothetical|unconfirmed|not established|rejected ideas|not approved|only a possibility)\b', blocks[index - 1][3], re.I):
            selected.add(index - 1)
    return sorted(selected)


class ContextBudgetError(LocalModelError):
    def __init__(self, stage, needed, budget):
        super().__init__(f'The relevant {stage} context needs approximately {needed} input tokens, exceeding the safe budget of {budget}. Narrow the request or increase its context budget within the model window. Required canon and constraints were not discarded.',
                         422, 'CONTEXT_BUDGET_EXCEEDED')


@dataclass
class ContextItem:
    kind: str
    id: str
    text: str
    score: float
    reason: str
    essential: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class RetrievedContext:
    query: str
    category: str
    entities: set = field(default_factory=set)
    related_entities: set = field(default_factory=set)
    items: list = field(default_factory=list)
    recent: list = field(default_factory=list)
    protected_recent_ids: set = field(default_factory=set)
    summary: str = ''
    section_windows: dict = field(default_factory=dict)
    section_versions: dict = field(default_factory=dict)
    existing_section_slugs: set = field(default_factory=set)
    reductions: list = field(default_factory=list)

    def render(self, kinds):
        return '\n\n'.join(item.text for item in self.items if item.kind in kinds) or 'None relevant retrieved.'


class ContextRetriever:
    def category(self, query, analysis=None):
        terms = tokens(query)
        if re.search(r'\b(?:name\w*|alias\w*|reserv\w*|surname\w*|called)\b', query, re.I):
            return 'naming'
        if terms & (TOPICS['political'] - {'geography', 'coast', 'coastal', 'northern'}):
            return 'political'
        if terms & TOPICS['character']:
            return 'character'
        if terms & TOPICS['planetary']:
            return 'planetary'
        if analysis and analysis.primary_intent == 'FACTUAL_RESEARCH':
            return 'factual_research'
        return 'canon_correction' if NEGATIVE.search(query) or (analysis and analysis.author_decisions) else 'worldbuilding'

    @staticmethod
    def _catalog(subjects, texts):
        names = {s.name for s in subjects if s.name.casefold() not in CATEGORY_NAMES and tokens(s.name) - GENERIC}
        for text in texts:
            for name in re.findall(r"\b(?:Planet|Star|Moon|Kingdom|Republic|Empire|Queen|King|House)\s+[A-Z][\w'’\-]*", text, re.I):
                if name.split()[-1].casefold() not in STOP:
                    names.add(name)
            names.update(re.findall(r'\b(?:northern|southern|eastern|western|coastal|central|upper|lower)\s+(?:kingdom|republic|empire|country|city|province|islands?)\b', text, re.I))
            for name in re.findall(r"\b[A-Z][\w'’\-]*(?:\s+(?:of\s+|the\s+)?[A-Z][\w'’\-]*){0,2}", text):
                if tokens(name) - GENERIC and not (tokens(name) <= STOP):
                    names.add(name)
        # Case variants must not eliminate each other as supposed substrings.
        unique = {name.casefold(): name for name in sorted(names, reverse=True)}
        unique.update({s.name.casefold(): s.name for s in subjects if tokens(s.name) - GENERIC})
        return sorted(unique.values(), key=lambda name: (-len(name), name))

    def retrieve(self, db, world_id, query, analysis=None, conversation=None,
                 focused_subject_id=None, exclude_message_ids=(), purpose='consultation'):
        settings.reload()
        result = RetrievedContext(query=query, category=self.category(query, analysis))
        from backend.services.entity_service import entity_service
        for eid, state, direct in entity_service.context(db, world_id, query, naming=result.category == 'naming'):
            result.items.append(ContextItem('structured_entity', eid, '[Authoritative structured entity; preserve each record status]\n' + state,
                120, 'direct structured identity' if direct else 'bounded entity relationship/name constraint', True))
        subjects = db.query(Subject).filter(Subject.world_id == world_id).all()
        subject_by_id = {s.id: s for s in subjects}
        facts = db.query(Fact).options(selectinload(Fact.subject_links), selectinload(Fact.section_links),
                                      selectinload(Fact.citations)).filter(Fact.world_id == world_id).all()
        sections = db.query(BibleSection).filter(BibleSection.world_id == world_id).order_by(BibleSection.display_order, BibleSection.id).all()
        result.section_versions = {s.id: s.version_number for s in sections}
        result.existing_section_slugs = {s.slug for s in sections}
        excluded = set(exclude_message_ids)
        recent = []
        if conversation:
            recent = db.query(Message).filter(Message.conversation_id == conversation.id,
                       Message.id.notin_(excluded)).order_by(Message.created_at.desc(), Message.id.desc()).limit(settings.context_recent_messages).all()[::-1]
            result.recent = recent
        corpus = [f'{f.title} {f.content}' for f in facts] + [s.markdown_content for s in sections]
        names = self._catalog(subjects, corpus)
        direct = {name for name in names if contains(query, name)}
        focus = subject_by_id.get(focused_subject_id or (conversation.focused_subject_id if conversation else None))
        if focus and tokens(focus.name) - GENERIC:
            direct.add(focus.name)
        if not direct and recent and re.search(r"\b(?:it|its|her|his|their|that|those|this|earlier|remember)\b", query, re.I):
            direct = {name for name in names if any(contains(m.content, name) for m in recent[-2:])}
        # Suppress partial aliases already covered by a complete entity phrase.
        direct = {n for n in direct if not any(n != longer and contains(longer, n) for longer in direct)}
        result.entities = direct
        alias_targets = set()
        for fact in facts:
            if fact.canon_status != 'ESTABLISHED_CANON':
                continue
            for sentence in re.split(r'(?<=[.!?])\s+|\n', fact.content):
                if not NEGATIVE.search(sentence) and re.search(r'\b(?:is (?:called|named|known as)|name is|also known as|alias)\b', sentence, re.I) and any(contains(sentence, name) for name in direct):
                    alias_targets.update(name for name in names if contains(sentence, name))
        anchors = direct | alias_targets
        query_terms = tokens(query)
        specific_terms = query_terms - GENERIC
        df = Counter(t for text in corpus for t in tokens(text))
        def lexical(text):
            hits = tokens(text) & query_terms
            return sum(1 + math.log(1 + len(corpus) / (1 + df[t])) for t in hits)
        def relevant(text):
            hits = tokens(text) & query_terms
            return bool(hits & specific_terms) or (not specific_terms and len(hits) >= 2)
        def mentions(text, entities):
            return any(contains(text, name) for name in entities)
        result.recent = [message for index, message in enumerate(recent) if index >= len(recent) - 2
                         or mentions(message.content, direct) or relevant(message.content)]
        result.protected_recent_ids = {message.id for message in result.recent if message.role == 'USER'
            and (NEGATIVE.search(message.content) or re.search(
                r"\b(?:is|are|was|were|has|have|decid\w*|chose|chosen|let['’]s use|establish\w*|actually|we agreed)\b",
                message.content, re.I))}
        if len(result.recent) < len(recent):
            result.reductions.append(f'irrelevant recent messages: {len(recent)} -> {len(result.recent)}; nearest exchange retained')
        topic = TOPICS.get(result.category, set())
        directly_linked = {s.id for s in subjects if any(contains(s.name, name) for name in anchors)}
        direct_facts = {f.id for f in facts if mentions(f'{f.title} {f.content}', anchors)
                        or any(link.subject_id in directly_linked for link in f.subject_links)}
        by_id = {f.id: f for f in facts}
        # Only one graph hop. Dependencies/constraints are essential even without lexical overlap.
        edges = db.query(FactRelationship).filter(FactRelationship.world_id == world_id).all()
        connected = {}
        chosen_edges = []
        for edge in edges:
            other = edge.target_fact_id if edge.source_fact_id in direct_facts else edge.source_fact_id if edge.target_fact_id in direct_facts else None
            if other and other in by_id:
                f = by_id[other]
                if edge.relationship_type in {'CONSTRAINS', 'CONTRADICTS', 'DEPENDS_ON'} or relevant(f.content) or tokens(f.content) & topic:
                    connected[other] = f'one-hop {edge.relationship_type}'
                    chosen_edges.append(edge)
        bridge_text = ' '.join([by_id[fid].content for fid in direct_facts | set(connected)]
            + [block for section in sections for _, _, _, block in markdown_blocks(section.markdown_content) if mentions(block, anchors)]
            + [subject_by_id[sid].description or '' for sid in directly_linked])
        related = {name for name in names if contains(bridge_text, name)} - direct
        # Parent subjects encode existing containment/cultural affiliation, not an invented graph.
        for sid in directly_linked:
            parent = subject_by_id.get(subject_by_id[sid].parent_subject_id)
            if parent and tokens(parent.name) - GENERIC:
                related.add(parent.name)
        related = {n for n in related if not any(n != longer and contains(longer, n) for longer in direct | related)}
        result.related_entities = related
        related_subject_ids = {s.id for s in subjects if s.name in related}
        for subject in subjects:
            if subject.name in direct | related:
                parent = subject_by_id.get(subject.parent_subject_id)
                result.items.append(ContextItem('entity', subject.id,
                    f'[Entity {subject.id} | name={subject.name} | parent={parent.name if parent else "unspecified"}] '
                    f'{subject.description or "No additional description recorded."}', 110,
                    'direct entity record' if subject.name in direct else 'related entity record', True))
        selected_facts = []
        for fact in facts:
            text = f'{fact.title} {fact.content} {fact.consistency_notes or ""}'
            direct_hit = fact.id in direct_facts
            constraint = bool(NEGATIVE.search(text)) or fact.canon_status == 'REJECTED'
            related_hit = (mentions(text, related) or any(link.subject_id in related_subject_ids for link in fact.subject_links)) and (constraint or bool(tokens(text) & topic))
            # Naming reservations are relevant even if their owner was not explicitly named.
            reservation = result.category == 'naming' and constraint and bool(re.search(r'\b(?:name|named|naming|reserved|reservation|alias|rejected)\b', text, re.I))
            hit = direct_hit or fact.id in connected or related_hit or reservation or relevant(text)
            if not hit:
                continue
            essential = direct_hit or fact.id in connected or related_hit or reservation or (
                relevant(text) and (fact.canon_status == 'ESTABLISHED_CANON' or constraint))
            score = (100 if direct_hit else 70 if fact.id in connected else 50 if related_hit or reservation else 0) + lexical(text)
            labels = ', '.join(subject_by_id[l.subject_id].name for l in fact.subject_links if l.subject_id in subject_by_id)
            label = (f'[Fact #{fact.id[:8]} | Status: {fact.canon_status} | Provenance: {fact.provenance_type} | '
                     f'Authority: {fact.authority_tier} | Consistency: {fact.consistency_status} | Subject: {labels or "General"}]')
            citations = []
            for citation in fact.citations:
                source = db.get(ReferenceChunk, citation.reference_chunk_id) if citation.reference_chunk_id else None
                web = db.get(ResearchSource, citation.research_source_id) if citation.research_source_id else None
                citations.append(f'[Source {citation.citation_source_type}: {citation.citation_note or ""}'
                    + ((f' | Book-p{source.book_page_start}' if source.book_page_start is not None else '')
                       + f'; PDF page {source.pdf_page_start}; {source.chapter_title}; {source.section_title}' if source else '')
                    + (f' | {web.title}; {web.url}' if web else '') + ']')
            citations = '\n'.join(citations)
            result.items.append(ContextItem('fact', fact.id, f'{label} {fact.title}: {fact.content}'
                + (f'\nConsistency note: {fact.consistency_notes}' if fact.consistency_notes else '')
                + (f'\nSuperseded by: {fact.superseded_by_fact_id}' if fact.superseded_by_fact_id else '')
                + (f'\n{citations}' if citations else ''), score,
                'direct entity' if direct_hit else connected.get(fact.id) or ('related constraint/context' if related_hit else 'naming reservation' if reservation else 'lexical relevance'), essential,
                {'status': fact.canon_status, 'provenance': fact.provenance_type}))
            selected_facts.append(fact)
        selected_ids = {f.id for f in selected_facts}
        if analysis and analysis.creative_permission.allow_invented_proper_nouns:
            # Retain the prior anti-collision function without shipping unrelated
            # biographies or planet properties. Specific queried names still pull
            # their complete owner/constraint records through normal retrieval.
            occupied = set()
            for fact in facts:
                if fact.canon_status != 'ESTABLISHED_CANON' or fact.provenance_type not in {'AUTHOR_CREATED', 'AI_GENERATED_AUTHOR_APPROVED'}:
                    continue
                for sentence in re.split(r'(?<=[.!?])\s+|\n', fact.content):
                    if NEGATIVE.search(sentence):
                        continue
                    occupied.update(re.findall(r"\b(?:is called|is named|name is|also known as) ([A-Z][\w'’-]*(?: [A-Z][\w'’-]*){0,3})", sentence))
            if occupied:
                result.items.append(ContextItem('naming_registry', 'occupied_names',
                    'Names already assigned in established canon (collision index only; no creative authorization): '
                    + ', '.join(sorted(occupied)), 85, 'compact established-name collision index', True))
        for edge in chosen_edges:
            if {edge.source_fact_id, edge.target_fact_id} <= selected_ids:
                result.items.append(ContextItem('relationship', edge.id,
                    f'[Relationship {edge.relationship_type} | created_by={edge.created_by} | resolved={edge.is_resolved}] '
                    f'Fact {edge.source_fact_id[:8]} -> Fact {edge.target_fact_id[:8]}: {edge.description or ""}', 90, 'one-hop edge', True))
        fact_content = {normalized(f.content) for f in selected_facts}
        for section in sections:
            blocks = markdown_blocks(section.markdown_content)
            indices = []
            for index, (_, _, heading, text) in enumerate(blocks):
                if text.startswith('#'):
                    continue
                direct_hit = mentions(text, anchors)
                related_hit = mentions(text, related) and (NEGATIVE.search(text) or tokens(text) & topic)
                reservation = result.category == 'naming' and NEGATIVE.search(text) and re.search(r'\b(?:name|reserved|naming|rejected)\b', text, re.I)
                if direct_hit or related_hit or reservation or relevant(text):
                    indices.append(index)
            indices = contextual_indices(blocks, indices)
            for index in indices:
                _, _, heading, text = blocks[index]
                essential = True
                if normalized(text) in fact_content:
                    result.reductions.append('section/fact duplicate')
                    continue
                result.items.append(ContextItem('section', f'{section.id}:{index}',
                    f'### World Bible Section: {section.title} | slug={section.slug} | v{section.version_number} | Provenance: {section.dominant_provenance}\n'
                    f'{heading}\n{text}', 100 + lexical(text) if essential else lexical(text),
                    'direct/related section passage' if essential else 'lexical section passage', essential,
                    {'section_id': section.id, 'slug': section.slug, 'block': index}))
            if indices:
                start, end = blocks[min(indices)][0], blocks[max(indices)][1]
                result.section_windows[section.slug] = {'id': section.id, 'title': section.title,
                    'version': section.version_number, 'provenance': section.dominant_provenance,
                    'start': start, 'end': end, 'base': section.markdown_content,
                    'excerpt': section.markdown_content[start:end]}
        for model, kind, fields in ((CanonProposal, 'proposal', ('proposed_title', 'proposed_content')),
                                     (UnresolvedQuestion, 'question', ('question_text', 'context_notes'))):
            rows = db.query(model).filter(model.world_id == world_id).all()
            for row in rows:
                text = ' '.join(getattr(row, key) or '' for key in fields)
                if mentions(text, direct | related) or relevant(text):
                    status = row.status
                    provenance = getattr(row, 'proposed_provenance', 'AUTHOR_QUESTION')
                    result.items.append(ContextItem(kind, row.id,
                        f'[{kind.upper()} | Status: {status} | Provenance: {provenance} | NOT CANON without an established fact] {text}',
                        60 + lexical(text), 'relevant unresolved/proposal record', mentions(text, direct)))
        established_content = {normalized(f.content) for f in selected_facts if f.canon_status == 'ESTABLISHED_CANON'}
        self._history(db, world_id, result, conversation, excluded, relevant, mentions, lexical, established_content)
        if re.search(r'\b(?:remember|previously decided|earlier decision)\b', query, re.I) and not any(i.kind == 'history' for i in result.items):
            result.items.append(ContextItem('notice', 'no_historical_match',
                'No matching older discussion was retrieved. Use established records and the recent exchange; if the requested memory is still missing, ask the author for the referent instead of inventing it.',
                100, 'requested historical context unavailable', True))
        if conversation:
            # A summary is a navigation aid, never independent canon or permission.
            summary = (conversation.rolling_summary or '').strip()
            covered = {normalized(m.content) for m in result.recent} | established_content
            blocks = [block for _, _, _, block in markdown_blocks(summary) if normalized(re.sub(
                r'^\s*(?:AUTHOR DECISION|ESTABLISHED(?: CANON)?|CANON DECISION)\s*:\s*', '', block, flags=re.I)) not in covered]
            result.summary = '\n\n'.join(blocks)
            if summary and not result.summary:
                result.reductions.append('summary duplicates selected records')
        result.items.sort(key=lambda item: (-item.score, item.id))
        return result

    def _history(self, db, world_id, result, conversation, excluded, relevant, mentions, lexical, represented_facts):
        recent_ids = {m.id for m in result.recent} | excluded
        # Stream persisted messages in batches: no all-history prompt or model call.
        messages = db.query(Message).join(Conversation).filter(Conversation.world_id == world_id,
                    Message.id.notin_(recent_ids)).order_by(Message.created_at.desc(), Message.id.desc()).yield_per(200)
        candidates = []
        seen = set()
        for message in messages:
            text = message.content
            direct = mentions(text, result.entities | result.related_entities)
            if not direct and not relevant(text):
                continue
            all_blocks = markdown_blocks(text)
            selected = contextual_indices(all_blocks, [index for index, (_, _, _, block) in enumerate(all_blocks)
                      if mentions(block, result.entities | result.related_entities) or relevant(block)])
            if not selected:
                continue
            # Include the next exchange to resolve approvals such as "Let's use that".
            next_messages = db.query(Message).filter(Message.conversation_id == message.conversation_id,
                Message.created_at > message.created_at).order_by(Message.created_at, Message.id).limit(2).all()
            excerpt = '\n\n'.join(all_blocks[index][3] for index in selected)
            key = (message.role, normalized(excerpt))
            if key in seen or normalized(excerpt) in represented_facts:
                continue
            seen.add(key)
            following = '\n'.join(f'[{m.role}; historical context, not fresh authority]: {m.content}' for m in next_messages if m.id not in recent_ids)
            label = 'AUTHOR STATEMENT: may be hypothetical/rejected; newer explicit decisions take precedence' if message.role == 'USER' else 'AI SUGGESTION: not author evidence'
            score = (50 if direct else 0) + lexical(text) + (5 if message.role == 'USER' else 0)
            decision_or_constraint = message.role == 'USER' and direct and bool(NEGATIVE.search(excerpt) or re.search(
                r"\b(?:decid\w*|chose|chosen|let['’]s use|is called|is named|name is|establish\w*|actually|we agreed)\b", excerpt, re.I))
            candidates.append(ContextItem('history', message.id,
                f'[Historical message {message.id} | conversation={message.conversation_id} | date={message.created_at.isoformat()} | role={message.role} | {label}]\n{excerpt}'
                + (f'\nFollowing exchange:\n{following}' if following else ''), score,
                'older entity discussion' if direct else 'older lexical discussion', decision_or_constraint,
                {'conversation_id': message.conversation_id, 'role': message.role}))
        candidates.sort(key=lambda item: (-item.score, item.id))
        essential = [item for item in candidates if item.essential]
        optional = [item for item in candidates if not item.essential][:settings.context_history_excerpts]
        result.items.extend(essential + optional)
        if len(candidates) > len(essential) + len(optional):
            result.reductions.append(f'history ranked: {len(candidates)} candidates -> {len(essential) + len(optional)} excerpts; author decisions/constraints retained')

    @staticmethod
    def select_research(book, online, queries):
        """Keep reference-book hits intact; filter/deduplicate supplementary excerpts."""
        terms = tokens(' '.join(queries))
        seen = set()
        passages = []
        for passage in book.get('passages', []):
            key = passage.get('chunk_id') or (passage['citation_id'], passage['content'])
            if key not in seen:
                seen.add(key)
                passages.append(passage)
        sources = []
        seen = set()
        for source in online.get('sources', []):
            if source['url'] in seen:
                continue
            text = source.get('title', '') + ' ' + source.get('excerpt', '')
            if terms and not tokens(text) & terms:
                continue
            seen.add(source['url'])
            blocks = markdown_blocks(source['excerpt'])
            hits = {i for i, (_, _, _, block) in enumerate(blocks) if tokens(block) & terms}
            indices = {j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(blocks)}
            # Complete paragraphs and immediate qualifications, never a character prefix.
            excerpt = '\n\n'.join(blocks[i][3] for i in sorted(indices)) if indices else source['excerpt']
            sources.append(dict(source, excerpt=excerpt))
        return dict(book, passages=passages), dict(online, sources=sources)

    @staticmethod
    def record(stage, bundle, components, budget, book_count=0, web_count=0):
        control = current_generation.get()
        diagnostic = {'request_id': control.request_id if control else None, 'stage': stage,
            'task_category': bundle.category, 'facts': sum(i.kind == 'fact' for i in bundle.items),
            'entities': len(bundle.entities), 'related_entities': len(bundle.related_entities),
            'structured_entities': sum(item.kind == 'structured_entity' for item in bundle.items),
            'conversation_excerpts': sum(i.kind == 'history' for i in bundle.items) + len(bundle.recent),
            'rolling_summary_included': bool(bundle.summary), 'book_passages': book_count,
            'external_excerpts': web_count, 'component_tokens': {key: estimate_tokens(value) for key, value in components.items()},
            'selected_record_tokens': {kind: sum(estimate_tokens(i.text) for i in bundle.items if i.kind == kind)
                                      for kind in ('entity', 'structured_entity', 'fact', 'section', 'relationship', 'history', 'proposal', 'question', 'naming_registry')},
            'budget': budget, 'reductions': list(bundle.reductions)}
        diagnostic['estimated_input_tokens'] = sum(diagnostic['component_tokens'].values())
        logger.info('context_selection %s', json.dumps(diagnostic, sort_keys=True))
        return diagnostic


context_retriever = ContextRetriever()
