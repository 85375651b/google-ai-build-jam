# Context architecture investigation (before this change)

The request owner in `generation_service._pipeline` calls `analyze_turn`,
`retrieve_knowledge`, `assemble_consultant_system_prompt`, and
`build_conversation_contents`, then streams the consultation answer. The existing
classification/permission contract and lifecycle remain the owners of authorization
and cancellation respectively.

* `analyze_turn`: latest author message plus two recent messages for references;
  produces the existing classification and permission contract.
* `retrieve_knowledge`: always queries the locally indexed reference book (FTS5
  BM25 plus lexical reranking, at most five passages), even for canon management.
  Online research supplements partial/low book coverage when factual retrieval is
  required. Citations are verified against these returned source records.
* `assemble_consultant_system_prompt`: immutable identity/permission/instruction
  layers, adaptive expertise and domain modules, reference passages and online
  sources, the 25 most recently updated established facts, every populated Bible
  section's first 1,500 characters, unranked hypothetical/open/rejected items,
  rolling summary, and a world-wide naming scan on authorized naming turns.
* `build_conversation_contents`: the same rolling summary again, up to 12 recent
  messages, and the current author message. Large messages lose their middle.
* `prepare_post_turn_extraction`: eight recent messages (minus the saved current
  pair), at most 30 unranked established facts, every complete Bible section,
  current author message, permission/author evidence and assistant reply.
* `extract_post_turn_proposals`: existing author-evidence gates and whole-section
  replacement/proposal application. Issue 2 prepares without writes, checks section
  versions, then applies all updates in one owned transaction. Selective extraction
  must not turn an excerpt into a destructive whole-section replacement.

Available persisted retrieval signals: Subjects and FactSubject links,
BibleSectionFact links, FactRelationship edges, fact status/provenance/revisions,
pending proposals, unresolved questions, all conversations/messages within a
world, reference-book FTS5, and fact citations. There is no separate entity/alias
graph. No additional model call or vector service is required to use these signals.

# Implemented selection path

`generation_service._pipeline` now calls `assemble_consultation`, which retrieves
one shared selection for the system instruction and conversation contents, applies
the consultation budget, and records diagnostics. The surrounding lifecycle,
authorization analysis, inference calls, and commit/cancellation boundaries are
unchanged.

`context_retrieval.ContextRetriever` is read-only. It uses explicit Subject names,
case-insensitive textual entity mentions, and confirmed naming aliases to find
anchors. Direct entity facts and FactSubject links outrank lexical support. Term
rarity weights lexical matches. Existing FactRelationship dependencies, constraints,
and contradictions expand by one hop; text co-occurrence and parent Subjects add
bounded connected context. Default category titles are excluded from entity anchors;
mentioning a destination category does not select its entire subject. Related
properties are filtered by task topic. Relevant
reservations and rejections are included even when the reserved name's owner was
not named in the request. Topic detection never changes creative permission.

Fact records retain their IDs, canon status, provenance, authority tier,
consistency notes, supersession links, and linked source citations. Established
facts, hypothetical/superseded material, pending proposals, open questions, rejected
ideas and historical AI suggestions remain distinguishable. Section prose lives in
its own qualified document layer, rather than being relabeled established atomic
canon. Author decisions govern fiction; the reference book remains the primary
factual source; online sources supplement it; general model knowledge is explicitly
unverified fallback knowledge.

Sections are selected as whole Markdown blocks with their heading path and source
section/version. Headings need not have a blank line after them. Adjacent dependent
qualifications and negations remain attached. Relevant text near a section's end
is searchable. Matching fact/section duplicates are removed. A very large indivisible
paragraph is retained or fails the budget explicitly; it is not clipped mid-fact.

Conversation continuity examines six recent messages by default, always keeps the
nearest exchange, and keeps other recent messages when relevant. Older discussions
are ranked across **all conversations in the same world, including archived ones**.
Historical excerpts include dates, roles and following exchanges for approvals or
corrections. Clear decision/constraint discussions about an anchor are protected
from optional-history limits. Older assistant prose is labeled non-author evidence.
If a requested earlier discussion is not found, the prompt says to ask rather than
invent a memory. The rolling summary is included only in conversation contents,
once, with exact normalized duplicates removed. Summary generation and model
routing are unchanged; summary hypotheses are never independently canon evidence.
Authorized naming turns also receive a compact index of names explicitly assigned
in established canon, without the unrelated owners' biographies or properties.

`retrieve_knowledge` skips book/search work for straightforward canon management,
including a noisy factual-retrieval flag, while explicit mixed research requests
still receive factual support. It does not mutate classification or permission
results. Otherwise the existing FTS5/BM25 book retrieval and coverage
policy remain in use. Whole book passages retain page ranges, PDF pages, chapter,
section and citation IDs. Supplementary sources are filtered for lexical relevance
and deduplicated; excerpts preserve complete paragraphs and surrounding qualifications.
New online source records and fresh search excerpts no longer lose everything after
character 2,000. Existing source records are not rewritten. Verified citation metadata
matches exact supplied IDs, so `Book-p1` cannot match `Book-p10` or an invented ID.

# Extraction and preservation of omitted text

`prepare_post_turn_extraction` retains the existing eligibility, reconciliation and
author-evidence safeguards. The model receives the original author message, full
assistant reply, classification/permission information, explicitly validated author
decisions, relevant prior discussion, entities, facts, relationships and constraints.
It receives only candidate section edit windows, not the entire Bible. Target
categories with no matching prose get an append-only window; their unrelated
contents are never sent merely because a domain was mentioned.

The existing schema's `updated_markdown` replaces **the supplied window only**.
`expand_extracted_section_edits` rebases it into the server-held original section
before the existing evidence gates and atomic persistence path run. Prefix and suffix
content is preserved byte-for-byte. An attempt to rewrite an existing section outside
the supplied scope fails explicitly. New custom sections and atomic proposal routing
retain their existing author gates. The existing complete section-version snapshot
still guards concurrent edits. No data model or transaction system was introduced.

A live model check exposed an empty `source_quote`, which the existing safeguard
correctly rejected. The inference-only extraction JSON schema now presents a shared
enum of the already validated author quotes (and any existing explicit authorization)
and requires the evidence field. The model copies an allowed literal rather than
omitting it. The persisted/API schemas and exact author/delegation validators are
unchanged; missing evidence is never filled or inferred by the backend. The Markdown
field's prompt description now explicitly describes edit windows, rather than
contradicting that contract by requesting an unseen complete section. This adds no
model call and the dynamic schema is included in the extraction token estimate.

# Budgets and diagnostics

| Environment variable | Default | Meaning |
|---|---:|---|
| `CONSULTATION_CONTEXT_TOKENS` | 12000 | Complete consultation input estimate |
| `EXTRACTION_CONTEXT_TOKENS` | 10000 | Extraction input, including instructions and JSON schema |
| `CONTEXT_RECENT_MESSAGES` | 6 | Recent continuity window examined |
| `CONTEXT_HISTORY_EXCERPTS` | 8 | Optional historical excerpts; relevant author decisions/constraints are additional |

The estimator uses `ceil(UTF-8 bytes / 3)` plus chat framing. These are estimates,
not exact `gpt-oss` tokenizer counts. Input budgets are also bounded by the unchanged
model context window minus the configured output allowance and a 1,024-token margin.
Optional supporting history/proposals are reduced first. Required canon,
prohibitions, instructions, author text and replies are never character-truncated.
Unsafe overflow yields `CONTEXT_BUDGET_EXCEEDED`, including required and available
estimates, before inference; the lifecycle surfaces that failure normally.

Each consultation/extraction assembly emits an INFO `context_selection` JSON record
with request ID when available, task category, counts of facts/entities/related
entities/history/book/web excerpts, summary inclusion, estimates by component and
record type, total estimate, budget, and reductions. An explicit application logger
handler makes these metadata records visible under the default development server.
No message bodies, source excerpts, manuscript prose or credentials are logged.

For read-only local development inspection, without inference or web calls:

```bash
.venv/bin/python scripts/inspect_context.py --world-id WORLD_UUID \
  --conversation-id CONVERSATION_UUID --message "Help me name Planet X"
```

This prints selected IDs, ranking reasons, statuses, provenance, estimates and entity
anchors. Add `--include-text` only when deliberately displaying the selected private
content locally. This does not create a public debug endpoint.

# Verification and limitations

`tests/test_context_retrieval.py` covers relevant entities, bounded connections,
reservations, old conversations, negative constraints, status/provenance, late section
passages, adjacent qualifications, aliases, summary deduplication, real local book
retrieval, research/citation integrity, extraction rebasing, budget failures and six
representative context-size regressions. `tests/context_scenarios.py` supplies a
synthetic world with 15 populated sections, older important canon and discussions,
and newer irrelevant facts/messages. See `context-benchmark.json` for measured
pre-change/current full-input estimates on that fixture. The factual case includes
real locally retrieved book passages. The comparison measures input size, not
wall-clock generation speed or a general model reasoning benchmark.

Retrieval is lexical and relationship based, not a semantic embedding model. It can
miss an unstated alias or a paraphrase with no shared terms or recorded relationship.
It scans existing world records and streams historical messages, so retrieval cost
grows with project size. Expansion is bounded by one recorded edge and a relevant
connected-entity pass; missing graph links cannot be inferred reliably. Exact
deduplication does not recognize every paraphrase. Multiple distant matches in one
section require a contiguous extraction window to preserve the existing schema;
that can require a narrower request when the span cannot fit safely. No uncertain
retrieval result establishes canon or grants creative permission.
