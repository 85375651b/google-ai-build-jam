# Existing decision-to-persistence path (investigated before editing)

`GenerationService._pipeline` classifies the original author message with Issue 1's
reconciliation/evidence gates, retrieves Issue 3's selective context, streams the
answer, and saves the complete conversation pair. It prepares extraction outside
the write transaction. Extraction returns Markdown edit windows and fact proposals;
the windows are rebased into full sections server-side. The owned `BEGIN IMMEDIATE`
transaction checks every section version, applies extraction, updates summary and
message metadata, and commits the terminal request result together (Issue 2).

`extract_post_turn_proposals` verifies author/delegation evidence but creates new
text Facts rather than updating entity properties. It applies section edits before
recording inferred contradictions. Fact helpers have individual commits, deferred
only by the lifecycle's ControlledSession. Other callers lack that protection.
`apply_agent_section_edit` versions section prose but omits the manual editor's
paragraph/Fact synchronization and does not accept an expected version.

Manual `update_section_directly` checks the loaded section version, writes a revision,
and synchronizes paragraphs to Facts. Facts have immutable IDs, provenance/status,
revisions and section/Subject links; Subjects mostly represent the 15 categories.
FactRelationship connects Fact IDs. Neither table represents authoritative entity
properties, name ownership or scoped reservations. Proposal approval creates or
updates Facts and appends prose. Export/import remaps IDs into a new project and
currently exports only part of the available revision/link metadata.

The incremental extension will retain these tables and Markdown, add stable entities,
property/name/relationship/constraint records plus batch/audit/synchronization records,
and reuse the existing owned commit boundary. Legacy text will not be interpreted as
entities merely because it contains capitalized words. A consistent SQLite backup of
the existing project was taken before schema changes; migration validation uses copies.

## Verification and completion (2026-11-13)

The working tree already contained the incremental implementation when this task
was resumed. This pass preserved it and corrected projection history: revisions
now record whether names were actually projected into Markdown. A direct author
edit suppresses projections, so the next agent update must not expect a generated
names block that was never written. The existing full-request retrieval regression
covers this path. The legacy-name regression now explicitly checks the replacement
canonical statement and retained former name.

Implementation map:
- `backend/models.py`, `backend/schemas.py`, `backend/migrations.py`: stable entity
  IDs, properties, scoped names, relationships, constraints, revisions, batches,
  naming policies and synchronization issues; additive migration with backup.
- `backend/services/entity_service.py`: validated operations, scoped canonical and
  reservation ownership, aliases/former/rejected names, dependency validation,
  optimistic versions, idempotency and SQLite transaction ownership.
- `backend/services/agent_orchestrator.py`, `generation_service.py`,
  `author_intent.py`, `canon_language.py`: author evidence and permission gates,
  extraction and application within the request commit boundary.
- `backend/services/bible_service.py`, `canon_archive.py`, `world_service.py`:
  targeted Markdown synchronization, preserved ambiguous author text, archives.
- `backend/services/persistence_reporting.py`: suppress premature success claims;
  committed outcomes supply persistence reporting.
- `backend/services/context_retrieval.py`: structured naming and dependency context.
- `tests/test_structured_canon.py`: identity, naming, reservations, related updates,
  rollback, concurrent sessions, authorization, synchronization, reporting,
  migration and archive regression coverage.

Relationships use stable source/target IDs. Focused validators cover parent/event
chronology, protected treaty properties and explicit constraints. Unknown chronology
requires clarification. These checks are deliberately not a general logic engine.
Unapproved, hypothetical and research operations remain provisional. Renaming keeps
the same entity/property identity and retains historical names. Related mutations
validate their final state before committing; failures roll back the batch.

Migration verification used SQLite backup to copy `data/worldbuilding.db` to a
temporary file, ran migration twice, and compared every pre-existing row in all
31 tables: unchanged. The production database was not migrated by this pass.
Legacy prose remains readable without automatic entity inference.

Limitations: natural-language synchronization recognizes a bounded grammar;
ambiguous text requires author clarification. Consistency is limited to supported
constraints, not arbitrary fictional logic. Passing mocked inference tests does
not establish the extraction accuracy of the live local model.
