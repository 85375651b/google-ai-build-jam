# Inference pipeline inspection

The current service paths are under `backend/services/`.

| Call | Trigger | Input | Output | Opportunity |
|---|---|---|---|---|
| Intent analysis | Every message | Current message and two recent messages | Validated intent, permission and search queries | Conservative deterministic cases; combination must retain pre-retrieval planning |
| Consultation | Every message | Selective World Bible, conversation and research context | Streamed prose | Could also generate typed operations, but requires a streaming structured envelope |
| Extraction | Reconciled extraction eligibility | Response, author evidence, selected context and edit windows | Typed proposals | Already skipped for ineligible turns; deterministic complete decisions may avoid it |
| Summary | More than twelve messages | Prior summary and up to twenty older messages | Summary prose | Needs persistent coverage metadata and maintenance scheduling |

The request saves the answer before extraction, then validates and commits canon
inside the owned transaction. Cancellation checks protect that boundary. Summary
currently runs before extraction and terminal completion. Combining calls must not
move database writes into model execution or bypass authorization reconciliation.

This pass adds privacy-preserving `ollama_metrics` records for completed synchronous
and asynchronous calls. Counters come directly from Ollama: prompt/generated token
counts, prefill/generation/total/load durations (nanoseconds). Missing runtime values
remain null rather than estimates. Logs include request identity and lifecycle phase;
they exclude prompts and generated content.

Status: instrumentation foundation only. No model-call reduction or latency benefit
has yet been established. Ten-scenario live baseline, combined response design,
summary coverage migration and complete regression verification remain outstanding.
