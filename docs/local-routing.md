# Local routing implementation status

Defaults remain gpt-oss:20b, 32768 allocated context, 4096 output cap,
low gpt-oss reasoning, and 10m keep-alive. Summarization requests 1200 output
tokens. Existing selective input budgets remain unchanged.

Optional environment variables LOCAL_CLASSIFICATION_MODEL,
LOCAL_EXTRACTION_MODEL and LOCAL_SUMMARIZATION_MODEL select local task models.
Empty values inherit LOCAL_MODEL. LOCAL_KEEP_ALIVE controls residency hints.
No additional model is enabled by default. Structured failures retry once against
the primary model with the same prompt and schema. Cancellation bypasses this
fallback. Backend authorization and transaction validation remain unchanged.
Streaming consultation continues using the primary model.

Runtime inspection found gpt-oss:20b (13 GB installed artifact) and llama3.2:1b
(1.3 GB installed artifact). These sizes are not measured runtime RAM requirements.
`ollama ps` showed no resident model at inspection time; this does not establish
whether GPU acceleration is available during inference. No models downloaded,
server settings changed, or hardware tuning adopted.

Research sources:
- https://ollama.com/library/qwen3 (potential future candidate, not benchmarked)
- https://github.com/ollama/ollama/blob/main/docs/faq.mdx
- https://github.com/ollama/ollama/blob/main/docs/gpu.mdx

This is a partial implementation, not a performance result. Outstanding work:
ten-workload live baseline and quality dataset; repeated candidate benchmarks;
RAM/CPU/shared GPU memory sampling; TTFT and switch overhead; controlled hardware
experiments; task context/output/reasoning controls; summary fallback; full
regression suite. Issue 5's call reduction remains incomplete. No measured speedup
or candidate quality claim is made. Keep all optional routes empty until validated.
