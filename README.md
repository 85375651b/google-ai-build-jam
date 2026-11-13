# Worldbuilding Studio — AI Worldbuilding Consultant & World Bible Workspace

An author-first, AI-powered worldbuilding consultant and structured **World Bible** studio designed for novelists, fantasy/sci-fi writers, conlangers, and worldbuilders.

## Core Design Principles

1. **The Author Is the Authority:** The consultant helps you understand real-world systems, trace implications, and implement your ideas—never taking creative control over your world or forcing conventional genre tropes.
2. **Opt-In Creative Generation:** Explaining real-world concepts, evaluating plausibility, and implementing your stated premises are strictly separated from creative brainstorming. Fictional options are generated only when explicitly requested and stay strictly within the requested scope.
3. **Subject-Specific Expertise Adaptation:** Adapts explanation depth independently across 12 worldbuilding domains (e.g., conversing at a peer technical level in *Linguistics & Constructed Languages* while explaining foundational concepts in *Political Science* or *Economics*).
4. **Grounded RAG + Supplementary Research Hierarchy:**
   - **Tier 1 (Fictional Canon):** Author-established world facts and direct World Bible edits.
   - **Tier 2 (Primary Real-World Reference):** Retrieval-Augmented Generation (RAG) indexed over Mark Rosenfelder's *The Planet Construction Kit* with chapter, section, and page citations.
   - **Tier 3 (Supplementary Online Research):** Automatic fallback to credible academic, historical, and encyclopedic sources via **Tavily** when a topic lies outside the reference book's coverage.
5. **Continuously Maintained Hybrid World Bible:** Edit your World Bible directly in Markdown or approve staged updates from conversation. Tracks provenance (`Author Created` vs. `Author-Approved AI`), revision history with one-click restore, cross-domain consistency checks, and standalone exports to **Markdown (`.md`)**, **PDF (`.pdf`)**, and **Word (`.docx`)**.

---

## Architecture & Tech Stack

- **Frontend:** React 18, Vite, CSS Design System (`src/App.jsx`, `src/styles.css`)
- **Backend:** Python 3, FastAPI, SQLAlchemy 2.0 (`backend/main.py`, `backend/services/`)
- **Database & Retrieval:** Local SQLite (WAL mode) + SQLite `FTS5` (BM25 full-text search) with lexical reranking + PyMuPDF (`pymupdf`) & Tesseract OCR fallback
- **Local LLM:** Ollama's native HTTP API for streamed consultation, schema-constrained turn analysis and World Bible extraction, and conversation summaries. No hosted model fallback.
- **Online Research:** Optional Tavily Search API. Without a key, the app skips online research and uses the local reference book.

---

## Quick Start

### 1. Configure local inference

Install [Ollama](https://ollama.com/download) if necessary. Start it if its desktop app or background service isn't already running:

```bash
ollama serve
```

In a separate terminal, download the default model if it isn't already installed:

```bash
ollama pull gpt-oss:20b
```

This repository defaults to `gpt-oss:20b`, which was already installed on the development machine. Use `ollama list` to inspect your installed models. For different hardware, download and select another local model; speed, memory consumption, and World Bible extraction quality depend on your model and context size.

For a new installation, copy `.env.example` to `.env`. Existing installations automatically use the local defaults; old Claude/Gemini keys are ignored. Preserve your database path, book path, and optional Tavily key.

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
LOCAL_MODEL=gpt-oss:20b
LOCAL_CONTEXT_WINDOW=32768
LOCAL_MAX_OUTPUT_TOKENS=4096
LOCAL_REQUEST_TIMEOUT=600
TAVILY_API_KEY=
```

Project Settings saves your chosen model to `.env`. Download the model before selecting it. Shell variables take precedence over `.env`; restart the backend after manual edits. Every model pass uses the selected model. The gpt-oss model uses low reasoning effort; reasoning text is excluded from chat and saved canon.

Book retrieval uses local SQLite full-text search, without an embedding API. Existing data and book indexes remain compatible. Leave `TAVILY_API_KEY` empty for offline operation after model downloads. Ollama must remain running while the app is in use.

The status indicator checks server reachability and whether the selected model is installed; it does not benchmark generation. Large World Bibles and long conversations may exceed local context capacity. The app uses a conservative context estimate and reports oversized input instead of deliberately discarding canon. Increase context size only within your model and memory limits. Output-limit and timeout errors suggest the corresponding environment setting or a shorter request. Existing deterministic analysis/extraction fallbacks remain in place, so validate your author's workflows with your chosen model.

### 2. Install Dependencies

Install the Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For the frontend:

```bash
npm install
```

### 3. Launch the Application

```bash
npm run dev
```

This starts both the **FastAPI backend** (`http://127.0.0.1:8000`) and the **Vite React frontend** (`http://127.0.0.1:5173`). Open **`http://127.0.0.1:5173`** in your browser.

---

## Running Tests

To run the automated integration and behavioral safeguard test suite:

```bash
.venv/bin/python -m pytest -q
```

Tests use a temporary database and disable online research. Controlled HTTP responses exercise the local model protocol and the chat-to-World-Bible flow without requiring Ollama. To exercise the running model as well, use:

```bash
.venv/bin/python scripts/check_local_model.py
```

The smoke check uses a temporary world and database, with web research disabled. It runs real analysis, streamed consultation, and extraction, checks that the author’s island count reaches the World Bible, then removes its test data. It may take several minutes on a CPU.

The provider implementation is `backend/services/llm_service.py`. `PUT /api/system/model` accepts `{"local_model":"gpt-oss:20b"}`. `GET /api/system/status` returns `local_model_status` and the selected model. The former embedding endpoint returns an explanatory 409 because retrieval uses full-text search.


## Generation lifecycle and cancellation

Each chat submission carries a UUID. A small additive `generation_requests` journal records its conversation, world, progress, and terminal result. The existing world/canon tables are unchanged. The backend rejects overlapping submissions in the same conversation; independent conversations can run concurrently. Reusing a completed request ID with the same content replays its result without re-running inference or persistence. Reusing an ID with different content is rejected. Use the existing single-worker development-server configuration; lifecycle ownership is process-local, with a durable journal for restart recovery.

The stream reports `request_started`, phase `progress`, `generation_started`, `delta`, `answer_completed`, `saving_started`, `response_saved`, applicable `postprocessing_started`, then exactly one `complete`, `cancelled`, or `error` terminal event. Events include request, conversation, and world IDs. Heartbeats report liveness and the actual phase, not a progress percentage. The UI ends its busy state on a valid terminal event without waiting for EOF. EOF without a terminal event is an error.

`POST /api/conversations/{conversation_id}/requests/{request_id}/cancel` requests cancellation and waits up to ten seconds for acknowledgment. It is idempotent and also supports cancellation arriving before submission. `GET` at the same path without `/cancel` reports request state and any terminal result. Stop propagates to asynchronous HTTPX calls and closes the active Ollama response; Ollama cancels its request context when that connection closes. No model-unload or invented runtime cancellation endpoint is used. A tab close, workspace switch, or disconnected SSE response also requests cancellation.

Cancellation before the answer is complete saves neither partial assistant text nor canon. A fully completed answer is saved in a short transaction; cancellation afterward retains it and marks the request cancelled/failed. Summarization and extraction prepare their results without holding a world-write transaction. Applying the resulting World Bible batch uses one transaction, including helper writes that previously committed individually. Cancelling before that boundary discards the batch. Once its short atomic commit has begun, cancellation observes its completed result rather than interrupting a commit halfway through. A Bible version change during extraction fails the update instead of overwriting newer edits. Startup marks interrupted journal entries as failed, preserving previously saved messages.

Timeouts are configurable in `.env` (restart the backend after edits):

| Setting | Default | Meaning |
|---|---:|---|
| `LOCAL_REQUEST_TIMEOUT` | 600 seconds | Ollama network inactivity between chunks; hidden reasoning calls use streaming transport too |
| `GENERATION_DEADLINE_SECONDS` | 3600 seconds | Overall request deadline, across every stage |
| `GENERATION_HEARTBEAT_SECONDS` | 5 seconds | SSE heartbeat interval |
| `GENERATION_CLIENT_IDLE_SECONDS` | 45 seconds | Browser watchdog without lifecycle events; at least three heartbeat intervals |

The browser watchdog requests backend cancellation and releases its owned UI state if the connection stops reporting progress. Long inference with working heartbeats is not mistaken for a disconnected stream; stalled Ollama I/O and the overall deadline are enforced independently by the backend. Non-model synchronous tasks reach cancellation checkpoints before the next model/persistence stage. A severely unavailable database must be repaired before journal operations can succeed.

Run backend and frontend regressions with:

```bash
.venv/bin/python -m pytest -q
npm run test:frontend
npm run build
```

Frontend tests include the actual React component in a DOM environment, covering the transition from Stop to post-processing and back to Send while the stream remains open.

## Selective context retrieval

Consultation and extraction now select relevant entities, facts, section passages,
constraints and older discussions from the existing database. The summary appears
once. Book evidence remains primary for factual support; simple canon-management
turns skip unnecessary research. No model, database schema or infrastructure change
is needed for context retrieval.

See [the context architecture and diagnostics guide](docs/context-retrieval.md)
for budgets, focused extraction editing, read-only inspection and limitations, and
[the context-size comparison](docs/context-benchmark.json) for six synthetic-world
before/after measurements. Run the context-specific tests with:

```bash
.venv/bin/python -m pytest -q tests/test_context_retrieval.py
```
