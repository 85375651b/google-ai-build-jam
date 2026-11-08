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
- **Database & Retrieval:** Local SQLite (WAL mode) + SQLite `FTS5` (BM25 full-text search) + vector similarity + PyMuPDF (`pymupdf`) & Tesseract OCR fallback
- **LLM & Research Integration:**
  - **Primary LLM:** Google Gemini API (`google-genai` SDK)
  - **Automatic Rate-Limit Fallback:** Anthropic Claude API (`/v1/messages` with streaming and structured JSON fallback if Gemini hits quota/rate limits)
  - **Online Research:** Tavily Search API

---

## Quick Start

### 1. Configure Environment Variables (`.env`)

Copy `.env.example` to `.env` (already created in the project root) and paste your API keys:

```env
# Primary LLM: Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# Automatic Fallback LLM (Used automatically if Gemini hits rate limits)
CLAUDE_API_KEY=your_claude_api_key_here
CLAUDE_MODEL=claude-sonnet-4-20250514

# Supplementary Online Research
TAVILY_API_KEY=your_tavily_api_key_here

# Local Storage & Reference Book
DATABASE_URL=sqlite:///./data/worldbuilding.db
PRIMARY_REFERENCE_BOOK_PATH="./The planet construction kit (Rosenfelder, Mark, author) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
```

### 2. Install Dependencies

Python virtual environment dependencies are installed in `.venv` (`pip install -r requirements.txt`). For the frontend:

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
.venv/bin/python -m pytest tests/test_mvp_integration.py -v
```
