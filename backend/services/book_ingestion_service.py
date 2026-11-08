"""Reference book ingestion, OCR fallback, structural chunking, and hybrid RAG retrieval."""

import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import pymupdf as fitz  # PyMuPDF
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from backend.config import settings
from backend.models import ReferenceChunk, ReferenceDocument
from backend.services.gemini_service import GeminiServiceError, gemini_service

logger = logging.getLogger(__name__)

# Known chapter landmarks in Mark Rosenfelder's "The Planet Construction Kit" (by printed page start)
ROSENFELDER_CHAPTER_MAP: List[Tuple[int, str]] = [
    (1, "Introduction"),
    (11, "E-Z Fantasy World"),
    (20, "Storytelling"),
    (35, "Astronomy and Geology"),
    (66, "Biology"),
    (100, "History"),
    (118, "Culture"),
    (152, "Daily Life"),
    (189, "Religion"),
    (218, "Magic"),
    (224, "Technology"),
    (243, "War"),
    (285, "Making Maps"),
    (300, "Illustrations"),
    (332, "3-D Modeling"),
    (356, "Further Reading & Index"),
]

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "what", "how", "why",
    "are", "was", "were", "have", "has", "had", "not", "but", "can", "could",
    "would", "should", "into", "about", "their", "there", "they", "them", "then",
    "than", "when", "where", "which", "who", "whom", "whose", "will", "More",
    "some", "such", "only", "other", "also", "very", "just", "like", "make",
    "world", "fictional", "system", "work", "works", "worked",
}


def _estimate_tokens(text_str: str) -> int:
    return max(1, int(len(text_str.split()) * 1.3))


def _clean_ocr_text(raw_text: str) -> str:
    """Repair common line-wrap hyphenations and normalize whitespace from scanned PDFs."""
    # Rejoin hyphenated line breaks: e.g. "civili-\nzation" -> "civilization"
    text_str = re.sub(r"([a-zA-Z])-\n([a-z])", r"\1\2", raw_text)
    # Normalize multiple blank lines
    text_str = re.sub(r"\r\n", "\n", text_str)
    text_str = re.sub(r"\n{3,}", "\n\n", text_str)
    return text_str.strip()


class BookIngestionService:
    """Ingests reference books (PDF/TXT/MD), preserves structure, and performs hybrid RAG retrieval."""

    def compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def ensure_primary_book_ingested(
        self, db: Session, generate_embeddings: bool = False
    ) -> Optional[ReferenceDocument]:
        """Ensure the default reference book in the repository is ingested into KnowledgeDB."""
        book_path = settings.primary_reference_book_path
        if not book_path.exists():
            logger.warning("Primary reference book not found at %s", book_path)
            return None

        file_hash = self.compute_file_hash(book_path)
        existing = (
            db.query(ReferenceDocument)
            .filter(ReferenceDocument.file_hash == file_hash)
            .first()
        )
        if existing and existing.ingestion_status == "INDEXED" and existing.total_chunks > 0:
            if generate_embeddings and existing.embeddings_count == 0 and settings.has_valid_gemini_key:
                self.generate_missing_embeddings(db, existing.id)
            return existing

        return self.ingest_document(
            db=db,
            file_path=book_path,
            title="The Planet Construction Kit",
            author="Mark Rosenfelder",
            generate_embeddings=generate_embeddings,
        )

    def ingest_document(
        self,
        db: Session,
        file_path: Path,
        title: Optional[str] = None,
        author: Optional[str] = None,
        generate_embeddings: bool = False,
    ) -> ReferenceDocument:
        """Execute the 9-stage ingestion pipeline on a reference document."""
        if not file_path.exists():
            raise FileNotFoundError(f"Reference document not found: {file_path}")

        file_hash = self.compute_file_hash(file_path)
        doc_record = (
            db.query(ReferenceDocument)
            .filter(ReferenceDocument.file_hash == file_hash)
            .first()
        )
        if not doc_record:
            doc_record = ReferenceDocument(
                title=title or file_path.stem,
                author=author or "Unknown Author",
                file_path=str(file_path),
                file_hash=file_hash,
                ingestion_status="PROCESSING",
            )
            db.add(doc_record)
            db.commit()
            db.refresh(doc_record)
        else:
            doc_record.ingestion_status = "PROCESSING"
            doc_record.error_message = None
            # Clear old chunks if re-ingesting
            db.query(ReferenceChunk).filter(
                ReferenceChunk.document_id == doc_record.id
            ).delete()
            db.execute(
                text("DELETE FROM reference_chunks_fts WHERE document_id = :did"),
                {"did": doc_record.id},
            )
            db.commit()

        try:
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                pages_data = self._extract_pdf_pages(file_path)
            elif suffix in {".txt", ".md"}:
                raw_txt = file_path.read_text(encoding="utf-8", errors="replace")
                pages_data = [
                    {
                        "pdf_page": 1,
                        "book_page": 1,
                        "chapter": doc_record.title,
                        "section": "General",
                        "text": _clean_ocr_text(raw_txt),
                        "ocr_repaired": False,
                    }
                ]
            else:
                raise ValueError(f"Unsupported reference document format: {suffix}")

            doc_record.total_pages = len(pages_data)
            chunks_created = self._chunk_and_index_pages(db, doc_record, pages_data)
            doc_record.total_chunks = chunks_created
            doc_record.ingestion_status = "INDEXED"
            db.commit()

            if generate_embeddings and settings.has_valid_gemini_key:
                self.generate_missing_embeddings(db, doc_record.id)

            db.refresh(doc_record)
            return doc_record
        except Exception as exc:
            logger.exception("Failed ingesting reference document %s", file_path)
            doc_record.ingestion_status = "FAILED"
            doc_record.error_message = str(exc)
            db.commit()
            raise

    def _extract_pdf_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text, printed book page numbers, chapters, and sections from a PDF."""
        doc = fitz.open(str(pdf_path))
        is_rosenfelder = "planet construction kit" in pdf_path.name.lower()
        has_tesseract = shutil.which("tesseract") is not None

        pages_data: List[Dict[str, Any]] = []
        current_section = "Overview"

        for page_idx in range(len(doc)):
            pdf_page_num = page_idx + 1
            page = doc[page_idx]
            raw_text = page.get_text("text") or ""
            ocr_repaired = False

            # Adaptive OCR fallback if a scanned page has practically no embedded text
            if len(raw_text.strip()) < 25 and has_tesseract and pdf_page_num > 6:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    proc = subprocess.run(
                        ["tesseract", "stdin", "stdout", "--oem", "1", "-l", "eng"],
                        input=img_bytes,
                        capture_output=True,
                        timeout=15,
                        check=False,
                    )
                    ocr_txt = proc.stdout.decode("utf-8", errors="replace").strip()
                    if len(ocr_txt) > len(raw_text.strip()):
                        raw_text = ocr_txt
                        ocr_repaired = True
                except Exception as ocr_exc:
                    logger.debug("Tesseract fallback skipped on page %d: %s", pdf_page_num, ocr_exc)

            cleaned = _clean_ocr_text(raw_text)
            if not cleaned:
                continue

            # Determine printed book page number (in Rosenfelder's scan, book page 1 starts at PDF page 9)
            if is_rosenfelder:
                book_page = max(1, pdf_page_num - 8)
                chapter_title = self._lookup_rosenfelder_chapter(book_page)
            else:
                book_page = pdf_page_num
                chapter_title = "Reference Text"

            # Detect subsection headers inside the page text
            detected_sec = self._detect_section_header(cleaned, current_section)
            if detected_sec:
                current_section = detected_sec

            pages_data.append(
                {
                    "pdf_page": pdf_page_num,
                    "book_page": book_page,
                    "chapter": chapter_title,
                    "section": current_section,
                    "text": cleaned,
                    "ocr_repaired": ocr_repaired,
                }
            )

        doc.close()
        return pages_data

    @staticmethod
    def _lookup_rosenfelder_chapter(book_page: int) -> str:
        current = "Front Matter"
        for start_page, chap_name in ROSENFELDER_CHAPTER_MAP:
            if book_page >= start_page:
                current = chap_name
            else:
                break
        return current

    @staticmethod
    def _detect_section_header(page_text: str, default_section: str) -> str:
        """Identify short title-case or standalone heading lines inside a page, skipping running headers."""
        lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
        for ln in lines[:10]:
            # Skip running page numbers or all-caps running headers like "BIOLOGY 97" or "124 CULTURE"
            if ln.isdigit() or len(ln) < 4 or len(ln) > 60:
                continue
            if re.match(r"^(\d+\s+[A-Z\s-]+|[A-Z\s-]+\s+\d+)$", ln):
                continue
            if ln.isupper():
                continue
            if (
                not ln.endswith((".", ",", ";", ":", "?", "!"))
                and ln[0].isupper()
                and not any(ch.isdigit() for ch in ln)
                and len(ln.split()) <= 6
            ):
                return ln
        return default_section

    def _chunk_and_index_pages(
        self,
        db: Session,
        doc_record: ReferenceDocument,
        pages_data: List[Dict[str, Any]],
    ) -> int:
        """Create parent-child semantic chunks and index child passages in SQLite FTS5."""
        total_child_chunks = 0

        # Group contiguous pages by chapter to build coherent parent sections & child passages
        idx = 0
        while idx < len(pages_data):
            start_item = pages_data[idx]
            chapter = start_item["chapter"]
            section = start_item["section"]
            pdf_start = start_item["pdf_page"]
            book_start = start_item["book_page"]

            combined_paragraphs: List[Tuple[str, int, int, str, bool]] = []
            # Accumulate up to 3 pages per parent block within the same chapter
            end_idx = idx
            while (
                end_idx < len(pages_data)
                and pages_data[end_idx]["chapter"] == chapter
                and (end_idx - idx) < 3
            ):
                p_item = pages_data[end_idx]
                for para in p_item["text"].split("\n\n"):
                    p_clean = para.strip()
                    if len(p_clean) >= 20:
                        combined_paragraphs.append(
                            (
                                p_clean,
                                p_item["pdf_page"],
                                p_item["book_page"],
                                p_item["section"],
                                p_item["ocr_repaired"],
                            )
                        )
                end_idx += 1

            if not combined_paragraphs:
                idx = end_idx
                continue

            parent_text = "\n\n".join(p[0] for p in combined_paragraphs)
            pdf_end = combined_paragraphs[-1][1]
            book_end = combined_paragraphs[-1][2]

            parent_chunk = ReferenceChunk(
                document_id=doc_record.id,
                parent_chunk_id=None,
                chunk_level="PARENT_SECTION",
                chapter_title=chapter,
                section_title=section,
                pdf_page_start=pdf_start,
                pdf_page_end=pdf_end,
                book_page_start=book_start,
                book_page_end=book_end,
                content=parent_text,
                token_count=_estimate_tokens(parent_text),
                ocr_repaired=any(p[4] for p in combined_paragraphs),
            )
            db.add(parent_chunk)
            db.flush()

            # Build child passages (~300-450 tokens) with 1-paragraph overlap
            cur_paras: List[Tuple[str, int, int, str, bool]] = []
            cur_tokens = 0

            for p_tuple in combined_paragraphs:
                p_tokens = _estimate_tokens(p_tuple[0])
                if cur_paras and (cur_tokens + p_tokens) > 420:
                    self._save_child_chunk(db, doc_record.id, parent_chunk.id, chapter, cur_paras)
                    total_child_chunks += 1
                    # Keep last paragraph as semantic overlap if reasonable length
                    overlap = [cur_paras[-1]] if _estimate_tokens(cur_paras[-1][0]) < 160 else []
                    cur_paras = overlap + [p_tuple]
                    cur_tokens = sum(_estimate_tokens(x[0]) for x in cur_paras)
                else:
                    cur_paras.append(p_tuple)
                    cur_tokens += p_tokens

            if cur_paras:
                self._save_child_chunk(db, doc_record.id, parent_chunk.id, chapter, cur_paras)
                total_child_chunks += 1

            idx = end_idx

        db.commit()
        return total_child_chunks

    def _save_child_chunk(
        self,
        db: Session,
        document_id: str,
        parent_id: str,
        chapter: str,
        paras: List[Tuple[str, int, int, str, bool]],
    ) -> ReferenceChunk:
        content = "\n\n".join(p[0] for p in paras)
        pdf_start = paras[0][1]
        pdf_end = paras[-1][1]
        book_start = paras[0][2]
        book_end = paras[-1][2]
        sec_title = paras[0][3]
        ocr_rep = any(p[4] for p in paras)

        child = ReferenceChunk(
            document_id=document_id,
            parent_chunk_id=parent_id,
            chunk_level="CHILD_PASSAGE",
            chapter_title=chapter,
            section_title=sec_title,
            pdf_page_start=pdf_start,
            pdf_page_end=pdf_end,
            book_page_start=book_start,
            book_page_end=book_end,
            content=content,
            token_count=_estimate_tokens(content),
            ocr_repaired=ocr_rep,
        )
        db.add(child)
        db.flush()

        db.execute(
            text(
                """
                INSERT INTO reference_chunks_fts (chunk_id, document_id, chapter_title, section_title, content)
                VALUES (:cid, :did, :chap, :sec, :content)
                """
            ),
            {
                "cid": child.id,
                "did": document_id,
                "chap": chapter,
                "sec": sec_title,
                "content": content,
            },
        )
        return child

    def generate_missing_embeddings(
        self, db: Session, document_id: str, max_chunks: int = 1500
    ) -> int:
        """Generate Gemini embeddings for child chunks that do not yet have vectors."""
        if not settings.has_valid_gemini_key:
            return 0
        unembedded = (
            db.query(ReferenceChunk)
            .filter(
                ReferenceChunk.document_id == document_id,
                ReferenceChunk.chunk_level == "CHILD_PASSAGE",
                ReferenceChunk.embedding_json.is_(None),
            )
            .limit(max_chunks)
            .all()
        )
        if not unembedded:
            return 0

        texts_to_embed = [
            f"[Book Chapter: {c.chapter_title} | Section: {c.section_title}]\n{c.content}"
            for c in unembedded
        ]
        try:
            vectors = gemini_service.embed_texts(
                texts_to_embed, task_type="RETRIEVAL_DOCUMENT", batch_size=50
            )
            for chunk_obj, vec in zip(unembedded, vectors):
                chunk_obj.embedding_json = json.dumps(vec)

            doc_obj = (
                db.query(ReferenceDocument)
                .filter(ReferenceDocument.id == document_id)
                .first()
            )
            if doc_obj:
                doc_obj.embeddings_count = (
                    db.query(ReferenceChunk)
                    .filter(
                        ReferenceChunk.document_id == document_id,
                        ReferenceChunk.embedding_json.is_not(None),
                    )
                    .count()
                )
            db.commit()
            return len(vectors)
        except GeminiServiceError as exc:
            logger.warning("Embedding generation deferred or failed: %s", exc.message)
            return 0

    # -------------------------------------------------------------------------
    # Hybrid Retrieval & Deterministic Coverage Evaluation (Step 1 & Step 2)
    # -------------------------------------------------------------------------

    def retrieve_passages(
        self,
        db: Session,
        queries: List[str],
        active_domains: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Retrieve relevant reference book passages using Hybrid BM25 + Vector + Reranking.

        Also evaluates deterministic Book Coverage (HIGH, PARTIAL, LOW) per Step 1 Section 4.3
        and Step 2 Section 6.1 (Guard 2: suppresses low-relevance chunks).
        """
        # Guard: Known intentional book gap in Rosenfelder's Planet Construction Kit
        # (Constructed languages / phonology / morphosyntax are in The Language Construction Kit)
        if active_domains and set(active_domains) == {"linguistics"}:
            combined_q = " ".join(queries).lower()
            conlang_markers = {
                "ergative", "absolutive", "phonology", "phonotactic", "allomorph",
                "morpheme", "vowel", "consonant", "fricative", "conlang", "grammar",
                "inflection", "agglutinative", "polysynthetic", "ipa", "syntax",
            }
            if any(m in combined_q for m in conlang_markers):
                return {
                    "coverage_status": "LOW",
                    "max_score": 0.15,
                    "passages": [],
                    "coverage_reason": (
                        "Mark Rosenfelder explicitly excludes constructed languages from "
                        "'The Planet Construction Kit' (page 10), deferring linguistics to "
                        "'The Language Construction Kit'."
                    ),
                }

        keywords = self._extract_query_terms(queries)
        if not keywords:
            return {
                "coverage_status": "LOW",
                "max_score": 0.0,
                "passages": [],
                "coverage_reason": "No searchable query terms.",
            }

        # 1. BM25 Full-Text Search via SQLite FTS5
        fts_candidates: Dict[str, float] = {}
        fts_query_str = " OR ".join(f'"{kw}"' for kw in keywords[:12])
        try:
            rows = db.execute(
                text(
                    """
                    SELECT chunk_id, bm25(reference_chunks_fts, 2.5, 2.0, 1.0) AS rank_score
                    FROM reference_chunks_fts
                    WHERE reference_chunks_fts MATCH :q
                    ORDER BY rank_score ASC
                    LIMIT 30
                    """
                ),
                {"q": fts_query_str},
            ).fetchall()
            for rank_idx, row in enumerate(rows):
                # RRF score component
                fts_candidates[row[0]] = 1.0 / (15 + rank_idx)
        except Exception as exc:
            logger.debug("FTS5 query fallback: %s", exc)

        # 2. Vector Cosine Similarity (if chunk embeddings exist and Gemini API key is configured)
        vec_candidates: Dict[str, float] = {}
        embedded_count = (
            db.query(ReferenceChunk)
            .filter(
                ReferenceChunk.chunk_level == "CHILD_PASSAGE",
                ReferenceChunk.embedding_json.is_not(None),
            )
            .count()
        )
        if embedded_count > 0 and settings.has_valid_gemini_key:
            try:
                q_vec = np.array(gemini_service.embed_query(" ".join(queries)), dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    all_embedded = (
                        db.query(ReferenceChunk.id, ReferenceChunk.embedding_json)
                        .filter(
                            ReferenceChunk.chunk_level == "CHILD_PASSAGE",
                            ReferenceChunk.embedding_json.is_not(None),
                        )
                        .all()
                    )
                    scored_vecs: List[Tuple[str, float]] = []
                    for cid, emb_json in all_embedded:
                        c_vec = np.array(json.loads(emb_json), dtype=np.float32)
                        denom = q_norm * np.linalg.norm(c_vec)
                        if denom > 0:
                            sim = float(np.dot(q_vec, c_vec) / denom)
                            scored_vecs.append((cid, sim))
                    scored_vecs.sort(key=lambda x: x[1], reverse=True)
                    for rank_idx, (cid, sim) in enumerate(scored_vecs[:30]):
                        if sim >= 0.45:
                            vec_candidates[cid] = (1.0 / (15 + rank_idx)) + (sim * 0.5)
            except Exception as exc:
                logger.debug("Vector query skipped, using BM25 + lexical reranker: %s", exc)

        candidate_ids = set(fts_candidates.keys()) | set(vec_candidates.keys())
        if not candidate_ids:
            return {
                "coverage_status": "LOW",
                "max_score": 0.0,
                "passages": [],
                "coverage_reason": "No matching passages found in the reference book.",
            }

        chunks = (
            db.query(ReferenceChunk)
            .filter(ReferenceChunk.id.in_(candidate_ids))
            .all()
        )
        doc_titles: Dict[str, str] = {
            d.id: d.title for d in db.query(ReferenceDocument).all()
        }

        # 3. Rerank candidates using combined RRF + query term coverage + header match
        scored_passages: List[Dict[str, Any]] = []
        kw_set = set(keywords)

        for chunk in chunks:
            content_lower = chunk.content.lower()
            header_lower = f"{chunk.chapter_title or ''} {chunk.section_title or ''}".lower()

            matched_kws = [kw for kw in kw_set if kw in content_lower or kw in header_lower]
            header_hits = [kw for kw in kw_set if kw in header_lower]

            lexical_coverage = len(matched_kws) / max(1, min(len(kw_set), 5))
            header_bonus = min(0.25, len(header_hits) * 0.12)
            rrf_component = fts_candidates.get(chunk.id, 0.0) * 4.0 + vec_candidates.get(chunk.id, 0.0) * 3.0

            rerank_score = min(1.0, (lexical_coverage * 0.55) + header_bonus + rrf_component)

            book_title = doc_titles.get(chunk.document_id, "The Planet Construction Kit")
            citation_id = f"Book-p{chunk.book_page_start}"
            citation_label = (
                f'{book_title}, "{chunk.chapter_title} — {chunk.section_title}", '
                f"p. {chunk.book_page_start} (PDF p. {chunk.pdf_page_start})"
            )

            scored_passages.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "book_title": book_title,
                    "chapter_title": chunk.chapter_title,
                    "section_title": chunk.section_title,
                    "book_page_start": chunk.book_page_start,
                    "book_page_end": chunk.book_page_end,
                    "pdf_page_start": chunk.pdf_page_start,
                    "pdf_page_end": chunk.pdf_page_end,
                    "content": chunk.content,
                    "rerank_score": round(rerank_score, 3),
                    "citation_id": citation_id,
                    "citation_label": citation_label,
                }
            )

        scored_passages.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Guard 2: Filter out low-relevance chunks (< 0.40) so we never force irrelevant book citations
        relevant_passages = [p for p in scored_passages if p["rerank_score"] >= 0.40][:top_k]
        max_score = scored_passages[0]["rerank_score"] if scored_passages else 0.0
        strong_count = sum(1 for p in relevant_passages if p["rerank_score"] >= 0.50)

        if max_score >= 0.65 and strong_count >= 2:
            coverage_status = "HIGH"
            coverage_reason = "Reference book provides direct multi-passage coverage."
        elif max_score >= 0.40 and relevant_passages:
            coverage_status = "PARTIAL"
            coverage_reason = "Reference book covers foundational aspects; supplementary research may add detail."
        else:
            coverage_status = "LOW"
            relevant_passages = []
            coverage_reason = "Topic is not adequately covered by the primary reference book."

        return {
            "coverage_status": coverage_status,
            "max_score": max_score,
            "passages": relevant_passages,
            "coverage_reason": coverage_reason,
        }

    @staticmethod
    def _extract_query_terms(queries: List[str]) -> List[str]:
        terms: List[str] = []
        seen = set()
        for q in queries:
            words = re.findall(r"[a-zA-Z]{3,}", q.lower())
            for w in words:
                if w not in STOPWORDS and w not in seen:
                    seen.add(w)
                    terms.append(w)
        return terms


book_ingestion_service = BookIngestionService()
