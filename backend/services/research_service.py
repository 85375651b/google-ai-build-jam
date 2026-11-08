"""Supplementary online research service integrating Tavily Search API."""

import logging
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
import httpx
from sqlalchemy.orm import Session
from backend.config import settings
from backend.models import ResearchSource

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

ACADEMIC_DOMAINS = (
    ".edu",
    ".ac.uk",
    ".ac.",
    "plato.stanford.edu",
    "jstor.org",
    "cambridge.org",
    "oxford",
    "harvard.edu",
    "mit.edu",
    "nature.com",
    "science.org",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
)

ENCYCLOPEDIC_DOMAINS = (
    "wikipedia.org",
    "britannica.com",
    "worldhistory.org",
    "wals.info",
    "glottolog.org",
    "ethnologue.com",
    "smithsonianmag.com",
    ".gov",
)


def _classify_credibility_tier(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower().removeprefix("www.")
    if any(marker in domain for marker in ACADEMIC_DOMAINS):
        return domain, "ACADEMIC_PEER_REVIEWED"
    if any(marker in domain for marker in ENCYCLOPEDIC_DOMAINS):
        return domain, "INSTITUTIONAL_ENCYCLOPEDIC"
    return domain or "web", "GENERAL_REFERENCE"


class ResearchService:
    """Executes supplementary online research via Tavily when book coverage is PARTIAL or LOW."""

    def search_online(
        self,
        db: Session,
        queries: List[str],
        max_results: int = 4,
    ) -> Dict[str, Any]:
        """Search credible online sources via Tavily and persist citations in research_sources.

        Never fabricates results if TAVILY_API_KEY is missing or if network calls fail.
        """
        settings.reload()
        if not settings.has_valid_tavily_key:
            return {
                "status": "UNCONFIGURED",
                "message": "Tavily API key not configured in .env; skipping online research.",
                "sources": [],
            }

        if not queries:
            return {"status": "SKIPPED", "message": "No search queries provided.", "sources": []}

        primary_query = queries[0]
        try:
            payload = {
                "api_key": settings.tavily_api_key,
                "query": primary_query,
                "search_depth": "basic",
                "include_answer": False,
                "max_results": max_results,
            }
            with httpx.Client(timeout=12.0) as client:
                resp = client.post(TAVILY_SEARCH_URL, json=payload)

            if resp.status_code != 200:
                logger.warning("Tavily API returned status %d: %s", resp.status_code, resp.text)
                return {
                    "status": "ERROR",
                    "message": f"Tavily API returned HTTP {resp.status_code}.",
                    "sources": [],
                }

            data = resp.json()
            raw_results = data.get("results", [])
            verified_sources: List[Dict[str, Any]] = []

            for idx, item in enumerate(raw_results, start=1):
                url = (item.get("url") or "").strip()
                title = (item.get("title") or "Untitled Source").strip()
                excerpt = (item.get("content") or "").strip()
                if not url or not excerpt:
                    continue

                domain, cred_tier = _classify_credibility_tier(url)

                # Upsert in research_sources table (separate from fictional canon)
                existing = (
                    db.query(ResearchSource).filter(ResearchSource.url == url).first()
                )
                if not existing:
                    existing = ResearchSource(
                        url=url,
                        title=title[:300],
                        domain=domain[:150],
                        source_credibility_tier=cred_tier,
                        extracted_excerpt=excerpt[:2000],
                        consensus_status="CONSENSUS",
                    )
                    db.add(existing)
                    db.flush()

                citation_id = f"Web-{idx}"
                verified_sources.append(
                    {
                        "source_id": existing.id,
                        "citation_id": citation_id,
                        "title": existing.title,
                        "url": existing.url,
                        "domain": existing.domain,
                        "credibility_tier": existing.source_credibility_tier,
                        "excerpt": existing.extracted_excerpt,
                        "citation_label": f"{existing.title} ({existing.domain})",
                    }
                )

            db.commit()
            # Sort so ACADEMIC_PEER_REVIEWED and INSTITUTIONAL_ENCYCLOPEDIC appear first
            tier_Order = {
                "ACADEMIC_PEER_REVIEWED": 0,
                "INSTITUTIONAL_ENCYCLOPEDIC": 1,
                "GENERAL_REFERENCE": 2,
            }
            verified_sources.sort(key=lambda s: tier_Order.get(s["credibility_tier"], 3))
            return {
                "status": "OK",
                "message": f"Retrieved {len(verified_sources)} online sources via Tavily.",
                "sources": verified_sources,
            }
        except Exception as exc:
            logger.warning("Online research via Tavily failed: %s", exc)
            return {
                "status": "ERROR",
                "message": f"Online research request failed: {exc}",
                "sources": [],
            }


research_service = ResearchService()
