"""Dedicated Google Gemini API service using the official google-genai SDK."""

import json
import logging
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Type, TypeVar
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError
from backend.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiServiceError(Exception):
    """Base exception for Gemini API errors."""

    def __init__(self, message: str, status_code: int = 500, error_code: str = "GEMINI_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class MissingAPIKeyError(GeminiServiceError):
    """Raised when GEMINI_API_KEY is missing or still set to a placeholder."""

    def __init__(self) -> None:
        super().__init__(
            message=(
                "GEMINI_API_KEY is not configured. Please add your Google Gemini API key "
                "to the .env file in the project root (GEMINI_API_KEY=your_key) and try again."
            ),
            status_code=401,
            error_code="MISSING_API_KEY",
        )


class InvalidAPIKeyError(GeminiServiceError):
    """Raised when the provided Gemini API key is rejected by Google."""

    def __init__(self, details: str = "") -> None:
        super().__init__(
            message=(
                "The configured GEMINI_API_KEY was rejected as invalid or unauthorized. "
                f"Please check your .env file. {details}".strip()
            ),
            status_code=401,
            error_code="INVALID_API_KEY",
        )


class RateLimitError(GeminiServiceError):
    """Raised when the Gemini API rate limit or quota is exceeded."""

    def __init__(self, details: str = "") -> None:
        super().__init__(
            message=(
                "Gemini API rate limit or quota reached. Please wait a moment and retry. "
                f"{details}".strip()
            ),
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
        )


class GeminiService:
    """Encapsulates all communication with Google's Gemini API."""

    def __init__(self) -> None:
        self._client: Optional[genai.Client] = None
        self._cached_key: Optional[str] = None

    def _get_client(self) -> genai.Client:
        """Return an initialized genai.Client, reloading .env if needed."""
        settings.reload()
        if not settings.has_valid_gemini_key:
            raise MissingAPIKeyError()

        if self._client is None or self._cached_key != settings.gemini_api_key:
            self._client = genai.Client(api_key=settings.gemini_api_key)
            self._cached_key = settings.gemini_api_key
        return self._client

    def _classify_api_error(self, exc: Exception) -> GeminiServiceError:
        """Convert SDK/network exceptions into structured user-understandable errors."""
        if isinstance(exc, GeminiServiceError):
            return exc
        err_str = str(exc)
        lower = err_str.lower()
        if (
            "api_key_invalid" in lower
            or "api key not valid" in lower
            or "401" in lower
            or "403" in lower
            or "permission_denied" in lower
            or "unauthenticated" in lower
        ):
            return InvalidAPIKeyError(err_str)
        if "429" in lower or "resource_exhausted" in lower or "quota" in lower or "rate limit" in lower:
            return RateLimitError(err_str)
        if isinstance(exc, APIError):
            return GeminiServiceError(
                message=f"Gemini API error: {err_str}",
                status_code=502,
                error_code="MODEL_API_ERROR",
            )
        return GeminiServiceError(
            message=f"Failed communicating with Gemini API: {err_str}",
            status_code=502,
            error_code="NETWORK_OR_MODEL_ERROR",
        )

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        model_override: Optional[str] = None,
    ) -> T:
        """Execute a structured JSON generation call and validate against response_schema."""
        client = self._get_client()
        model_name = model_override or settings.gemini_model
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_schema,
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            raw_text = (response.text or "").strip()
            if not raw_text:
                raise GeminiServiceError(
                    "Gemini returned an empty structured response.",
                    status_code=502,
                    error_code="EMPTY_MODEL_RESPONSE",
                )
            data = json.loads(raw_text)
            return response_schema.model_validate(data)
        except Exception as exc:
            raise self._classify_api_error(exc) from exc

    def generate_text(
        self,
        contents: List[types.Content] | str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 4096,
        model_override: Optional[str] = None,
    ) -> str:
        """Execute a synchronous text generation call."""
        client = self._get_client()
        model_name = model_override or settings.gemini_model
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            text_out = (response.text or "").strip()
            if not text_out:
                raise GeminiServiceError(
                    "Gemini returned an empty response.",
                    status_code=502,
                    error_code="EMPTY_MODEL_RESPONSE",
                )
            return text_out
        except Exception as exc:
            raise self._classify_api_error(exc) from exc

    def stream_text(
        self,
        contents: List[types.Content] | str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 4096,
        model_override: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream text chunks from Gemini progressively."""
        client = self._get_client()
        model_name = model_override or settings.gemini_model
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if system_instruction:
                config.system_instruction = system_instruction

            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
            yielded_any = False
            for chunk in stream:
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
            if not yielded_any:
                raise GeminiServiceError(
                    "Gemini stream completed without producing any text.",
                    status_code=502,
                    error_code="EMPTY_MODEL_RESPONSE",
                )
        except Exception as exc:
            raise self._classify_api_error(exc) from exc

    def embed_texts(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 50,
    ) -> List[List[float]]:
        """Generate vector embeddings for a list of texts in batches."""
        if not texts:
            return []
        client = self._get_client()
        embed_model = settings.gemini_embedding_model
        all_embeddings: List[List[float]] = []
        try:
            for idx in range(0, len(texts), batch_size):
                batch = texts[idx : idx + batch_size]
                response = client.models.embed_content(
                    model=embed_model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                if not response.embeddings:
                    raise GeminiServiceError(
                        "Gemini returned empty embeddings.",
                        status_code=502,
                        error_code="EMPTY_EMBEDDING_RESPONSE",
                    )
                for emb in response.embeddings:
                    all_embeddings.append(list(emb.values))
            return all_embeddings
        except Exception as exc:
            raise self._classify_api_error(exc) from exc

    def embed_query(self, query: str) -> List[float]:
        """Generate a vector embedding for a single retrieval query."""
        vectors = self.embed_texts([query], task_type="RETRIEVAL_QUERY", batch_size=1)
        return vectors[0] if vectors else []


gemini_service = GeminiService()
