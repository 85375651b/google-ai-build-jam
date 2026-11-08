"""LLM service powered exclusively by Anthropic Claude API via CLAUDE_API_KEY in .env."""

import json
import logging
import re
from typing import Any, Dict, Generator, List, Optional, Type, TypeVar
import httpx
from pydantic import BaseModel
from backend.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
CANDIDATE_CLAUDE_MODELS = (
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-haiku-5-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-5-5",
)


class GeminiServiceError(Exception):
    """Base exception for LLM API errors."""

    def __init__(self, message: str, status_code: int = 500, error_code: str = "LLM_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class MissingAPIKeyError(GeminiServiceError):
    """Raised when CLAUDE_API_KEY / ANTHROPIC_API_KEY is missing."""

    def __init__(self) -> None:
        super().__init__(
            message=(
                "CLAUDE_API_KEY is not configured. Please add your Claude API key "
                "to the .env file in the project root (CLAUDE_API_KEY=your_key) and try again."
            ),
            status_code=401,
            error_code="MISSING_API_KEY",
        )


class InvalidAPIKeyError(GeminiServiceError):
    """Raised when the provided Claude API key is rejected."""

    def __init__(self, details: str = "") -> None:
        super().__init__(
            message=(
                "The configured CLAUDE_API_KEY was rejected as invalid or unauthorized. "
                f"Please check your .env file. {details}".strip()
            ),
            status_code=401,
            error_code="INVALID_API_KEY",
        )


class RateLimitError(GeminiServiceError):
    """Raised when the Claude API rate limit is exceeded."""

    def __init__(self, details: str = "") -> None:
        super().__init__(
            message=(
                "Claude API rate limit or quota reached. Please wait a moment and retry. "
                f"{details}".strip()
            ),
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
        )


class GeminiService:
    """Executes all structured, synchronous, and streaming LLM calls using Claude API."""

    def _classify_http_error(self, status_code: int, body_text: str) -> GeminiServiceError:
        if status_code in (401, 403):
            return InvalidAPIKeyError(f"HTTP {status_code}: {body_text}")
        if status_code == 429:
            return RateLimitError(f"HTTP 429: {body_text}")
        return GeminiServiceError(
            message=f"Claude API error (HTTP {status_code}): {body_text}",
            status_code=502,
            error_code="MODEL_API_ERROR",
        )

    @staticmethod
    def _contents_to_claude_messages(contents: Any) -> List[Dict[str, str]]:
        """Convert contents list or string into Anthropic messages format."""
        if isinstance(contents, str):
            return [{"role": "user", "content": contents}]

        messages: List[Dict[str, str]] = []
        for item in contents:
            if isinstance(item, dict):
                raw_role = item.get("role", "user")
                text_val = item.get("content", "")
            else:
                raw_role = getattr(item, "role", "user")
                parts = getattr(item, "parts", []) or []
                text_val = "\n".join(
                    getattr(p, "text", "") for p in parts if getattr(p, "text", None)
                ).strip()
            role = "assistant" if raw_role in ("model", "assistant") else "user"
            if not text_val:
                continue
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + text_val
            else:
                messages.append({"role": role, "content": text_val})

        if not messages:
            messages.append({"role": "user", "content": "Hello"})
        if messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "Continue consultation."})
        return messages

    def _get_candidate_models(self, model_override: Optional[str] = None) -> List[str]:
        primary = model_override or settings.claude_model or "claude-sonnet-5-5"
        models = [primary]
        for m in CANDIDATE_CLAUDE_MODELS:
            if m not in models:
                models.append(m)
        return models

    def _call_claude_sync(
        self,
        contents: Any,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 4096,
        model_override: Optional[str] = None,
    ) -> str:
        settings.reload()
        if not settings.has_valid_claude_key:
            raise MissingAPIKeyError()

        claude_messages = self._contents_to_claude_messages(contents)
        headers = {
            "x-api-key": settings.claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        last_status = 500
        last_body = ""
        for model_name in self._get_candidate_models(model_override):
            payload: Dict[str, Any] = {
                "model": model_name,
                "max_tokens": max_output_tokens,
                "messages": claude_messages,
            }
            if system_instruction:
                payload["system"] = system_instruction

            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(ANTHROPIC_MESSAGES_URL, headers=headers, json=payload)
            except Exception as exc:
                raise GeminiServiceError(f"Network error calling Claude API: {exc}", 502) from exc

            if resp.status_code == 200:
                data = resp.json()
                blocks = data.get("content", [])
                text_out = "".join(
                    b.get("text", "") for b in blocks if b.get("type") == "text"
                ).strip()
                if text_out:
                    if model_name != settings.claude_model:
                        settings.update_model(model_name)
                    return text_out
                raise GeminiServiceError("Claude returned an empty response.", status_code=502)

            last_status = resp.status_code
            last_body = resp.text
            if resp.status_code == 404:
                continue
            break

        raise self._classify_http_error(last_status, last_body)

    def _stream_claude(
        self,
        contents: Any,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 4096,
        model_override: Optional[str] = None,
    ) -> Generator[str, None, None]:
        settings.reload()
        if not settings.has_valid_claude_key:
            raise MissingAPIKeyError()

        claude_messages = self._contents_to_claude_messages(contents)
        headers = {
            "x-api-key": settings.claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        for model_name in self._get_candidate_models(model_override):
            payload: Dict[str, Any] = {
                "model": model_name,
                "max_tokens": max_output_tokens,
                "stream": True,
                "messages": claude_messages,
            }
            if system_instruction:
                payload["system"] = system_instruction

            yielded_any = False
            with httpx.Client(timeout=90.0) as client:
                with client.stream(
                    "POST", ANTHROPIC_MESSAGES_URL, headers=headers, json=payload
                ) as resp:
                    if resp.status_code == 404:
                        continue
                    if resp.status_code != 200:
                        err_body = resp.read().decode("utf-8", errors="replace")
                        raise self._classify_http_error(resp.status_code, err_body)

                    if model_name != settings.claude_model:
                        settings.update_model(model_name)

                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            ev = json.loads(data_str)
                        except Exception:
                            continue
                        if ev.get("type") == "content_block_delta":
                            delta = ev.get("delta", {})
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                yielded_any = True
                                yield delta["text"]
                    if yielded_any:
                        return

        raise GeminiServiceError("No compatible Claude model succeeded for streaming.", 502)

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        model_override: Optional[str] = None,
    ) -> T:
        """Execute a structured JSON generation call via Claude."""
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        claude_sys = (
            (system_instruction or "")
            + "\n\nCRITICAL: Respond ONLY with a single raw valid JSON object conforming to this JSON Schema. "
            "Do NOT wrap in markdown code fences or add any explanation outside the JSON:\n"
            + schema_json
        ).strip()
        raw_claude = self._call_claude_sync(
            contents=prompt,
            system_instruction=claude_sys,
            temperature=temperature,
            max_output_tokens=2048,
            model_override=model_override,
        )
        cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_claude.strip())
        # Extract first {...} block if model added surrounding text
        start_idx = cleaned_json.find("{")
        end_idx = cleaned_json.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned_json = cleaned_json[start_idx : end_idx + 1]
        data = json.loads(cleaned_json)
        return response_schema.model_validate(data)

    def generate_text(
        self,
        contents: Any,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 4096,
        model_override: Optional[str] = None,
    ) -> str:
        return self._call_claude_sync(
            contents=contents,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model_override=model_override,
        )

    def stream_text(
        self,
        contents: Any,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 4096,
        model_override: Optional[str] = None,
    ) -> Generator[str, None, None]:
        yield from self._stream_claude(
            contents=contents,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model_override=model_override,
        )

    def embed_texts(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 50,
    ) -> List[List[float]]:
        return []

    def embed_query(self, query: str) -> List[float]:
        return []


gemini_service = GeminiService()
