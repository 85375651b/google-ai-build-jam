"""Ollama inference for consultation, structured extraction, and summaries."""

import asyncio
import json
import logging
from typing import Any, Dict, Generator, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from backend.config import settings
from backend.services.generation_control import current_generation

T = TypeVar("T", bound=BaseModel)


class LocalModelError(Exception):
    def __init__(self, message: str, status_code: int = 502, error_code: str = "LOCAL_MODEL_ERROR"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class LocalModelService:
    """Use a single configured local model, without hosted-provider fallbacks."""

    def _client(self, timeout: Optional[float] = None) -> httpx.Client:
        return httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(timeout or settings.local_request_timeout, connect=5.0),
            trust_env=False,
        )

    @staticmethod
    def _connection_error(exc: httpx.HTTPError) -> LocalModelError:
        if isinstance(exc, httpx.TimeoutException):
            return LocalModelError(
                "Local inference timed out. Try a smaller model or increase LOCAL_REQUEST_TIMEOUT.",
                504, "LOCAL_MODEL_TIMEOUT",
            )
        return LocalModelError(
            "Cannot reach Ollama. Start `ollama serve` and check OLLAMA_BASE_URL.",
            503, "LOCAL_MODEL_UNAVAILABLE",
        )

    @staticmethod
    def _check_response(response: httpx.Response, model: str) -> None:
        if response.is_success:
            return
        if response.status_code == 404:
            raise LocalModelError(
                f"Local model '{model}' is unavailable. Download it with `ollama pull {model}`.",
                503, "LOCAL_MODEL_NOT_FOUND",
            )
        raise LocalModelError(
            f"Ollama rejected the request (HTTP {response.status_code}). "
            "Check the Ollama logs, model compatibility, and available memory.",
        )

    def get_status(self) -> Dict[str, Any]:
        """Inspect installed models without loading them into memory."""
        settings.reload()
        try:
            with self._client(timeout=2.0) as client:
                response = client.get("/api/tags")
            self._check_response(response, settings.local_model)
            models = [entry["name"] for entry in response.json()["models"]]
            selected = settings.local_model
            ready = selected in models or (":" not in selected and f"{selected}:latest" in models)
            return {
                "ready": ready, "available_models": models,
                "message": "Ready" if ready else f"Download the model: ollama pull {selected}",
            }
        except httpx.HTTPError as exc:
            return {"ready": False, "available_models": [], "message": self._connection_error(exc).message}
        except (LocalModelError, ValueError, KeyError, TypeError):
            return {"ready": False, "available_models": [], "message": "Ollama returned an invalid status response."}

    def _payload(
        self, contents: Any, system_instruction: Optional[str], temperature: float,
        max_output_tokens: Optional[int], model_override: Optional[str], stream: bool,
    ) -> Dict[str, Any]:
        settings.reload()
        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        else:
            messages.extend({"role": m["role"], "content": m["content"]} for m in contents)
        limit = min(max_output_tokens or settings.local_max_output_tokens, settings.local_max_output_tokens)
        # Conservative heuristic, not a tokenizer. Avoid deliberately dropping canon
        # or the latest author input to fit a smaller local context window.
        estimated_tokens = sum(len(m["content"].encode("utf-8")) // 3 + 8 for m in messages)
        if estimated_tokens + limit > settings.local_context_window:
            raise LocalModelError(
                "Input exceeds the estimated local context budget. Shorten the conversation or "
                "input, or increase LOCAL_CONTEXT_WINDOW within your model and hardware limits.",
                413, "LOCAL_CONTEXT_LIMIT",
            )
        model = model_override or settings.local_model
        logging.getLogger(__name__).info("local_route model=%s context=%s output=%s", model, settings.local_context_window, limit)
        payload = {
            "model": model, "messages": messages, "stream": stream,
            "options": {"temperature": temperature, "num_ctx": settings.local_context_window, "num_predict": limit},
            "keep_alive": settings.local_keep_alive,
        }
        # The installed gpt-oss model exposes low/medium/high rather than a boolean.
        if model.split(":")[0] == "gpt-oss":
            payload["think"] = "low"
        return payload

    @staticmethod
    def _record_metrics(data):
        """Record runtime counters only; never prompts, output, or credentials."""
        if not data.get("done"):
            return
        control = current_generation.get()
        metrics = {key: data.get(key) for key in (
            "prompt_eval_count", "prompt_eval_duration", "eval_count",
            "eval_duration", "total_duration", "load_duration")}
        metrics["request_id"] = getattr(control, "request_id", None)
        metrics["stage"] = getattr(control, "state", None)
        logging.getLogger(__name__).info("ollama_metrics %s", json.dumps(metrics))

    @staticmethod
    def _validate_event(data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict) or data.get("error"):
            raise LocalModelError("Ollama failed during generation. Check its logs and available memory.")
        message = data.get("message", {})
        if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
            raise LocalModelError("Ollama returned an invalid message.")
        if data.get("done_reason") == "length":
            raise LocalModelError(
                "Local inference reached its output limit. Increase LOCAL_MAX_OUTPUT_TOKENS "
                "or request a shorter response.", 502, "LOCAL_OUTPUT_LIMIT",
            )
        return message

    def _generate(self, payload: Dict[str, Any]) -> str:
        if control := current_generation.get():
            return control.wait_for(self._generate_async(payload))
        try:
            with self._client() as client:
                response = client.post("/api/chat", json=payload)
            self._check_response(response, payload["model"])
            data = response.json()
            self._record_metrics(data)
            message = self._validate_event(data)
            text = message.get("content", "").strip()
            if not data.get("done") or not text:
                raise LocalModelError("The local model returned an empty or incomplete response.")
            return text
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc
        except (ValueError, TypeError) as exc:
            raise LocalModelError("Ollama returned malformed JSON.") from exc

    def generate_text(
        self, contents: Any, system_instruction: Optional[str] = None,
        temperature: float = 0.4, max_output_tokens: Optional[int] = None,
        model_override: Optional[str] = None, task: str = "consultation",
    ) -> str:
        settings.reload()
        selected = model_override or settings.local_task_models.get(task) or settings.local_model
        return self._generate(self._payload(contents, system_instruction, temperature, max_output_tokens, selected, False))

    def generate_structured(
        self, prompt: str, response_schema: Type[T], system_instruction: Optional[str] = None,
        temperature: float = 0.1, model_override: Optional[str] = None, task: str = "extraction",
    ) -> T:
        settings.reload()
        selected = model_override or settings.local_task_models.get(task) or settings.local_model
        schema = response_schema.model_json_schema()
        instruction = (system_instruction or "") + "\nReturn only JSON matching this schema:\n" + json.dumps(schema)
        payload = self._payload(prompt, instruction, temperature, None, selected, False)
        payload["format"] = schema
        try:
            text = self._generate(payload)
            return response_schema.model_validate_json(text)
        except (ValidationError, LocalModelError) as exc:
            if selected != settings.local_model and model_override is None:
                logging.getLogger(__name__).warning("local_route_fallback task=%s model=%s", task, selected)
                return self.generate_structured(prompt, response_schema, system_instruction,
                    temperature, model_override=settings.local_model, task=task)
            if isinstance(exc, LocalModelError):
                raise
            raise LocalModelError(
                "The local model returned invalid structured data. Try a more capable model.",
                502, "LOCAL_STRUCTURED_OUTPUT_INVALID",
            ) from exc

    def stream_text(
        self, contents: Any, system_instruction: Optional[str] = None,
        temperature: float = 0.4, max_output_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> Generator[str, None, None]:
        payload = self._payload(contents, system_instruction, temperature, max_output_tokens, model_override, True)
        if control := current_generation.get():
            iterator = self._stream_async(payload)
            try:
                while True:
                    try:
                        yield control.wait_for(iterator.__anext__())
                    except StopAsyncIteration:
                        return
            finally:
                # Also close the generator if the consumer stops at a cancellation boundary.
                future = asyncio.run_coroutine_threadsafe(self._close_iterator(iterator), control.loop)
                future.result(timeout=10)
            return
        yielded_any = False
        completed = False
        try:
            with self._client() as client:
                with client.stream("POST", "/api/chat", json=payload) as response:
                    self._check_response(response, payload["model"])
                    for line in response.iter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        self._record_metrics(data)
                        message = self._validate_event(data)
                        text = message.get("content", "")
                        if text:
                            yielded_any = True
                            yield text
                        if data.get("done"):
                            completed = True
                            break
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc
        except (ValueError, TypeError) as exc:
            raise LocalModelError("Ollama returned malformed streaming data.") from exc
        if not completed or not yielded_any:
            raise LocalModelError("The local model returned an empty or interrupted stream.")


    @staticmethod
    async def _close_iterator(iterator):
        # Cancellation of the cross-thread Future can precede async context cleanup.
        for _ in range(1000):
            try:
                await iterator.aclose()
                return
            except RuntimeError as exc:
                if "already running" not in str(exc):
                    raise
                await asyncio.sleep(0.005)
        raise LocalModelError("The model connection did not close after cancellation.")

    def _async_client(self):
        return httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(settings.local_request_timeout, connect=5.0),
            trust_env=False,
        )

    async def _stream_async(self, payload):
        # Stream even hidden structured/reasoning calls: network inactivity is then
        # measured between actual runtime chunks rather than the entire model call.
        payload = dict(payload, stream=True)
        yielded_any = False
        completed = False
        try:
            async with self._async_client() as client:
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    self._check_response(response, payload["model"])
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        self._record_metrics(data)
                        message = self._validate_event(data)
                        content = message.get("content", "")
                        if content:
                            yielded_any = True
                            yield content
                        if data.get("done"):
                            completed = True
                            break
        except httpx.HTTPError as exc:
            raise self._connection_error(exc) from exc
        except (ValueError, TypeError) as exc:
            raise LocalModelError("Ollama returned malformed streaming data.") from exc
        if not completed or not yielded_any:
            raise LocalModelError("The local model returned an empty or interrupted stream.")

    async def _generate_async(self, payload):
        return "".join([part async for part in self._stream_async(payload)]).strip()


llm_service = LocalModelService()
