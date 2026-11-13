"""Configuration management loading from local .env file (local Ollama inference)."""

import os
from pathlib import Path
from dotenv import load_dotenv, set_key

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

PLACEHOLDERS = {
    "your_api_key_here",
    "your_tavily_api_key_here",
    "paste_your_key_here",
    "todo",
}


def _clean_env_val(raw: str) -> str:
    """Strip whitespace and accidental surrounding quotes from .env values."""
    val = (raw or "").strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
        val = val[1:-1].strip()
    return val


class Settings:
    """Application settings loaded dynamically from environment / .env."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Reload configuration from .env and environment variables."""
        if ENV_FILE_PATH.exists():
            load_dotenv(dotenv_path=ENV_FILE_PATH, override=False)

        self.local_model = (
            _clean_env_val(os.environ.get("LOCAL_MODEL", "gpt-oss:20b")) or "gpt-oss:20b"
        )
        self.local_task_models = {
            role: _clean_env_val(os.environ.get(f"LOCAL_{role.upper()}_MODEL", ""))
            for role in ("classification", "extraction", "summarization")
        }
        self.local_keep_alive = _clean_env_val(os.environ.get("LOCAL_KEEP_ALIVE", "10m")) or "10m"
        self.ollama_base_url = _clean_env_val(
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.local_context_window = max(2048, int(os.environ.get("LOCAL_CONTEXT_WINDOW", "32768")))
        self.local_max_output_tokens = max(256, min(
            int(os.environ.get("LOCAL_MAX_OUTPUT_TOKENS", "4096")),
            self.local_context_window // 2,
        ))
        self.local_request_timeout = max(1.0, float(os.environ.get("LOCAL_REQUEST_TIMEOUT", "600")))
        self.consultation_context_tokens = max(256, int(os.environ.get("CONSULTATION_CONTEXT_TOKENS", "12000")))
        self.extraction_context_tokens = max(256, int(os.environ.get("EXTRACTION_CONTEXT_TOKENS", "10000")))
        self.context_recent_messages = max(2, int(os.environ.get("CONTEXT_RECENT_MESSAGES", "6")))
        self.context_history_excerpts = max(0, int(os.environ.get("CONTEXT_HISTORY_EXCERPTS", "8")))

        # Overall duration is distinct from Ollama's per-chunk network timeout.
        self.generation_deadline_seconds = max(1.0, float(os.environ.get("GENERATION_DEADLINE_SECONDS", "3600")))
        self.generation_heartbeat_seconds = max(0.1, float(os.environ.get("GENERATION_HEARTBEAT_SECONDS", "5")))
        self.generation_client_idle_seconds = max(15.0, self.generation_heartbeat_seconds * 3,
                                                 float(os.environ.get("GENERATION_CLIENT_IDLE_SECONDS", "45")))

        self.tavily_api_key: str = _clean_env_val(os.environ.get("TAVILY_API_KEY", ""))

        default_db_path = PROJECT_ROOT / "data" / "worldbuilding.db"
        self.database_url: str = (
            _clean_env_val(os.environ.get("DATABASE_URL", ""))
            or f"sqlite:///{default_db_path}"
        )

        default_book = (
            PROJECT_ROOT
            / "The planet construction kit (Rosenfelder, Mark, author) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
        )
        raw_book_path = _clean_env_val(os.environ.get("PRIMARY_REFERENCE_BOOK_PATH", ""))
        if raw_book_path:
            self.primary_reference_book_path: Path = (PROJECT_ROOT / raw_book_path).resolve()
        else:
            self.primary_reference_book_path = default_book

    @property
    def has_valid_tavily_key(self) -> bool:
        return bool(self.tavily_api_key) and self.tavily_api_key.lower() not in PLACEHOLDERS

    def update_model(self, model_name: str) -> None:
        """Persist selection while preserving other settings and secrets."""
        model_name = model_name.strip()
        set_key(str(ENV_FILE_PATH), "LOCAL_MODEL", model_name)
        os.environ["LOCAL_MODEL"] = model_name
        self.local_model = model_name


settings = Settings()
