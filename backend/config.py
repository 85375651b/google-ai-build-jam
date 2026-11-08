"""Configuration management loading from local .env file (Claude-powered)."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

PLACEHOLDERS = {
    "your_api_key_here",
    "your_claude_api_key_here",
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
            load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)

        self.claude_api_key: str = _clean_env_val(
            os.environ.get("CLAUDE_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.claude_model: str = (
            _clean_env_val(os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"))
            or "claude-sonnet-5-5"
        )
        # Keep aliases for compatibility across routers/tests
        self.gemini_api_key: str = ""
        self.gemini_model: str = self.claude_model
        self.gemini_embedding_model: str = "fts5-bm25"

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
    def has_valid_claude_key(self) -> bool:
        return bool(self.claude_api_key) and self.claude_api_key.lower() not in PLACEHOLDERS

    @property
    def has_valid_gemini_key(self) -> bool:
        # Maintained for internal status compatibility; false since we use Claude exclusively
        return False

    @property
    def has_any_valid_llm_key(self) -> bool:
        return self.has_valid_claude_key

    @property
    def has_valid_tavily_key(self) -> bool:
        return bool(self.tavily_api_key) and self.tavily_api_key.lower() not in PLACEHOLDERS

    def update_model(self, model_name: str, embedding_model: Optional[str] = None) -> None:
        self.claude_model = model_name.strip()
        self.gemini_model = self.claude_model
        os.environ["CLAUDE_MODEL"] = self.claude_model


settings = Settings()
