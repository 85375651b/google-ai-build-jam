"""Configuration management loading from local .env file."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings:
    """Application settings loaded dynamically from environment / .env."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Reload configuration from .env and environment variables."""
        if ENV_FILE_PATH.exists():
            load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)

        self.gemini_api_key: str = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        self.gemini_model: str = (
            os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
            or "gemini-2.5-flash"
        )
        self.gemini_embedding_model: str = (
            os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
            or "gemini-embedding-001"
        )
        self.tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "").strip()

        default_db_path = PROJECT_ROOT / "data" / "worldbuilding.db"
        self.database_url: str = (
            os.environ.get("DATABASE_URL", "").strip()
            or f"sqlite:///{default_db_path}"
        )

        default_book = (
            PROJECT_ROOT
            / "The planet construction kit (Rosenfelder, Mark, author) (z-library.sk, 1lib.sk, z-lib.sk).pdf"
        )
        raw_book_path = os.environ.get("PRIMARY_REFERENCE_BOOK_PATH", "").strip()
        if raw_book_path:
            resolved = (PROJECT_ROOT / raw_book_path).resolve()
            self.primary_reference_book_path: Path = resolved
        else:
            self.primary_reference_book_path = default_book

    @property
    def has_valid_gemini_key(self) -> bool:
        """Check whether a non-placeholder Gemini API key is configured."""
        if not self.gemini_api_key:
            return False
        placeholders = {
            "your_api_key_here",
            "your_gemini_api_key_here",
            "paste_your_key_here",
            "todo",
        }
        return self.gemini_api_key.lower() not in placeholders

    @property
    def has_valid_tavily_key(self) -> bool:
        """Check whether a non-placeholder Tavily API key is configured."""
        if not self.tavily_api_key:
            return False
        placeholders = {
            "your_api_key_here",
            "your_tavily_api_key_here",
            "paste_your_key_here",
            "todo",
        }
        return self.tavily_api_key.lower() not in placeholders

    def update_model(self, model_name: str, embedding_model: Optional[str] = None) -> None:
        """Update the active Gemini model at runtime and persist to .env if present."""
        self.gemini_model = model_name.strip()
        os.environ["GEMINI_MODEL"] = self.gemini_model
        if embedding_model:
            self.gemini_embedding_model = embedding_model.strip()
            os.environ["GEMINI_EMBEDDING_MODEL"] = self.gemini_embedding_model


settings = Settings()
