import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 150))
    top_k: int = int(os.getenv("TOP_K", 4))
    # If best retrieved chunk scores below this, we refuse instead of guessing.
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", 0.25))
    # Answers whose sentences are weakly supported by context get flagged.
    groundedness_threshold: float = float(os.getenv("GROUNDEDNESS_THRESHOLD", 0.45))
    storage_dir: Path = ROOT / "storage"
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    llm_model: str = os.getenv("LLM_MODEL", "")


settings = Settings()
