"""Central configuration for the LangGraph Mobile Security Assistant.

All runtime choices are read from environment variables first, then fall back to
safe defaults. This removes hardcoded setup assumptions from the application
logic and makes the project easier to run on another machine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps config importable even before dependencies are installed.
    load_dotenv = None

if load_dotenv:
    load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    response_model: str
    guard_model: str
    embeddings_model: str
    memory_embeddings_model: str
    data_folder: str
    qdrant_path: str
    collection_name: str
    masvs_file: str
    memory_index_path: str
    chunk_size: int
    chunk_overlap: int
    cache_ttl_minutes: int
    cache_max_size: int
    compression_threshold: int
    enable_cache: bool
    enable_web_search: bool
    enable_langsmith: bool
    langsmith_endpoint: str
    langsmith_project: str
    langsmith_api_key: str
    tavily_api_key: str


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_config() -> AppConfig:
    base_dir = Path(os.getenv("APP_BASE_DIR", ".")).resolve()
    langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
    langsmith_project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "mobile-security-assistant")
    langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    return AppConfig(
        response_model=os.getenv("RESPONSE_MODEL", "mistral:latest"),
        guard_model=os.getenv("GUARD_MODEL", "llama3:latest"),
        embeddings_model=os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest"),
        memory_embeddings_model=os.getenv("MEMORY_EMBEDDINGS_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        data_folder=os.getenv("DATA_FOLDER", str(base_dir / "owasp_rag_data")),
        qdrant_path=os.getenv("QDRANT_PATH", str(base_dir / "qdrant_local")),
        collection_name=os.getenv("QDRANT_COLLECTION", "security_assistant"),
        masvs_file=os.getenv("MASVS_FILE", str(base_dir / "masvs.json")),
        memory_index_path=os.getenv("MEMORY_INDEX_PATH", str(base_dir / "faiss_memory_enhanced")),
        chunk_size=_get_int("CHUNK_SIZE", 900),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
        cache_ttl_minutes=_get_int("CACHE_TTL_MINUTES", 60),
        cache_max_size=_get_int("CACHE_MAX_SIZE", 100),
        compression_threshold=_get_int("MEMORY_COMPRESSION_THRESHOLD", 30),
        enable_cache=_get_bool("ENABLE_CACHE", True),
        enable_web_search=_get_bool("ENABLE_WEB_SEARCH", False),
        enable_langsmith=(
            _get_bool("ENABLE_LANGSMITH", False)
            or _get_bool("LANGSMITH_TRACING", False)
            or _get_bool("LANGCHAIN_TRACING_V2", False)
        ),
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_api_key=langsmith_key,
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
    )
