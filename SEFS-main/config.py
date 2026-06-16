"""Central configuration and utility helpers for SEFS."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

ROOT_DIR: Optional[Path] = None

RAW_SUBDIR = "_raw"
INTERNAL_DIR = ".sefs"
UNCATEGORIZED_DIR = "Uncategorized"
VECTOR_STORE_SUBDIR = "vector_store"
LOCK_FILE_NAME = "sefs.lock"
CONFIG_FILE_NAME = "config.yaml"
OVERRIDES_FILE_NAME = "overrides.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sefs_files"
MAX_TEXT_LENGTH = 10_000
MIN_TEXT_LENGTH = 50
SNIPPET_LENGTH = 200

HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"
PAIR_CLUSTER_SIMILARITY_THRESHOLD = 0.52
SIMILARITY_FALLBACK_THRESHOLD = 0.42
TFIDF_FALLBACK_THRESHOLD = 0.04
FORCED_PAIR_MIN_SIMILARITY = 0.30

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
REFRESH_INTERVAL_MS = 3000
DEFAULT_SEARCH_TOP_K = 5
MAX_SEARCH_TOP_K = 10
SEARCH_CANDIDATE_MULTIPLIER = 4
SEARCH_MAX_CANDIDATES = 30
SEARCH_SEMANTIC_WEIGHT = 0.55
SEARCH_LEXICAL_WEIGHT = 0.45
SEARCH_FALLBACK_SEMANTIC_WEIGHT = 0.20
SEARCH_FALLBACK_LEXICAL_WEIGHT = 0.80
SEARCH_MIN_COMBINED_SCORE = 0.12
SEARCH_MIN_LEXICAL_SCORE = 0.05
SEARCH_MIN_TOKEN_LENGTH = 3
SEARCH_QUERY_NOISE_TOKENS = {
    "about",
    "answer",
    "define",
    "definition",
    "describe",
    "explain",
    "find",
    "give",
    "how",
    "meaning",
    "please",
    "question",
    "search",
    "show",
    "summarize",
    "summary",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "why",
}
RAG_CONTEXT_DOCS = 4
RAG_MAX_CONTEXT_CHARS = 4_000

DEBOUNCE_SECONDS = 2.0
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".py", ".docx", ".pptx", ".csv"}
TEMP_FILE_PREFIXES = (".", "~")
TEMP_FILE_SUFFIXES = (
    ".tmp",
    ".swp",
    ".swx",
    ".part",
    ".crdownload",
)

INVALID_FOLDER_CHARS_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
MAX_FOLDER_NAME_LENGTH = 80

OLLAMA_ENABLE_ENV = "SEFS_ENABLE_OLLAMA_NAMING"
OLLAMA_MODEL_ENV = "SEFS_OLLAMA_MODEL"
OLLAMA_TIMEOUT_ENV = "SEFS_OLLAMA_TIMEOUT_SECONDS"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 8
OLLAMA_RAG_ENABLE_ENV = "SEFS_ENABLE_OLLAMA_RAG"
OLLAMA_RAG_MODEL_ENV = "SEFS_OLLAMA_RAG_MODEL"
OLLAMA_RAG_TIMEOUT_ENV = "SEFS_OLLAMA_RAG_TIMEOUT_SECONDS"
DEFAULT_OLLAMA_RAG_TIMEOUT_SECONDS = 45
RAG_PROVIDER_ENV = "SEFS_RAG_PROVIDER"
DEFAULT_RAG_PROVIDER = "auto"
HF_RAG_MODEL_ENV = "SEFS_HF_RAG_MODEL"
DEFAULT_HF_RAG_MODEL = "google/flan-t5-small"
HF_RAG_MAX_NEW_TOKENS_ENV = "SEFS_HF_RAG_MAX_NEW_TOKENS"
DEFAULT_HF_RAG_MAX_NEW_TOKENS = 220
HF_LOCAL_ONLY_ENV = "SEFS_HF_LOCAL_ONLY"

LOG_LEVEL_ENV = "SEFS_LOG_LEVEL"

AMBIGUITY_DELTA_THRESHOLD = 0.04
AMBIGUITY_MIN_SIMILARITY = 0.18
DUPLICATE_SIMILARITY_THRESHOLD = 0.95
MAX_DUPLICATE_PAIRS = 40
EDGE_SIMILARITY_THRESHOLD = 0.72
MAX_SIMILARITY_EDGES = 140
MAX_TIMELINE_POINTS = 240
MAX_EVENT_HISTORY = 400
MAX_RECENT_EVENTS_IN_SNAPSHOT = 14
CLUSTER_SUMMARY_KEYWORDS = 3



def is_supported_file(path: Path) -> bool:
    """Return True when the path is a supported text-bearing file."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS



def is_temporary_file(path: Path) -> bool:
    """Return True when the file name looks temporary/editor-generated."""
    name = path.name
    lowered = name.lower()
    return (
        name.startswith(TEMP_FILE_PREFIXES)
        or name.endswith("~")
        or lowered.endswith(TEMP_FILE_SUFFIXES)
    )



def sanitize_folder_name(name: str, fallback: str) -> str:
    """Convert arbitrary cluster text into a filesystem-safe folder name."""
    sanitized = INVALID_FOLDER_CHARS_RE.sub("_", name)
    sanitized = re.sub(r"\s+", "_", sanitized).strip(" ._")
    sanitized = re.sub(r"_+", "_", sanitized)

    if not sanitized:
        sanitized = fallback

    if len(sanitized) > MAX_FOLDER_NAME_LENGTH:
        sanitized = sanitized[:MAX_FOLDER_NAME_LENGTH].rstrip("._ ")

    if not sanitized:
        return fallback
    return sanitized



def get_log_level(default: str = "INFO") -> str:
    """Resolve runtime log level from environment."""
    return os.getenv(LOG_LEVEL_ENV, default).upper()
