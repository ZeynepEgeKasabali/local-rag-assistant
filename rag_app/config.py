from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = Path(os.getenv("RAG_DOCUMENTS_DIR", ROOT / "data" / "course_documents"))
DB_PATH = Path(os.getenv("RAG_DB_PATH", ROOT / "data" / "course.sqlite3"))
MODEL_CACHE_DIR = Path(os.getenv("RAG_MODEL_CACHE_DIR", ROOT / "models"))
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "phi-4-mini")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
INDEX_VERSION = "course-v4"
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.42"))
SCORE_MARGIN = float(os.getenv("RAG_SCORE_MARGIN", "0.10"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
