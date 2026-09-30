from __future__ import annotations

import sqlite3
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Match:
    source: str
    chunk_index: int
    content: str
    score: float
    lexical_score: float = 0.0


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, source TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL, embedding_model TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                embedding BLOB NOT NULL, dimension INTEGER NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
        """)

    def close(self) -> None:
        self.connection.close()

    def is_current(self, source: str, content_hash: str, model: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM documents WHERE source=? AND content_hash=? AND embedding_model=?",
            (source, content_hash, model),
        ).fetchone()
        return row is not None

    def replace_document(self, source: str, content_hash: str, model: str, chunks: list[tuple[str, list[float]]]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM documents WHERE source=?", (source,))
            cursor = self.connection.execute(
                "INSERT INTO documents(source, content_hash, embedding_model) VALUES(?,?,?)",
                (source, content_hash, model),
            )
            document_id = cursor.lastrowid
            self.connection.executemany(
                "INSERT INTO chunks(document_id, chunk_index, content, embedding, dimension) VALUES(?,?,?,?,?)",
                [(document_id, i, text, np.asarray(vector, dtype=np.float32).tobytes(), len(vector))
                 for i, (text, vector) in enumerate(chunks)],
            )

    def count_chunks(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def document_sources(self) -> list[str]:
        return [row[0] for row in self.connection.execute("SELECT source FROM documents ORDER BY source")]

    def delete_documents_not_in(self, sources: set[str]) -> int:
        existing = self.document_sources()
        removed = [source for source in existing if source not in sources]
        with self.connection:
            self.connection.executemany("DELETE FROM documents WHERE source=?", [(source,) for source in removed])
        return len(removed)

    def search(self, query: list[float], limit: int = 3, query_text: str | None = None) -> list[Match]:
        q = np.asarray(query, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return []
        ranked: list[tuple[float, Match]] = []
        query_tokens = self._tokens(query_text or "")
        rows = self.connection.execute(
            "SELECT d.source, c.chunk_index, c.content, c.embedding, c.dimension FROM chunks c JOIN documents d ON d.id=c.document_id"
        )
        for source, index, content, blob, dimension in rows:
            vector = np.frombuffer(blob, dtype=np.float32, count=dimension)
            denom = q_norm * float(np.linalg.norm(vector))
            score = float(np.dot(q, vector) / denom) if denom else 0.0
            content_tokens = self._tokens(content)
            lexical = len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0.0
            match = Match(source, index, content, score, lexical)
            ranked.append((score + 0.20 * lexical, match))
        return [match for _, match in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {"bir", "bu", "ile", "için", "kaç", "mı", "mi", "mu", "mü", "ne", "nedir", "zaman"}
        return {token for token in re.findall(r"[a-zçğıöşü0-9]+", text.casefold()) if token not in stop and (len(token) > 2 or token.isdigit())}
