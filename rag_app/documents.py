from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Document:
    path: Path
    source: str
    text: str
    content_hash: str


def read_documents(folder: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8-sig")
        elif path.suffix.lower() == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            continue
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            documents.append(Document(path, path.name, text, digest))
    return documents


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Parça boyutu pozitif, örtüşme ise daha küçük olmalıdır.")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        start = 0
        while start < len(paragraph):
            end = min(start + size, len(paragraph))
            if end < len(paragraph):
                boundary = paragraph.rfind(" ", start + size // 2, end)
                if boundary > start:
                    end = boundary
            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(paragraph):
                break
            start = max(start + 1, end - overlap)
    return chunks
