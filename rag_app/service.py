from __future__ import annotations

from dataclasses import dataclass
import re

from . import config
from .documents import chunk_text, read_documents
from .store import Match, Store

INSUFFICIENT = "Bu bilgi ders belgelerinde yer almıyor."


def expand_query(question: str) -> str:
    """Add document-style forms for common Turkish ordinal assignment phrases."""
    ordinals = {"birinci": "1", "ilk": "1", "ikinci": "2", "üçüncü": "3"}
    expanded = question
    for word, number in ordinals.items():
        if re.search(rf"\b{word}\s+ödev\b", question, flags=re.IGNORECASE):
            if re.search(r"\b(ne zaman|teslim|tarih)", question, flags=re.IGNORECASE):
                return f"Ödev {number} son teslim tarihi"
            expanded += f" Ödev {number}"
    return expanded


def explicitly_unavailable(question: str, matches: list[Match]) -> bool:
    markers = ("yer almıyor", "yer almaz", "verilmemiş", "belirtilmemiş", "tanımlanmamış", "yoktur", "bilgi verilmemiş")
    generic = {"ders", "final", "sınav", "sınavı", "sınavının", "kapsam", "kapsamı", "bilgi"}
    question_stems = {token[:5] for token in Store._tokens(question) if token not in generic}
    for match in matches:
        for sentence in re.split(r"(?<=[.!?])\s+", match.content):
            lowered = sentence.casefold()
            content_stems = {token[:5] for token in Store._tokens(sentence)}
            if question_stems & content_stems and any(marker in lowered for marker in markers):
                return True
    return False


def select_evidence(answer_text: str, candidates: list[Match]) -> list[Match]:
    if not candidates:
        return []
    answer_tokens = Store._tokens(answer_text)
    scored = [(len(answer_tokens & Store._tokens(match.content)) / max(len(answer_tokens), 1), match) for match in candidates]
    best = max(score for score, _ in scored)
    return [match for score, match in scored if score >= best - 0.15]


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[str]
    matches: list[Match]
    evidence: list[Match]


def ingest(runtime, store: Store) -> tuple[int, int, int]:
    docs = read_documents(config.DOCUMENTS_DIR)
    if not docs:
        raise ValueError(f"İndekslenecek TXT/PDF belgesi bulunamadı: {config.DOCUMENTS_DIR}")
    indexed = skipped = chunks_total = 0
    index_model = f"{config.EMBEDDING_MODEL}:{config.INDEX_VERSION}"
    for doc in docs:
        if store.is_current(doc.source, doc.content_hash, index_model):
            skipped += 1
            continue
        texts = chunk_text(doc.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        vectors = runtime.embed(texts, config.EMBEDDING_MODEL)
        store.replace_document(doc.source, doc.content_hash, index_model, list(zip(texts, vectors)))
        indexed += 1
        chunks_total += len(texts)
    store.delete_documents_not_in({doc.source for doc in docs})
    return indexed, skipped, chunks_total


def ask(runtime, store: Store, question: str) -> Answer:
    question = question.strip()
    if not question:
        raise ValueError("Soru boş olamaz.")
    if store.count_chunks() == 0:
        raise ValueError("Veritabanı boş. Önce 'python main.py ingest' çalıştırın.")
    expanded_query = expand_query(question)
    query = runtime.embed([expanded_query], config.EMBEDDING_MODEL)[0]
    matches = store.search(query, config.TOP_K, query_text=expanded_query)
    if explicitly_unavailable(question, matches):
        return Answer(INSUFFICIENT, [], matches, [])
    best_rank = max((m.score + 0.20 * m.lexical_score for m in matches), default=0.0)
    accepted = [m for m in matches if m.score >= config.MIN_SIMILARITY and m.score + 0.20 * m.lexical_score >= best_rank - config.SCORE_MARGIN]
    if accepted:
        best_lexical = max(m.lexical_score for m in accepted)
        if best_lexical > 0:
            accepted = [m for m in accepted if m.lexical_score > best_lexical - 0.15]
    if not accepted:
        return Answer(INSUFFICIENT, [], matches, [])
    context = "\n\n".join(f"[Kaynak: {m.source}, parça {m.chunk_index}]\n{m.content}" for m in accepted)
    text = runtime.answer(question, context, config.CHAT_MODEL)
    if INSUFFICIENT.casefold().rstrip(".") in text.casefold().rstrip("."):
        return Answer(INSUFFICIENT, [], matches, [])
    evidence = select_evidence(text, accepted)
    sources = list(dict.fromkeys(m.source for m in evidence))
    return Answer(text, sources, matches, evidence)
