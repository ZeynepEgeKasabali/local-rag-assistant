from __future__ import annotations

import argparse
import sys

from rag_app import config
from rag_app.foundry import FoundryRuntime
from rag_app.service import ask, ingest
from rag_app.store import Store


def print_answer(result) -> None:
    print(f"\nCevap: {result.text}")
    print("Kaynaklar: " + (", ".join(result.sources) if result.sources else "yok"))
    if result.matches:
        print("Benzerlikler: " + ", ".join(f"{m.source}#{m.chunk_index}={m.score:.3f}" for m in result.matches))


def main() -> int:
    parser = argparse.ArgumentParser(description="Tamamen yerel Foundry Local RAG asistanı")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="TXT/PDF belgelerini indeksle")
    ask_parser = sub.add_parser("ask", help="Tek soru sor")
    ask_parser.add_argument("question", nargs="?", default="")
    sub.add_parser("chat", help="Etkileşimli soru-cevap")
    args = parser.parse_args()

    runtime = None
    store = Store(config.DB_PATH)
    try:
        runtime = FoundryRuntime(config.MODEL_CACHE_DIR)
        if args.command == "ingest":
            indexed, skipped, chunks = ingest(runtime, store)
            print(f"Tamamlandı: {indexed} belge / {chunks} parça indekslendi, {skipped} değişmemiş belge atlandı.")
        elif args.command == "ask":
            print_answer(ask(runtime, store, args.question))
        else:
            print("Çıkmak için boş satır veya 'çık' yazın.")
            while True:
                try:
                    question = input("\nSoru> ").strip()
                except EOFError:
                    print()
                    break
                if not question or question.lower() in {"çık", "exit", "quit"}:
                    break
                try:
                    print_answer(ask(runtime, store, question))
                except ValueError as exc:
                    print(f"Hata: {exc}")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Foundry Local hatası: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
