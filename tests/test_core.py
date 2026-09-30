import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag_app.documents import chunk_text
from rag_app.service import INSUFFICIENT, ask, expand_query, explicitly_unavailable, select_evidence
from rag_app.store import Match
from rag_app.store import Store


class CoreTests(unittest.TestCase):
    def test_chunking_has_overlap_and_preserves_text(self):
        chunks = chunk_text("bir iki üç dört beş altı yedi sekiz", 18, 5)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_replace_is_idempotent_and_cosine_search_orders(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.sqlite3")
            store.replace_document("a.txt", "h1", "model", [("yakın", [1.0, 0.0]), ("uzak", [0.0, 1.0])])
            store.replace_document("a.txt", "h1", "model", [("yakın", [1.0, 0.0]), ("uzak", [0.0, 1.0])])
            self.assertEqual(store.count_chunks(), 2)
            self.assertEqual(store.search([1.0, 0.0], 1)[0].content, "yakın")
            store.close()

    def test_zero_query_returns_no_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.sqlite3")
            store.replace_document("a", "h", "m", [("x", np.ones(2).tolist())])
            self.assertEqual(store.search([0.0, 0.0]), [])
            store.close()

    def test_removed_documents_are_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.sqlite3")
            store.replace_document("eski.txt", "h", "m", [("x", [1.0])])
            self.assertEqual(store.delete_documents_not_in({"yeni.txt"}), 1)
            self.assertEqual(store.count_chunks(), 0)
            store.close()

    def test_low_similarity_skips_chat_and_sources(self):
        class Runtime:
            def embed(self, texts, alias):
                return [[-1.0, 0.0]]

            def answer(self, question, context, alias):
                raise AssertionError("chat düşük skorda çağrılmamalı")

        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.sqlite3")
            store.replace_document("ders.txt", "h", "m", [("içerik", [1.0, 0.0])])
            result = ask(Runtime(), store, "belgede olmayan soru")
            self.assertEqual(result.text, INSUFFICIENT)
            self.assertEqual(result.sources, [])
            self.assertEqual(result.evidence, [])
            store.close()

    def test_turkish_assignment_ordinal_is_expanded(self):
        self.assertIn("Ödev 2", expand_query("İkinci ödev ne zaman teslim?"))

    def test_explicitly_missing_fact_is_detected(self):
        matches = [Match("sinav.txt", 0, "Bütünleme tarihi bu belgelerde tanımlanmamıştır.", 0.4)]
        self.assertTrue(explicitly_unavailable("Bütünleme sınavı ne zaman?", matches))

    def test_evidence_must_support_generated_answer(self):
        candidates = [
            Match("kurallar.txt", 0, "Geç teslim kuralı 48 saattir.", 0.7),
            Match("odevler.txt", 1, "Ödev 2 son teslim: 6 Kasım 2026 Cuma, 23.59.", 0.6),
        ]
        evidence = select_evidence("Ödev 2, 6 Kasım 2026 Cuma 23.59'da teslim edilir.", candidates)
        self.assertEqual([item.source for item in evidence], ["odevler.txt"])


if __name__ == "__main__":
    unittest.main()
