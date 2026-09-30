from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from foundry_local_sdk import (
    ChatSession, Configuration, EmbeddingsSession, FoundryLocalManager,
    Request, TextItem, TextItemType,
)


class FoundryRuntime:
    def __init__(self, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        FoundryLocalManager.initialize(Configuration(
            "local-rag-assistant", model_cache_dir=str(cache_dir),
            disable_nonessential_telemetry=True,
        ))
        self.manager = FoundryLocalManager.instance
        assert self.manager is not None
        self._models: dict[str, object] = {}

    def close(self) -> None:
        self.manager.close()

    def _model(self, alias: str):
        if alias not in self._models:
            model = self.manager.catalog.get_model(alias)
            if model is None:
                available = ", ".join(m.alias for m in self.manager.catalog.list_models())
                raise RuntimeError(f"Model alias bulunamadı: {alias}. Kullanılabilir alias'lar: {available}")
            if not model.is_cached:
                print(f"Model indiriliyor: {alias}")
                last = [-10]
                def progress(value: float) -> None:
                    current = int(value)
                    if current >= last[0] + 10:
                        print(f"  %{current}")
                        last[0] = current
                model.download(progress)
            if not model.is_loaded:
                print(f"Model yükleniyor: {alias}")
                model.load()
            self._models[alias] = model
        return self._models[alias]

    @staticmethod
    def _run(session_type, model, payload: dict) -> dict:
        with session_type(model) as session, Request() as request:
            request.add_item(TextItem(json.dumps(payload, ensure_ascii=False), TextItemType.OPENAI_JSON))
            with session.process_request(request) as response:
                return json.loads(response.get_item(0).text)

    def embed(self, texts: Sequence[str], alias: str) -> list[list[float]]:
        if not texts:
            return []
        model = self._model(alias)
        payload = {"model": model.id, "input": list(texts)}
        result = self._run(EmbeddingsSession, model, payload)
        return [item["embedding"] for item in sorted(result["data"], key=lambda x: x["index"])]

    def answer(self, question: str, context: str, alias: str) -> str:
        model = self._model(alias)
        messages = [
            {"role": "system", "content": "Sen PY101 ders asistanısın. Sadece aşağıdaki belge alıntılarındaki açık bilgiyi kullan ve sorunun doğrudan yanıtını en fazla üç cümleyle Türkçe ver. Alıntıda soruyu yanıtlayan bir cümle varsa onu aktar. 'Hayır', 'gerekmez' veya 'kapalıdır' gibi olumsuz ifadeler geçerli, destekli cevaplardır; bunları bilgi eksikliği sayma. Yalnızca alıntı istenen bilginin verilmediğini ya da tanımlanmadığını açıkça söylüyorsa 'Bu bilgi ders belgelerinde yer almıyor.' yaz. Tarih, kişi, bağlantı, iletişim bilgisi veya sınav sorusu uydurma. Kaynak adı ekleme."},
            {"role": "user", "content": f"BELGE ALINTILARI:\n{context}\n\nYANITLANACAK SORU: {question}\n\nYalnızca doğrudan yanıt:"},
        ]
        payload = {"model": model.id, "messages": messages, "temperature": 0, "max_tokens": 160}
        result = self._run(ChatSession, model, payload)
        return result["choices"][0]["message"]["content"].strip()
