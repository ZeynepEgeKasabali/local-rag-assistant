# PY101 Yerel Ders Asistanı

Microsoft Foundry Local ile tamamen cihazda çalışan bir RAG uygulamasıdır. Beş kurgusal PY101 belgesinden ders içeriği, ödevler, sınavlar, notlandırma ve kurallar hakkında yanıt üretir. Belgeler yalnızca demo içindir; gerçek bir üniversitenin resmî bilgileri değildir.

Uygulama TXT/PDF belgelerini parçalar, `qwen3-embedding-0.6b` ile vektörleştirir, SQLite'ta saklar ve cosine similarity ile arar. Tarih ve ödev numarası gibi kesin terimler için yerel sözcük eşleşmesiyle sıralamayı güçlendirip en ilgili üç parçayı seçer. Yeterli kanıt varsa `phi-4-mini` yanıt üretir. Dış API, web araması, bulut servisi veya harici vektör veritabanı kullanılmaz.

## Kurulum (Windows PowerShell)

```powershell
cd C:\Users\User1\Desktop\local-rag-assistant
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Güvenli indeks oluşturma

Ders belgeleri `data/course_documents/`, ders indeksi `data/course.sqlite3` konumundadır. Eski Mars ve kütüphane örnekleri `data/examples/` altında tutulur ve ders indeksine girmez. Eski `data/rag.sqlite3` dosyası da ayrı kaldığı için sonuçlara karışmaz.

```powershell
python main.py ingest
python main.py ask "İkinci ödev ne zaman teslim?"
python main.py chat
```

Değişmeyen belgeler yeniden eklenmez; değiştirilen belgeler yenilenir ve klasörden kaldırılan belgeler indeksten silinir.

## Tarayıcı arayüzü

```powershell
python -m streamlit run app.py --server.address localhost --server.port 8501
```

Tarayıcı adresi: `http://localhost:8501`

Sohbet geçmişi tarayıcı oturumu boyunca korunur. Her desteklenen yanıtın altında retrieval tarafından seçilmiş gerçek dosya adları ve kısa alıntılar gösterilir. Kanıt yetersizse kaynak gösterilmez.

## Testler

```powershell
python -m unittest discover -s tests -v
```

## Yapılandırma

Gerekirse PowerShell ortam değişkenleri kullanılabilir:

```powershell
$env:RAG_CHAT_MODEL = "phi-4-mini"
$env:RAG_EMBEDDING_MODEL = "qwen3-embedding-0.6b"
$env:RAG_MIN_SIMILARITY = "0.42"
$env:RAG_DOCUMENTS_DIR = "C:\tam\belge\yolu"
$env:RAG_DB_PATH = "C:\tam\indeks\yolu.sqlite3"
```
