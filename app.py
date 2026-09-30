from __future__ import annotations

import atexit
from html import escape

import streamlit as st

from rag_app import config
from rag_app.foundry import FoundryRuntime
from rag_app.service import Answer, ask
from rag_app.store import Store


st.set_page_config(
    page_title="PY101 Ders Asistanı",
    page_icon="🐍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 5rem; }
  .subtitle { color: #52606d; margin-top: -.7rem; }
  .demo-note { padding: .8rem 1rem; border-radius: .8rem; background: #fff7df; border: 1px solid #f1d58a; color: #594515; }
  .source-card { margin: .4rem 0; padding: .7rem .85rem; border-left: 4px solid #5b8def; border-radius: .35rem; background: rgba(91,141,239,.08); }
  .source-name { font-weight: 650; }
  .quote { color: #465362; font-size: .92rem; margin-top: .25rem; }
  @media (max-width: 600px) {
    .block-container { padding: 1rem .75rem 4.5rem; }
    h1 { font-size: 1.75rem !important; }
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_runtime() -> FoundryRuntime:
    runtime = FoundryRuntime(config.MODEL_CACHE_DIR)
    atexit.register(runtime.close)
    return runtime


def answer_question(question: str) -> Answer:
    runtime = get_runtime()
    store = Store(config.DB_PATH)
    try:
        return ask(runtime, store, question)
    finally:
        store.close()


def excerpt(text: str, limit: int = 260) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit].rstrip() + "…"


def render_sources(answer: Answer) -> None:
    if not answer.evidence:
        st.caption("Bu yanıt için destekleyici kaynak bulunamadı.")
        return
    with st.expander(f"Kaynaklar ve alıntılar ({len(answer.evidence)})", expanded=False):
        for match in answer.evidence:
            st.markdown(
                f'<div class="source-card"><div class="source-name">{escape(match.source)}</div>'
                f'<div class="quote">“{escape(excerpt(match.content))}”</div></div>',
                unsafe_allow_html=True,
            )


st.title("🐍 PY101 Ders Asistanı")
st.markdown('<p class="subtitle">Ders içeriği, ödevler, sınavlar, notlandırma ve kurallar hakkında sorun.</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="demo-note">Bu uygulama kurgusal demo belgelerini kullanır; gerçek bir üniversitenin resmî ders bilgi sistemi değildir.</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message["answer"])

question = st.chat_input("Dersle ilgili sorunuzu yazın…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.status("Yerel modeller hazırlanıyor ve ders belgeleri taranıyor…", expanded=False) as status:
                result = answer_question(question)
                status.update(label="Yanıt hazır", state="complete")
            st.markdown(result.text)
            render_sources(result)
            st.session_state.messages.append({"role": "assistant", "content": result.text, "answer": result})
        except Exception as exc:
            message = f"Yanıt hazırlanamadı: {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "answer": Answer(message, [], [], [])})
