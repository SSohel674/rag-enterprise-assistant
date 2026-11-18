"""Streamlit UI:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from rag import RAGPipeline
from rag.llm import LLMError

st.set_page_config(page_title="Enterprise Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


rag = get_pipeline()

st.title("📄 Enterprise Document Assistant")
st.caption("RAG with citations and a hallucination check. Answers come only from your documents.")

with st.sidebar:
    st.header("Knowledge base")
    files = st.file_uploader("Upload PDF / DOCX / TXT / MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not files):
        for f in files:
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / f.name
                p.write_bytes(f.getvalue())
                n = rag.ingest_file(p)
            st.success(f"{f.name}: {n} chunks" if n else f"{f.name}: already indexed")
    if st.button("Load sample documents"):
        res = rag.ingest_directory(Path(__file__).parent / "data" / "sample_docs")
        st.success(f"Indexed: {res}")
    st.divider()
    st.write("**Indexed documents**")
    for s in rag.store.sources() or ["(none yet)"]:
        st.write(f"• {s}")
    if st.button("Clear knowledge base"):
        rag.store.clear()
        st.rerun()
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, 4)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about your documents…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not rag.store.sources():
            st.warning("Index some documents first (use the sidebar).")
        else:
            try:
                with st.spinner("Thinking…"):
                    res = rag.ask(question, top_k=top_k)
                st.markdown(res["answer"])
                if res["groundedness"] is not None:
                    if res["grounded"]:
                        st.success(f"Grounded in sources (score {res['groundedness']})")
                    else:
                        st.warning(f"Low groundedness ({res['groundedness']}): verify against sources.")
                for s in res["sources"]:
                    with st.expander(f"[{s['id']}] {s['source']} · page {s['page']} · relevance {s['score']}"):
                        st.write(s["snippet"] + "…")
                st.session_state.history.append({"role": "assistant", "content": res["answer"]})
            except LLMError as e:
                st.error(str(e))
