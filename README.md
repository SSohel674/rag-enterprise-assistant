# Marginalia: RAG-Based Enterprise Document Assistant

Ask questions over your company documents (PDF, DOCX, TXT, MD) and get **answers with source citations** and a **hallucination check**. Built from scratch (no heavy framework) so every stage of the RAG pipeline is explicit and explainable.

## Architecture

```
 Documents ─► Loader ─► Chunker ─► Embedder ─► FAISS index (persisted)
                                                    │
 Question ─► Embedder ─► Top-k retrieval ◄──────────┘
                              │
                 relevance gate (refuse if too low)
                              │
              Prompt with numbered passages ─► LLM ─► Answer with [1][2] citations
                                                          │
                                       Groundedness check (answer vs. context)
                                                          │
                                   Answer + sources + confidence + grounded flag
```

## Key features
- **Citations**: every answer references numbered passages (file + page).
- **Relevance gate**: if the best match is below `MIN_RELEVANCE`, the assistant refuses rather than guessing.
- **Hallucination check**: each answer sentence is embedded and compared to retrieved chunks; low mean max-similarity flags the answer as weakly grounded.
- **Pluggable LLM**: Ollama (free, local), OpenAI-compatible, or Anthropic via `.env`.
- **Custom web UI** with a source-passage panel (click a citation to see the evidence), plus a REST API.
- **Idempotent ingestion**: files are hashed; unchanged files aren't re-indexed.
- **Evaluation script**: Hit@k and MRR for retrieval, keyword accuracy for answers.
- **Dockerised**.

## Tech stack
Python · sentence-transformers (all-MiniLM-L6-v2) · FAISS · pypdf · python-docx · FastAPI · vanilla JS frontend · Docker · pytest

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # Windows: copy .env.example .env
```

**LLM option A: free & local (Ollama)**
```bash
# install from https://ollama.com
ollama pull llama3.2
```
**LLM option B: API**: set `LLM_PROVIDER=openai` (or `anthropic`) and the key in `.env`.

**Run the web app**
```bash
uvicorn api:app --reload        # then open http://localhost:8000
```
Click **Try with sample policies**, then ask e.g. *"How many days of annual leave do I get?"*. Click any numbered marker in an answer to see the exact passage it came from. REST docs are at `/docs`:
```bash
curl -F "files=@data/sample_docs/hr_leave_policy.md" localhost:8000/ingest
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How long is maternity leave?"}'
```
(`app.py` is the earlier Streamlit prototype and still works: `streamlit run app.py`.)

**Index a folder from CLI**: `python ingest_cli.py path/to/docs`

**Evaluate**
```bash
python eval/evaluate.py             # retrieval metrics (no LLM needed)
python eval/evaluate.py --answers   # + answer accuracy
```
Run it once and paste your real numbers into the Results section below.

**Tests**: `pytest -q`

**Docker**
```bash
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env rag-assistant
```
(When using Ollama from Docker, set `OLLAMA_URL=http://host.docker.internal:11434`.)

## Results (fill in after running `eval/evaluate.py`)
| Metric | Value |
|---|---|
| Hit@4 | 100.00% |
| MRR | 1.000 |
| Answer accuracy | 100.00% |

## Project structure
```
frontend/index.html Custom web UI (served by FastAPI)
api.py              FastAPI service + serves the UI
app.py              Legacy Streamlit prototype
ingest_cli.py       Batch indexing
rag/
  config.py         Settings via env vars
  loader.py         PDF/DOCX/TXT/MD loading
  chunker.py        Paragraph-aware chunking with overlap
  embedder.py       Sentence-transformer embeddings
  vectorstore.py    FAISS store + metadata persistence
  llm.py            Ollama / OpenAI / Anthropic client
  pipeline.py       Ingest, retrieve, generate, groundedness
eval/               Evaluation set + script
tests/              Unit tests
data/sample_docs/   Demo HR and IT policy docs
```

## Design decisions (interview talking points)
- **Why no LangChain?** Writing the pipeline directly keeps each stage inspectable and avoids framework churn. The components map 1:1 to LangChain/LlamaIndex abstractions (loader, splitter, embeddings, vector store, retriever, chain).
- **Chunking**: 800 chars with 150 overlap, split on paragraph then sentence boundaries to avoid cutting facts in half. Tunable via `.env`.
- **Cosine via inner product**: embeddings are L2-normalised and stored in `IndexFlatIP` (exact search; fine up to ~100k chunks, switch to HNSW/IVF beyond).
- **Hallucination mitigation**: strict prompt + low temperature + relevance gate + groundedness score.
- **Limitations / future work**: add a cross-encoder reranker, hybrid BM25 + dense retrieval, per-document delete, conversation memory/query rewriting, auth and multi-tenant indices, and LLM-as-judge evaluation (e.g. RAGAS).

## Resume bullets (edit with your own numbers)
- Built an end-to-end RAG assistant (FAISS + sentence-transformers + LLM) answering questions over PDF/DOCX documents with inline source citations; achieved **X% Hit@4** on a custom evaluation set.
- Implemented a groundedness check and relevance gate that cut unsupported answers, refusing out-of-scope queries instead of hallucinating.
- Exposed the system via a FastAPI service with a custom web UI that shows the source passage behind every answer, containerised with Docker, with unit tests and an automated retrieval evaluation harness.
