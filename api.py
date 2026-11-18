"""Web app + REST API:   uvicorn api:app --reload   then open http://localhost:8000
Interactive API docs at /docs"""
import tempfile
from collections import Counter
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import RAGPipeline
from rag.llm import LLMError
from rag.loader import SUPPORTED

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Marginalia: document Q&A", version="2.0.0")
rag = RAGPipeline()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(6, ge=1, le=12)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(rag.store.sources()), "chunks": len(rag.store.meta)}


@app.get("/sources")
def sources():
    counts = Counter(m["source"] for m in rag.store.meta)
    return {"sources": [{"name": n, "passages": c} for n, c in sorted(counts.items())]}


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    results = {}
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(400, f"{f.filename}: unsupported file type. Use PDF, DOCX, TXT or MD.")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / Path(f.filename).name
            p.write_bytes(await f.read())
            results[f.filename] = rag.ingest_file(p)
    return {"chunks_added": results}


@app.post("/ingest_samples")
def ingest_samples():
    return {"chunks_added": rag.ingest_directory(ROOT / "data" / "sample_docs")}


@app.post("/clear")
def clear():
    rag.store.clear()
    return {"status": "cleared"}


@app.post("/ask")
def ask(req: AskRequest):
    if not rag.store.sources():
        raise HTTPException(400, "No documents yet. Add a file first.")
    try:
        return rag.ask(req.question, req.top_k)
    except LLMError as e:
        raise HTTPException(502, str(e))
