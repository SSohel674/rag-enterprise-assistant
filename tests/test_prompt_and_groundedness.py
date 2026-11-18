import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

from rag.pipeline import RAGPipeline, _sentences  # noqa: E402


def test_sentences_strip_citations():
    s = _sentences("Employees get 24 days of leave [1]. Unused days carry forward up to ten [2].")
    assert all("[" not in x for x in s) and len(s) == 2


def test_prompt_contains_numbered_context():
    hits = [{"source": "a.md", "page": 1, "text": "foo bar"}, {"source": "b.md", "page": 2, "text": "baz"}]
    p = RAGPipeline._build_prompt("q?", hits)
    assert "[1] (source: a.md, page 1)" in p and "[2] (source: b.md, page 2)" in p
