import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import split_text  # noqa: E402


def test_short_text_single_chunk():
    assert split_text("Hello world.", 800, 100) == ["Hello world."]


def test_chunks_respect_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    chunks = split_text(text, 400, 80)
    assert len(chunks) > 1
    # allow overlap tail + separator headroom
    assert all(len(c) <= 400 + 80 + 2 for c in chunks)


def test_overlap_carries_context():
    text = "\n\n".join(f"Unique sentence number {i} here." + " filler" * 30 for i in range(6))
    chunks = split_text(text, 300, 60)
    assert len(chunks) >= 2
    assert chunks[0][-20:].strip().split()[-1] in chunks[1]


def test_giant_sentence_is_cut():
    chunks = split_text("x" * 2000, 500, 50)
    assert len(chunks) >= 4


def test_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
