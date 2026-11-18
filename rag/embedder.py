from typing import List

import numpy as np

from .config import settings


class Embedder:
    """Wraps a sentence-transformers model. Vectors are L2-normalised so inner product = cosine."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embed_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")
