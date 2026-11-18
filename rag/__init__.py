__all__ = ["RAGPipeline"]


def __getattr__(name):  # lazy import keeps light modules (chunker, loader) usable without heavy deps
    if name == "RAGPipeline":
        from .pipeline import RAGPipeline

        return RAGPipeline
    raise AttributeError(name)
