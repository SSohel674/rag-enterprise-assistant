"""Index a folder from the command line:  python ingest_cli.py data/sample_docs"""
import sys

from rag import RAGPipeline

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs"
    result = RAGPipeline().ingest_directory(folder)
    for name, n in result.items():
        print(f"{name}: {n} chunks" if n else f"{name}: already indexed")
