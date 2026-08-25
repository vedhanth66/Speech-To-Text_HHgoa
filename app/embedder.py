"""
Embedder Module — Implements TARGET_INTERFACE contract for rag-local-eval-loop.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_BACKEND = os.path.join(_REPO_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from retrieval.dense import DenseRetriever

_retriever = None


def get_model():
    """Initializes and returns the embedding model."""
    global _retriever
    if _retriever is None:
        model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _retriever = DenseRetriever(model_name=model_name)
    return _retriever.model


def embed(texts: list[str]) -> np.ndarray:
    """Returns array-like embeddings with shape (len(texts), dim)."""
    global _retriever
    if _retriever is None:
        get_model()
    return _retriever._embed(texts)


def embed_one(text: str) -> np.ndarray:
    """Returns array-like embedding with shape (dim,)."""
    global _retriever
    if _retriever is None:
        get_model()
    vecs = _retriever._embed([text])
    if len(vecs) > 0:
        return vecs[0]
    return np.zeros((_retriever.vector_dim,), dtype=np.float32)
