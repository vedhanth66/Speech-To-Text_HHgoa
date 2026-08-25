"""
VectorDBEngine — Backward-Compatible Wrapper
============================================
Wraps the new DenseRetriever for any code that still calls
VectorDBEngine directly (e.g. /api/chunking/compare, tests).

Also adds load_index_from_disk / save_index_to_disk methods
used by main.py for the fallback path.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger("vector_db")


class VectorDBEngine:
    """
    Thin wrapper around DenseRetriever, kept for backward compatibility.

    New code should use retrieval.DenseRetriever directly.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        from retrieval.dense import DenseRetriever
        self._dense = DenseRetriever(model_name=embedding_model_name)
        self.embedding_model_name = embedding_model_name

    # ── Backward-compat properties ────────────────────────────────────────────

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        return self._dense.chunks

    @property
    def embeddings_matrix(self) -> Optional[np.ndarray]:
        return self._dense.embeddings_matrix

    @property
    def vector_dim(self) -> int:
        return self._dense.vector_dim

    # ── Index Building ────────────────────────────────────────────────────────

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index chunks in-memory (backward compat)."""
        result = self._dense.index_chunks(chunks)
        return {
            "indexed_count": result.get("indexed_count", 0),
            "vector_dim": self._dense.vector_dim,
            "indexing_time_ms": result.get("indexing_ms", 0.0),
        }

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self._dense._embed(texts)

    # ── Disk I/O ──────────────────────────────────────────────────────────────

    def load_index_from_disk(self, index_dir: str) -> Dict[str, Any]:
        """Load pre-built FAISS index from disk."""
        loaded = self._dense.load_from_disk(index_dir)
        return {
            "loaded": loaded,
            "chunks_count": len(self._dense.chunks) if loaded else 0,
        }

    def save_index_to_disk(self, index_dir: str):
        """Save current in-memory index to disk."""
        self._dense.save_to_disk(index_dir)

    # ── Search (backward compat signature) ───────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Search for top-k similar chunks.
        Returns dict matching original VectorDBEngine.search() shape.
        """
        query_key = query.strip().lower()
        is_cached = query_key in self._dense._query_cache

        results, retrieval_ms = self._dense.search(
            query, top_k=top_k, similarity_threshold=similarity_threshold
        )

        # Convert new schema to old schema for compatibility
        old_results = []
        for r in results:
            old_results.append({
                "rank": r["rank"],
                "chunk_id": r["chunk_id"],
                "similarity_score": round(r.get("dense_score", 0.0), 4),
                "text": r["text"],
                "parent_text": r.get("parent_text", r["text"]),
                "metadata": r.get("metadata", {}),
                "is_above_threshold": r.get("dense_score", 0.0) >= similarity_threshold,
            })

        top_score = old_results[0]["similarity_score"] if old_results else 0.0

        return {
            "results": old_results,
            "top_score": round(top_score, 4),
            "retrieval_latency_ms": retrieval_ms,
            "is_cached": is_cached,
        }
