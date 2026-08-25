"""
Generator Module — Implements TARGET_INTERFACE contract for rag-local-eval-loop.
"""
import os
import sys
import time
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_BACKEND = os.path.join(_REPO_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from generation.extractive_synthesizer import ExtractiveSynthesizer

_synthesizer = ExtractiveSynthesizer()


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str = "extractive-synthesizer"


def generate_answer(query: str, results: list) -> GeneratedAnswer:
    """
    Generates an answer from duck-typed search results.
    Each result item has .text, .source, and .score attributes.
    Returns GeneratedAnswer(text, grounded, generation_ms, model).

    Grounding policy:
      - grounded=True  → system has a confident, passage-supported answer
      - grounded=False → system declines (no passages, low relevance, or no match)
    The eval harness uses this flag for the reliability / lying-factor check.
    """
    t0 = time.perf_counter()
    if not results:
        return GeneratedAnswer(
            text="The provided documents do not contain information to answer this query.",
            grounded=False,
            generation_ms=round((time.perf_counter() - t0) * 1000, 2),
            model="extractive-synthesizer",
        )

    # --- Relevance gate ---
    # FAISS inner-product scores for all-MiniLM-L6-v2 (L2-normalised) are
    # cosine similarities in [-1, 1].  Empirically, a score below ~0.25
    # means the top retrieved passage is not about the query at all.
    # Refuse early so the ExtractiveSynthesizer's fallback cannot fabricate
    # an answer from unrelated passage text.
    MIN_RELEVANCE_SCORE = 0.25
    top_score = getattr(results[0], "score", 1.0)
    if top_score < MIN_RELEVANCE_SCORE:
        return GeneratedAnswer(
            text="The provided documents do not contain sufficient information to answer this question.",
            grounded=False,
            generation_ms=round((time.perf_counter() - t0) * 1000, 2),
            model="extractive-synthesizer",
        )

    # Convert duck-typed context objects to dictionary format
    candidates = []
    for r in results:
        text = getattr(r, "text", "")
        source = getattr(r, "source", "")
        score = getattr(r, "score", 1.0)
        candidates.append({
            "text": text,
            "chunk_id": source,
            "similarity_score": score,
        })

    extracted_text, synth_ms = _synthesizer.synthesize(
        query, candidates,
        use_fallback=False,   # Refuse rather than return unrelated first-sentence fallback
    )

    if extracted_text and len(extracted_text.strip()) > 5:
        return GeneratedAnswer(
            text=extracted_text,
            grounded=True,
            generation_ms=round(synth_ms, 2),
            model="extractive-synthesizer",
        )
    else:
        return GeneratedAnswer(
            text="The provided documents do not contain sufficient information to answer this question.",
            grounded=False,
            generation_ms=round(synth_ms, 2),
            model="extractive-synthesizer",
        )
