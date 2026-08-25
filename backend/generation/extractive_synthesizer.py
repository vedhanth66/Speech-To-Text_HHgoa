"""
Non-LLM Extractive Answer Synthesizer
======================================
Produces accurate, concise, grounded answers in 1–3 ms without any cloud API network latency.
Supports English and all Indic languages (Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, etc.).
"""

import re
import time
import unicodedata
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("generation.synthesizer")

# Indic + English sentence boundary regex
SENTENCE_SPLIT_REGEX = re.compile(r"[\n\r]+|(?<=[.!?।॥])\s+")

STOPWORDS = {
    "what", "is", "the", "of", "and", "a", "to", "in", "for", "are", "on", "with",
    "as", "by", "at", "from", "how", "where", "who", "which", "why", "when", "does", "do", "did",
    "i", "me", "my", "you", "your", "we", "us", "can", "could", "would", "should", "please",
    "tell", "show", "give", "find", "get", "want", "need", "know", "explain", "detail", "details",
    "batao", "bataiye", "saanga", "sang", "that", "this", "these", "those", "say", "said", "saying",
    "का", "के", "की", "है", "हैं", "में", "से", "को", "पर", "यह", "और", "एक", "क्या",
    "ఉంది", "యొక్క", "మరియు", "అనేది", "ஆகும்", "மற்றும்", "என்பது"
}

GENERIC_QUESTION_WORDS = {
    "capital", "city", "country", "state", "name", "definition", "meaning", "meaning of",
    "list", "tell", "explain", "about", "called", "known", "as", "serves", "serve",
    "located", "situated", "acts", "act", "means", "stands", "defined", "named", "idea",
    "say", "said", "saying", "think", "thought"
}


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences supporting Indic punctuation (danda ।) and Latin punctuation."""
    if not text:
        return []
    raw = SENTENCE_SPLIT_REGEX.split(text.strip())
    sentences = []
    for s in raw:
        cleaned = s.strip()
        if len(cleaned) > 10:
            sentences.append(cleaned)
    return sentences if sentences else [text.strip()]


def extract_query_keywords(query: str) -> Tuple[set, set, List[str]]:
    """Extract content words, subject keywords, and multi-word phrases from query."""
    cleaned = unicodedata.normalize("NFKC", query).lower()
    tokens = re.findall(r"[\w\u0900-\u0D7F]+", cleaned)
    
    all_keywords = {t for t in tokens if t not in STOPWORDS and len(t) > 1}
    subject_keywords = {t for t in all_keywords if t not in GENERIC_QUESTION_WORDS}

    # Extract 2-word & 3-word subject phrases (e.g. "south africa", "new delhi")
    phrases = []
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        if w1 in subject_keywords and w2 in subject_keywords:
            phrases.append(f"{w1} {w2}")
        if i < len(tokens) - 2:
            w3 = tokens[i+2]
            if w1 in subject_keywords and w2 in subject_keywords and w3 in subject_keywords:
                phrases.append(f"{w1} {w2} {w3}")

    return all_keywords, subject_keywords, phrases


class ExtractiveSynthesizer:
    """
    Extracts the most relevant, concise, and coherent answer from retrieved passages.
    Target latency: < 2 ms.
    """

    def synthesize(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        max_sentences: int = 2,
        use_fallback: bool = True,
        min_sentence_score: float = 0.0,
    ) -> Tuple[str, float]:
        """
        Synthesize answer from retrieved passages with strict subject/phrase matching
        and single-passage sentence coherence.
        Returns (answer_string, latency_ms).
        """
        t0 = time.perf_counter()
        if not retrieved_results:
            return "", round((time.perf_counter() - t0) * 1000, 3)

        all_keywords, subject_keywords, query_phrases = extract_query_keywords(query)

        # ── Subject entity presence guard ──────────────────────────────────────
        # If the query has named subject keywords (e.g. "telangana", "pakistan"),
        # verify at least one appears in ANY of the top-3 retrieved passages.
        # If none match → corpus doesn't know about this entity → return ""
        # which causes is_refusal=True ("data not available").
        if subject_keywords:
            subject_found_in_corpus = any(
                any(kw in (r.get("parent_text") or r.get("text", "")).lower()
                    for kw in subject_keywords)
                for r in retrieved_results[:3]
            )
            if not subject_found_in_corpus:
                logger.debug(
                    "Synthesizer: subject %s absent from all retrieved passages → refusal",
                    subject_keywords,
                )
                return "", round((time.perf_counter() - t0) * 1000, 3)

        # Collect all candidate sentences from top passages
        scored_sentences = []
        seen_sentences = set()

        for passage_rank, r in enumerate(retrieved_results[:3]):
            passage_text = r.get("parent_text") or r.get("text", "")
            base_score = r.get("reranker_score", r.get("rrf_score", r.get("dense_score", 0.5)))
            sentences = split_into_sentences(passage_text)

            for s_idx, sentence in enumerate(sentences):
                norm_s = sentence.lower()
                if norm_s in seen_sentences:
                    continue
                seen_sentences.add(norm_s)

                s_tokens = set(re.findall(r"[\w\u0900-\u0D7F]+", norm_s))

                # Check multi-word phrase matching
                phrase_match = any(p in norm_s for p in query_phrases) if query_phrases else False
                
                long_phrases = [p for p in query_phrases if len(p.split()) >= 3]
                if long_phrases and not phrase_match:
                    all_phrase_words_in_sentence = any(
                        set(p.split()).issubset(s_tokens) for p in long_phrases
                    )
                    if all_phrase_words_in_sentence:
                        phrase_match = True

                def matches_token(kw: str, st: str) -> bool:
                    if len(st) < 3 or len(kw) < 3:
                        return kw == st
                    if kw in st:
                        return True
                    if len(kw) >= 4 and len(st) >= 4 and kw[:4] == st[:4]:
                        return True
                    return False

                # Subject keyword match check
                subject_matches = sum(1 for kw in subject_keywords if any(matches_token(kw, st) for st in s_tokens))
                
                # General keyword overlap score
                overlap_count = sum(1 for kw in all_keywords if any(matches_token(kw, st) for st in s_tokens))
                overlap_ratio = overlap_count / max(len(all_keywords), 1)

                # Require at least one subject keyword to match the sentence.
                # No score-based exception: the corpus-level check above already
                # confirmed the subject exists somewhere — so per-sentence matching must hold.
                if subject_keywords and subject_matches == 0:
                    continue

                # Position bias (first sentence in passage often contains main definition)
                position_bonus = 0.35 if s_idx == 0 else (0.15 if s_idx == 1 else 0.0)

                # Passage rank weighting
                rank_weight = 1.0 / (1.0 + 0.5 * passage_rank)

                # Total relevance score
                total_score = (
                    overlap_ratio * 2.0 + 
                    (2.0 if phrase_match else 0.0) +
                    subject_matches * 1.5 + 
                    position_bonus + 
                    base_score * 0.5
                ) * rank_weight

                scored_sentences.append({
                    "text": sentence,
                    "score": total_score,
                    "overlap": overlap_count,
                    "subject_matches": subject_matches,
                    "passage_rank": passage_rank,
                    "sentence_idx": s_idx,
                })

        if not scored_sentences and retrieved_results:
            if not use_fallback:
                # Caller requested no fallback (e.g. eval harness strict mode):
                # returning "" causes generator to mark answer as ungrounded.
                return "", round((time.perf_counter() - t0) * 1000, 3)
            # Fallback: extract first sentence from the top retrieved passage
            top_passage_text = retrieved_results[0].get("parent_text") or retrieved_results[0].get("text", "")
            fallback_sents = split_into_sentences(top_passage_text)
            if fallback_sents:
                return fallback_sents[0], round((time.perf_counter() - t0) * 1000, 3)
            return "", round((time.perf_counter() - t0) * 1000, 3)

        # Sort by total score descending
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)

        # Minimum internal-score gate (optional, for eval strict mode)
        if min_sentence_score > 0.0 and scored_sentences[0]["score"] < min_sentence_score:
            return "", round((time.perf_counter() - t0) * 1000, 3)

        top_cand = scored_sentences[0]
        selected = [top_cand["text"]]

        # Single-passage coherence: only select a 2nd sentence if it comes from the SAME passage
        if max_sentences > 1 and len(scored_sentences) > 1:
            top_passage_rank = top_cand["passage_rank"]
            for cand in scored_sentences[1:]:
                if cand["passage_rank"] == top_passage_rank:
                    words1 = set(selected[0].split())
                    words2 = set(cand["text"].split())
                    jaccard = len(words1 & words2) / max(len(words1 | words2), 1)
                    if jaccard < 0.6:
                        selected.append(cand["text"])
                        break

        answer = " ".join(selected).strip()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return answer, elapsed_ms
