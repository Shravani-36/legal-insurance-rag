"""
BM25 Lexical Keyword Search for Legal & Insurance Documents.
"""
from typing import List, Dict, Any, Tuple
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class BM25Retriever:
    """
    BM25Okapi indexer and searcher for exact keyword and term matching.
    """

    def __init__(self):
        self.bm25: Any = None
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text into lowercase alphanumeric tokens.
        """
        clean = re.sub(r"[^\w\s\-]", " ", text.lower())
        tokens = [t for t in clean.split() if len(t) > 1]
        return tokens

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Builds the BM25 index over a list of chunk dictionaries.
        """
        self.corpus_chunks = chunks
        self.tokenized_corpus = [
            self.tokenize(f"{c.get('section_title', '')} {c.get('text', '')}")
            for c in chunks
        ]

        if BM25Okapi is not None and self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries BM25 and returns ranked (chunk, normalized_score) tuples.
        """
        if not self.corpus_chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return [(c, 0.0) for c in self.corpus_chunks[:top_k]]

        if self.bm25 is not None:
            raw_scores = self.bm25.get_scores(query_tokens)
        else:
            # Fallback simple TF keyword scoring if rank_bm25 not yet loaded
            raw_scores = []
            for doc_tokens in self.tokenized_corpus:
                score = sum(doc_tokens.count(qt) for qt in query_tokens)
                raw_scores.append(float(score))

        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0

        # Sort indices by score descending
        ranked_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx in ranked_indices[:top_k]:
            normalized_score = float(raw_scores[idx] / max_score) if max_score > 0 else 0.0
            results.append((self.corpus_chunks[idx], round(normalized_score, 4)))

        return results
