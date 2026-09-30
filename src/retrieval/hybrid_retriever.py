"""
Hybrid Retriever combining FAISS Semantic Search and BM25 Lexical Keyword Search
using Reciprocal Rank Fusion (RRF) and Convex Weighted Score Combination.
"""
from typing import List, Dict, Any, Tuple, Optional
from src.retrieval.vector_search import FAISSRetriever
from src.retrieval.bm25_search import BM25Retriever
import config


class HybridRetriever:
    """
    Combines dense FAISS semantic search and sparse BM25 keyword search.
    Provides Reciprocal Rank Fusion (RRF) and tunable alpha weighting.
    """

    def __init__(self, embedding_model: str = config.DEFAULT_EMBEDDING_MODEL):
        self.vector_retriever = FAISSRetriever(model_name=embedding_model)
        self.bm25_retriever = BM25Retriever()
        self.chunks: List[Dict[str, Any]] = []

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Indexes chunks into both the FAISS vector index and BM25 index.
        """
        self.chunks = chunks
        if not chunks:
            return

        self.vector_retriever.fit(chunks)
        self.bm25_retriever.fit(chunks)

    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        alpha: float = config.DEFAULT_HYBRID_ALPHA,
        use_rrf: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid search over the indexed chunks.

        Args:
            query: The user prompt or legal question
            top_k: Number of top results to return
            alpha: Weight for FAISS dense search (1.0 = 100% dense, 0.0 = 100% BM25 keyword)
            use_rrf: If True, uses Reciprocal Rank Fusion; if False, uses normalized linear scoring.

        Returns:
            List[Dict[str, Any]]: Ranked chunks with hybrid scores and individual sub-scores.
        """
        if not self.chunks:
            return []

        # Retrieve a broader pool of candidates from each engine
        candidate_k = min(len(self.chunks), max(top_k * 3, 10))
        vector_results = self.vector_retriever.search(query, top_k=candidate_k)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)

        # Map chunk_id to scores and rank positions
        chunk_map: Dict[str, Dict[str, Any]] = {}
        vector_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}
        vector_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(vector_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            vector_ranks[cid] = rank + 1
            vector_scores[cid] = score

        for rank, (chunk, score) in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            bm25_ranks[cid] = rank + 1
            bm25_scores[cid] = score

        final_ranked: List[Dict[str, Any]] = []

        if use_rrf:
            # Reciprocal Rank Fusion
            rrf_k = config.RRF_K
            for cid, chunk in chunk_map.items():
                v_rank = vector_ranks.get(cid, candidate_k + 50)
                b_rank = bm25_ranks.get(cid, candidate_k + 50)

                rrf_score = (alpha / (rrf_k + v_rank)) + ((1.0 - alpha) / (rrf_k + b_rank))

                result_entry = dict(chunk)
                result_entry["hybrid_score"] = round(rrf_score * 100.0, 4)
                result_entry["vector_score"] = vector_scores.get(cid, 0.0)
                result_entry["bm25_score"] = bm25_scores.get(cid, 0.0)
                result_entry["vector_rank"] = vector_ranks.get(cid, -1)
                result_entry["bm25_rank"] = bm25_ranks.get(cid, -1)
                final_ranked.append(result_entry)

            final_ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
        else:
            # Linear Convex Combination
            for cid, chunk in chunk_map.items():
                v_score = vector_scores.get(cid, 0.0)
                b_score = bm25_scores.get(cid, 0.0)
                linear_score = (alpha * v_score) + ((1.0 - alpha) * b_score)

                result_entry = dict(chunk)
                result_entry["hybrid_score"] = round(linear_score, 4)
                result_entry["vector_score"] = v_score
                result_entry["bm25_score"] = b_score
                result_entry["vector_rank"] = vector_ranks.get(cid, -1)
                result_entry["bm25_rank"] = bm25_ranks.get(cid, -1)
                final_ranked.append(result_entry)

            final_ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return final_ranked[:top_k]
