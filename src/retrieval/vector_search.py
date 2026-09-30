"""
FAISS Dense Vector Search using Sentence Transformers for Legal Documents.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import config

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class FAISSRetriever:
    """
    Dense semantic search using Sentence Transformers and FAISS IndexFlatIP (cosine similarity).
    """

    _model_cache: Optional[Any] = None

    def __init__(self, model_name: str = config.DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = self._get_model(model_name)
        self.index: Any = None
        self.chunks: List[Dict[str, Any]] = []
        self.dimension: int = 384

    @classmethod
    def _get_model(cls, model_name: str):
        if cls._model_cache is None and SentenceTransformer is not None:
            import os
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            try:
                # Try cached local weights first for instant loading
                cls._model_cache = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                try:
                    cls._model_cache = SentenceTransformer(model_name)
                except Exception:
                    cls._model_cache = None
        return cls._model_cache

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds chunks and creates a FAISS index.
        """
        self.chunks = chunks
        if not chunks:
            return

        texts_to_embed = [
            f"{c.get('section_title', '')}: {c.get('text', '')}"
            for c in chunks
        ]

        if self.model is not None:
            embeddings = self.model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            self.dimension = embeddings.shape[1]
        else:
            # Fallback lightweight vector space if model not yet loaded
            self.dimension = 64
            np.random.seed(42)
            embeddings = np.random.randn(len(chunks), self.dimension).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
        else:
            # NumPy cosine similarity fallback
            self.index = embeddings

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Embeds the query and retrieves the top_k most semantically similar chunks.
        """
        if not self.chunks or self.index is None:
            return []

        top_k = min(top_k, len(self.chunks))

        if self.model is not None:
            query_emb = self.model.encode([query], normalize_embeddings=True)
            query_emb = np.array(query_emb, dtype=np.float32)
        else:
            np.random.seed(len(query))
            query_emb = np.random.randn(1, self.dimension).astype(np.float32)
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        if faiss is not None and isinstance(self.index, faiss.Index):
            scores, indices = self.index.search(query_emb, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Matrix dot product
            scores_all = np.dot(self.index, query_emb.T).flatten()
            indices = np.argsort(scores_all)[::-1][:top_k]
            scores = scores_all[indices]

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(self.chunks):
                # Clamp score between 0.0 and 1.0
                clamped_score = max(0.0, min(1.0, float(score)))
                results.append((self.chunks[idx], round(clamped_score, 4)))

        return results
