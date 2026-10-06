"""Dual-mode vectorization engine: Sparse TF-IDF & Dense Sentence-Transformers."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

_DENSE_MODEL_CACHE = {}


class BaseEmbedder:
    """Abstract interface for embeddings."""

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def transform_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def get_telemetry(self) -> Dict[str, Any]:
        raise NotImplementedError


class TFIDFEmbedder(BaseEmbedder):
    """Base Mode (Exploratory): Scikit-Learn TfidfVectorizer with vocabulary telemetry."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            norm="l2",
            ngram_range=(1, 2)
        )
        self.matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if not texts:
            self.matrix = np.empty((0, 0))
            return self.matrix

        sparse_matrix = self.vectorizer.fit_transform(texts)
        self.matrix = sparse_matrix.toarray()
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        self.is_fitted = True
        return self.matrix

    def transform_query(self, query: str) -> np.ndarray:
        if not self.is_fitted or not query.strip():
            return np.zeros((1, len(self.feature_names)))
        sparse_vec = self.vectorizer.transform([query])
        return sparse_vec.toarray()

    def get_top_keywords_for_chunk(self, chunk_index: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Return top-N weighted TF-IDF terms for a chunk."""
        if self.matrix is None or chunk_index >= len(self.matrix):
            return []

        row = self.matrix[chunk_index]
        non_zero_indices = np.where(row > 0)[0]
        if len(non_zero_indices) == 0:
            return []

        sorted_indices = non_zero_indices[np.argsort(row[non_zero_indices])[::-1]]
        top_indices = sorted_indices[:top_n]
        return [(self.feature_names[i], round(float(row[i]), 4)) for i in top_indices]

    def get_telemetry(self) -> Dict[str, Any]:
        """Compute vocabulary dimension and matrix sparsity percentage."""
        if self.matrix is None or self.matrix.size == 0:
            return {
                "mode": "Sparse TF-IDF",
                "n_chunks": 0,
                "dimension": 0,
                "sparsity_pct": 100.0,
                "total_vocabulary": 0,
                "dense_dimension": None,
            }

        n_chunks, n_features = self.matrix.shape
        total_elements = n_chunks * n_features
        zero_elements = np.sum(self.matrix == 0.0)
        sparsity_pct = (
            round((zero_elements / total_elements) * 100.0, 2)
            if total_elements > 0
            else 0.0
        )

        return {
            "mode": "Sparse TF-IDF",
            "n_chunks": n_chunks,
            "dimension": n_features,
            "sparsity_pct": sparsity_pct,
            "total_vocabulary": n_features,
            "dense_dimension": None,
        }


class DenseEmbedder(BaseEmbedder):
    """Advanced Mode (Semantic): Transformer dense embeddings (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = self._get_cached_model(model_name)
        self.matrix: Optional[np.ndarray] = None
        self.is_fitted = False

    @staticmethod
    def _get_cached_model(model_name: str):
        if model_name not in _DENSE_MODEL_CACHE:
            from sentence_transformers import SentenceTransformer
            _DENSE_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        return _DENSE_MODEL_CACHE[model_name]

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if not texts:
            self.matrix = np.empty((0, 384))
            return self.matrix

        # Encode and normalize for L2 cosine dot products
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.matrix = np.asarray(embeddings, dtype=np.float32)
        self.is_fitted = True
        return self.matrix

    def transform_query(self, query: str) -> np.ndarray:
        if not query.strip():
            dim = self.matrix.shape[1] if self.matrix is not None and self.matrix.size > 0 else 384
            return np.zeros((1, dim), dtype=np.float32)
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(q_emb, dtype=np.float32)

    def get_telemetry(self) -> Dict[str, Any]:
        dim = self.matrix.shape[1] if self.matrix is not None and self.matrix.size > 0 else 384
        n_chunks = self.matrix.shape[0] if self.matrix is not None else 0
        return {
            "mode": f"Dense Transformer ({self.model_name.split('/')[-1]})",
            "n_chunks": n_chunks,
            "dimension": dim,
            "sparsity_pct": 0.0,
            "total_vocabulary": "Continuous Latent Space",
            "dense_dimension": dim,
        }


def get_embedder(mode: str = "Dense (Semantic)") -> BaseEmbedder:
    """Factory to retrieve TF-IDF or Dense embedder."""
    if "tfidf" in mode.lower() or "sparse" in mode.lower() or "base" in mode.lower():
        return TFIDFEmbedder()
    return DenseEmbedder()
