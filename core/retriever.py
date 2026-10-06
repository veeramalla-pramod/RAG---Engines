"""In-memory vector retrieval and cosine similarity ranking engine."""

import numpy as np
from typing import List, Dict, Any, Tuple
from core.chunker import Chunk


class VectorRetriever:
    """In-memory cosine similarity search and ranking engine."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.doc_matrix: np.ndarray = np.empty((0, 0))

    def index(self, chunks: List[Chunk], doc_matrix: np.ndarray):
        """Index chunks with their corresponding embedding matrix."""
        self.chunks = chunks
        self.doc_matrix = doc_matrix

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Compute cosine similarity for query vector against indexed chunks.

        Returns:
            ranked_results: List of dicts with chunk data, score, margin_to_next.
            all_scores: 1D array of cosine similarities for all indexed chunks.
        """
        if len(self.chunks) == 0 or self.doc_matrix.size == 0 or query_vector.size == 0:
            return [], np.array([])

        q_vec = query_vector.flatten()
        q_norm = np.linalg.norm(q_vec)

        # Handle zero query vector (e.g. empty query or OOV in TF-IDF)
        if q_norm == 0.0:
            all_scores = np.zeros(len(self.chunks), dtype=np.float32)
            return [], all_scores

        # Compute cosine similarity across all document vectors
        doc_norms = np.linalg.norm(self.doc_matrix, axis=1)
        # Avoid division by zero
        doc_norms[doc_norms == 0.0] = 1e-12

        dot_products = np.dot(self.doc_matrix, q_vec)
        all_scores = dot_products / (doc_norms * q_norm)
        # Clip to [0.0, 1.0] for non-negative representation
        all_scores = np.clip(all_scores, 0.0, 1.0)

        # Sort indices in descending order
        sorted_indices = np.argsort(all_scores)[::-1]

        # Filter by threshold and take top_k
        filtered_indices = [idx for idx in sorted_indices if all_scores[idx] >= threshold]
        top_indices = filtered_indices[:top_k]

        ranked_results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices):
            score = float(all_scores[idx])
            chunk = self.chunks[idx]

            # Compute margin to next rank
            if rank < len(top_indices) - 1:
                next_score = float(all_scores[top_indices[rank + 1]])
                margin = round(score - next_score, 4)
            else:
                margin = 0.0

            ranked_results.append({
                "rank": rank + 1,
                "chunk_id": chunk.chunk_id,
                "chunk_index": idx,
                "score": round(score, 4),
                "margin_to_next": margin,
                "text": chunk.text,
                "word_count": chunk.word_count,
                "token_count": chunk.token_count,
                "start_idx": chunk.start_idx,
                "end_idx": chunk.end_idx,
                "snippet": (chunk.text[:140] + "...") if len(chunk.text) > 140 else chunk.text,
            })

        return ranked_results, all_scores
