"""Sliding-window chunking engine with configurable size, overlap, and boundary tracking."""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Chunk:
    chunk_id: int
    text: str
    word_count: int
    token_count: int
    start_idx: int
    end_idx: int
    overlap_with_next: str = ""


class SlidingWindowChunker:
    """Configurable sliding-window text chunker supporting word and token-based chunking."""

    def __init__(self, mode: str = "words"):
        """
        Args:
            mode: 'words' or 'tokens'
        """
        self.mode = mode

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 100,
        overlap_percentage: float = 0.20,
        token_counter=None
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Raw input text.
            chunk_size: Chunk size in words (or tokens if mode='tokens').
            overlap_percentage: Overlap ratio [0.0, 0.50].
            token_counter: Optional callable(str) -> int.
        """
        if not text or not text.strip():
            return []

        # Clean normalized whitespace but preserve sentence structure
        words = text.split()
        total_units = len(words)

        if total_units == 0:
            return []

        # Bound inputs
        chunk_size = max(10, int(chunk_size))
        overlap_percentage = max(0.0, min(0.50, float(overlap_percentage)))
        overlap_units = int(chunk_size * overlap_percentage)
        step_size = max(1, chunk_size - overlap_units)

        chunks: List[Chunk] = []
        start_idx = 0
        chunk_id = 1

        while start_idx < total_units:
            end_idx = min(start_idx + chunk_size, total_units)
            chunk_words = words[start_idx:end_idx]
            chunk_str = " ".join(chunk_words)

            t_count = token_counter(chunk_str) if token_counter else len(chunk_words)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    word_count=len(chunk_words),
                    token_count=t_count,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )

            chunk_id += 1
            if end_idx >= total_units:
                break
            start_idx += step_size

        # Compute overlap text with next chunk for contiguous inspection
        for i in range(len(chunks) - 1):
            curr_c = chunks[i]
            next_c = chunks[i + 1]
            overlap_count = max(0, curr_c.end_idx - next_c.start_idx)
            if overlap_count > 0:
                overlapping_words = words[next_c.start_idx:curr_c.end_idx]
                curr_c.overlap_with_next = " ".join(overlapping_words)

        return chunks

    @staticmethod
    def get_overlap_diff(chunk_a: Chunk, chunk_b: Chunk) -> dict:
        """
        Compute diff / shared text between two contiguous chunks.
        """
        words_a = chunk_a.text.split()
        words_b = chunk_b.text.split()

        # Find overlapping sequence from end of A matching start of B
        max_overlap_len = min(len(words_a), len(words_b))
        best_overlap_len = 0

        for l in range(1, max_overlap_len + 1):
            if words_a[-l:] == words_b[:l]:
                best_overlap_len = l

        if best_overlap_len > 0:
            unique_a = " ".join(words_a[:-best_overlap_len])
            shared = " ".join(words_a[-best_overlap_len:])
            unique_b = " ".join(words_b[best_overlap_len:])
        else:
            unique_a = chunk_a.text
            shared = ""
            unique_b = chunk_b.text

        return {
            "chunk_a_id": chunk_a.chunk_id,
            "chunk_b_id": chunk_b.chunk_id,
            "unique_a": unique_a,
            "shared": shared,
            "unique_b": unique_b,
            "overlap_words": best_overlap_len,
        }
