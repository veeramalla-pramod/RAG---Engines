"""Token counting and telemetry module using tiktoken with budget analytics."""

from typing import List, Dict, Any
import numpy as np

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _ENCODER = None


def count_tokens(text: str) -> int:
    """Return exact token count using cl100k_base or word approximation fallback."""
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Approximate 1 token ~= 0.75 words (or ~1.33 tokens per word)
    return int(len(text.split()) * 1.33)


def calculate_token_telemetry(raw_text: str, chunks: list, prompt_overhead: int = 250, completion_budget: int = 500, max_context_window: int = 4096) -> Dict[str, Any]:
    """
    Compute comprehensive token telemetry as specified in PRD Section 5.1.
    """
    raw_token_count = count_tokens(raw_text)
    
    if not chunks:
        return {
            "raw_token_count": raw_token_count,
            "total_chunk_tokens": 0,
            "min_chunk_tokens": 0,
            "max_chunk_tokens": 0,
            "mean_chunk_tokens": 0.0,
            "inflation_ratio": 1.0,
            "prompt_overhead": prompt_overhead,
            "completion_budget": completion_budget,
            "max_context_window": max_context_window,
        }

    chunk_tokens = [c.token_count for c in chunks]
    total_chunk_tokens = sum(chunk_tokens)
    inflation_ratio = (
        round(total_chunk_tokens / raw_token_count, 3)
        if raw_token_count > 0
        else 1.0
    )

    return {
        "raw_token_count": raw_token_count,
        "total_chunk_tokens": total_chunk_tokens,
        "min_chunk_tokens": int(np.min(chunk_tokens)),
        "max_chunk_tokens": int(np.max(chunk_tokens)),
        "mean_chunk_tokens": round(float(np.mean(chunk_tokens)), 1),
        "inflation_ratio": inflation_ratio,
        "prompt_overhead": prompt_overhead,
        "completion_budget": completion_budget,
        "max_context_window": max_context_window,
    }


def compute_context_budget(retrieved_chunk_tokens: int, prompt_overhead: int = 250, completion_budget: int = 500, max_context: int = 4096) -> Dict[str, Any]:
    """
    Calculates context consumption for the budget progress gauge.
    """
    total_consumed = prompt_overhead + retrieved_chunk_tokens + completion_budget
    utilization_pct = min(100.0, round((total_consumed / max_context) * 100, 2))

    return {
        "prompt_overhead": prompt_overhead,
        "retrieved_tokens": retrieved_chunk_tokens,
        "completion_budget": completion_budget,
        "total_consumed": total_consumed,
        "max_context": max_context,
        "remaining_tokens": max(0, max_context - total_consumed),
        "utilization_pct": utilization_pct,
    }
