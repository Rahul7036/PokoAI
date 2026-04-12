"""
Debounce Layer — prevents duplicate AI responses.

Two-level deduplication:
    1. **Exact hash match**: normalized text is SHA-256 hashed. Identical
       questions within the time window are suppressed.
    2. **Similarity match**: Jaccard similarity of word sets. Near-duplicate
       questions (e.g. same question with minor STT differences) are caught.
"""

import hashlib
import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)


class Debounce:
    """
    Time-window deduplication for utterances.

    Parameters
    ----------
    window_seconds : float
        How long (in seconds) to remember processed utterances.
    similarity_threshold : float
        Jaccard similarity ≥ this value counts as a duplicate (0.0–1.0).
    """

    def __init__(
        self,
        window_seconds: float = 15.0,
        similarity_threshold: float = 0.8,
    ):
        self._window = window_seconds
        self._similarity_threshold = similarity_threshold
        self._hash_log: Dict[str, float] = {}   # hash  → timestamp
        self._text_log: Dict[str, float] = {}   # text  → timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_process(self, text: str) -> bool:
        """
        Return ``True`` if *text* is **not** a duplicate.

        Must be called before sending the utterance to the LLM.
        """
        now = time.monotonic()
        self._cleanup(now)

        text_hash = self._normalize_hash(text)

        # 1) Exact-duplicate guard
        if text_hash in self._hash_log:
            logger.debug(f"Debounce: exact dup blocked — '{text[:40]}…'")
            return False

        # 2) Similarity guard
        for prev_text, ts in list(self._text_log.items()):
            if now - ts < self._window:
                if self._jaccard(text, prev_text) >= self._similarity_threshold:
                    logger.debug(f"Debounce: similar dup blocked — '{text[:40]}…'")
                    return False

        # Record
        self._hash_log[text_hash] = now
        self._text_log[text] = now
        return True

    def reset(self):
        """Clear all deduplication state."""
        self._hash_log.clear()
        self._text_log.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_hash(text: str) -> str:
        normalized = " ".join(text.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _cleanup(self, now: float):
        expired = [h for h, t in self._hash_log.items() if now - t > self._window]
        for h in expired:
            del self._hash_log[h]

        expired_t = [t for t, ts in self._text_log.items() if now - ts > self._window]
        for t in expired_t:
            del self._text_log[t]
