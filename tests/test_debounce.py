"""
Tests for the Debounce layer — validates exact-duplicate blocking,
similarity-based blocking, time-window expiry, and reset.
"""

import time
import pytest
from core.debounce import Debounce


class TestExactDuplicate:
    def test_exact_duplicate_blocked(self):
        db = Debounce(window_seconds=10)
        assert db.should_process("What is your experience?") is True
        assert db.should_process("What is your experience?") is False

    def test_case_insensitive_duplicate(self):
        db = Debounce(window_seconds=10)
        assert db.should_process("Tell me about Python") is True
        assert db.should_process("tell me about python") is False

    def test_whitespace_normalized(self):
        db = Debounce(window_seconds=10)
        assert db.should_process("What  is   your experience") is True
        assert db.should_process("What is your experience") is False


class TestSimilarityDuplicate:
    def test_similar_text_blocked(self):
        db = Debounce(window_seconds=10, similarity_threshold=0.7)
        assert db.should_process("What is your experience with Python") is True
        # Very similar question
        assert db.should_process("What is your experience with Python programming") is False

    def test_different_text_passes(self):
        db = Debounce(window_seconds=10)
        assert db.should_process("What is your experience with Python?") is True
        assert db.should_process("Tell me about your hobbies.") is True

    def test_completely_different_passes(self):
        db = Debounce(window_seconds=10)
        assert db.should_process("Explain microservices architecture") is True
        assert db.should_process("What salary are you expecting") is True


class TestTimeWindow:
    def test_expired_entry_allows_reprocessing(self):
        db = Debounce(window_seconds=0.2)  # 200ms window
        assert db.should_process("What is Python?") is True
        assert db.should_process("What is Python?") is False

        time.sleep(0.3)  # Wait past window
        assert db.should_process("What is Python?") is True

    def test_within_window_still_blocked(self):
        db = Debounce(window_seconds=1.0)
        assert db.should_process("Question one") is True
        time.sleep(0.1)
        assert db.should_process("Question one") is False


class TestReset:
    def test_reset_clears_state(self):
        db = Debounce(window_seconds=10)
        assert db.should_process("What is X?") is True
        assert db.should_process("What is X?") is False

        db.reset()
        assert db.should_process("What is X?") is True


class TestEdgeCases:
    def test_empty_text(self):
        db = Debounce()
        assert db.should_process("") is True
        # Empty string hashes the same way
        assert db.should_process("") is False

    def test_many_different_questions(self):
        db = Debounce(window_seconds=10)
        for i in range(50):
            assert db.should_process(f"Unique question number {i}") is True

    def test_rapid_duplicates(self):
        """Simulates rapid-fire identical transcripts from STT."""
        db = Debounce(window_seconds=10)
        text = "What is your experience with Docker and Kubernetes?"
        assert db.should_process(text) is True
        for _ in range(10):
            assert db.should_process(text) is False
