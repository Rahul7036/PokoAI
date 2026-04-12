"""
Tests for the SpeakerFilter — validates auto-assignment, manual overrides,
and passthrough behavior when diarization data is missing.
"""

import pytest
from core.speaker_filter import SpeakerFilter


@pytest.fixture
def sf():
    return SpeakerFilter()





class TestManualAssignment:
    def test_manual_interviewer_tag(self, sf):
        sf.set_interviewer_tag(2)
        assert sf.is_interviewer(2) is True
        assert sf.is_interviewer(1) is False

    def test_manual_candidate_tag(self, sf):
        sf.set_candidate_tag(1)
        sf.set_interviewer_tag(2)
        assert sf.is_interviewer(1) is False


class TestPassthrough:
    def test_no_tags_passes_through(self, sf):
        """When no speaker tags are available, pass all speech through."""
        assert sf.is_interviewer(None) is True

    def test_empty_words_returns_none(self, sf):
        tag = sf.process_speaker_tags([])
        assert tag is None
        assert sf.is_interviewer(tag) is True

    def test_zero_tags_ignored(self, sf):
        """Words with speaker_tag=0 (unassigned) are ignored."""
        words = [
            {"word": "Hello", "speaker_tag": 0},
            {"word": "world", "speaker_tag": 0},
        ]
        tag = sf.process_speaker_tags(words)
        assert tag is None


class TestDominantSpeaker:
    def test_majority_speaker_wins(self, sf):
        words = [
            {"word": "What", "speaker_tag": 1},
            {"word": "is", "speaker_tag": 1},
            {"word": "your", "speaker_tag": 2},  # minority
            {"word": "experience", "speaker_tag": 1},
        ]
        tag = sf.process_speaker_tags(words)
        assert tag == 1  # speaker 1 is dominant


class TestReset:
    def test_reset_clears_assignments(self, sf):
        sf.set_interviewer_tag(1)
        sf.set_candidate_tag(2)
        sf.reset()

        # After reset, everything passes through
        assert sf.is_interviewer(1) is True
        assert sf.is_interviewer(2) is True
