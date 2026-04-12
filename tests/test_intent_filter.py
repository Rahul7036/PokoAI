"""
Tests for the IntentFilter — validates filler detection, question detection,
and edge cases.
"""

import pytest
from core.intent_filter import IntentFilter


@pytest.fixture
def intent_filter():
    return IntentFilter(min_word_count=3)


class TestFillerDetection:
    """Fillers and acknowledgments must be filtered out."""

    @pytest.mark.parametrize("filler", [
        "yeah", "okay", "hmm", "right", "sure", "got it",
        "uh huh", "mhm", "thanks", "hello", "bye",
        "cool", "awesome", "perfect", "interesting",
        "sorry", "pardon", "no", "nope",
        "Yeah", "OKAY", "Hmm",  # case insensitive
        "yeah.", "okay!", "hmm...",  # with punctuation
    ])
    def test_fillers_are_not_meaningful(self, intent_filter, filler):
        assert intent_filter.is_meaningful(filler) is False

    @pytest.mark.parametrize("filler", [
        "i see", "i understand", "got you", "gotcha",
        "one moment", "just a moment", "hold on",
        "can you hear me", "am i audible",
        "let me think", "moving on", "please continue",
    ])
    def test_multi_word_fillers_filtered(self, intent_filter, filler):
        assert intent_filter.is_meaningful(filler) is False


class TestQuestionDetection:
    """Meaningful questions and commands must pass through."""

    @pytest.mark.parametrize("question", [
        "What is your experience with Python?",
        "How do you handle tight deadlines?",
        "Tell me about yourself.",
        "Can you explain your approach to testing?",
        "Why did you leave your previous job?",
        "Describe a challenging project you worked on.",
        "Walk me through your resume.",
        "What are your biggest strengths and weaknesses?",
        "Could you share an example of a time you led a team?",
        "Do you have experience with microservices architecture?",
    ])
    def test_real_questions_are_meaningful(self, intent_filter, question):
        assert intent_filter.is_meaningful(question) is True

    @pytest.mark.parametrize("command", [
        "Give me a code example in Python.",
        "Explain the difference between REST and GraphQL.",
        "List three design patterns you've used.",
        "Share your experience with cloud platforms.",
    ])
    def test_commands_are_meaningful(self, intent_filter, command):
        assert intent_filter.is_meaningful(command) is True


class TestShortUtterances:
    """Short utterances (< min_word_count) need question indicators to pass."""

    def test_short_question_passes(self, intent_filter):
        assert intent_filter.is_meaningful("What happened?") is True

    def test_short_non_question_blocked(self, intent_filter):
        assert intent_filter.is_meaningful("good morning") is False

    def test_single_word_question(self, intent_filter):
        assert intent_filter.is_meaningful("Why?") is True

    def test_single_word_non_question(self, intent_filter):
        assert intent_filter.is_meaningful("Python") is False


class TestEdgeCases:
    def test_empty_string(self, intent_filter):
        assert intent_filter.is_meaningful("") is False

    def test_whitespace_only(self, intent_filter):
        assert intent_filter.is_meaningful("   ") is False

    def test_none_is_not_meaningful(self, intent_filter):
        # Passing None should not crash
        assert intent_filter.is_meaningful(None) is False

    def test_long_filler_sentence_passes(self, intent_filter):
        """Even if individual words are filly, a long enough sentence passes."""
        assert intent_filter.is_meaningful(
            "So I was thinking about the Docker deployment pipeline"
        ) is True

    def test_mixed_filler_and_content(self, intent_filter):
        """A real question with filler words embedded should pass."""
        assert intent_filter.is_meaningful(
            "So like what is your approach to database optimization"
        ) is True
