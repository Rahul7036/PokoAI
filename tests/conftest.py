"""
Shared fixtures and configuration for the PokoAI test suite.
"""

import asyncio
import sys
import os
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class FakeModel:
    """
    Lightweight fake for ``vertexai.generative_models.GenerativeModel``.

    Supports both streaming and non-streaming calls.
    """

    def __init__(self, responses=None, stream_chunks=None):
        self._responses = responses or ["This is a test answer."]
        self._stream_chunks = stream_chunks or ["This ", "is ", "a ", "test ", "answer."]
        self._call_count = 0

    def generate_content(self, prompt, stream=False):
        self._call_count += 1
        self._last_prompt = prompt
        if stream:
            return iter(
                [_FakeChunk(c) for c in self._stream_chunks]
            )
        return _FakeResponse(self._responses[0])


class _FakeChunk:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.text = text


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def no_answer_model():
    """Model that returns NO_ANSWER."""
    return FakeModel(
        responses=["NO_ANSWER"],
        stream_chunks=["NO_ANSWER"],
    )
