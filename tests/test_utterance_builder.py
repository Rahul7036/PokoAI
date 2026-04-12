"""
Tests for the UtteranceBuilder — validates silence detection, buffering,
mid-sentence pause handling, flush, and reset.
"""

import asyncio
import pytest
from core.utterance_builder import UtteranceBuilder


@pytest.mark.asyncio
async def test_single_final_emits_after_silence():
    """A single final chunk should emit after the silence threshold."""
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=200)
    ub.set_callback(on_utterance)

    await ub.add_final("What is your experience")
    assert emitted == []  # Not yet

    await asyncio.sleep(0.35)  # Wait past threshold
    assert len(emitted) == 1
    assert emitted[0] == "What is your experience"


@pytest.mark.asyncio
async def test_multiple_finals_merge_into_one_utterance():
    """Consecutive finals within the silence window merge into one utterance."""
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=300)
    ub.set_callback(on_utterance)

    await ub.add_final("Tell me about")
    await asyncio.sleep(0.1)
    await ub.add_final("your experience with Python")

    await asyncio.sleep(0.5)
    assert len(emitted) == 1
    assert emitted[0] == "Tell me about your experience with Python"


@pytest.mark.asyncio
async def test_interim_resets_timer():
    """Interim results should reset the silence timer (speaker still talking)."""
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=200)
    ub.set_callback(on_utterance)

    await ub.add_final("What is")
    await asyncio.sleep(0.1)
    await ub.on_interim()  # Reset timer — speaker still talking
    await asyncio.sleep(0.1)
    await ub.on_interim()  # Reset again
    await asyncio.sleep(0.1)
    assert emitted == []  # Should not have emitted yet

    await ub.add_final("your biggest strength")
    await asyncio.sleep(0.35)
    assert len(emitted) == 1
    assert "What is" in emitted[0]
    assert "your biggest strength" in emitted[0]


@pytest.mark.asyncio
async def test_mid_sentence_pause_then_continue():
    """A mid-sentence pause followed by more speech should NOT emit early."""
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=300)
    ub.set_callback(on_utterance)

    await ub.add_final("Can you tell me")
    await asyncio.sleep(0.15)  # Short pause (< threshold)
    await ub.add_final("about your experience with Docker")

    assert emitted == []  # Should not have emitted mid-sentence

    await asyncio.sleep(0.5)  # Now wait for real silence
    assert len(emitted) == 1
    assert "Can you tell me about your experience with Docker" == emitted[0]


@pytest.mark.asyncio
async def test_two_separate_utterances():
    """Two sentences with a real silence gap should emit separately."""
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=150)
    ub.set_callback(on_utterance)

    await ub.add_final("First question")
    await asyncio.sleep(0.3)  # Wait past threshold → emits
    assert len(emitted) == 1

    await ub.add_final("Second question")
    await asyncio.sleep(0.3)
    assert len(emitted) == 2
    assert emitted[0] == "First question"
    assert emitted[1] == "Second question"


@pytest.mark.asyncio
async def test_flush_force_emits():
    """flush() should emit whatever is buffered immediately."""
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=5000)  # Very long threshold
    ub.set_callback(on_utterance)

    await ub.add_final("Buffered text")
    assert emitted == []

    await ub.flush()
    assert len(emitted) == 1
    assert emitted[0] == "Buffered text"


@pytest.mark.asyncio
async def test_flush_empty_buffer_does_nothing():
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=200)
    ub.set_callback(on_utterance)

    await ub.flush()
    assert emitted == []


@pytest.mark.asyncio
async def test_reset_discards_buffer():
    """reset() should discard buffered text without emitting."""
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=200)
    ub.set_callback(on_utterance)

    await ub.add_final("This should be discarded")
    ub.reset()

    await asyncio.sleep(0.4)
    assert emitted == []  # Nothing emitted


@pytest.mark.asyncio
async def test_empty_text_ignored():
    """Empty or whitespace-only finals should not be buffered."""
    emitted = []

    async def on_utterance(text):
        emitted.append(text)

    ub = UtteranceBuilder(silence_threshold_ms=200)
    ub.set_callback(on_utterance)

    await ub.add_final("")
    await ub.add_final("   ")
    await asyncio.sleep(0.4)
    assert emitted == []


@pytest.mark.asyncio
async def test_no_callback_set():
    """Should not crash if no callback is registered."""
    ub = UtteranceBuilder(silence_threshold_ms=100)
    await ub.add_final("Test")
    await asyncio.sleep(0.2)
    # No assertion — just verifying no exception
