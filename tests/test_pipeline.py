"""
Integration tests for the InterviewPipeline — exercises the full flow 
from transcript → filter → LLM → WebSocket using mocks.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.session import Session
from core.pipeline import InterviewPipeline
from core.llm_stream import LLMStreamer
from core.state_machine import AIState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeWebSocket:
    """Captures messages sent to the WebSocket."""

    def __init__(self):
        self.messages = []

    async def send_text(self, data: str):
        self.messages.append(json.loads(data))

    def get_messages_of_type(self, msg_type: str):
        return [m for m in self.messages if m.get("type") == msg_type]


class FakeSTT:
    """Minimal STT stub — pipeline calls set_*_callback and start/stop."""

    def __init__(self):
        self._on_transcript = None
        self._on_error = None

    def set_transcript_callback(self, cb):
        self._on_transcript = cb

    def set_error_callback(self, cb):
        self._on_error = cb

    def start(self, loop):
        pass

    def stop(self):
        pass

    def feed_audio(self, data):
        pass

    async def fire_transcript(self, text, is_final, speaker_words=None):
        if self._on_transcript:
            await self._on_transcript(text, is_final, speaker_words)


class FakeModel:
    """Fake Vertex AI model."""

    def __init__(self, chunks):
        self._chunks = chunks

    def generate_content(self, prompt, stream=False):
        if stream:
            return iter([type("C", (), {"text": c})() for c in self._chunks])
        return type("R", (), {"text": "".join(self._chunks)})()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session():
    s = Session("test-session", "user@test.com")
    s.update_context(
        resume="Experienced Python developer.",
        jd="Looking for a Python engineer.",
        company="TestCorp",
    )
    return s


@pytest.fixture
def ws():
    return FakeWebSocket()


@pytest.fixture
def stt():
    return FakeSTT()


def make_pipeline(session, ws, stt, model_chunks=None):
    chunks = model_chunks or ["Great ", "question! ", "I have ", "5 years of experience."]
    model = FakeModel(chunks)
    llm = LLMStreamer(model)
    return InterviewPipeline(session, ws, stt, llm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_question_produces_streamed_response(self, session, ws, stt):
        """A real interview question should produce a streamed answer."""
        pipeline = make_pipeline(session, ws, stt)
        await pipeline.start()

        # Simulate: final transcript → triggers utterance builder
        await stt.fire_transcript(
            "What is your experience with Python?", is_final=True
        )

        # Wait for silence threshold + processing
        await asyncio.sleep(1.8)

        await pipeline.stop()

        # Should have answer chunks
        chunks = ws.get_messages_of_type("answer_chunk")
        assert len(chunks) > 0

        # Should have answer_complete
        completes = ws.get_messages_of_type("answer_complete")
        assert len(completes) == 1
        assert "question" in completes[0]

        # Should have state updates
        states = ws.get_messages_of_type("state")
        state_values = [s["state"] for s in states]
        assert "listening" in state_values

    @pytest.mark.asyncio
    async def test_filler_does_not_trigger_response(self, session, ws, stt):
        """Fillers like 'okay' should be filtered out."""
        pipeline = make_pipeline(session, ws, stt)
        await pipeline.start()

        await stt.fire_transcript("okay", is_final=True)
        await asyncio.sleep(1.5)

        await pipeline.stop()

        chunks = ws.get_messages_of_type("answer_chunk")
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_duplicate_questions_produce_single_response(self, session, ws, stt):
        """The same question repeated quickly should only produce one response."""
        pipeline = make_pipeline(session, ws, stt)
        await pipeline.start()

        await stt.fire_transcript(
            "Tell me about your Docker experience.", is_final=True
        )
        await asyncio.sleep(0.05)
        await stt.fire_transcript(
            "Tell me about your Docker experience.", is_final=True
        )

        await asyncio.sleep(2.0)
        await pipeline.stop()

        completes = ws.get_messages_of_type("answer_complete")
        assert len(completes) == 1  # Only one response

    @pytest.mark.asyncio
    async def test_candidate_speech_filtered_by_speaker_tag(self, session, ws, stt):
        """Speech from the candidate (speaker_tag 2) should be ignored."""
        pipeline = make_pipeline(session, ws, stt)
        await pipeline.start()

        # First: interviewer speaks (tag 1) to establish roles
        await stt.fire_transcript(
            "Tell me about yourself.",
            is_final=True,
            speaker_words=[
                {"word": "Tell", "speaker_tag": 1},
                {"word": "me", "speaker_tag": 1},
                {"word": "about", "speaker_tag": 1},
                {"word": "yourself", "speaker_tag": 1},
            ],
        )
        await asyncio.sleep(1.5)

        # Now: candidate speaks (tag 2) — should be filtered
        ws.messages.clear()
        await stt.fire_transcript(
            "I have five years of experience with Python.",
            is_final=True,
            speaker_words=[
                {"word": "I", "speaker_tag": 2},
                {"word": "have", "speaker_tag": 2},
                {"word": "five", "speaker_tag": 2},
                {"word": "years", "speaker_tag": 2},
            ],
        )
        await asyncio.sleep(1.5)
        await pipeline.stop()

        # No answer should be generated for candidate speech
        chunks = ws.get_messages_of_type("answer_chunk")
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_no_answer_response_suppressed(self, session, ws, stt):
        """LLM returning NO_ANSWER should not produce answer messages."""
        pipeline = make_pipeline(
            session, ws, stt, model_chunks=["NO_ANSWER"]
        )
        await pipeline.start()

        await stt.fire_transcript(
            "What do you think about the weather today?", is_final=True
        )
        await asyncio.sleep(1.8)
        await pipeline.stop()

        completes = ws.get_messages_of_type("answer_complete")
        assert len(completes) == 0

    @pytest.mark.asyncio
    async def test_interim_results_relayed_but_not_processed(self, session, ws, stt):
        """Interim transcripts are sent to the client but don't trigger AI."""
        pipeline = make_pipeline(session, ws, stt)
        await pipeline.start()

        await stt.fire_transcript("What is your", is_final=False)
        await asyncio.sleep(0.5)

        await pipeline.stop()

        transcripts = ws.get_messages_of_type("transcript")
        assert len(transcripts) >= 1
        assert transcripts[0]["is_final"] is False

        # No answer chunks (only interim, no final)
        chunks = ws.get_messages_of_type("answer_chunk")
        assert len(chunks) == 0


class TestStateTransitions:
    @pytest.mark.asyncio
    async def test_state_goes_through_full_lifecycle(self, session, ws, stt):
        """State should go LISTENING → PROCESSING → RESPONDING → LISTENING."""
        pipeline = make_pipeline(session, ws, stt)
        await pipeline.start()

        await stt.fire_transcript(
            "Explain your experience with microservices.", is_final=True
        )
        await asyncio.sleep(2.0)
        await pipeline.stop()

        states = ws.get_messages_of_type("state")
        values = [s["state"] for s in states]
        # Should see listening at start and end at minimum
        assert values[0] == "listening"
        assert values[-1] == "listening"


class TestMultiSentenceQuestion:
    @pytest.mark.asyncio
    async def test_multi_part_question_merged(self, session, ws, stt):
        """Multiple finals within silence window should merge into one utterance."""
        pipeline = make_pipeline(session, ws, stt)
        await pipeline.start()

        await stt.fire_transcript("So tell me,", is_final=True)
        await asyncio.sleep(0.2)
        await stt.fire_transcript(
            "what is your experience with Kubernetes?", is_final=True
        )

        await asyncio.sleep(2.0)
        await pipeline.stop()

        completes = ws.get_messages_of_type("answer_complete")
        assert len(completes) == 1
        # The question should contain both parts
        q = completes[0]["question"]
        assert "tell me" in q.lower()
        assert "kubernetes" in q.lower()
