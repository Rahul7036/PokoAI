"""
Interview Pipeline — orchestrates the complete audio-to-response flow.

    Audio bytes
      → STT Manager  (streaming recognition, auto-restart)
      → Transcript relay to client (interim + final)
      → Utterance Builder (pause-based sentence assembly)
      → Speaker Filter (ignore candidate speech)
      → Intent Filter (ignore fillers)
      → Debounce (suppress duplicates)
      → State Machine gate (only process when LISTENING)
      → Streaming LLM (token-by-token response)
      → WebSocket streaming to UI

The pipeline runs entirely within one asyncio event loop + one STT
background thread, with explicit locks to prevent race conditions.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

from fastapi import WebSocket

from .llm_stream import LLMStreamer
from .session import Session
from .stt_manager import STTManager
from .state_machine import AIState

logger = logging.getLogger(__name__)

# How many characters to buffer before checking for NO_ANSWER
_NO_ANSWER_BUFFER_SIZE = 12


class InterviewPipeline:
    """
    Wires all pipeline stages together for a single user session.

    Parameters
    ----------
    session : Session
        The per-user session holding state, context, and sub-components.
    websocket : WebSocket
        FastAPI WebSocket for sending data to the client.
    stt_manager : STTManager
        Google Speech-to-Text streaming manager.
    llm_streamer : LLMStreamer
        Vertex AI streaming response generator.
    """

    def __init__(
        self,
        session: Session,
        websocket: WebSocket,
        stt_manager: STTManager,
        llm_streamer: LLMStreamer,
    ):
        self._session = session
        self._ws = websocket
        self._stt = stt_manager
        self._llm = llm_streamer
        self._running = False
        self._pending: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None

        # Wire callbacks
        self._stt.set_transcript_callback(self._on_transcript)
        self._stt.set_error_callback(self._on_stt_error)
        self._session.utterance_builder.set_callback(self._on_complete_utterance)
        self._session.state_machine.set_callback(self._on_state_change)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        self._running = True
        loop = asyncio.get_running_loop()
        self._stt.start(loop)
        self._processor_task = asyncio.create_task(self._utterance_processor())
        await self._send_state(AIState.LISTENING)
        logger.info(f"Pipeline started — session {self._session.session_id}")

    async def stop(self):
        self._running = False
        self._stt.stop()
        await self._session.utterance_builder.flush()
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Pipeline stopped — session {self._session.session_id}")

    def feed_audio(self, data: bytes):
        """Forward raw audio bytes into the STT stream."""
        self._stt.feed_audio(data)

    # ------------------------------------------------------------------
    # Stage 1 — STT transcript callback
    # ------------------------------------------------------------------

    async def _on_transcript(
        self,
        text: str,
        is_final: bool,
        speaker_words: Optional[List[Dict]],
    ):
        if not self._running:
            return
        try:
            # Relay to client for live display
            await self._ws.send_text(
                json.dumps(
                    {"type": "transcript", "transcript": text, "is_final": is_final}
                )
            )

            if is_final:
                # Speaker diarization check
                if speaker_words:
                    tag = self._session.speaker_filter.process_speaker_tags(
                        speaker_words
                    )
                    if not self._session.speaker_filter.is_interviewer(tag):
                        logger.debug(f"Speaker filter: candidate speech ignored")
                        return

                # Feed into utterance builder (waits for silence)
                await self._session.utterance_builder.add_final(text)
            else:
                # Interim → reset silence timer
                await self._session.utterance_builder.on_interim()

        except Exception as exc:
            logger.error(f"Transcript handling error: {exc}")

    # ------------------------------------------------------------------
    # Stage 2 — complete utterance callback
    # ------------------------------------------------------------------

    async def _on_complete_utterance(self, utterance: str):
        """Called by UtteranceBuilder when a silence gap is detected."""
        if not self._running:
            return

        logger.info(f"Complete utterance ({len(utterance)} chars): {utterance[:80]}…")

        # Intent filter
        if not self._session.intent_filter.is_meaningful(utterance):
            logger.info(f"Intent filter: skipped")
            return

        # Debounce
        if not self._session.debounce.should_process(utterance):
            logger.info(f"Debounce: duplicate skipped")
            return

        # Enqueue for state-aware processing
        await self._pending.put(utterance)

    # ------------------------------------------------------------------
    # Stage 3 — state-aware utterance processor
    # ------------------------------------------------------------------

    async def _utterance_processor(self):
        """
        Continuously pulls from the pending queue and generates responses,
        respecting the state machine.  Drains stale items to stay current.
        """
        while self._running:
            try:
                utterance = await asyncio.wait_for(
                    self._pending.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Wait until AI is available (LISTENING)
            patience = 0
            while not self._session.state_machine.is_available() and self._running:
                await asyncio.sleep(0.1)
                patience += 1
                if patience > 150:  # 15-second hard cap
                    logger.warning("State machine blocked >15 s — forcing reset")
                    await self._session.state_machine.force_reset()
                    break

            if not self._running:
                break

            # Drain queue — keep only the latest utterance
            latest = utterance
            while not self._pending.empty():
                try:
                    latest = self._pending.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Transition → PROCESSING
            if not await self._session.state_machine.transition_to(AIState.PROCESSING):
                logger.warning("Could not enter PROCESSING")
                continue

            await self._generate_response(latest)

    # ------------------------------------------------------------------
    # Stage 4 — streaming LLM response
    # ------------------------------------------------------------------

    async def _generate_response(self, question: str):
        """Stream an LLM answer back to the client."""
        try:
            # Notify UI
            await self._ws.send_text(
                json.dumps(
                    {"type": "status", "state": "processing", "message": "Thinking…"}
                )
            )

            # Transition → RESPONDING
            await self._session.state_machine.transition_to(AIState.RESPONDING)

            buffer: list = []
            buffer_text = ""
            streaming_started = False
            full_chunks: list = []

            async for chunk in self._llm.generate_stream(
                question=question,
                context=self._session.context,
                history=self._session.conversation_history,
            ):
                if not self._running:
                    break

                # --- NO_ANSWER detection (buffered) ---
                if not streaming_started:
                    buffer.append(chunk)
                    buffer_text += chunk
                    if len(buffer_text) >= _NO_ANSWER_BUFFER_SIZE:
                        if "NO_ANSWER" in buffer_text.strip():
                            logger.info(
                                f"LLM declined: {question[:50]}…"
                            )
                            return  # finally-block resets state
                        # Flush buffer to client
                        streaming_started = True
                        combined = "".join(buffer)
                        full_chunks.append(combined)
                        await self._ws.send_text(
                            json.dumps(
                                {
                                    "type": "answer_chunk",
                                    "chunk": combined,
                                    "question": question,
                                }
                            )
                        )
                        buffer.clear()
                    continue

                full_chunks.append(chunk)
                await self._ws.send_text(
                    json.dumps({"type": "answer_chunk", "chunk": chunk})
                )

            # Handle case where stream ended while still buffering
            if not streaming_started and buffer:
                combined = "".join(buffer)
                if "NO_ANSWER" in combined.strip():
                    logger.info(f"LLM declined (short): {question[:50]}…")
                    return
                full_chunks.append(combined)
                await self._ws.send_text(
                    json.dumps(
                        {
                            "type": "answer_chunk",
                            "chunk": combined,
                            "question": question,
                        }
                    )
                )

            final_answer = "".join(full_chunks).strip()
            if not final_answer:
                return

            # Send completion marker
            await self._ws.send_text(
                json.dumps(
                    {
                        "type": "answer_complete",
                        "question": question,
                        "answer": final_answer,
                    }
                )
            )

            # Persist Q&A
            self._session.add_to_history(question, final_answer)

        except Exception as exc:
            logger.error(f"Response generation error: {exc}")
            try:
                await self._ws.send_text(
                    json.dumps({"type": "error", "message": "Failed to generate response"})
                )
            except Exception:
                pass
        finally:
            await self._session.state_machine.force_reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _on_state_change(self, new_state: AIState):
        await self._send_state(new_state)

    async def _send_state(self, state: AIState):
        try:
            await self._ws.send_text(
                json.dumps({"type": "state", "state": state.value})
            )
        except Exception:
            pass

    async def _on_stt_error(self, exc: Exception):
        logger.error(f"STT error relayed: {exc}")
        try:
            await self._ws.send_text(
                json.dumps(
                    {"type": "error", "message": "Speech recognition reconnecting…"}
                )
            )
        except Exception:
            pass
