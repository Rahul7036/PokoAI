"""
Utterance Builder — accumulates STT chunks and detects complete utterances.

Strategy:
    1. Buffer every ``is_final`` transcript from Google STT.
    2. On each new speech event (interim *or* final), reset a silence timer.
    3. When the timer fires (no speech for ``silence_threshold_ms``), 
       the buffered fragments are joined and emitted as one complete utterance.
    4. This prevents mid-sentence AI triggers — the AI only sees 
       full questions after the speaker pauses.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable, List

logger = logging.getLogger(__name__)


class UtteranceBuilder:
    """
    Pause-based utterance assembler for streaming STT results.

    Parameters
    ----------
    silence_threshold_ms : int
        Milliseconds of silence required before emitting an utterance.
        Recommended range: 800–1200 ms.
    """

    def __init__(self, silence_threshold_ms: int = 1000):
        self._buffer: List[str] = []
        self._silence_threshold: float = silence_threshold_ms / 1000.0
        self._timer_task: Optional[asyncio.Task] = None
        self._on_utterance: Optional[Callable[[str], Awaitable[None]]] = None
        self._last_activity: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_callback(self, callback: Callable[[str], Awaitable[None]]):
        """Register the handler called when a complete utterance is ready."""
        self._on_utterance = callback

    async def add_final(self, text: str):
        """
        Append a *final* STT result to the buffer and (re)start the timer.
        """
        async with self._lock:
            cleaned = text.strip()
            if not cleaned:
                return
            self._buffer.append(cleaned)
            self._last_activity = time.monotonic()
            self._schedule_timer()

    async def on_interim(self):
        """
        Signal that an *interim* result was received (speaker still talking).
        Resets the silence timer without adding text.
        """
        async with self._lock:
            self._last_activity = time.monotonic()
            self._schedule_timer()

    async def flush(self):
        """Force-emit whatever is in the buffer right now."""
        async with self._lock:
            self._cancel_timer()
            await self._emit()

    def reset(self):
        """Discard all buffered text and cancel the timer (no emission)."""
        self._cancel_timer()
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _schedule_timer(self):
        """Cancel any pending timer and start a fresh one."""
        self._cancel_timer()
        self._timer_task = asyncio.ensure_future(self._silence_timer())

    def _cancel_timer(self):
        if self._timer_task is not None and not self._timer_task.done():
            self._timer_task.cancel()
        self._timer_task = None

    async def _silence_timer(self):
        """Wait for silence, then emit the buffered utterance."""
        try:
            await asyncio.sleep(self._silence_threshold)
            async with self._lock:
                await self._emit()
        except asyncio.CancelledError:
            pass  # Timer was reset or cancelled — expected

    async def _emit(self):
        """Join buffered fragments and deliver via callback."""
        if not self._buffer:
            return
        utterance = " ".join(self._buffer)
        self._buffer.clear()
        if self._on_utterance and utterance.strip():
            try:
                await self._on_utterance(utterance)
            except Exception as exc:
                logger.error(f"Utterance callback error: {exc}")
