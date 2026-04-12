"""
Session Management — per-user isolated state.

Every WebSocket connection gets a ``Session`` instance that owns its own:
    • Interview context (resume, JD, company)
    • Conversation history
    • Pipeline components (state machine, utterance builder, debounce, etc.)

The ``SessionStore`` maps user-email → session and supports reconnect
(restoring an existing session for the same user).

This module replaces the old global ``USER_CONTEXT`` dict, eliminating
cross-user data leakage entirely.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .debounce import Debounce
from .intent_filter import IntentFilter
from .speaker_filter import SpeakerFilter
from .state_machine import AIState, StateMachine
from .utterance_builder import UtteranceBuilder

logger = logging.getLogger(__name__)


class Session:
    """
    Fully isolated per-user interview session.

    Every mutable piece of state lives here — nothing is shared globally.
    """

    def __init__(self, session_id: str, user_email: str):
        self.session_id: str = session_id
        self.user_email: str = user_email

        # Interview context (per-user, no global sharing)
        self.context: Dict[str, str] = {"resume": "", "jd": "", "company": ""}

        # Conversation history (list of (question, answer) tuples)
        self.conversation_history: List[Tuple[str, str]] = []

        # Pipeline components — all per-session
        self.state_machine = StateMachine()
        self.utterance_builder = UtteranceBuilder(silence_threshold_ms=1000)
        self.speaker_filter = SpeakerFilter()
        self.intent_filter = IntentFilter(min_word_count=3)
        self.debounce = Debounce(window_seconds=15.0)

        # Timestamps
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_active: datetime = datetime.now(timezone.utc)
        self._connected_at: Optional[float] = None  # monotonic time

        # Concurrency guard
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def update_context(
        self,
        resume: str = "",
        jd: str = "",
        company: str = "",
    ):
        """Update interview context fields (only non-empty values overwrite)."""
        if resume:
            self.context["resume"] = resume
        if jd:
            self.context["jd"] = jd
        if company:
            self.context["company"] = company
        self.last_active = datetime.now(timezone.utc)
        logger.info(
            f"Session {self.session_id}: context updated "
            f"(company={self.context['company']!r})"
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_to_history(self, question: str, answer: str):
        self.conversation_history.append((question, answer))
        self.last_active = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Timing / usage
    # ------------------------------------------------------------------

    def mark_connected(self):
        """Record the moment the WebSocket is authenticated and active."""
        self._connected_at = time.monotonic()

    def get_session_duration_seconds(self) -> int:
        """Actual wall-clock seconds since the connection was authenticated."""
        if self._connected_at is None:
            return 0
        return int(time.monotonic() - self._connected_at)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self):
        """Release resources held by pipeline components."""
        self.utterance_builder.reset()
        self.debounce.reset()
        self.speaker_filter.reset()
        await self.state_machine.force_reset()


# ======================================================================
# Session Store
# ======================================================================


class SessionStore:
    """
    In-memory session store with per-user isolation.

    Thread/task-safe via an ``asyncio.Lock``.  For multi-server deployments,
    swap this class with a Redis-backed implementation sharing the same
    interface.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}          # session_id → Session
        self._user_sessions: Dict[str, str] = {}          # email     → session_id
        self._lock: asyncio.Lock = asyncio.Lock()

    async def create_or_restore(self, user_email: str) -> Session:
        """
        Return the existing session for *user_email*, or create a new one.

        This enables **session persistence on reconnect** — if a user
        disconnects and reconnects, their context and history are preserved.
        """
        async with self._lock:
            sid = self._user_sessions.get(user_email)
            if sid and sid in self._sessions:
                session = self._sessions[sid]
                logger.info(f"Restored session {sid} for {user_email}")
                return session

            sid = str(uuid.uuid4())
            session = Session(session_id=sid, user_email=user_email)
            self._sessions[sid] = session
            self._user_sessions[user_email] = sid
            logger.info(f"Created session {sid} for {user_email}")
            return session

    async def get_by_email(self, email: str) -> Optional[Session]:
        sid = self._user_sessions.get(email)
        if sid:
            return self._sessions.get(sid)
        return None

    async def get_by_id(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def remove(self, session_id: str):
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                await session.cleanup()
                self._user_sessions.pop(session.user_email, None)
                logger.info(f"Removed session {session_id}")

    async def active_count(self) -> int:
        return len(self._sessions)
