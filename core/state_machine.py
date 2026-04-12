"""
State Machine for AI assistant lifecycle.

States:
    LISTENING   → Waiting for interviewer speech
    PROCESSING  → Building response from LLM
    RESPONDING  → Streaming response to client

Valid transitions:
    LISTENING  → PROCESSING
    PROCESSING → RESPONDING
    PROCESSING → LISTENING  (if LLM decides NO_ANSWER)
    RESPONDING → LISTENING

Any state → LISTENING via force_reset() for error recovery.
"""

import asyncio
import enum
import logging
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class AIState(enum.Enum):
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"


# Valid state transitions
_VALID_TRANSITIONS = {
    AIState.LISTENING: {AIState.PROCESSING},
    AIState.PROCESSING: {AIState.RESPONDING, AIState.LISTENING},
    AIState.RESPONDING: {AIState.LISTENING},
}


class StateMachine:
    """
    Thread-safe state machine controlling the AI response lifecycle.
    
    Prevents race conditions by using an asyncio Lock for all state transitions.
    Broadcasts state changes via an optional callback for UI updates.
    """

    def __init__(self):
        self._state: AIState = AIState.LISTENING
        self._lock: asyncio.Lock = asyncio.Lock()
        self._on_state_change: Optional[Callable[[AIState], Awaitable[None]]] = None

    @property
    def state(self) -> AIState:
        """Current state (read without locking for status checks)."""
        return self._state

    def set_callback(self, callback: Callable[[AIState], Awaitable[None]]):
        """Set the callback invoked on every state transition."""
        self._on_state_change = callback

    async def transition_to(self, new_state: AIState) -> bool:
        """
        Attempt a state transition. Returns True if successful.
        
        Only valid transitions (per _VALID_TRANSITIONS) are allowed.
        The lock guarantees at most one transition is in-flight at a time.
        """
        async with self._lock:
            allowed = _VALID_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                logger.debug(
                    f"Invalid transition: {self._state.value} → {new_state.value}"
                )
                return False
            old_state = self._state
            self._state = new_state

        logger.debug(f"State: {old_state.value} → {new_state.value}")
        if self._on_state_change:
            try:
                await self._on_state_change(new_state)
            except Exception as exc:
                logger.error(f"State change callback error: {exc}")
        return True

    def is_available(self) -> bool:
        """Check if the AI is ready to process a new utterance."""
        return self._state == AIState.LISTENING

    async def force_reset(self):
        """
        Force the state machine back to LISTENING.
        
        Used for error recovery and cleanup. Bypasses transition validation.
        """
        async with self._lock:
            self._state = AIState.LISTENING

        if self._on_state_change:
            try:
                await self._on_state_change(AIState.LISTENING)
            except Exception as exc:
                logger.error(f"State reset callback error: {exc}")
