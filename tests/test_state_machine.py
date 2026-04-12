"""
Tests for the StateMachine — validates state transitions, concurrency safety,
and callback behavior.
"""

import asyncio
import pytest
from core.state_machine import StateMachine, AIState


@pytest.mark.asyncio
async def test_initial_state():
    sm = StateMachine()
    assert sm.state == AIState.LISTENING
    assert sm.is_available()


@pytest.mark.asyncio
async def test_valid_transition_listening_to_processing():
    sm = StateMachine()
    result = await sm.transition_to(AIState.PROCESSING)
    assert result is True
    assert sm.state == AIState.PROCESSING
    assert not sm.is_available()


@pytest.mark.asyncio
async def test_valid_transition_processing_to_responding():
    sm = StateMachine()
    await sm.transition_to(AIState.PROCESSING)
    result = await sm.transition_to(AIState.RESPONDING)
    assert result is True
    assert sm.state == AIState.RESPONDING


@pytest.mark.asyncio
async def test_valid_transition_responding_to_listening():
    sm = StateMachine()
    await sm.transition_to(AIState.PROCESSING)
    await sm.transition_to(AIState.RESPONDING)
    result = await sm.transition_to(AIState.LISTENING)
    assert result is True
    assert sm.state == AIState.LISTENING
    assert sm.is_available()


@pytest.mark.asyncio
async def test_valid_transition_processing_to_listening():
    """Processing → Listening is valid (for NO_ANSWER path)."""
    sm = StateMachine()
    await sm.transition_to(AIState.PROCESSING)
    result = await sm.transition_to(AIState.LISTENING)
    assert result is True
    assert sm.state == AIState.LISTENING


@pytest.mark.asyncio
async def test_invalid_transition_listening_to_responding():
    sm = StateMachine()
    result = await sm.transition_to(AIState.RESPONDING)
    assert result is False
    assert sm.state == AIState.LISTENING  # unchanged


@pytest.mark.asyncio
async def test_invalid_transition_responding_to_processing():
    sm = StateMachine()
    await sm.transition_to(AIState.PROCESSING)
    await sm.transition_to(AIState.RESPONDING)
    result = await sm.transition_to(AIState.PROCESSING)
    assert result is False
    assert sm.state == AIState.RESPONDING


@pytest.mark.asyncio
async def test_invalid_transition_listening_to_listening():
    sm = StateMachine()
    result = await sm.transition_to(AIState.LISTENING)
    assert result is False


@pytest.mark.asyncio
async def test_force_reset_from_any_state():
    sm = StateMachine()
    await sm.transition_to(AIState.PROCESSING)
    await sm.transition_to(AIState.RESPONDING)
    assert sm.state == AIState.RESPONDING

    await sm.force_reset()
    assert sm.state == AIState.LISTENING
    assert sm.is_available()


@pytest.mark.asyncio
async def test_callback_on_transition():
    sm = StateMachine()
    states_seen = []

    async def on_change(new_state):
        states_seen.append(new_state)

    sm.set_callback(on_change)
    await sm.transition_to(AIState.PROCESSING)
    await sm.transition_to(AIState.RESPONDING)
    await sm.transition_to(AIState.LISTENING)

    assert states_seen == [AIState.PROCESSING, AIState.RESPONDING, AIState.LISTENING]


@pytest.mark.asyncio
async def test_callback_not_called_on_invalid_transition():
    sm = StateMachine()
    called = []

    async def on_change(s):
        called.append(s)

    sm.set_callback(on_change)
    await sm.transition_to(AIState.RESPONDING)  # invalid
    assert called == []


@pytest.mark.asyncio
async def test_callback_on_force_reset():
    sm = StateMachine()
    called = []

    async def on_change(s):
        called.append(s)

    sm.set_callback(on_change)
    await sm.transition_to(AIState.PROCESSING)
    await sm.force_reset()

    assert AIState.LISTENING in called


@pytest.mark.asyncio
async def test_concurrent_transitions():
    """Multiple tasks racing to transition should not corrupt state."""
    sm = StateMachine()
    results = []

    async def try_transition():
        r = await sm.transition_to(AIState.PROCESSING)
        results.append(r)

    await asyncio.gather(*[try_transition() for _ in range(10)])

    # Exactly one should succeed
    assert results.count(True) == 1
    assert results.count(False) == 9
    assert sm.state == AIState.PROCESSING


@pytest.mark.asyncio
async def test_full_lifecycle():
    """LISTENING → PROCESSING → RESPONDING → LISTENING (full cycle)."""
    sm = StateMachine()
    assert sm.is_available()

    assert await sm.transition_to(AIState.PROCESSING)
    assert not sm.is_available()

    assert await sm.transition_to(AIState.RESPONDING)
    assert not sm.is_available()

    assert await sm.transition_to(AIState.LISTENING)
    assert sm.is_available()
