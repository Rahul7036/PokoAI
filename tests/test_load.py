"""
Load / concurrency tests — verifies session isolation and safety under
simulated multi-user load.
"""

import asyncio
import json
import pytest
from core.session import Session, SessionStore
from core.state_machine import StateMachine, AIState
from core.utterance_builder import UtteranceBuilder
from core.debounce import Debounce
from core.intent_filter import IntentFilter


class TestConcurrentSessionCreation:
    @pytest.mark.asyncio
    async def test_100_sessions_unique_and_isolated(self):
        """Create 100 sessions concurrently and verify no overlap."""
        store = SessionStore()
        N = 100

        async def create_session(i):
            email = f"loaduser{i}@test.com"
            s = await store.create_or_restore(email)
            s.update_context(
                resume=f"Resume-{i}",
                jd=f"JD-{i}",
                company=f"Company-{i}",
            )
            s.add_to_history(f"Q-{i}", f"A-{i}")
            return s

        sessions = await asyncio.gather(*[create_session(i) for i in range(N)])

        # All unique session IDs
        ids = {s.session_id for s in sessions}
        assert len(ids) == N

        # Verify data integrity (no cross-contamination)
        for i, s in enumerate(sessions):
            assert s.context["resume"] == f"Resume-{i}", (
                f"Data leak! Session {i} has resume={s.context['resume']}"
            )
            assert s.context["company"] == f"Company-{i}"
            assert s.conversation_history[-1] == (f"Q-{i}", f"A-{i}")

        assert await store.active_count() == N


class TestConcurrentStateMachineRaces:
    @pytest.mark.asyncio
    async def test_many_concurrent_transition_attempts(self):
        """Only one of N concurrent LISTENING→PROCESSING transitions succeeds."""
        sm = StateMachine()
        results = await asyncio.gather(
            *[sm.transition_to(AIState.PROCESSING) for _ in range(50)]
        )
        assert results.count(True) == 1
        assert results.count(False) == 49

    @pytest.mark.asyncio
    async def test_state_machine_per_session_independence(self):
        """State machines in different sessions don't interfere."""
        store = SessionStore()
        s1 = await store.create_or_restore("a@test.com")
        s2 = await store.create_or_restore("b@test.com")

        await s1.state_machine.transition_to(AIState.PROCESSING)
        assert s1.state_machine.state == AIState.PROCESSING
        assert s2.state_machine.state == AIState.LISTENING


class TestConcurrentUtteranceBuilders:
    @pytest.mark.asyncio
    async def test_independent_utterance_builders(self):
        """Each session's utterance builder operates independently."""
        emitted_1 = []
        emitted_2 = []

        async def cb1(text):
            emitted_1.append(text)

        async def cb2(text):
            emitted_2.append(text)

        ub1 = UtteranceBuilder(silence_threshold_ms=200)
        ub2 = UtteranceBuilder(silence_threshold_ms=200)
        ub1.set_callback(cb1)
        ub2.set_callback(cb2)

        await ub1.add_final("Question for user 1")
        await ub2.add_final("Question for user 2")

        await asyncio.sleep(0.4)

        assert len(emitted_1) == 1
        assert len(emitted_2) == 1
        assert emitted_1[0] == "Question for user 1"
        assert emitted_2[0] == "Question for user 2"


class TestDebounceIsolation:
    def test_separate_debounce_instances_independent(self):
        """Each session's debounce is independent."""
        db1 = Debounce(window_seconds=10)
        db2 = Debounce(window_seconds=10)

        assert db1.should_process("Same question?") is True
        assert db2.should_process("Same question?") is True  # Independent!


class TestIntentFilterThreadSafety:
    def test_filter_is_stateless(self):
        """IntentFilter has no mutable state — safe for concurrent use."""
        f = IntentFilter()
        # Same filter, concurrent reads are fine
        results = [f.is_meaningful("What is X?") for _ in range(100)]
        assert all(r is True for r in results)


class TestReconnectPersistence:
    @pytest.mark.asyncio
    async def test_session_survives_reconnect(self):
        """Disconnecting and reconnecting should restore session state."""
        store = SessionStore()
        email = "reconnect@test.com"

        # First connection
        s1 = await store.create_or_restore(email)
        s1.update_context(resume="Important Resume", company="BigCo")
        s1.add_to_history("Q1", "A1")

        # Simulate disconnect (don't remove session)
        # …

        # Reconnect
        s2 = await store.create_or_restore(email)
        assert s2.session_id == s1.session_id
        assert s2.context["resume"] == "Important Resume"
        assert s2.conversation_history == [("Q1", "A1")]
