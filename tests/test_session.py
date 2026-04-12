"""
Tests for Session and SessionStore — validates isolation, persistence,
context management, and cleanup.
"""

import asyncio
import pytest
from core.session import Session, SessionStore


class TestSession:
    def test_initial_state(self):
        s = Session("id-1", "user@test.com")
        assert s.session_id == "id-1"
        assert s.user_email == "user@test.com"
        assert s.context == {"resume": "", "jd": "", "company": ""}
        assert s.conversation_history == []

    def test_update_context(self):
        s = Session("id-1", "user@test.com")
        s.update_context(resume="My Resume", jd="Job Desc", company="Acme")
        assert s.context["resume"] == "My Resume"
        assert s.context["jd"] == "Job Desc"
        assert s.context["company"] == "Acme"

    def test_partial_context_update(self):
        s = Session("id-1", "user@test.com")
        s.update_context(resume="Resume 1")
        s.update_context(company="Acme")
        # resume should not be overwritten
        assert s.context["resume"] == "Resume 1"
        assert s.context["company"] == "Acme"

    def test_empty_string_does_not_overwrite(self):
        s = Session("id-1", "user@test.com")
        s.update_context(resume="Original Resume")
        s.update_context(resume="")  # empty should not overwrite
        assert s.context["resume"] == "Original Resume"

    def test_add_to_history(self):
        s = Session("id-1", "user@test.com")
        s.add_to_history("Q1", "A1")
        s.add_to_history("Q2", "A2")
        assert len(s.conversation_history) == 2
        assert s.conversation_history[0] == ("Q1", "A1")

    def test_session_duration_before_connect(self):
        s = Session("id-1", "user@test.com")
        assert s.get_session_duration_seconds() == 0

    def test_session_duration_after_connect(self):
        import time
        s = Session("id-1", "user@test.com")
        s.mark_connected()
        time.sleep(0.1)
        assert s.get_session_duration_seconds() >= 0

    @pytest.mark.asyncio
    async def test_cleanup(self):
        s = Session("id-1", "user@test.com")
        s.update_context(resume="Test")
        await s.cleanup()
        # State machine should be reset
        from core.state_machine import AIState
        assert s.state_machine.state == AIState.LISTENING


class TestSessionStore:
    @pytest.mark.asyncio
    async def test_create_new_session(self):
        store = SessionStore()
        session = await store.create_or_restore("user@test.com")
        assert session is not None
        assert session.user_email == "user@test.com"

    @pytest.mark.asyncio
    async def test_restore_existing_session(self):
        store = SessionStore()
        s1 = await store.create_or_restore("user@test.com")
        s1.update_context(resume="Saved Resume")

        s2 = await store.create_or_restore("user@test.com")
        assert s1.session_id == s2.session_id
        assert s2.context["resume"] == "Saved Resume"

    @pytest.mark.asyncio
    async def test_different_users_get_different_sessions(self):
        store = SessionStore()
        s1 = await store.create_or_restore("user1@test.com")
        s2 = await store.create_or_restore("user2@test.com")
        assert s1.session_id != s2.session_id

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """Critical: user1's data must never leak to user2."""
        store = SessionStore()
        s1 = await store.create_or_restore("user1@test.com")
        s2 = await store.create_or_restore("user2@test.com")

        s1.update_context(resume="User1 Resume", company="Company1")
        s2.update_context(resume="User2 Resume", company="Company2")

        assert s1.context["resume"] == "User1 Resume"
        assert s2.context["resume"] == "User2 Resume"
        assert s1.context["company"] != s2.context["company"]

    @pytest.mark.asyncio
    async def test_remove_session(self):
        store = SessionStore()
        s = await store.create_or_restore("user@test.com")
        sid = s.session_id

        await store.remove(sid)
        assert await store.get_by_id(sid) is None
        assert await store.get_by_email("user@test.com") is None

    @pytest.mark.asyncio
    async def test_active_count(self):
        store = SessionStore()
        await store.create_or_restore("u1@test.com")
        await store.create_or_restore("u2@test.com")
        assert await store.active_count() == 2

    @pytest.mark.asyncio
    async def test_get_by_email(self):
        store = SessionStore()
        s = await store.create_or_restore("user@test.com")
        found = await store.get_by_email("user@test.com")
        assert found is s

    @pytest.mark.asyncio
    async def test_get_by_email_nonexistent(self):
        store = SessionStore()
        assert await store.get_by_email("noone@test.com") is None


class TestSessionIsolationUnderLoad:
    @pytest.mark.asyncio
    async def test_50_concurrent_sessions_no_leakage(self):
        """Simulate 50 users creating sessions concurrently."""
        store = SessionStore()

        async def create_and_verify(i):
            email = f"user{i}@test.com"
            session = await store.create_or_restore(email)
            session.update_context(
                resume=f"Resume_{i}",
                jd=f"JD_{i}",
                company=f"Company_{i}",
            )
            await asyncio.sleep(0.01)  # Simulate real work
            # Verify data integrity
            assert session.context["resume"] == f"Resume_{i}"
            assert session.context["jd"] == f"JD_{i}"
            assert session.context["company"] == f"Company_{i}"
            return session

        sessions = await asyncio.gather(
            *[create_and_verify(i) for i in range(50)]
        )

        # All sessions unique
        ids = [s.session_id for s in sessions]
        assert len(set(ids)) == 50

        # Cross-check no leakage
        for i, s in enumerate(sessions):
            assert s.context["resume"] == f"Resume_{i}"
