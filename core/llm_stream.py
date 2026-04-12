"""
Streaming LLM — generates AI responses token-by-token using Vertex AI.

Design:
    • The Vertex AI ``generate_content(stream=True)`` call blocks in a
      background thread.  Chunks are pushed to an ``asyncio.Queue`` so the
      caller can ``async for`` them with sub-second latency.
    • The prompt is carefully crafted to:
        – Keep the AI in the *candidate* role (not the interviewer).
        – Produce concise, bullet-point answers (2–4 lines).
        – Emit the sentinel ``NO_ANSWER`` for non-questions.
"""

import asyncio
import logging
import threading
from typing import AsyncGenerator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sentinel pushed into the queue when the stream finishes
_STREAM_END = object()
_STREAM_ERROR = object()


class LLMStreamer:
    """
    Wrapper around Vertex AI's ``GenerativeModel`` that yields tokens
    asynchronously.

    Parameters
    ----------
    model : vertexai.generative_models.GenerativeModel or None
        The Vertex AI model.  ``None`` disables generation (returns error text).
    max_history : int
        Maximum number of recent Q\u0026A pairs to include in the prompt.
    """

    def __init__(self, model, max_history: int = 10):
        self._model = model
        self._max_history = max_history

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        context: Dict[str, str],
        history: List[Tuple[str, str]],
    ) -> str:
        company = context.get("company") or "the company"
        resume = context.get("resume") or ""
        jd = context.get("jd") or ""

        hist_block = ""
        if history:
            pairs = history[-self._max_history :]
            hist_block = "CONVERSATION SO FAR:\n" + "\n".join(
                f"Interviewer: {q}\nYou answered: {a}" for q, a in pairs
            )

        return (
            f"You are a JOB CANDIDATE being interviewed at {company}.\n"
            f"Answer the INTERVIEWER's question using YOUR RESUME as the source of truth.\n\n"
            f"YOUR RESUME:\n{resume}\n\n"
            f"JOB DESCRIPTION:\n{jd}\n\n"
            f"{hist_block}\n\n"
            f'LIVE MICROPHONE TRANSCRIPT:\n"{question}"\n\n'
            "RULES:\n"
            "1. The transcript may be a mix of the candidate speaking (echoes of previous answers) and the interviewer asking a question.\n"
            "2. IGNORE the candidate's speech. ONLY answer the real question directed at the candidate by the interviewer.\n"
            "3. If the transcript contains NO actual question for the candidate, reply EXACTLY: NO_ANSWER\n"
            "4. Answer in FIRST PERSON as the candidate.\n"
            "5. Respond naturally and conversationally, like a real human speaking in an interview.\n"
            "6. Explain technical concepts clearly but verbally. Do NOT use bullet points or markdown tables.\n"
            "7. Stick strictly to resume facts; align with JD when relevant.\n"
            '8. NEVER say "Based on my resume" or reference having a resume.\n'
            "9. NEVER repeat the question back.\n"
            "10. NEVER generate follow-up questions as the interviewer."
        )

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        question: str,
        context: Dict[str, str],
        history: List[Tuple[str, str]],
    ) -> AsyncGenerator[str, None]:
        """
        Yield LLM response chunks asynchronously.

        The underlying blocking iterator runs in a daemon thread; each chunk
        is pushed to an ``asyncio.Queue`` and yielded immediately, achieving
        sub-second first-token latency.
        """
        if not self._model:
            yield "Error: AI model not initialized."
            return

        prompt = self._build_prompt(question, context, history)
        chunk_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_in_thread():
            try:
                responses = self._model.generate_content(prompt, stream=True)
                for chunk in responses:
                    text = getattr(chunk, "text", None)
                    if text:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, text)
            except Exception as exc:
                logger.error(f"LLM stream thread error: {exc}")
                loop.call_soon_threadsafe(chunk_queue.put_nowait, _STREAM_ERROR)
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, _STREAM_END)

        thread = threading.Thread(target=_run_in_thread, daemon=True, name="llm-stream")
        thread.start()

        while True:
            item = await chunk_queue.get()
            if item is _STREAM_END:
                break
            if item is _STREAM_ERROR:
                yield "I apologize, I'm having trouble formulating my response right now."
                break
            yield item

    # ------------------------------------------------------------------
    # Non-streaming fallback
    # ------------------------------------------------------------------

    async def generate_full(
        self,
        question: str,
        context: Dict[str, str],
        history: List[Tuple[str, str]],
    ) -> str:
        """Synchronous (non-streaming) generation — fallback only."""
        if not self._model:
            return "Error: AI model not initialized."

        prompt = self._build_prompt(question, context, history)
        try:
            response = await asyncio.to_thread(
                self._model.generate_content, prompt
            )
            return response.text.strip()
        except Exception as exc:
            logger.error(f"LLM full-gen error: {exc}")
            return "I apologize, I'm having trouble formulating my response right now."
