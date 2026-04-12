"""
STT Manager — Google Cloud Speech-to-Text streaming with automatic restart.

Key features:
    • Proactive stream restart before the 5-minute API limit.
    • No transcript loss during restarts (audio queue is persistent).
    • Optional speaker diarization (``speaker_tag`` on words).
    • Automatic punctuation for cleaner transcripts.
    • Thread-safe audio ingestion via ``queue.Queue``.
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Callable, Awaitable, List, Dict, Optional

from google.cloud import speech

logger = logging.getLogger(__name__)

# Restart the stream 30 s before the hard 5-min limit
_MAX_STREAM_SECONDS = 270  # 4 min 30 s


class STTManager:
    """
    Manages a long-running Google Speech-to-Text streaming session.

    Parameters
    ----------
    credentials : google.oauth2.service_account.Credentials
        GCP credentials for the Speech client.
    sample_rate : int
        Audio sample rate in Hz (must match client-side encoding).
    language_code : str
        BCP-47 language code.
    enable_diarization : bool
        Whether to request speaker diarization from the API.
    """

    def __init__(
        self,
        credentials,
        sample_rate: int = 16000,
        language_code: str = "en-US",
        enable_diarization: bool = True,
    ):
        self._credentials = credentials
        self._sample_rate = sample_rate
        self._language_code = language_code
        self._enable_diarization = enable_diarization

        self._audio_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks (async, called on the event loop)
        self._on_transcript: Optional[
            Callable[[str, bool, Optional[List[Dict]]], Awaitable[None]]
        ] = None
        self._on_error: Optional[Callable[[Exception], Awaitable[None]]] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_transcript_callback(
        self,
        callback: Callable[[str, bool, Optional[List[Dict]]], Awaitable[None]],
    ):
        """
        Register the transcript handler.

        Signature: ``async def handler(text, is_final, speaker_words)``
        where *speaker_words* is ``[{"word": str, "speaker_tag": int}, ...]``
        or ``None`` when unavailable.
        """
        self._on_transcript = callback

    def set_error_callback(
        self, callback: Callable[[Exception], Awaitable[None]]
    ):
        self._on_error = callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop):
        """Spawn the background recognition thread."""
        self._loop = loop
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._recognition_loop, daemon=True, name="stt-worker"
        )
        self._thread.start()
        logger.info("STT Manager started")

    def stop(self):
        """Signal the recognition thread to quit and wait for it."""
        self._stop_event.set()
        self._audio_queue.put(None)  # unblock the generator
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("STT Manager stopped")

    def feed_audio(self, data: bytes):
        """Enqueue raw audio bytes for recognition."""
        if not self._stop_event.is_set():
            self._audio_queue.put(data)

    # ------------------------------------------------------------------
    # Internal — runs in a dedicated thread
    # ------------------------------------------------------------------

    def _build_config(self) -> speech.StreamingRecognitionConfig:
        diarization = None
        if self._enable_diarization:
            diarization = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=2,
            )

        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self._sample_rate,
            language_code=self._language_code,
            enable_automatic_punctuation=True,
            diarization_config=diarization,
            enable_word_time_offsets=self._enable_diarization,
        )
        return speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
        )

    def _audio_generator(self, stream_start: float):
        """Yield audio requests until stop or time limit."""
        while not self._stop_event.is_set():
            if time.monotonic() - stream_start >= _MAX_STREAM_SECONDS:
                logger.info("Proactive STT stream restart (approaching 5-min limit)")
                return

            try:
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if data is None:
                return
            yield speech.StreamingRecognizeRequest(audio_content=data)

    def _recognition_loop(self):
        """Outer loop: reconnects the stream automatically."""
        while not self._stop_event.is_set():
            try:
                client = speech.SpeechClient(credentials=self._credentials)
                config = self._build_config()
                stream_start = time.monotonic()

                logger.info("Opening new STT stream")
                requests = self._audio_generator(stream_start)
                responses = client.streaming_recognize(config, requests)

                for response in responses:
                    if self._stop_event.is_set():
                        break
                    self._handle_response(response)

            except Exception as exc:
                if self._stop_event.is_set():
                    break

                msg = str(exc)
                if any(tok in msg for tok in ("400", "out of range", "DEADLINE")):
                    logger.warning(f"STT stream ended (retriable): {msg[:120]}")
                else:
                    logger.error(f"STT stream error: {msg[:200]}")
                    self._fire_error(exc)

            # Brief pause before reconnecting
            if not self._stop_event.is_set():
                time.sleep(0.3)

    def _handle_response(self, response):
        """Extract transcript + speaker data from a streaming response."""
        if not response.results:
            return
        result = response.results[0]
        if not result.alternatives:
            return

        alt = result.alternatives[0]
        transcript = alt.transcript
        is_final = result.is_final

        speaker_words = None
        if is_final and alt.words:
            speaker_words = [
                {
                    "word": w.word,
                    "speaker_tag": getattr(w, "speaker_tag", 0),
                }
                for w in alt.words
            ]

        self._fire_transcript(transcript, is_final, speaker_words)

    # ------------------------------------------------------------------
    # Thread-safe callback dispatch
    # ------------------------------------------------------------------

    def _fire_transcript(self, text, is_final, speaker_words):
        if self._on_transcript and self._loop and not self._stop_event.is_set():
            asyncio.run_coroutine_threadsafe(
                self._on_transcript(text, is_final, speaker_words),
                self._loop,
            )

    def _fire_error(self, exc):
        if self._on_error and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._on_error(exc), self._loop
            )
