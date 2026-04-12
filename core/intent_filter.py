"""
Intent Filter — separates meaningful interviewer questions from fillers.

Design:
    • A curated set of filler phrases (single-word and multi-word) is matched
      after case-folding and stripping punctuation.
    • Short utterances (< ``min_word_count``) must look like questions 
      (interrogative opener or trailing ``?``) to pass.
    • Longer utterances pass by default — they are more likely to be
      substantive speech.
"""

import re
import logging
from typing import Set, List, Pattern

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filler / acknowledgment phrases that should NOT trigger the AI
# ---------------------------------------------------------------------------
FILLER_PHRASES: Set[str] = {
    # Affirmatives
    "yeah", "yes", "yep", "yup", "ya", "yea", "aye",
    # Acknowledgments
    "okay", "ok", "k", "okie",
    "hmm", "hm", "hmmm", "um", "uh", "umm", "uhh", "er", "erm",
    "right", "alright", "all right",
    "sure", "sure sure",
    "got it", "got you", "gotcha", "got that",
    "i see", "i understand", "understood",
    "uh huh", "uh-huh", "uhhuh", "mhm", "mm-hmm", "mmhmm", "mm hm",
    # Negatives
    "no", "nope", "nah",
    # Pleasantries
    "thanks", "thank you", "thank you so much",
    "great", "good", "nice", "cool", "awesome", "perfect", "wonderful",
    "absolutely", "definitely", "exactly", "precisely",
    "hello", "hi", "hey", "hi there", "hey there",
    "bye", "goodbye", "good bye", "see you", "take care",
    # Fillers
    "so", "well", "like", "you know", "i mean",
    "let me think", "let me see", "lets see", "let's see",
    "one moment", "one second", "just a moment", "hold on",
    "can you hear me", "am i audible", "is my audio working",
    "sorry", "pardon", "excuse me",
    # Transitional
    "moving on", "next", "go ahead", "please continue", "continue",
    "interesting", "i see that", "noted",
}

# ---------------------------------------------------------------------------
# Patterns that strongly indicate a question or command
# ---------------------------------------------------------------------------
_QUESTION_OPENERS = (
    r"^(what|how|why|when|where|who|which|whose|whom"
    r"|could|can|would|will|shall|should|may|might"
    r"|do|does|did|don't|doesn't|didn't"
    r"|is|are|was|were|isn't|aren't|wasn't|weren't"
    r"|have|has|had|haven't|hasn't|hadn't"
    r"|tell|explain|describe|walk|give|share|elaborate"
    r"|define|compare|list|name|mention|discuss"
    r"|show|demonstrate|clarify|justify|outline)\b"
)

_QUESTION_PATTERNS: List[Pattern] = [
    re.compile(r"\?\s*$"),                  # Ends with question mark
    re.compile(_QUESTION_OPENERS, re.I),    # Interrogative / imperative opener
]


class IntentFilter:
    """
    Decides whether an utterance is substantive enough to warrant an AI response.

    Parameters
    ----------
    min_word_count : int
        Utterances shorter than this must match a question pattern.
    """

    def __init__(self, min_word_count: int = 3):
        self._min_word_count = min_word_count

    def is_meaningful(self, text: str) -> bool:
        """
        Return ``True`` if *text* should be forwarded to the LLM.
        """
        if not text or not text.strip():
            return False

        cleaned = text.strip().lower()
        # Strip punctuation for filler comparison
        cleaned_no_punct = re.sub(r"[^\w\s]", "", cleaned).strip()

        # ---- Filler check ----
        if cleaned_no_punct in FILLER_PHRASES:
            logger.debug(f"Filtered filler: '{text}'")
            return False

        # ---- Word count gate ----
        words = cleaned.split()
        if len(words) < self._min_word_count:
            is_q = self._is_question(cleaned)
            if not is_q:
                logger.debug(f"Filtered short non-question: '{text}'")
            return is_q

        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _is_question(text: str) -> bool:
        """Heuristic: does *text* look like a question or command?"""
        for pattern in _QUESTION_PATTERNS:
            if pattern.search(text):
                return True
        return False
