"""
Speaker Filter — separates interviewer speech from candidate speech.

Uses Google Speech-to-Text speaker diarization tags (``speaker_tag`` on
each ``WordInfo``).  The filter auto-assigns speaker roles:

    • We currently do NOT auto-assign speaker roles because the order of who speaks first is unreliable in real interviews.
    • Instead, we pass all distinct speakers through and defer to the LLM prompt for role disambiguation and ignoring candidate echoes.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SpeakerFilter:
    """
    Determines whether an utterance belongs to the interviewer or candidate
    based on Google Speech ``speaker_tag`` metadata.
    """

    def __init__(self):
        self._interviewer_tag: Optional[int] = None
        self._candidate_tag: Optional[int] = None
        self._tag_history: List[int] = []
        self._tag_history: List[int] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_interviewer_tag(self, tag: int):
        """Manually override which speaker tag belongs to the interviewer."""
        self._interviewer_tag = tag
        logger.info(f"Interviewer tag manually set to {tag}")

    def set_candidate_tag(self, tag: int):
        """Manually override which speaker tag belongs to the candidate."""
        self._candidate_tag = tag
        logger.info(f"Candidate tag manually set to {tag}")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_speaker_tags(self, words_with_tags: List[Dict]) -> Optional[int]:
        """
        Analyze word-level speaker tags and return the **dominant** speaker
        tag for the utterance.

        Parameters
        ----------
        words_with_tags : list of dict
            Each dict must have ``"word"`` (str) and ``"speaker_tag"`` (int).

        Returns
        -------
        int or None
            The most frequent speaker tag, or ``None`` if no tags are present.
        """
        if not words_with_tags:
            return None

        tags = [
            w["speaker_tag"]
            for w in words_with_tags
            if w.get("speaker_tag", 0) > 0
        ]
        if not tags:
            return None

        dominant_tag = Counter(tags).most_common(1)[0][0]
        self._tag_history.append(dominant_tag)

        return dominant_tag

    def is_interviewer(self, dominant_tag: Optional[int]) -> bool:
        """
        Return ``True`` if the dominant speaker is (likely) the interviewer.

        Falls back to ``True`` when diarization data is insufficient so that
        the LLM prompt can serve as a secondary filter.
        """
        if dominant_tag is None:
            return True  # No diarization → pass through
        if self._interviewer_tag is None:
            return True  # Not enough data to assign yet
        return dominant_tag == self._interviewer_tag

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all learned speaker assignments."""
        self._interviewer_tag = None
        self._candidate_tag = None
        self._tag_history.clear()
        self._tag_history.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

