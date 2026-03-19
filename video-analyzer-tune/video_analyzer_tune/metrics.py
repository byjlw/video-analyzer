"""LLM-as-judge metric for evaluating video analysis quality."""

import logging
from typing import List

import dspy

from .signatures import DescriptionJudgeSignature, FrameNoteJudgeSignature

logger = logging.getLogger(__name__)

# Sample at most this many frames when scoring frame notes to keep evaluation fast
MAX_FRAME_SAMPLES = 5


class VideoAnalysisMetric:
    """LLM-as-judge metric scoring final description and optionally frame notes.

    Scores the final description against the ideal (always), and optionally
    scores frame notes against ideal frame notes when the user provided them.

    Args:
        description_weight: How much the final description score contributes
            to the overall score (0.0 to 1.0). The remainder weights frame
            note quality. Only has effect when ideal frame notes are provided.
            Default 0.7 — the final description is the primary goal.
    """

    def __init__(self, description_weight: float = 0.7):
        if not 0.0 <= description_weight <= 1.0:
            raise ValueError("description_weight must be between 0.0 and 1.0")
        self.description_weight = description_weight
        self.frame_weight = 1.0 - description_weight
        self.judge_description = dspy.Predict(DescriptionJudgeSignature)
        self.judge_frame = dspy.Predict(FrameNoteJudgeSignature)

    def __call__(self, example, prediction, trace=None) -> float:
        """Score a prediction against the ideal outputs in example.

        Args:
            example: dspy.Example with ideal_description, frame_notes_list,
                     user_question, and has_ideal_frame_notes fields
            prediction: dspy.Prediction with description and frame_notes_list fields

        Returns:
            Score between 0.0 and 1.0
        """
        desc_score = self._score_description(
            ideal=getattr(example, "ideal_description", ""),
            candidate=getattr(prediction, "description", ""),
            user_question=getattr(example, "user_question", ""),
        )

        if not getattr(example, "has_ideal_frame_notes", False):
            return desc_score

        frame_score = self._score_frame_notes(
            ideal_notes=getattr(example, "frame_notes_list", []),
            candidate_notes=getattr(prediction, "frame_notes_list", []),
        )

        return self.description_weight * desc_score + self.frame_weight * frame_score

    def _score_description(self, ideal: str, candidate: str, user_question: str) -> float:
        if not ideal or not candidate:
            return 0.0
        try:
            pred = self.judge_description(
                ideal=ideal,
                candidate=candidate,
                user_question=user_question,
            )
            return self._parse_score(pred.score) / 5.0
        except Exception as e:
            logger.warning(f"Description scoring failed: {e}")
            return 0.0

    def _score_frame_notes(self, ideal_notes: List[str], candidate_notes: List[str]) -> float:
        pairs = [
            (ideal, candidate)
            for ideal, candidate in zip(ideal_notes, candidate_notes)
            if ideal and candidate
        ]
        if not pairs:
            return 0.0

        # Evenly sample up to MAX_FRAME_SAMPLES to keep evaluation fast
        step = max(1, len(pairs) // MAX_FRAME_SAMPLES)
        sampled = pairs[::step][:MAX_FRAME_SAMPLES]

        scores = []
        for ideal, candidate in sampled:
            try:
                pred = self.judge_frame(ideal=ideal, candidate=candidate)
                scores.append(self._parse_score(pred.score) / 5.0)
            except Exception as e:
                logger.warning(f"Frame note scoring failed: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _parse_score(self, score_value) -> int:
        """Parse a score value, clamping to [1, 5]."""
        try:
            val = int(str(score_value).strip().split()[0])
            return max(1, min(5, val))
        except (ValueError, TypeError, IndexError):
            return 1
