"""DSPy pipeline mirroring the video-analyzer two-stage analysis process."""

import logging
from typing import List, Dict, Any

import dspy

from .signatures import FrameAnalysisSignature, ReconstructionSignature

logger = logging.getLogger(__name__)


class VideoAnalysisPipeline(dspy.Module):
    """DSPy pipeline that mirrors the video-analyzer two-stage process:
    sequential per-frame analysis followed by full video reconstruction.

    The accumulation logic and prompt token formats exactly match
    VideoAnalyzer in video_analyzer/analyzer.py so that optimized prompts
    are drop-in compatible.
    """

    def __init__(self):
        self.analyze_frame = dspy.Predict(FrameAnalysisSignature)
        self.reconstruct = dspy.Predict(ReconstructionSignature)

    def forward(
        self,
        frames: List[Dict[str, Any]],
        user_question: str,
        transcript: str,
    ) -> dspy.Prediction:
        """Run the full pipeline: analyze each frame then reconstruct.

        Args:
            frames: List of dicts with keys: index (int), timestamp (float),
                    image (dspy.Image or None)
            user_question: Formatted user question, e.g. "I want to know X"
                           or empty string — matches VideoAnalyzer._format_user_prompt()
            transcript: Audio transcript text, or empty string if none

        Returns:
            Prediction with 'description' (str) and 'frame_notes_list' (List[str])
        """
        # Accumulate notes exactly as VideoAnalyzer._format_previous_analyses() does
        accumulated_notes: List[str] = []
        frame_notes_list: List[str] = []

        for i, frame in enumerate(frames):
            # Build previous_frames text matching _format_previous_analyses() format:
            # "Frame {i}\n{note}\n" joined with "\n"
            if accumulated_notes:
                previous_frames_text = "\n".join(
                    f"Frame {j}\n{note}\n"
                    for j, note in enumerate(accumulated_notes)
                )
            else:
                previous_frames_text = ""

            try:
                pred = self.analyze_frame(
                    image=frame.get("image"),
                    previous_frames=previous_frames_text,
                    user_question=user_question,
                )
                frame_note = pred.frame_note
            except Exception as e:
                logger.warning(f"Frame {i} analysis failed: {e}")
                frame_note = f"Error analyzing frame {i}"

            accumulated_notes.append(frame_note)
            frame_notes_list.append(frame_note)

        # Build frame_notes in the exact format from VideoAnalyzer.reconstruct_video():
        # "Frame {i} ({timestamp:.2f}s):\n{note}" joined with "\n\n"
        all_frame_notes = "\n\n".join(
            f"Frame {i} ({frame['timestamp']:.2f}s):\n{note}"
            for i, (frame, note) in enumerate(zip(frames, frame_notes_list))
        )
        first_frame_note = frame_notes_list[0] if frame_notes_list else ""

        try:
            result = self.reconstruct(
                frame_notes=all_frame_notes,
                first_frame_note=first_frame_note,
                transcript=transcript,
                user_question=user_question,
            )
            description = result.description
        except Exception as e:
            logger.warning(f"Reconstruction failed: {e}")
            description = "Error during reconstruction"

        return dspy.Prediction(
            description=description,
            frame_notes_list=frame_notes_list,
        )
