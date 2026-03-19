"""DSPy Signatures for the two video-analyzer prompts."""

import dspy


class FrameAnalysisSignature(dspy.Signature):
    """Analyze a single video frame and produce concise notes about what is visible,
    connecting observations to the context of previously analyzed frames."""

    image: dspy.Image = dspy.InputField(
        desc="The video frame image to analyze"
    )
    previous_frames: str = dspy.InputField(
        desc="Notes from previously analyzed frames in chronological order"
    )
    user_question: str = dspy.InputField(
        desc="The user's question or focus area for the analysis"
    )
    frame_note: str = dspy.OutputField(
        desc="Concise notes about this frame including setting, action, and key continuation points"
    )


class ReconstructionSignature(dspy.Signature):
    """Synthesize chronological frame notes and audio transcript into a cohesive video description."""

    frame_notes: str = dspy.InputField(
        desc="Chronological notes from all analyzed frames"
    )
    first_frame_note: str = dspy.InputField(
        desc="Analysis of the first frame to anchor the description"
    )
    transcript: str = dspy.InputField(
        desc="Audio transcript from the video (may be empty if no audio was detected)"
    )
    user_question: str = dspy.InputField(
        desc="The user's question or focus area for the description"
    )
    description: str = dspy.OutputField(
        desc="Comprehensive, coherent description of the video content"
    )


class DescriptionJudgeSignature(dspy.Signature):
    """Rate the quality of a video description compared to an ideal reference."""

    ideal: str = dspy.InputField(
        desc="The ideal reference description"
    )
    candidate: str = dspy.InputField(
        desc="The candidate description to evaluate"
    )
    user_question: str = dspy.InputField(
        desc="The user's original question or focus area"
    )
    score: int = dspy.OutputField(
        desc="Quality score from 1 (poor) to 5 (excellent), considering coverage, accuracy, and relevance to the user's question"
    )


class FrameNoteJudgeSignature(dspy.Signature):
    """Rate the quality of a frame analysis note compared to an ideal reference."""

    ideal: str = dspy.InputField(
        desc="The ideal frame note for reference"
    )
    candidate: str = dspy.InputField(
        desc="The candidate frame note to evaluate"
    )
    score: int = dspy.OutputField(
        desc="Quality score from 1 (poor) to 5 (excellent), considering detail, accuracy, and usefulness for video reconstruction"
    )
