"""Tests for DSPy Signature definitions."""

from video_analyzer_tune.signatures import (
    DescriptionJudgeSignature,
    FrameAnalysisSignature,
    FrameNoteJudgeSignature,
    ReconstructionSignature,
)


def test_frame_analysis_input_fields():
    inputs = list(FrameAnalysisSignature.input_fields.keys())
    assert "image" in inputs
    assert "previous_frames" in inputs
    assert "user_question" in inputs


def test_frame_analysis_output_fields():
    outputs = list(FrameAnalysisSignature.output_fields.keys())
    assert "frame_note" in outputs


def test_reconstruction_input_fields():
    inputs = list(ReconstructionSignature.input_fields.keys())
    assert "frame_notes" in inputs
    assert "first_frame_note" in inputs
    assert "transcript" in inputs
    assert "user_question" in inputs


def test_reconstruction_output_fields():
    outputs = list(ReconstructionSignature.output_fields.keys())
    assert "description" in outputs


def test_description_judge_fields():
    inputs = list(DescriptionJudgeSignature.input_fields.keys())
    assert "ideal" in inputs
    assert "candidate" in inputs
    assert "user_question" in inputs
    outputs = list(DescriptionJudgeSignature.output_fields.keys())
    assert "score" in outputs


def test_frame_note_judge_fields():
    inputs = list(FrameNoteJudgeSignature.input_fields.keys())
    assert "ideal" in inputs
    assert "candidate" in inputs
    outputs = list(FrameNoteJudgeSignature.output_fields.keys())
    assert "score" in outputs
