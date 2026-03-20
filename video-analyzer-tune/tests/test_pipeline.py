"""Tests for VideoAnalysisPipeline."""

from unittest.mock import MagicMock

import dspy
import pytest

from video_analyzer_tune.pipeline import VideoAnalysisPipeline


@pytest.fixture
def mock_frames():
    return [
        {"index": 0, "timestamp": 0.0, "image": MagicMock()},
        {"index": 1, "timestamp": 1.5, "image": MagicMock()},
    ]


def make_pipeline_with_mocks(frame_note="Test note", description="Test description"):
    pipeline = VideoAnalysisPipeline()
    frame_pred = MagicMock()
    frame_pred.frame_note = frame_note
    recon_pred = MagicMock()
    recon_pred.description = description
    pipeline.analyze_frame = MagicMock(return_value=frame_pred)
    pipeline.reconstruct = MagicMock(return_value=recon_pred)
    return pipeline


def test_returns_prediction(mock_frames):
    pipeline = make_pipeline_with_mocks()
    result = pipeline.forward(frames=mock_frames, user_question="", transcript="")
    assert isinstance(result, dspy.Prediction)


def test_returns_description(mock_frames):
    pipeline = make_pipeline_with_mocks(description="Final description")
    result = pipeline.forward(frames=mock_frames, user_question="", transcript="")
    assert result.description == "Final description"


def test_returns_frame_notes_list(mock_frames):
    pipeline = make_pipeline_with_mocks(frame_note="A note")
    result = pipeline.forward(frames=mock_frames, user_question="", transcript="")
    assert len(result.frame_notes_list) == len(mock_frames)
    assert all(n == "A note" for n in result.frame_notes_list)


def test_analyze_frame_called_once_per_frame(mock_frames):
    pipeline = make_pipeline_with_mocks()
    pipeline.forward(frames=mock_frames, user_question="", transcript="")
    assert pipeline.analyze_frame.call_count == len(mock_frames)


def test_reconstruct_called_once(mock_frames):
    pipeline = make_pipeline_with_mocks()
    pipeline.forward(frames=mock_frames, user_question="", transcript="")
    assert pipeline.reconstruct.call_count == 1


def test_first_frame_receives_empty_previous_frames(mock_frames):
    captured = []

    def capture(**kwargs):
        captured.append(kwargs.get("previous_frames", "MISSING"))
        pred = MagicMock()
        pred.frame_note = f"Note {len(captured)}"
        return pred

    pipeline = VideoAnalysisPipeline()
    pipeline.analyze_frame = MagicMock(side_effect=capture)
    pipeline.reconstruct = MagicMock(return_value=MagicMock(description="done"))

    pipeline.forward(frames=mock_frames, user_question="", transcript="")
    assert captured[0] == ""


def test_subsequent_frames_receive_accumulated_previous_frames(mock_frames):
    captured = []

    def capture(**kwargs):
        captured.append(kwargs.get("previous_frames", ""))
        pred = MagicMock()
        pred.frame_note = f"Note {len(captured)}"
        return pred

    pipeline = VideoAnalysisPipeline()
    pipeline.analyze_frame = MagicMock(side_effect=capture)
    pipeline.reconstruct = MagicMock(return_value=MagicMock(description="done"))

    pipeline.forward(frames=mock_frames, user_question="", transcript="")
    assert "Frame 0" in captured[1]


def test_handles_empty_frames_list():
    pipeline = VideoAnalysisPipeline()
    pipeline.analyze_frame = MagicMock()
    pipeline.reconstruct = MagicMock(return_value=MagicMock(description="empty"))

    result = pipeline.forward(frames=[], user_question="", transcript="")
    assert pipeline.analyze_frame.call_count == 0
    assert result.description == "empty"
    assert result.frame_notes_list == []


def test_handles_analyze_frame_exception(mock_frames):
    pipeline = VideoAnalysisPipeline()
    pipeline.analyze_frame = MagicMock(side_effect=Exception("LM error"))
    pipeline.reconstruct = MagicMock(return_value=MagicMock(description="recovered"))

    result = pipeline.forward(frames=mock_frames, user_question="", transcript="")
    assert len(result.frame_notes_list) == len(mock_frames)
    assert all("Error" in note for note in result.frame_notes_list)
