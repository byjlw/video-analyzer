"""Tests for VideoAnalysisMetric."""

from unittest.mock import MagicMock

import pytest

from video_analyzer_tune.metrics import VideoAnalysisMetric


def make_example(ideal_desc, ideal_frames=None, user_question="", has_ideal_frame_notes=None):
    ex = MagicMock()
    ex.ideal_description = ideal_desc
    ex.frame_notes_list = ideal_frames or []
    ex.user_question = user_question
    ex.has_ideal_frame_notes = (
        has_ideal_frame_notes
        if has_ideal_frame_notes is not None
        else bool(ideal_frames)
    )
    return ex


def make_prediction(desc, frames=None):
    pred = MagicMock()
    pred.description = desc
    pred.frame_notes_list = frames or []
    return pred


def make_metric_with_scores(desc_score=5, frame_score=5, description_weight=0.7):
    metric = VideoAnalysisMetric(description_weight=description_weight)
    desc_result = MagicMock()
    desc_result.score = desc_score
    frame_result = MagicMock()
    frame_result.score = frame_score
    metric.judge_description = MagicMock(return_value=desc_result)
    metric.judge_frame = MagicMock(return_value=frame_result)
    return metric


# --- Weight and scoring ---

def test_description_only_when_no_frame_notes():
    metric = make_metric_with_scores(desc_score=4)
    example = make_example("ideal", has_ideal_frame_notes=False)
    prediction = make_prediction("candidate")

    score = metric(example, prediction)

    assert score == pytest.approx(4 / 5.0)
    metric.judge_frame.assert_not_called()


def test_weighted_combination_with_frame_notes():
    metric = make_metric_with_scores(desc_score=5, frame_score=3, description_weight=0.7)
    example = make_example("ideal", ideal_frames=["frame note"], has_ideal_frame_notes=True)
    prediction = make_prediction("candidate", frames=["candidate frame"])

    score = metric(example, prediction)

    # 0.7 * (5/5) + 0.3 * (3/5) = 0.7 + 0.18 = 0.88
    assert score == pytest.approx(0.7 * 1.0 + 0.3 * 0.6)


def test_50_50_weight():
    metric = make_metric_with_scores(desc_score=4, frame_score=2, description_weight=0.5)
    example = make_example("ideal", ideal_frames=["note"], has_ideal_frame_notes=True)
    prediction = make_prediction("candidate", frames=["candidate"])

    score = metric(example, prediction)

    # 0.5 * (4/5) + 0.5 * (2/5) = 0.4 + 0.2 = 0.6
    assert score == pytest.approx(0.5 * 0.8 + 0.5 * 0.4)


def test_100_description_weight_ignores_frame_notes():
    metric = make_metric_with_scores(desc_score=3, frame_score=5, description_weight=1.0)
    example = make_example("ideal", ideal_frames=["note"], has_ideal_frame_notes=True)
    prediction = make_prediction("candidate", frames=["candidate"])

    score = metric(example, prediction)

    assert score == pytest.approx(3 / 5.0)
    metric.judge_frame.assert_not_called()


# --- Error handling ---

def test_judge_exception_returns_zero():
    metric = VideoAnalysisMetric()
    metric.judge_description = MagicMock(side_effect=Exception("LM error"))

    example = make_example("ideal", has_ideal_frame_notes=False)
    prediction = make_prediction("candidate")

    score = metric(example, prediction)
    assert score == 0.0


def test_empty_ideal_returns_zero():
    metric = VideoAnalysisMetric()
    metric.judge_description = MagicMock()

    example = make_example("", has_ideal_frame_notes=False)
    prediction = make_prediction("candidate")

    score = metric(example, prediction)
    assert score == 0.0
    metric.judge_description.assert_not_called()


# --- Score parsing ---

def test_parse_score_integer():
    metric = VideoAnalysisMetric()
    assert metric._parse_score(4) == 4


def test_parse_score_string():
    metric = VideoAnalysisMetric()
    assert metric._parse_score("4") == 4


def test_parse_score_string_with_extra_text():
    metric = VideoAnalysisMetric()
    assert metric._parse_score("3 out of 5") == 3


def test_parse_score_clamps_below_1():
    metric = VideoAnalysisMetric()
    assert metric._parse_score(0) == 1
    assert metric._parse_score(-5) == 1


def test_parse_score_clamps_above_5():
    metric = VideoAnalysisMetric()
    assert metric._parse_score(10) == 5
    assert metric._parse_score(6) == 5


def test_parse_score_invalid_returns_1():
    metric = VideoAnalysisMetric()
    assert metric._parse_score("not a number") == 1
    assert metric._parse_score(None) == 1


# --- Validation ---

def test_invalid_weight_above_1_raises():
    with pytest.raises(ValueError):
        VideoAnalysisMetric(description_weight=1.5)


def test_invalid_weight_below_0_raises():
    with pytest.raises(ValueError):
        VideoAnalysisMetric(description_weight=-0.1)
