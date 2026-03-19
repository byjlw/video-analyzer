"""Tests for training_data.py — loading and validating analysis.json files."""

import json
import logging
from pathlib import Path

import pytest

from video_analyzer_tune.training_data import (
    TrainingExample,
    TrainingFrame,
    load_training_data,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_analysis():
    with open(FIXTURES_DIR / "sample_analysis.json") as f:
        return json.load(f)


@pytest.fixture
def output_dir(tmp_path, sample_analysis):
    """Valid output directory with analysis.json and dummy frame images."""
    out = tmp_path / "output"
    out.mkdir()
    frames_dir = out / "frames"
    frames_dir.mkdir()
    for i in range(len(sample_analysis["frame_analyses"])):
        (frames_dir / f"frame_{i:03d}.jpg").write_bytes(b"fake_jpeg")
    with open(out / "analysis.json", "w") as f:
        json.dump(sample_analysis, f)
    return out


@pytest.fixture
def training_data_file(tmp_path, output_dir):
    """training_data.json pointing at output_dir."""
    data = {"examples": [{"output_dir": str(output_dir)}]}
    path = tmp_path / "training_data.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# --- Happy path ---

def test_load_valid_example(training_data_file):
    examples = load_training_data(str(training_data_file))
    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, TrainingExample)
    assert ex.ideal_description != ""
    assert ex.user_question == "What is happening?"
    assert ex.transcript == "No dialogue detected."
    assert len(ex.frames) == 2


def test_frame_objects_are_correct_type(training_data_file):
    examples = load_training_data(str(training_data_file))
    for f in examples[0].frames:
        assert isinstance(f, TrainingFrame)


def test_frame_indices_and_timestamps(training_data_file):
    examples = load_training_data(str(training_data_file))
    frames = examples[0].frames
    assert frames[0].index == 0
    assert frames[0].timestamp == 0.0
    assert frames[1].index == 1
    assert frames[1].timestamp == 1.5


def test_frame_image_paths_resolved(training_data_file, output_dir):
    examples = load_training_data(str(training_data_file))
    for frame in examples[0].frames:
        assert frame.image_path is not None
        assert frame.image_path.exists()


def test_has_ideal_frame_notes_true(training_data_file):
    examples = load_training_data(str(training_data_file))
    assert examples[0].has_ideal_frame_notes is True


def test_multiple_examples(tmp_path, sample_analysis):
    dirs = []
    for i in range(2):
        out = tmp_path / f"output_{i}"
        out.mkdir()
        with open(out / "analysis.json", "w") as f:
            json.dump(sample_analysis, f)
        dirs.append(out)

    data = {"examples": [{"output_dir": str(d)} for d in dirs]}
    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump(data, f)

    examples = load_training_data(str(td_path))
    assert len(examples) == 2


# --- Optional fields ---

def test_missing_transcript_uses_empty_string(tmp_path, sample_analysis):
    sample_analysis.pop("transcript", None)
    out = tmp_path / "output"
    out.mkdir()
    with open(out / "analysis.json", "w") as f:
        json.dump(sample_analysis, f)

    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"examples": [{"output_dir": str(out)}]}, f)

    examples = load_training_data(str(td_path))
    assert examples[0].transcript == ""


def test_null_transcript_uses_empty_string(tmp_path, sample_analysis):
    sample_analysis["transcript"] = None
    out = tmp_path / "output"
    out.mkdir()
    with open(out / "analysis.json", "w") as f:
        json.dump(sample_analysis, f)

    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"examples": [{"output_dir": str(out)}]}, f)

    examples = load_training_data(str(td_path))
    assert examples[0].transcript == ""


def test_missing_prompt_uses_empty_string(tmp_path, sample_analysis):
    sample_analysis.pop("prompt", None)
    out = tmp_path / "output"
    out.mkdir()
    with open(out / "analysis.json", "w") as f:
        json.dump(sample_analysis, f)

    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"examples": [{"output_dir": str(out)}]}, f)

    examples = load_training_data(str(td_path))
    assert examples[0].user_question == ""


def test_missing_frames_dir_warns_and_image_paths_none(tmp_path, sample_analysis, caplog):
    out = tmp_path / "output"
    out.mkdir()
    with open(out / "analysis.json", "w") as f:
        json.dump(sample_analysis, f)

    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"examples": [{"output_dir": str(out)}]}, f)

    with caplog.at_level(logging.WARNING):
        examples = load_training_data(str(td_path))

    assert any("frames" in r.message.lower() for r in caplog.records)
    assert all(frame.image_path is None for frame in examples[0].frames)


# --- Validation errors ---

def test_missing_video_description_raises(tmp_path, sample_analysis):
    sample_analysis["video_description"]["response"] = ""
    out = tmp_path / "output"
    out.mkdir()
    with open(out / "analysis.json", "w") as f:
        json.dump(sample_analysis, f)

    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"examples": [{"output_dir": str(out)}]}, f)

    with pytest.raises(ValueError, match="video_description.response is empty"):
        load_training_data(str(td_path))


def test_missing_output_dir_raises(tmp_path):
    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"examples": [{"output_dir": str(tmp_path / "nonexistent")}]}, f)

    with pytest.raises(ValueError):
        load_training_data(str(td_path))


def test_missing_examples_key_raises(tmp_path):
    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"not_examples": []}, f)

    with pytest.raises(ValueError, match="examples"):
        load_training_data(str(td_path))


def test_empty_examples_list_raises(tmp_path):
    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"examples": []}, f)

    with pytest.raises(ValueError):
        load_training_data(str(td_path))


def test_nonexistent_path_raises():
    with pytest.raises(ValueError, match="not found"):
        load_training_data("/nonexistent/path/training_data.json")


# --- Direct path inputs ---

def test_load_from_output_directory_directly(output_dir):
    """Point at the output dir itself, no training_data.json needed."""
    examples = load_training_data(str(output_dir))
    assert len(examples) == 1
    assert examples[0].ideal_description != ""


def test_load_from_analysis_json_directly(output_dir):
    """Point at analysis.json directly."""
    examples = load_training_data(str(output_dir / "analysis.json"))
    assert len(examples) == 1
    assert examples[0].ideal_description != ""


def test_load_from_directory_and_wrapper_produce_same_result(output_dir, training_data_file):
    """All three input forms load the same data."""
    from_dir = load_training_data(str(output_dir))
    from_file = load_training_data(str(output_dir / "analysis.json"))
    from_wrapper = load_training_data(str(training_data_file))

    assert from_dir[0].ideal_description == from_file[0].ideal_description == from_wrapper[0].ideal_description
    assert len(from_dir[0].frames) == len(from_file[0].frames) == len(from_wrapper[0].frames)


def test_missing_output_dir_field_raises(tmp_path):
    td_path = tmp_path / "td.json"
    with open(td_path, "w") as f:
        json.dump({"examples": [{"wrong_key": "value"}]}, f)

    with pytest.raises(ValueError, match="output_dir"):
        load_training_data(str(td_path))
