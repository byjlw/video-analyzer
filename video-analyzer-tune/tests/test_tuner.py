"""Tests for PromptTuner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
import pytest

from video_analyzer_tune.training_data import TrainingExample, TrainingFrame
from video_analyzer_tune.tuner import PromptTuner


def make_training_example(has_frame_notes=True, num_frames=2, user_question="What is happening?"):
    frames = [
        TrainingFrame(
            index=i,
            timestamp=float(i),
            image_path=None,
            ideal_note=f"Frame {i} note" if has_frame_notes else None,
        )
        for i in range(num_frames)
    ]
    return TrainingExample(
        video_path="test.mp4",
        user_question=user_question,
        transcript="No audio",
        ideal_description="A person walks into a room.",
        has_ideal_frame_notes=has_frame_notes,
        frames=frames,
    )


@pytest.fixture
def ollama_config():
    return {
        "type": "ollama",
        "model": "llama3.2-vision",
        "api_base": "http://localhost:11434",
        "api_key": None,
    }


@pytest.fixture
def openai_config():
    return {
        "type": "openai_api",
        "model": "gpt-4o",
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-test",
    }


# --- dspy.Example building ---

def test_build_dspy_examples_count(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    result = tuner._build_dspy_examples([make_training_example()])
    assert len(result) == 1


def test_build_dspy_examples_ideal_description(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    result = tuner._build_dspy_examples([make_training_example()])
    assert result[0].ideal_description == "A person walks into a room."


def test_build_dspy_examples_formats_user_question(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    result = tuner._build_dspy_examples([make_training_example(user_question="What is happening?")])
    # Must match VideoAnalyzer._format_user_prompt()
    assert result[0].user_question == "I want to know What is happening?"


def test_build_dspy_examples_empty_user_question(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    result = tuner._build_dspy_examples([make_training_example(user_question="")])
    assert result[0].user_question == ""


def test_build_dspy_examples_none_image_when_no_file(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    result = tuner._build_dspy_examples([make_training_example()])
    for frame in result[0].frames:
        assert frame["image"] is None


def test_build_dspy_examples_frame_count(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    result = tuner._build_dspy_examples([make_training_example(num_frames=3)])
    assert len(result[0].frames) == 3


# --- Train/val split ---

def test_split_small_dataset_reuses_as_val(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    examples = [MagicMock(), MagicMock()]
    trainset, valset = tuner._split_examples(examples)
    assert trainset == examples
    assert valset == examples


def test_split_single_example_reuses_as_val(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    examples = [MagicMock()]
    trainset, valset = tuner._split_examples(examples)
    assert trainset == valset


def test_split_large_dataset_80_20(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    examples = [MagicMock() for _ in range(5)]
    trainset, valset = tuner._split_examples(examples)
    assert len(trainset) == 4
    assert len(valset) == 1


def test_split_large_dataset_no_overlap(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    examples = [MagicMock() for _ in range(10)]
    trainset, valset = tuner._split_examples(examples)
    assert not any(e in valset for e in trainset)


# --- LM configuration ---

def test_configure_lm_ollama(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    with patch("dspy.LM") as mock_lm, patch("dspy.configure"):
        mock_lm.return_value = MagicMock()
        tuner._configure_lm()
        mock_lm.assert_called_once_with(
            model="ollama/llama3.2-vision",
            api_base="http://localhost:11434",
            api_key="ollama",
        )


def test_configure_lm_openai(openai_config):
    tuner = PromptTuner(lm_config=openai_config)
    with patch("dspy.LM") as mock_lm, patch("dspy.configure"):
        mock_lm.return_value = MagicMock()
        tuner._configure_lm()
        mock_lm.assert_called_once_with(
            model="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key="sk-test",
        )


# --- Full optimize() call ---

def test_optimize_calls_miprov2_compile(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    mock_optimized = MagicMock()

    with patch.object(tuner, "_configure_lm"), \
         patch.object(tuner, "_build_dspy_examples", return_value=[MagicMock()]), \
         patch("video_analyzer_tune.tuner.MIPROv2") as mock_mipro:

        mock_optimizer = MagicMock()
        mock_optimizer.compile.return_value = mock_optimized
        mock_mipro.return_value = mock_optimizer

        result = tuner.optimize([make_training_example()])

        mock_optimizer.compile.assert_called_once()
        assert result == mock_optimized


def test_optimize_passes_trainset_and_valset(ollama_config):
    tuner = PromptTuner(lm_config=ollama_config)
    fake_examples = [MagicMock() for _ in range(5)]

    with patch.object(tuner, "_configure_lm"), \
         patch.object(tuner, "_build_dspy_examples", return_value=fake_examples), \
         patch("video_analyzer_tune.tuner.MIPROv2") as mock_mipro:

        mock_optimizer = MagicMock()
        mock_optimizer.compile.return_value = MagicMock()
        mock_mipro.return_value = mock_optimizer

        tuner.optimize([make_training_example() for _ in range(5)])

        compile_kwargs = mock_optimizer.compile.call_args.kwargs
        assert "trainset" in compile_kwargs
        assert "valset" in compile_kwargs
        # num_trials belongs in compile(), not MIPROv2.__init__()
        assert "num_trials" in compile_kwargs
        init_kwargs = mock_mipro.call_args.kwargs
        assert "num_trials" not in init_kwargs
