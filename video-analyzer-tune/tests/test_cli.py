"""Tests for cli.py entry point."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import video_analyzer_tune.cli as cli


def write_training_data(tmp_path, content=None):
    td_path = tmp_path / "td.json"
    td_path.write_text(json.dumps(content or {"examples": []}))
    return td_path


# --- Validation failures ---

def test_missing_api_key_for_openai_exits(tmp_path):
    td_path = write_training_data(tmp_path)
    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(td_path),
        "--client", "openai_api",
        "--api-url", "https://api.openai.com/v1",
    ]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1


def test_missing_api_url_for_openai_exits(tmp_path):
    td_path = write_training_data(tmp_path)
    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(td_path),
        "--client", "openai_api",
        "--api-key", "sk-test",
    ]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1


def test_api_key_without_client_infers_openai(tmp_path):
    """Providing --api-key without --client should infer openai_api, matching video-analyzer."""
    td_path = write_training_data(tmp_path)
    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(td_path),
        "--api-key", "sk-test",
        # --api-url missing → should exit with error, not ollama error
        # This confirms openai_api was inferred (ollama wouldn't care about api-url)
    ]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1


def test_invalid_description_weight_exits(tmp_path):
    td_path = write_training_data(tmp_path)
    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(td_path),
        "--description-weight", "1.5",
    ]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1


def test_nonexistent_training_data_exits(tmp_path):
    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(tmp_path / "nonexistent.json"),
    ]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1


# --- Happy path ---

def test_main_happy_path(tmp_path):
    td_path = write_training_data(tmp_path)
    mock_examples = [MagicMock()]
    mock_pipeline = MagicMock()
    mock_instructions = {
        "frame_analysis_instruction": "test",
        "reconstruction_instruction": "test",
    }

    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(td_path),
        "--output-dir", str(tmp_path / "out"),
    ]), \
    patch("video_analyzer_tune.cli.load_training_data", return_value=mock_examples), \
    patch("video_analyzer_tune.cli.PromptTuner") as mock_tuner_cls, \
    patch("video_analyzer_tune.cli.extract_optimized_instructions", return_value=mock_instructions), \
    patch("video_analyzer_tune.cli.write_prompt_files"), \
    patch("video_analyzer_tune.cli.print_config_snippet"):
        mock_tuner = MagicMock()
        mock_tuner.optimize.return_value = mock_pipeline
        mock_tuner_cls.return_value = mock_tuner

        cli.main()

        mock_tuner.optimize.assert_called_once_with(mock_examples)


def test_main_default_output_dir_is_tuned_prompts(tmp_path):
    td_path = write_training_data(tmp_path)
    captured = []

    def capture_write(instructions, output_dir):
        captured.append(Path(output_dir))

    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(td_path),
    ]), \
    patch("video_analyzer_tune.cli.load_training_data", return_value=[MagicMock()]), \
    patch("video_analyzer_tune.cli.PromptTuner") as mock_tuner_cls, \
    patch("video_analyzer_tune.cli.extract_optimized_instructions",
          return_value={"frame_analysis_instruction": "", "reconstruction_instruction": ""}), \
    patch("video_analyzer_tune.cli.write_prompt_files", side_effect=capture_write), \
    patch("video_analyzer_tune.cli.print_config_snippet"):
        mock_tuner_cls.return_value.optimize.return_value = MagicMock()
        cli.main()

    assert captured[0].name == "tuned_prompts"


def test_main_passes_description_weight_to_tuner(tmp_path):
    td_path = write_training_data(tmp_path)

    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(td_path),
        "--description-weight", "0.5",
    ]), \
    patch("video_analyzer_tune.cli.load_training_data", return_value=[MagicMock()]), \
    patch("video_analyzer_tune.cli.PromptTuner") as mock_tuner_cls, \
    patch("video_analyzer_tune.cli.extract_optimized_instructions",
          return_value={"frame_analysis_instruction": "", "reconstruction_instruction": ""}), \
    patch("video_analyzer_tune.cli.write_prompt_files"), \
    patch("video_analyzer_tune.cli.print_config_snippet"):
        mock_tuner_cls.return_value.optimize.return_value = MagicMock()
        cli.main()

    call_kwargs = mock_tuner_cls.call_args.kwargs
    assert call_kwargs["description_weight"] == pytest.approx(0.5)


def test_main_builds_lm_config_for_ollama(tmp_path):
    td_path = write_training_data(tmp_path)

    with patch("sys.argv", [
        "video-analyzer-tune",
        "--training-data", str(td_path),
        "--model", "llava:13b",
        "--ollama-url", "http://localhost:11434",
        # no --client or --api-key → should infer ollama
    ]), \
    patch("video_analyzer_tune.cli.load_training_data", return_value=[MagicMock()]), \
    patch("video_analyzer_tune.cli.PromptTuner") as mock_tuner_cls, \
    patch("video_analyzer_tune.cli.extract_optimized_instructions",
          return_value={"frame_analysis_instruction": "", "reconstruction_instruction": ""}), \
    patch("video_analyzer_tune.cli.write_prompt_files"), \
    patch("video_analyzer_tune.cli.print_config_snippet"):
        mock_tuner_cls.return_value.optimize.return_value = MagicMock()
        cli.main()

    lm_config = mock_tuner_cls.call_args.kwargs["lm_config"]
    assert lm_config["type"] == "ollama"
    assert lm_config["model"] == "llava:13b"
    assert lm_config["api_base"] == "http://localhost:11434"
