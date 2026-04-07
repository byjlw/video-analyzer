import argparse
import json

from video_analyzer.config import Config
from video_analyzer.cli import resolve_max_frames


def test_update_from_args_maps_output_to_output_dir(tmp_path):
    config_path = tmp_path / "config.json"

    config = Config(str(config_path))
    args = argparse.Namespace(
        video_path="video.mp4",
        config=str(config_path),
        output="custom-output",
        client=None,
        ollama_url=None,
        api_key=None,
        api_url=None,
        model=None,
        duration=None,
        keep_frames=False,
        whisper_model=None,
        start_stage=1,
        max_frames=10,
        log_level="INFO",
        prompt="",
        language=None,
        device="cpu",
        temperature=None,
    )

    config.update_from_args(args)

    assert config.get("output_dir") == "custom-output"


def test_update_from_args_preserves_keep_frames_when_flag_omitted(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"keep_frames": True}))

    config = Config(str(config_path))
    args = argparse.Namespace(
        video_path="video.mp4",
        config=str(config_path),
        output=None,
        client=None,
        ollama_url=None,
        api_key=None,
        api_url=None,
        model=None,
        duration=None,
        keep_frames=None,
        whisper_model=None,
        start_stage=1,
        max_frames=10,
        log_level="INFO",
        prompt="",
        language=None,
        device="cpu",
        temperature=None,
    )

    config.update_from_args(args)

    assert config.get("keep_frames") is True


def test_update_from_args_sets_keep_frames_when_flag_passed(tmp_path):
    config_path = tmp_path / "config.json"

    config = Config(str(config_path))
    args = argparse.Namespace(
        video_path="video.mp4",
        config=str(config_path),
        output=None,
        client=None,
        ollama_url=None,
        api_key=None,
        api_url=None,
        model=None,
        duration=None,
        keep_frames=True,
        whisper_model=None,
        start_stage=1,
        max_frames=10,
        log_level="INFO",
        prompt="",
        language=None,
        device="cpu",
        temperature=None,
    )

    config.update_from_args(args)

    assert config.get("keep_frames") is True


def test_config_supports_frames_max_frames(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"frames": {"max_frames": 7}}))

    config = Config(str(config_path))

    assert config.get("frames", {}).get("max_frames") == 7


def test_update_from_args_does_not_override_frames_max_frames_when_omitted(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"frames": {"max_frames": 7}}))

    config = Config(str(config_path))
    args = argparse.Namespace(
        video_path="video.mp4",
        config=str(config_path),
        output=None,
        client=None,
        ollama_url=None,
        api_key=None,
        api_url=None,
        model=None,
        duration=None,
        keep_frames=None,
        whisper_model=None,
        start_stage=1,
        max_frames=None,
        log_level="INFO",
        prompt="",
        language=None,
        device="cpu",
        temperature=None,
    )

    config.update_from_args(args)

    assert config.get("frames", {}).get("max_frames") == 7


def test_resolve_max_frames_uses_config_when_cli_omitted(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"frames": {"max_frames": 7}}))

    config = Config(str(config_path))
    args = argparse.Namespace(max_frames=None)

    assert resolve_max_frames(args, config) == 7


def test_resolve_max_frames_prefers_cli_over_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"frames": {"max_frames": 7}}))

    config = Config(str(config_path))
    args = argparse.Namespace(max_frames=3)

    assert resolve_max_frames(args, config) == 3


def test_load_config_preserves_default_nested_values_when_user_config_is_partial(
    tmp_path,
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"audio": {"whisper_model": "large"}}))

    config = Config(str(config_path))
    audio_config = config.get("audio", {})

    assert audio_config.get("whisper_model") == "large"
    assert audio_config.get("device") == "cpu"
    assert audio_config.get("sample_rate") == 16000


def test_load_config_allows_nested_user_values_to_override_defaults(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "clients": {
                    "default": "openai_api",
                    "openai_api": {"model": "custom-model"},
                }
            }
        )
    )

    config = Config(str(config_path))

    assert config.get("clients", {}).get("default") == "openai_api"
    assert (
        config.get("clients", {}).get("openai_api", {}).get("model") == "custom-model"
    )
    assert (
        config.get("clients", {}).get("openai_api", {}).get("api_url")
        == "https://openrouter.ai/api/v1"
    )


def test_load_config_normalizes_non_list_prompts_to_empty_list(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"prompts": "not-a-list"}))

    config = Config(str(config_path))

    assert config.get("prompts") == []


def test_loads_packaged_defaults_when_custom_config_file_is_missing(tmp_path):
    config = Config(str(tmp_path / "missing-config.json"))

    assert config.get("output_dir") == "output"


def test_custom_config_file_path_merges_on_top_of_packaged_defaults(tmp_path):
    config_path = tmp_path / "custom-config.json"
    config_path.write_text(json.dumps({"output_dir": "custom-output"}))

    config = Config(str(config_path))

    assert config.get("output_dir") == "custom-output"
    assert config.get("clients", {}).get("default") == "ollama"
