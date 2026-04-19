from pathlib import Path

import pytest

from video_analyzer.prompt import PromptLoader


PROMPTS = [
    {
        "name": "Frame Analysis",
        "path": "frame_analysis/frame_analysis.txt",
    },
    {
        "name": "Video Reconstruction",
        "path": "frame_analysis/describe.txt",
    },
]


def test_prompt_loading_from_package_resources():
    loader = PromptLoader("", PROMPTS)

    assert loader.get_by_name("Frame Analysis")
    assert loader.get_by_name("Video Reconstruction")


def test_prompt_loading_from_custom_prompt_dir(tmp_path: Path):
    prompt_dir = tmp_path / "test_prompts"
    frame_dir = prompt_dir / "frame_analysis"
    frame_dir.mkdir(parents=True)

    (frame_dir / "frame_analysis.txt").write_text("Test frame analysis")
    (frame_dir / "describe.txt").write_text("Test description")

    loader = PromptLoader(str(prompt_dir), PROMPTS)

    assert loader.get_by_name("Frame Analysis") == "Test frame analysis"
    assert loader.get_by_name("Video Reconstruction") == "Test description"


def test_missing_prompt_name_raises_value_error():
    loader = PromptLoader("", PROMPTS)

    with pytest.raises(ValueError, match="Prompt with name 'Missing Prompt' not found"):
        loader.get_by_name("Missing Prompt")
