"""Tests for prompt_writer.py."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video_analyzer_tune.prompt_writer import (
    extract_optimized_instructions,
    print_config_snippet,
    write_prompt_files,
)


def make_mock_pipeline(
    frame_instruction="Optimized frame instruction",
    recon_instruction="Optimized recon instruction",
):
    pipeline = MagicMock()
    frame_sig = MagicMock()
    frame_sig.instructions = frame_instruction
    pipeline.analyze_frame.extended_signature = frame_sig
    recon_sig = MagicMock()
    recon_sig.instructions = recon_instruction
    pipeline.reconstruct.extended_signature = recon_sig
    return pipeline


# --- Instruction extraction ---

def test_extract_instructions_has_both_keys():
    pipeline = make_mock_pipeline()
    result = extract_optimized_instructions(pipeline)
    assert "frame_analysis_instruction" in result
    assert "reconstruction_instruction" in result


def test_extract_instructions_values():
    pipeline = make_mock_pipeline(
        frame_instruction="Better frame text",
        recon_instruction="Better recon text",
    )
    result = extract_optimized_instructions(pipeline)
    assert result["frame_analysis_instruction"] == "Better frame text"
    assert result["reconstruction_instruction"] == "Better recon text"


# --- Prompt file contents ---

def test_frame_analysis_file_contains_previous_frames_token(tmp_path):
    instructions = {
        "frame_analysis_instruction": "My instruction",
        "reconstruction_instruction": "Other instruction",
    }
    write_prompt_files(instructions, tmp_path)
    content = (tmp_path / "frame_analysis_tuned.txt").read_text()
    assert "{PREVIOUS_FRAMES}" in content


def test_frame_analysis_file_contains_prompt_token(tmp_path):
    instructions = {
        "frame_analysis_instruction": "My instruction",
        "reconstruction_instruction": "Other instruction",
    }
    write_prompt_files(instructions, tmp_path)
    content = (tmp_path / "frame_analysis_tuned.txt").read_text()
    assert "{prompt}" in content


def test_frame_analysis_file_contains_instruction(tmp_path):
    instructions = {
        "frame_analysis_instruction": "My custom instruction",
        "reconstruction_instruction": "Other",
    }
    write_prompt_files(instructions, tmp_path)
    content = (tmp_path / "frame_analysis_tuned.txt").read_text()
    assert "My custom instruction" in content


def test_describe_file_contains_frame_notes_token(tmp_path):
    instructions = {
        "frame_analysis_instruction": "inst",
        "reconstruction_instruction": "inst",
    }
    write_prompt_files(instructions, tmp_path)
    content = (tmp_path / "describe_tuned.txt").read_text()
    assert "{FRAME_NOTES}" in content


def test_describe_file_contains_transcript_token(tmp_path):
    instructions = {
        "frame_analysis_instruction": "inst",
        "reconstruction_instruction": "inst",
    }
    write_prompt_files(instructions, tmp_path)
    content = (tmp_path / "describe_tuned.txt").read_text()
    assert "{TRANSCRIPT}" in content


def test_describe_file_contains_prompt_token(tmp_path):
    instructions = {
        "frame_analysis_instruction": "inst",
        "reconstruction_instruction": "inst",
    }
    write_prompt_files(instructions, tmp_path)
    content = (tmp_path / "describe_tuned.txt").read_text()
    assert "{prompt}" in content


def test_describe_file_contains_first_frame_token(tmp_path):
    instructions = {
        "frame_analysis_instruction": "inst",
        "reconstruction_instruction": "inst",
    }
    write_prompt_files(instructions, tmp_path)
    content = (tmp_path / "describe_tuned.txt").read_text()
    assert "{FIRST_FRAME}" in content


def test_describe_file_contains_instruction(tmp_path):
    instructions = {
        "frame_analysis_instruction": "frame inst",
        "reconstruction_instruction": "My recon instruction",
    }
    write_prompt_files(instructions, tmp_path)
    content = (tmp_path / "describe_tuned.txt").read_text()
    assert "My recon instruction" in content


# --- Output directory creation ---

def test_write_creates_nested_output_dir(tmp_path):
    instructions = {
        "frame_analysis_instruction": "inst",
        "reconstruction_instruction": "inst",
    }
    nested = tmp_path / "a" / "b" / "c"
    write_prompt_files(instructions, nested)
    assert nested.exists()
    assert (nested / "frame_analysis_tuned.txt").exists()
    assert (nested / "describe_tuned.txt").exists()


# --- Config snippet ---

def test_config_snippet_contains_file_names(tmp_path, capsys):
    print_config_snippet(tmp_path / "tuned_prompts")
    captured = capsys.readouterr()
    assert "frame_analysis_tuned.txt" in captured.out
    assert "describe_tuned.txt" in captured.out


def test_config_snippet_contains_prompt_dir(tmp_path, capsys):
    print_config_snippet(tmp_path / "tuned_prompts")
    captured = capsys.readouterr()
    assert "prompt_dir" in captured.out


def test_config_snippet_contains_prompts_key(tmp_path, capsys):
    print_config_snippet(tmp_path / "tuned_prompts")
    captured = capsys.readouterr()
    assert "prompts" in captured.out
