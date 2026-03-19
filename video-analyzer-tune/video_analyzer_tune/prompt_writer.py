"""Extract optimized instructions from a compiled DSPy pipeline and write prompt files."""

import logging
from pathlib import Path

import dspy

logger = logging.getLogger(__name__)

# Prompt file templates preserving the exact token positions that
# video_analyzer/analyzer.py uses for string replacement.
#
# frame_analysis.txt tokens: {PREVIOUS_FRAMES}, {prompt}
# describe.txt tokens:       {FRAME_NOTES}, {TRANSCRIPT}, {FIRST_FRAME}, {prompt}

_FRAME_ANALYSIS_TEMPLATE = """{PREVIOUS_FRAMES}

{instruction}

{prompt}"""

_DESCRIBE_TEMPLATE = """{FRAME_NOTES}

{TRANSCRIPT}

{instruction}

{prompt}

{FIRST_FRAME}"""


def _extract_instruction(predict_module: dspy.Predict) -> str:
    """Extract the optimized instruction text from a DSPy Predict module.

    Tries multiple attribute paths to handle DSPy version differences.
    Falls back to the original signature docstring if nothing is found.
    """
    for attr in ("extended_signature", "signature"):
        sig = getattr(predict_module, attr, None)
        if sig is not None:
            instructions = getattr(sig, "instructions", None)
            if instructions:
                return str(instructions).strip()

    # Fallback: use the original signature class docstring
    sig_class = getattr(predict_module, "signature", None)
    if sig_class is not None and getattr(sig_class, "__doc__", None):
        return sig_class.__doc__.strip()

    return "Analyze the provided inputs and produce the requested output."


def extract_optimized_instructions(optimized_pipeline) -> dict:
    """Extract optimized instruction text from a compiled DSPy pipeline.

    Returns:
        Dict with keys 'frame_analysis_instruction' and 'reconstruction_instruction'
    """
    return {
        "frame_analysis_instruction": _extract_instruction(optimized_pipeline.analyze_frame),
        "reconstruction_instruction": _extract_instruction(optimized_pipeline.reconstruct),
    }


def write_prompt_files(instructions: dict, output_dir: Path) -> None:
    """Write tuned prompt files to output_dir.

    Each file is a drop-in replacement for the originals — it preserves all
    {TOKEN} placeholders that video-analyzer uses for string replacement, with
    the DSPy-optimized instruction text embedded.

    Args:
        instructions: Dict from extract_optimized_instructions()
        output_dir: Directory to write prompt files into
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_analysis_path = output_dir / "frame_analysis_tuned.txt"
    frame_analysis_path.write_text(
        _FRAME_ANALYSIS_TEMPLATE.format(
            instruction=instructions["frame_analysis_instruction"]
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote frame analysis prompt: {frame_analysis_path}")

    describe_path = output_dir / "describe_tuned.txt"
    describe_path.write_text(
        _DESCRIBE_TEMPLATE.format(
            instruction=instructions["reconstruction_instruction"]
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote reconstruction prompt: {describe_path}")


def print_config_snippet(output_dir: Path) -> None:
    """Print the config.json snippet for the user to paste in."""
    print(f"""
Tuning complete! Add the following to your config/config.json:

  "prompt_dir": "{output_dir}",
  "prompts": [
    {{"name": "Frame Analysis", "path": "frame_analysis_tuned.txt"}},
    {{"name": "Video Reconstruction", "path": "describe_tuned.txt"}}
  ]
""")
