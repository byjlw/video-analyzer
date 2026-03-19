"""Load and validate training examples from video-analyzer output directories."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingFrame:
    """A single frame from a training example."""
    index: int
    timestamp: float
    image_path: Optional[Path]
    ideal_note: Optional[str]


@dataclass
class TrainingExample:
    """A complete training example loaded from an analysis.json output."""
    video_path: str
    user_question: str
    transcript: str
    ideal_description: str
    has_ideal_frame_notes: bool
    frames: List[TrainingFrame] = field(default_factory=list)


def _find_frame_images(frames_dir: Path, count: int) -> List[Optional[Path]]:
    """Find frame image files in frames_dir, sorted by name, aligned to count.

    Matches the naming convention used by video-analyzer when extracting frames.
    """
    if not frames_dir.exists():
        logger.warning(f"Frames directory not found: {frames_dir}")
        return [None] * count

    images = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: p.name,
    )

    result = []
    for i in range(count):
        result.append(images[i] if i < len(images) else None)
    return result


def _load_example(output_dir: Path) -> TrainingExample:
    """Load a single training example from an output directory."""
    analysis_file = output_dir / "analysis.json"
    if not analysis_file.exists():
        raise ValueError(f"analysis.json not found in: {output_dir}")

    with open(analysis_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required: ideal final description
    video_desc = data.get("video_description") or {}
    ideal_description = (video_desc.get("response") or "").strip()
    if not ideal_description:
        raise ValueError(
            f"video_description.response is empty in {analysis_file}. "
            "Please edit this field to provide your ideal video description."
        )

    # Optional fields — missing or null is fine
    user_question = (data.get("prompt") or "").strip()
    transcript_data = data.get("transcript") or {}
    transcript = (transcript_data.get("text") or "").strip()
    video_path = data.get("video_path") or str(output_dir)

    # Frame analyses
    frame_analyses = data.get("frame_analyses") or []
    if not frame_analyses:
        raise ValueError(f"No frame_analyses found in {analysis_file}")

    # Reconstruct frame image paths from the frames/ subdirectory
    frames_dir = output_dir / "frames"
    image_paths = _find_frame_images(frames_dir, len(frame_analyses))

    frames = []
    has_ideal_frame_notes = False
    for i, (analysis, image_path) in enumerate(zip(frame_analyses, image_paths)):
        ideal_note = (analysis.get("response") or "").strip()
        if ideal_note:
            has_ideal_frame_notes = True

        frames.append(TrainingFrame(
            index=i,
            timestamp=float(analysis.get("timestamp", 0.0)),
            image_path=image_path,
            ideal_note=ideal_note or None,
        ))

    return TrainingExample(
        video_path=video_path,
        user_question=user_question,
        transcript=transcript,
        ideal_description=ideal_description,
        has_ideal_frame_notes=has_ideal_frame_notes,
        frames=frames,
    )


def load_training_data(training_data_path: str) -> List[TrainingExample]:
    """Load all training examples from a training_data.json file.

    Each example points to an output directory containing an analysis.json
    that has been edited to reflect ideal outputs.

    Args:
        training_data_path: Path to training_data.json

    Returns:
        List of TrainingExample objects

    Raises:
        ValueError: If training data is invalid or required fields are missing
    """
    training_data_path = Path(training_data_path)
    if not training_data_path.exists():
        raise ValueError(f"Training data file not found: {training_data_path}")

    with open(training_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples_config = data.get("examples")
    if not examples_config or not isinstance(examples_config, list):
        raise ValueError(
            "training_data.json must contain an 'examples' key with a non-empty list.\n"
            'Example: {"examples": [{"output_dir": "output"}]}'
        )

    base_dir = training_data_path.parent
    examples = []

    for i, entry in enumerate(examples_config):
        if "output_dir" not in entry:
            raise ValueError(f"Example {i} is missing required 'output_dir' field")

        output_dir = Path(entry["output_dir"])
        if not output_dir.is_absolute():
            output_dir = (base_dir / output_dir).resolve()

        if not output_dir.exists():
            raise ValueError(f"output_dir does not exist: {output_dir}")

        try:
            example = _load_example(output_dir)
            examples.append(example)
            frame_note_status = "with ideal frame notes" if example.has_ideal_frame_notes else "without ideal frame notes"
            logger.info(f"Loaded example from {output_dir} ({frame_note_status})")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error loading example from {output_dir}: {e}") from e

    return examples
