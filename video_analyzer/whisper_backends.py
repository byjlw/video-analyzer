import logging
from typing import Any, Dict, List, Optional, Protocol


logger = logging.getLogger(__name__)


class WhisperBackend(Protocol):
    def transcribe(self, audio_path: str, language: Optional[str] = None): ...


class FasterWhisperBackend:
    def __init__(self, model_size_or_path: str = "medium", device: str = "cpu"):
        from faster_whisper import WhisperModel

        cache_dir = __import__("pathlib").Path.home() / ".cache" / "huggingface" / "hub"
        logger.debug(f"Using HuggingFace cache directory: {cache_dir}")
        logger.debug(f"Using device: {device}")

        self.model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type="float32",
        )
        logger.info(
            f"Initialized faster-whisper backend with model/path: {model_size_or_path}, device: {device}"
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None):
        return self.model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            language=language,
        )


class MlxWhisperBackend:
    def __init__(self, model_size_or_path: str = "mlx-community/whisper-turbo"):
        import mlx_whisper

        self.mlx_whisper = mlx_whisper
        self.model_size_or_path = model_size_or_path
        logger.info(
            f"Initialized mlx-whisper backend with model/path: {model_size_or_path}"
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None):
        result = self.mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.model_size_or_path,
            word_timestamps=True,
            temperature=0,
            language=language,
        )
        return result


def create_whisper_backend(
    backend: str,
    model_size_or_path: str,
    device: str,
) -> WhisperBackend:
    if backend == "faster-whisper":
        return FasterWhisperBackend(
            model_size_or_path=model_size_or_path, device=device
        )
    if backend == "mlx-whisper":
        return MlxWhisperBackend(model_size_or_path=model_size_or_path)
    raise ValueError(f"Unknown whisper backend: {backend}")
