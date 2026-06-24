import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Pegasus enforces a max_tokens floor; requests below this are rejected.
MIN_MAX_TOKENS = 512


class TwelveLabsClient(LLMClient):
    """Client for TwelveLabs' Pegasus video understanding model.

    Unlike the frame-by-frame vision clients (Ollama / OpenAI-compatible),
    Pegasus natively ingests an entire video and reasons jointly over its
    visuals, motion, and audio. This collapses the project's three-stage
    frame-extraction + transcription + reconstruction pipeline into a single
    ``analyze`` call, so no local frame sampling or Whisper transcription is
    required when this client is selected.

    The whole-video entry point is :meth:`analyze_video`. ``generate`` is
    implemented for interface compatibility but Pegasus does not analyze
    individual still frames, so it raises ``NotImplementedError``.
    """

    def __init__(self, api_key: str, model: str = "pegasus1.5"):
        if not api_key:
            raise ValueError("API key is required when using the TwelveLabs client")
        # Imported lazily so the dependency is only needed when this client is used.
        try:
            from twelvelabs import TwelveLabs
        except ImportError as e:
            raise ImportError(
                "The 'twelvelabs' package is required to use the TwelveLabs client. "
                "Install it with: pip install twelvelabs"
            ) from e
        self.client = TwelveLabs(api_key=api_key)
        self.model = model

    @staticmethod
    def _is_url(video: str) -> bool:
        parsed = urlparse(str(video))
        return parsed.scheme in ("http", "https")

    def _build_video_context(self, video: str):
        """Build a Pegasus VideoContext from a local file path or a public URL."""
        from twelvelabs.types.video_context import (
            VideoContext_Base64String,
            VideoContext_Url,
        )

        if self._is_url(video):
            logger.debug("Using remote video URL for TwelveLabs analysis")
            return VideoContext_Url(url=video)

        path = Path(video)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video}")
        logger.debug(f"Encoding local video {path} for TwelveLabs analysis")
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return VideoContext_Base64String(base_64_string=encoded)

    def analyze_video(
        self,
        video: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Analyze an entire video in a single Pegasus call.

        Args:
            video: Path to a local video file or a public http(s) URL.
                URLs are fetched server-side by TwelveLabs.
            prompt: Natural-language instruction describing what to extract.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens in the response. Values below
                ``MIN_MAX_TOKENS`` are raised to that floor since Pegasus
                rejects smaller limits.

        Returns:
            Dict with a "response" key holding the generated description,
            matching the shape returned by the other LLM clients.
        """
        video_context = self._build_video_context(video)
        response = self.client.analyze(
            model_name=self.model,
            video=video_context,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max(max_tokens, MIN_MAX_TOKENS),
        )
        if getattr(response, "error", None):
            raise Exception(f"TwelveLabs analysis error: {response.error}")
        return {"response": response.data or ""}

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "pegasus1.5",
        temperature: float = 0.2,
        num_predict: int = 256,
    ) -> Dict[Any, Any]:
        """Not supported: Pegasus analyzes whole videos, not single frames.

        Use :meth:`analyze_video` instead. The whole-video path is wired into
        the CLI, which bypasses per-frame analysis when this client is active.
        """
        raise NotImplementedError(
            "TwelveLabsClient analyzes whole videos via analyze_video(); "
            "it does not support per-frame generate() calls."
        )
