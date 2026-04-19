import sys
import types

import pytest

from video_analyzer.whisper_backends import MlxWhisperBackend, create_whisper_backend


def test_create_whisper_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown whisper backend"):
        create_whisper_backend("unknown", "medium", "cpu")


def test_mlx_whisper_backend_forwards_expected_transcribe_options(monkeypatch):
    captured = {}

    def fake_transcribe(audio_path, **kwargs):
        captured["audio_path"] = audio_path
        captured.update(kwargs)
        return {"text": "hello", "segments": [{"text": "hello"}], "language": "en"}

    fake_module = types.SimpleNamespace(transcribe=fake_transcribe)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)

    backend = MlxWhisperBackend("mlx-community/whisper-large-v3-turbo-asr-fp16")
    result = backend.transcribe("audio.wav", language="en")

    assert result["text"] == "hello"
    assert captured == {
        "audio_path": "audio.wav",
        "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo-asr-fp16",
        "word_timestamps": True,
        "temperature": 0,
        "language": "en",
    }
