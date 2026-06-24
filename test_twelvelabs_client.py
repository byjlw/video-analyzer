#!/usr/bin/env python3
"""Tests for the TwelveLabs (Pegasus) client.

The unit tests run without network access or the SDK installed. The live
test calls the real Pegasus API and is skipped unless TWELVELABS_API_KEY is set.
"""
import os

from video_analyzer.clients.twelvelabs import TwelveLabsClient, MIN_MAX_TOKENS

# A short, publicly hosted sample video that TwelveLabs can fetch server-side.
SAMPLE_VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/"
    "Big_Buck_Bunny_720_10s_1MB.mp4"
)


def test_url_detection():
    """URLs and local paths are classified correctly."""
    assert TwelveLabsClient._is_url("https://example.com/video.mp4")
    assert TwelveLabsClient._is_url("http://example.com/video.mp4")
    assert not TwelveLabsClient._is_url("/tmp/video.mp4")
    assert not TwelveLabsClient._is_url("video.mp4")


def test_empty_api_key_rejected():
    """An empty API key raises a clear error before any network call."""
    try:
        TwelveLabsClient("")
        assert False, "Expected ValueError for empty API key"
    except ValueError:
        pass


def test_generate_not_supported():
    """Per-frame generate() is unsupported; whole-video analyze_video() is used."""
    key = os.environ.get("TWELVELABS_API_KEY")
    if not key:
        print("Skipping test_generate_not_supported (TWELVELABS_API_KEY not set)")
        return
    client = TwelveLabsClient(key)
    try:
        client.generate(prompt="hi", image_path="frame.jpg")
        assert False, "Expected NotImplementedError from generate()"
    except NotImplementedError:
        pass


def test_min_max_tokens_constant():
    """The Pegasus max_tokens floor is exposed as a constant."""
    assert MIN_MAX_TOKENS == 512


def test_analyze_video_live():
    """End-to-end Pegasus call against a public sample video (network + key)."""
    key = os.environ.get("TWELVELABS_API_KEY")
    if not key:
        print("Skipping test_analyze_video_live (TWELVELABS_API_KEY not set)")
        return
    client = TwelveLabsClient(key)
    result = client.analyze_video(
        video=SAMPLE_VIDEO_URL,
        prompt="Describe this video in one sentence.",
        max_tokens=512,
    )
    assert "response" in result
    assert isinstance(result["response"], str)
    assert result["response"].strip(), "Expected a non-empty description"
    print(f"Live analysis result: {result['response'][:200]}")


if __name__ == "__main__":
    test_url_detection()
    test_empty_api_key_rejected()
    test_min_max_tokens_constant()
    test_generate_not_supported()
    test_analyze_video_live()
    print("All tests passed!")
