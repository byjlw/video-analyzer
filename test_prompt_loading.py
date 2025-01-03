#!/usr/bin/env python3
"""Test script for the prompt loading system."""
import tempfile
from pathlib import Path
from video_analyzer.prompt import PromptLoader

def test_prompt_loading():
    """Test prompt loading with package and custom prompts."""
    prompts = [
        {
            "name": "Frame Analysis",
            "path": "frame_analysis/frame_analysis.txt"
        },
        {
            "name": "Video Reconstruction", 
            "path": "frame_analysis/describe.txt"
        }
    ]
    
    # Test default package prompts
    loader = PromptLoader("", prompts)
    assert loader.get_by_name("Frame Analysis"), "Failed to load package frame analysis prompt"
    assert loader.get_by_name("Video Reconstruction"), "Failed to load package describe prompt"
    
    # Test custom prompts in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / 'test_prompts'
        frame_dir = test_dir / 'frame_analysis'
        frame_dir.mkdir(parents=True)
        
        (frame_dir / 'frame_analysis.txt').write_text('Test frame analysis')
        (frame_dir / 'describe.txt').write_text('Test description')
        
        loader = PromptLoader(str(test_dir), prompts)
        assert loader.get_by_name("Frame Analysis") == 'Test frame analysis'
        assert loader.get_by_name("Video Reconstruction") == 'Test description'

if __name__ == "__main__":
    test_prompt_loading()
    print("All tests passed!")
