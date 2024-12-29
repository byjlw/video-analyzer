#!/usr/bin/env python3
"""
Test script for the prompt loading system.

This script verifies that prompts can be loaded correctly in various scenarios:
1. Default Package Prompts: Using built-in prompts
2. Absolute Path: Using full system path
3. Relative Path: Using path relative to current directory
4. User Home Path: Using ~/test_prompts
5. Different Directory: Running from another location

Usage:
    # Setup test environment
    mkdir -p test_prompts/frame_analysis
    cp video_analyzer/prompts/frame_analysis/*.txt test_prompts/frame_analysis/

    # Run tests
    python3 test_prompt_loading.py

Expected Results:
- All test cases should pass (✓)
- First line of each prompt should be displayed
- Tests run from different directories should work
- Cleanup happens automatically
"""
import os
from pathlib import Path
import json
from video_analyzer.prompt import PromptLoader

def test_prompt_loading(prompt_dir: str, test_name: str):
    """Test prompt loading with given directory."""
    print(f"\nTesting {test_name}")
    print(f"Prompt directory: {prompt_dir}")
    
    # Sample prompts configuration
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
    
    try:
        loader = PromptLoader(prompt_dir, prompts)
        
        # Test loading by name
        frame_analysis = loader.get_by_name("Frame Analysis")
        print("✓ Successfully loaded Frame Analysis prompt")
        print(f"  First line: {frame_analysis.split(os.linesep)[0]}")
        
        describe = loader.get_by_name("Video Reconstruction")
        print("✓ Successfully loaded Video Reconstruction prompt")
        print(f"  First line: {describe.split(os.linesep)[0]}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    # Get absolute path to test directory
    test_dir = Path(__file__).parent / 'test_prompts'
    
    # Test scenarios
    print("=== Prompt Loading Test Cases ===")
    
    # 1. Default package prompts (empty prompt_dir)
    test_prompt_loading("", "Default Package Prompts")
    
    # 2. Absolute path
    test_prompt_loading(str(test_dir.absolute()), "Absolute Path")
    
    # 3. Relative path from current directory
    test_prompt_loading("test_prompts", "Relative Path (CWD)")
    
    # 4. User home directory path
    home_prompts = Path.home() / "test_prompts"
    if not home_prompts.exists():
        os.system(f"cp -r test_prompts {home_prompts}")
    test_prompt_loading("~/test_prompts", "User Home Path")
    
    # 5. Test from different directory
    print("\nTesting from different directory:")
    original_dir = os.getcwd()
    os.chdir("/tmp")
    test_prompt_loading(str(test_dir.absolute()), "Different Directory (Absolute)")
    test_prompt_loading(str(test_dir.relative_to(original_dir)), "Different Directory (Relative)")
    os.chdir(original_dir)
    
    # Cleanup
    os.system(f"rm -rf {home_prompts}")
    print("\nTests completed!")

if __name__ == "__main__":
    main()
