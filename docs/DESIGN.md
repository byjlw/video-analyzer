# Video Analyzer Design
![Design](design.png)
## Core Workflow

1. Frame Extraction
   - Uses OpenCV to extract frames from video
   - Calculates frame differences to identify key moments
   - Saves frames as JPEGs for LLM analysis
   - Adaptive sampling based on video length and target frames per minute

   ### Frame Selection Algorithm
   1. Target Frame Calculation
      - Calculates target frames based on video duration and frames_per_minute
      - Respects optional max_frames limit
      - Ensures at least 1 frame and no more than total video frames

   2. Adaptive Sampling
      - Uses sampling interval = total_frames / (target_frames * 2)
      - Reduces processing load while maintaining coverage
      - Samples more frequently than target to ensure enough candidates

   3. Frame Difference Analysis
      - Converts frames to grayscale for efficient comparison
      - Uses OpenCV's absdiff to calculate absolute difference
      - Compares against FRAME_DIFFERENCE_THRESHOLD (default 10.0)
      - Stores frame number, image data, and difference score

   4. Final Selection Process
      - Selects frames with highest difference scores
      - Takes top N frames based on target frame count
      - If max_frames specified, samples evenly across selected frames
      - Ensures most significant changes are captured

   ### Limitations
   - Frames between sampling intervals may be missed
   - Rapid sequences may only have one frame selected
   - High-scoring frames may be excluded if outranked by others
   - Even sampling with max_frames may skip some significant changes

2. Audio Processing
   - Extracts audio using FFmpeg
   - Uses Whisper for transcription
   - Handles poor quality audio by checking confidence scores
   - Segments audio for better context in final analysis

3. Frame Analysis
   - Each frame is analyzed independently using vision LLM
   - Uses frame_analysis.txt prompt to guide LLM analysis
   - Captures timestamp, visual elements, and actions
   - Maintains chronological order for narrative flow

4. Video Reconstruction
   - Combines frame analyses chronologically
   - Integrates audio transcript if available
   - Uses video_reconstruction.txt prompt to create technical description
   - Uses narrate_storyteller.txt to transform into engaging narrative

## LLM Integration

### Base Client (llm_client.py)
```python
class LLMClient:
    def encode_image(self, image_path: str) -> str:
        # Common base64 encoding for all clients
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @abstractmethod
    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256) -> Dict[Any, Any]:
        pass
```

### Client Implementations

1. Ollama (ollama.py)
   - Uses local Ollama API
   - Sends images as base64 in "images" array
   - Returns raw response from Ollama

2. Generic OpenAI API (generic_openai_api.py)
   - Compatible with OpenAI-style APIs (OpenAI, OpenRouter, etc.)
   - Configurable API URL (e.g. OpenRouter: https://openrouter.ai/api/v1, OpenAI: https://api.openai.com/v1)
   - Sends images as content array with type "image_url"
   - Requires API key and service URL
   - Returns standardized response format

## Configuration System

Uses cascade priority:
1. Command line args
2. User config (config.json)
3. Default config (default_config.json)

Key configuration groups:
```json
{
    "clients": {
        "default": "ollama",
        "ollama": {
            "url": "http://localhost:11434",
            "model": "llama3.2-vision"
        },
        "openai_api": {
            "api_key": "",
            "api_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free"
        }
    },
    "frames": {
        "per_minute": 60,
        "analysis_threshold": 10.0,
        "min_difference": 5.0,
        "max_count": 30
    }
}
```

## Prompt System

### Prompt Files

Two key prompts:

1. frame_analysis.txt
   - Analyzes single frame
   - Includes timestamp context
   - Focuses on visual elements and actions
   - Supports user questions through {prompt} token

2. describe.txt
   - Combines frame analyses
   - Uses 1 frame
   - Integrates transcript
   - Creates a description of the video based on all the past frames
   - Supports user questions through {prompt} token

Both prompts support user questions via the --prompt flag. When a question is provided, it is prefixed with "I want to know" and injected into the prompts using the {prompt} token. This allows users to ask specific questions about the video that guide both the frame analysis and final description.

### Prompt Loading System

The prompt loading system supports flexible prompt file locations and custom prompts:

1. Path Resolution:
   - User-specified directory (via config `prompt_dir`):
     * Absolute paths: `/path/to/prompts`
     * User home paths: `~/prompts`
     * Relative paths: Checked against:
       1. Current working directory
       2. Package root directory
   - Package resources (fallback)

2. Development Workflow:
   - Install in dev mode: `pip install -e .`
   - Modify prompt files directly
   - Changes reflect immediately without reinstall
   - Works from any directory

3. Configuration:
```json
{
    "prompt_dir": "/absolute/path/to/prompts",  // Absolute path
    // or "~/prompts"                          // User home directory
    // or "prompts"                            // Relative path
    // or ""                                   // Use package prompts only
    "prompts": [
        {
            "name": "Frame Analysis",
            "path": "frame_analysis/frame_analysis.txt"
        },
        {
            "name": "Video Reconstruction",
            "path": "frame_analysis/describe.txt"
        }
    ]
}
```

The system prioritizes user-specified prompts over package prompts, enabling customization while maintaining reliable fallbacks.

## Sample output
[Sample Output](sample_analysis.json)

## Common Issues & Solutions

1. Frame Analysis Failures
   - Ollama: Check if service is running and model is loaded
   - OpenRouter: Verify API key and check response format
   - Both: Ensure image encoding is correct for each API

2. Memory Usage
   - Adjust frames_per_minute based on video length
   - Clean up frames after analysis
   - Use appropriate Whisper model size

3. Poor Analysis Quality
   - Check frame extraction threshold
   - Verify prompt templates
   - Ensure correct model is being used

## Adding New Features

1. New LLM Provider
   - Inherit from LLMClient
   - Implement correct image format for API
   - Add client config to default_config.json
   - Update create_client() in video_analyzer.py

2. Custom Analysis
   - Add new prompt template
   - Update VideoAnalyzer methods
   - Modify output format in results
