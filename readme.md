# Video Analysis using vision models like Llama3.2 Vision and OpenAI's Whisper Models

A video analysis tool that combines vision models like Llama's 11B vision model and Whisper to create a description by taking key frames, feeding them to the vision model to get details. It uses the details from each frame and the transcript, if available, to describe what's happening in the video. 

## Table of Contents
- [Features](#features)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Sample Output](#sample-output)
  - [Complete Usage Guide](docs/USAGES.md)
- [Requirements](#requirements)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Ollama Setup](#ollama-setup)
  - [OpenAI-compatible API Setup](#openai-compatible-api-setup-optional)
- [Design](#design)
  - [Detailed Design Documentation](docs/DESIGN.md)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Output](#output)
- [Uninstallation](#uninstallation)
- [License](#license)
- [Contributing](#contributing)

## Features
- 💻 Can run completely locally - no cloud services or API keys needed
- ☁️  Or, leverage any OpenAI API compatible LLM service (openrouter, openai, etc) for speed and scale
- 🎬 Intelligent key frame extraction from videos
- 🔊 High-quality audio transcription using OpenAI's Whisper
- 👁️ Frame analysis using Ollama and Llama3.2 11B Vision Model
- 📝 Natural language descriptions of video content
- 🔄 Automatic handling of poor quality audio
- 📊 Detailed JSON output of analysis results
- ⚙️ Highly configurable through command line arguments or config file

## Design
The system operates in three stages:

1. Frame Extraction & Audio Processing
   - Uses OpenCV to extract key frames
   - Processes audio using Whisper for transcription
   - Handles poor quality audio with confidence checks

2. Frame Analysis
   - Analyzes each frame using vision LLM
   - Each analysis includes context from previous frames
   - Maintains chronological progression
   - Uses frame_analysis.txt prompt template

3. Video Reconstruction
   - Combines frame analyses chronologically
   - Integrates audio transcript
   - Uses first frame to set the scene
   - Creates comprehensive video description

![Design](docs/design.png)

## Requirements

### System Requirements
- Python 3.11 or higher
- FFmpeg (required for audio processing)
- When running LLMs locally (not necessary when using openrouter)
  - At least 16GB RAM (32GB recommended)
  - GPU at least 12GB of VRAM or Apple M Series with at least 32GB

### Installation

1. Clone the repository:
```bash
git clone https://github.com/byjlw/video-analyzer.git
cd video-analyzer
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install .  # For regular installation
# OR
pip install -e .  # For development installation
```

4. Install FFmpeg:
- Ubuntu/Debian:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```
- macOS:
  ```bash
  brew install ffmpeg
  ```
- Windows:
  ```bash
  choco install ffmpeg
  ```

### Ollama Setup

1. Install Ollama following the instructions at [ollama.ai](https://ollama.ai)

2. Pull the default vision model:
```bash
ollama pull llama3.2-vision
```

3. Start the Ollama service:
```bash
ollama serve
```

### OpenAI-compatible API Setup (Optional)

If you want to use OpenAI-compatible APIs (like OpenRouter or OpenAI) instead of Ollama:

1. Get an API key from your provider:
   - [OpenRouter](https://openrouter.ai)
   - [OpenAI](https://platform.openai.com)

2. Configure via command line:
   ```bash
   # For OpenRouter
   video-analyzer video.mp4 --client openai_api --api-key your-key --api-url https://openrouter.ai/api/v1 --model gpt-4o-mini

   # For OpenAI
   video-analyzer video.mp4 --client openai_api --api-key your-key --api-url https://api.openai.com/v1 --model meta-llama/llama-3.2-11b-vision-instruct
   ```

   Or add to config/config.json:
   ```json
   {
     "clients": {
       "default": "openai_api",
       "openai_api": {
         "api_key": "your-api-key",
         "api_url": "https://openrouter.ai/api/v1"  # or https://api.openai.com/v1
       }
     }
   }
   ```

Note: With OpenRouter, you can use llama 3.2 11b vision for free by adding :free to the model name

## Project Structure

```
video-analyzer/
├── config/
│   └── default_config.json
├── prompts/
│   └── frame_analysis/
│       ├── frame_analysis.txt
│       └── describe.txt
├── output/             # Generated during runtime
├── video_analyzer/     # Package source code
└── setup.py            # Package installation configuration
```

For detailed information about the project's design and implementation, including how to make changes, see [docs/DESIGN.md](docs/DESIGN.md).

## Usage

For detailed usage instructions and all available options, see [docs/USAGES.md](docs/USAGES.md).

### Quick Start

```bash
# Local analysis with Ollama (default)
video-analyzer video.mp4

# Cloud analysis with OpenRouter
video-analyzer video.mp4 \
    --client openai_api \
    --api-key your-key \
    --api-url https://openrouter.ai/api/v1 \
    --model meta-llama/llama-3.2-11b-vision-instruct:free

# Analysis with custom prompt
video-analyzer video.mp4 \
    --prompt "What activities are happening in this video?" \
    --whisper-model large
```

### Sample Output
```
Video Summary**\n\nDuration: 5 minutes and 67 seconds\n\nThe video begins with a person with long blonde hair, wearing a pink t-shirt and yellow shorts, standing in front of a black plastic tub or container on wheels. The ground appears to be covered in wood chips.\n\nAs the video progresses, the person remains facing away from the camera, looking down at something inside the tub. Their left hand is resting on their hip, while their right arm hangs loosely by their side. There are no new objects or people visible in this frame, but there appears to be some greenery and possibly fruit scattered around the ground behind the person.\n\nThe black plastic tub on wheels is present throughout the video, and the wood chips covering the ground remain consistent with those seen in Frame 0. The person's pink t-shirt matches the color of the shirt worn by the person in Frame 0.\n\nAs the video continues, the person remains stationary, looking down at something inside the tub. There are no significant changes or developments in this frame.\n\nThe key continuation point is to watch for the person to pick up an object from the tub and examine it more closely.\n\n**Key Continuation Points:**\n\n*   The person's pink t-shirt matches the color of the shirt worn by the person in Frame 0.\n*   The black plastic tub on wheels is also present in Frame 0.\n*   The wood chips covering the ground are consistent with those seen in Frame 0.
```


## Configuration

The tool uses a cascading configuration system with command line arguments taking highest priority, followed by user config (config/config.json), and finally the default config. See [docs/USAGES.md](docs/USAGES.md) for detailed configuration options.

## Output

The tool generates a JSON file (`analysis.json`) containing:
- Metadata about the analysis
- Audio transcript (if available)
- Frame-by-frame analysis
- Final video description

### Example Output Structure


## Uninstallation

To uninstall the package:
```bash
pip uninstall video-analyzer
```

## License

Apache License

## Contributing

We welcome contributions! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines on how to:
- Review the project design
- Propose changes through GitHub Discussions
- Submit pull requests
