# Video Analysis using Llama3.2 Vision and OpenAI's Whisper Models locally

A video analysis tool that combines Llama's 11B vision model and Whisper to create a description by taking key frames, feeding them to the vision model to get details. It uses the details from each frame and the transcript, if available, to describe what's happening in the video. 

## Features
- 💻 Can run completely locally - no cloud services or API keys needed
- ☁️  Or, Leverage openrouter's LLM service for speed and scale
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
python3.11 -m venv .venv
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

### OpenRouter Setup (Optional)

If you want to use OpenRouter instead of Ollama:

*Currently you can use llama 3.2 11b vision for free, and the default config uses this version automatically!*

1. Get an API key from [OpenRouter](https://openrouter.ai)
2. Either:
   - Pass it via command line: `--openrouter-key your-api-key`
   - Or add it to config/config.json:
     ```json
     {
       "clients": {
         "default": "openrouter",
         "openrouter": {
           "api_key": "your-api-key"
         }
       }
     }
     ```

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

## Usage

### Basic Usage

Using Ollama (default):
```bash
video-analyzer path/to/video.mp4
```

Using OpenRouter:
```bash
video-analyzer path/to/video.mp4 --openrouter-key your-api-key
```

#### Sample Output
```
Video Summary**\n\nDuration: 5 minutes and 67 seconds\n\nThe video begins with a person with long blonde hair, wearing a pink t-shirt and yellow shorts, standing in front of a black plastic tub or container on wheels. The ground appears to be covered in wood chips.\n\nAs the video progresses, the person remains facing away from the camera, looking down at something inside the tub. Their left hand is resting on their hip, while their right arm hangs loosely by their side. There are no new objects or people visible in this frame, but there appears to be some greenery and possibly fruit scattered around the ground behind the person.\n\nThe black plastic tub on wheels is present throughout the video, and the wood chips covering the ground remain consistent with those seen in Frame 0. The person's pink t-shirt matches the color of the shirt worn by the person in Frame 0.\n\nAs the video continues, the person remains stationary, looking down at something inside the tub. There are no significant changes or developments in this frame.\n\nThe key continuation point is to watch for the person to pick up an object from the tub and examine it more closely.\n\n**Key Continuation Points:**\n\n*   The person's pink t-shirt matches the color of the shirt worn by the person in Frame 0.\n*   The black plastic tub on wheels is also present in Frame 0.\n*   The wood chips covering the ground are consistent with those seen in Frame 0.
```

### Advanced Usage

```bash
video-analyzer path/to/video.mp4 \
    --config custom_config.json \
    --output ./custom_output \
    --client openrouter \
    --openrouter-key your-api-key \
    --model llama3.2-vision \
    --frames-per-minute 15 \
    --duration 60 \
    --whisper-model medium \
    --keep-frames
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `video_path` | Path to the input video file | (Required) |
| `--config` | Path to configuration directory | config/ |
| `--output` | Output directory for analysis results | output/ |
| `--client` | Client to use (ollama or openrouter) | ollama |
| `--ollama-url` | URL for the Ollama service | http://localhost:11434 |
| `--openrouter-key` | API key for OpenRouter service | None |
| `--model` | Name of the vision model to use | llama3.2-vision |
| `--frames-per-minute` | Target number of frames to extract | 10 |
| `--duration` | Duration in seconds to process | None (full video) |
| `--whisper-model` | Whisper model size | medium |
| `--keep-frames` | Keep extracted frames after analysis | False |
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |

## Configuration

The tool uses a cascading configuration system:
1. Command line arguments (highest priority)
2. User config (config/config.json)
3. Default config [config/default_config.json](config/default_config.json)

### Configuration Options

#### General Settings
- `clients.default`: Default client to use (ollama or openrouter)
- `clients.ollama.url`: URL for the Ollama service
- `clients.ollama.model`: Vision model to use with Ollama
- `clients.openrouter.api_key`: API key for OpenRouter service
- `clients.openrouter.model`: Vision model to use with OpenRouter
- `prompt_dir`: Directory containing prompt files
- `output_dir`: Directory for output files
- `frames.per_minute`: Target number of frames to extract per minute
- `whisper_model`: Whisper model size (tiny, base, small, medium, large)
- `keep_frames`: Whether to keep extracted frames after analysis

#### Frame Analysis Settings
- `frames.analysis_threshold`: Threshold for key frame detection
- `frames.min_difference`: Minimum difference between frames
- `frames.max_count`: Maximum number of frames to extract

#### Response Length Settings
- `response_length.frame`: Maximum length for frame analysis
- `response_length.reconstruction`: Maximum length for video reconstruction
- `response_length.narrative`: Maximum length for enhanced narrative

#### Audio Settings
- `audio.sample_rate`: Audio sample rate
- `audio.channels`: Number of audio channels
- `audio.quality_threshold`: Minimum quality threshold for transcription
- `audio.chunk_length`: Length of audio chunks for processing
- `audio.language_confidence_threshold`: Confidence threshold for language detection

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

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
