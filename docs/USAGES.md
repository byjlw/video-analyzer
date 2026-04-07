# Video Analyzer Usage Guide

This guide covers all configuration options and command line arguments for the video analyzer tool, along with practical examples for different use cases.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Command Line Arguments](#command-line-arguments)
- [Configuration System](#configuration-system)
- [Common Use Cases](#common-use-cases)
- [Advanced Examples](#advanced-examples)

## Basic Usage

### Local Analysis with Ollama (Default)
```bash
video-analyzer path/to/video.mp4
```

### Using OpenAI-Compatible API (OpenRouter/OpenAI)
```bash
video-analyzer path/to/video.mp4 --client openai_api --api-key your-key --api-url https://openrouter.ai/api/v1
```

## Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `video_path` | Path to the input video file | (Required) | `video.mp4` |
| `--config` | Path to user configuration JSON file | config/config.json | `--config /path/to/config.json` |
| `--output` | Output directory for analysis results | output/ | `--output ./results/` |
| `--client` | Client to use (ollama or openai_api) | ollama | `--client openai_api` |
| `--ollama-url` | URL for the Ollama service | http://localhost:11434 | `--ollama-url http://localhost:11434` |
| `--api-key` | API key for OpenAI-compatible service | None | `--api-key sk-xxx...` |
| `--api-url` | API URL for OpenAI-compatible API | None | `--api-url https://openrouter.ai/api/v1` |
| `--model` | Name of the vision model to use | llama3.2-vision | `--model gpt-4-vision-preview` |
| `--duration` | Duration in seconds to process | None (full video) | `--duration 60` |
| `--keep-frames` | Keep extracted frames after analysis | False | `--keep-frames` |
| `--whisper-model` | Whisper model size or model path | medium | `--whisper-model large` |
| `--start-stage` | Stage to start processing from (1-3) | 1 | `--start-stage 2` |
| `--max-frames` | Maximum number of frames to process. When specified, frames are sampled evenly across the video duration rather than just taking the first N frames. | sys.maxsize (effectively no limit) | `--max-frames 100` |
| `--log-level` | Set logging level | INFO | `--log-level DEBUG` |
| `--prompt` | Question to ask about the video | "" | `--prompt "What activities are shown?"` |
| `--language` | Set language for transcription | None (auto-detect) | `--language en` |
| `--device` | Select device for Whisper model | cpu | `--device cuda` |
| `--temperature` | Temperature for LLM generation | 0.2 | `--temperature 0.2` |

### Processing Stages
The `--start-stage` argument allows you to begin processing from a specific stage:
1. Frame and Audio Processing
2. Frame Analysis
3. Video Reconstruction

`--start-stage` and `--max-frames` are currently command-line-only controls. They are not read from `config.json` at runtime.

## Configuration System

The tool uses a cascading configuration system with the following priority:
1. Command line arguments (highest priority)
2. User config (`config/config.json` by default, or the file passed to `--config`)
3. Default config (`<package_dir>/config/default_config.json`)

### Configuration File Structure

By default, the CLI loads user configuration from `config/config.json`. Passing `--config` lets you point to a different user configuration JSON file.

```json
{
  "clients": {
    "default": "ollama",
    "temperature": 0.2,
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
  "prompt_dir": "",
  "prompts": [
    {
      "name": "Frame Analysis",
      "path": "frame_analysis/frame_analysis.txt"
    },
    {
      "name": "Video Reconstruction",
      "path": "frame_analysis/describe.txt"
    }
  ],
  "output_dir": "output",
  "frames": {
    "per_minute": 10,
    "analysis_threshold": 10.0,
    "min_difference": 5.0,
    "max_count": 30
  },
  "response_length": {
    "frame": 256,
    "reconstruction": 512,
    "narrative": 1024
  },
  "audio": {
    "whisper_model": "medium",
    "sample_rate": 16000,
    "channels": 1,
    "quality_threshold": 0.5,
    "chunk_length": 30,
    "language_confidence_threshold": 0.5,
    "language": null,
    "device": "cpu"
  },
  "keep_frames": false,
  "prompt": ""
}
```

This JSON block is an example of the configuration shape, not an authoritative list of shipped default values.

### Configuration Options Explained

#### Client Settings
- `clients.default`: Default LLM client (ollama/openai_api)
- `clients.temperature`: Temperature for LLM generation (0.0-1.0, higher values = more creative)
- `clients.ollama.url`: Ollama service URL
- `clients.ollama.model`: Vision model for Ollama
- `clients.openai_api.api_key`: API key for OpenAI-compatible services
- `clients.openai_api.api_url`: API endpoint URL
- `clients.openai_api.model`: Vision model for API service

#### Frame Analysis Settings
- `frames.per_minute`: Target frames to extract per minute
- `frames.analysis_threshold`: Threshold for key frame detection
- `frames.min_difference`: Minimum difference between frames
- `frames.max_count`: Maximum frames to extract

#### Response Length Settings
- `response_length.frame`: Max length for frame analysis
- `response_length.reconstruction`: Max length for video reconstruction
- `response_length.narrative`: Max length for enhanced narrative

#### Audio Processing Settings
- `audio.whisper_model`: Whisper model size or local model path
- `audio.sample_rate`: Audio sample rate in Hz
- `audio.channels`: Number of audio channels
- `audio.quality_threshold`: Minimum quality for transcription
- `audio.chunk_length`: Audio chunk processing length
- `audio.language_confidence_threshold`: Language detection confidence
- `audio.language`: Force specific language (null for auto-detect)
- `audio.device`: Device for Whisper inference (for example `cpu` or `cuda`)

#### General Settings
- `prompt_dir`: Custom prompt directory path
- `prompts`: Prompt definitions loaded by `PromptLoader`
- `output_dir`: Analysis output directory
- `keep_frames`: Retain extracted frames
- `prompt`: Custom analysis prompt

### Current Behavior Notes

- `--config` expects a path to a user configuration JSON file, not a directory.

## Common Use Cases

### Quick Local Analysis
```bash
video-analyzer video.mp4
```

### High-Quality Cloud Analysis with Custom Prompt
```bash
video-analyzer video.mp4 \
    --client openai_api \
    --api-key your-key \
    --api-url https://openrouter.ai/api/v1 \
    --model meta-llama/llama-3.2-11b-vision-instruct:free \
    --whisper-model large \
    --prompt "What activities are happening in this video?"
```

### Resume from Frame Analysis Stage
```bash
video-analyzer video.mp4 \
    --start-stage 2 \
    --max-frames 50 \
    --keep-frames
```

### Analyze Video with Evenly Sampled Frames
```bash
video-analyzer video.mp4 \
    --max-frames 5 \
    --keep-frames
```
This will extract frames evenly spaced across the video duration. For example, in a 5-minute video, it would sample approximately one frame per minute rather than taking the first 5 frames.

### Specific Language Processing
```bash
video-analyzer video.mp4 \
    --language es \
    --whisper-model large
```

### GPU-Accelerated Processing
```bash
video-analyzer video.mp4 \
    --device cuda \
    --whisper-model large
```

## Advanced Examples

### Full Configuration with OpenRouter
```bash
video-analyzer video.mp4 \
    --config /path/to/custom-config.json \
    --output ./analysis_results \
    --client openai_api \
    --api-key your-key \
    --api-url https://openrouter.ai/api/v1 \
    --model meta-llama/llama-3.2-11b-vision-instruct:free \
    --duration 120 \
    --whisper-model large \
    --keep-frames \
    --log-level DEBUG \
    --prompt "Focus on the interactions between people"
```

### Local Processing with Frame Limits
```bash
video-analyzer video.mp4 \
    --client ollama \
    --ollama-url http://localhost:11434 \
    --model llama3.2-vision \
    --max-frames 30 \
    --whisper-model medium \
    --device cuda \
    --language en
```

### Resume Analysis from Specific Stage
```bash
video-analyzer video.mp4 \
    --start-stage 2 \
    --output ./custom_output \
    --keep-frames \
    --max-frames 50 \
    --prompt "Describe the main events"
```

### Using Local Whisper Model
```bash
video-analyzer video.mp4 \
    --whisper-model /path/to/whisper/model \
    --device cuda \
    --start-stage 1
