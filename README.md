# Video Analyzer

A comprehensive video analysis tool that combines computer vision, audio transcription, and natural language processing to generate detailed descriptions of video content. This tool extracts key frames from videos, transcribes audio content, and produces natural language descriptions of the video's content.

## Features

- Frame extraction and analysis
- Audio transcription
- Face detection
- Object detection
- Graphical User Interface (GUI)
- Command Line Interface (CLI)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/byjlw/video-analyzer.git
cd video-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Graphical User Interface (GUI)

To start the application in GUI mode:

```bash
python main.py --gui
```

The GUI provides an intuitive interface with the following features:

- Video player with playback controls
- Configuration panel for all analysis settings
- Real-time preview of video content
- Persistent settings between sessions

GUI Controls:
1. **Video Input**: Click 'Browse' to select a video file
2. **Frame Extraction**: Set frame interval and maximum frames
3. **Audio Transcription**: Enable/disable transcription and set language
4. **Analysis Options**: Configure face detection, object detection, and confidence thresholds
5. **Video Playback**: Use the player controls to preview the video
6. **Analysis**: Click 'Analyze Video' to start processing

### Command Line Interface

[Original CLI documentation remains here]

## Configuration

Settings can be configured through:
1. Command line arguments
2. GUI interface
3. Configuration files

GUI settings are automatically saved to `~/.video-analyzer/gui_settings.json`

[Rest of original README content]