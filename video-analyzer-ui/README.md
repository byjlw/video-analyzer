# Video Analyzer UI

A lightweight web interface for the video-analyzer tool.

## Features
- Simple, intuitive interface for video analysis
- Real-time command output streaming
- Drag-and-drop video upload
- Results visualization and download
- Session management and cleanup

## Prerequisites
- Python 3.8 or higher
- video-analyzer package installed
- FFmpeg (required by video-analyzer)

## Installation

```bash
pip install video-analyzer-ui
```

## Quick Start

1. Start the server:
   ```bash
   video-analyzer-ui
   ```

2. Open in browser:
   http://localhost:5000

## Usage

### Development Mode
```bash
video-analyzer-ui --dev
```
- Auto-reload on code changes
- Debug logging
- Development error pages

### Production Mode
```bash
video-analyzer-ui --host 0.0.0.0 --port 5000
```
- Optimized for performance
- Error logging to file
- Production-ready security

### Command Line Options
- `--dev`: Enable development mode
- `--host`: Bind address (default: localhost)
- `--port`: Port number (default: 5000)
- `--log-file`: Log file path
- `--config`: Custom config file path

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/video-analyzer-ui.git
   cd video-analyzer-ui
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Run development server:
   ```bash
   video-analyzer-ui --dev
   ```

## How It Works

1. Upload Video:
   - Drag and drop or select a video file
   - File is temporarily stored in a session-specific directory

2. Configure Analysis:
   - Required fields are marked with *
   - Optional parameters can be left empty
   - Real-time command preview shows what will be executed

3. Run Analysis:
   - Progress is shown in real-time
   - Output is streamed as it becomes available
   - Results are stored in session directory

4. View Results:
   - Download analysis.json
   - View extracted frames (if kept)
   - Access transcripts and other outputs

## License

Apache License
