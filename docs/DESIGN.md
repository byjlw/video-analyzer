# Video Analyzer Design Document

## Architecture

The Video Analyzer is built with a modular architecture that supports both command-line and graphical interfaces. The system is composed of the following main components:

### Core Components

- Frame Extractor: Handles video frame extraction
- Audio Processor: Manages audio transcription
- Vision Analyzer: Processes frames for object and face detection
- Description Generator: Creates natural language descriptions

### Interface Components

#### Command Line Interface (CLI)
- Provides direct access to all analyzer features
- Supports batch processing and automation
- Configurable through command line arguments and config files

#### Graphical User Interface (GUI)
- Built using PyQt6 for cross-platform compatibility
- Components:
  - MainWindow: Primary application window and layout manager
  - ConfigPanel: User interface for all configuration options
  - VideoPlayer: Integrated video playback using python-vlc
  - Settings: Persistent storage of user preferences

## Configuration Management

The system follows a hierarchical configuration approach:

1. Command Line Arguments (highest priority)
2. GUI Settings (if in GUI mode)
3. User Config File (~/.video-analyzer/config.json)
4. Default Configuration

GUI-specific settings are stored separately in ~/.video-analyzer/gui_settings.json

## Data Flow

1. Video Input
   - CLI: Direct file path input
   - GUI: File selection dialog

2. Configuration
   - CLI: Command line args -> Config file -> Defaults
   - GUI: User Interface -> Saved Settings -> Defaults

3. Processing Pipeline
   - Frame Extraction
   - Audio Transcription
   - Vision Analysis
   - Description Generation

4. Output Handling
   - CLI: Console output and file generation
   - GUI: Interactive display and file saving

## Dependencies

Core Dependencies:
- OpenCV: Video processing
- PyTorch: Machine learning models
- Transformers: NLP processing

GUI Dependencies:
- PyQt6: GUI framework
- python-vlc: Video playback

## Future Considerations

- Real-time analysis mode
- Plugin system for custom analyzers
- Network-based processing
- Multi-language support
- Batch processing in GUI mode
