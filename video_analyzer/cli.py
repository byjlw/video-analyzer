import argparse
import logging
import sys

from .commands import analyze_video_command, show_prompts_command

# Initialize logger at module level
logger = logging.getLogger(__name__)

def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant."""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)

def main():
    """Main entry point for the CLI application."""
    # Special command handling
    if len(sys.argv) > 1 and sys.argv[1] == "show-prompts":
        # Handle show-prompts command
        parser = argparse.ArgumentParser(description="Show prompt file locations")
        parser.add_argument("--config", type=str, default="config",
                            help="Path to configuration directory")
        args = parser.parse_args(sys.argv[2:])  # Parse remaining args
        return show_prompts_command(args)

    # Standard video analysis command (original functionality)
    parser = argparse.ArgumentParser(description="Analyze video using Vision models")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--config", type=str, default="config",
                        help="Path to configuration directory")
    parser.add_argument("--output", type=str, help="Output directory for analysis results")
    parser.add_argument("--client", type=str, help="Client to use (ollama or openrouter)")
    parser.add_argument("--ollama-url", type=str, help="URL for the Ollama service")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI-compatible service")
    parser.add_argument("--api-url", type=str, help="API URL for OpenAI-compatible API")
    parser.add_argument("--model", type=str, help="Name of the vision model to use")
    parser.add_argument("--duration", type=float, help="Duration in seconds to process")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames after analysis")
    parser.add_argument("--whisper-model", type=str, help="Whisper model size (tiny, base, small, medium, large), or path to local Whisper model snapshot")
    parser.add_argument("--start-stage", type=int, default=1, help="Stage to start processing from (1-3)")
    parser.add_argument("--max-frames", type=int, default=sys.maxsize, help="Maximum number of frames to process")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Question to ask about the video")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    return analyze_video_command(args)

if __name__ == "__main__":
    main()
