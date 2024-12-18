import argparse
from pathlib import Path
import json
import logging
import shutil
import sys
from typing import Optional
import torch
import torch.backends.mps

from config import Config, get_client, get_model
from frame import VideoProcessor
from prompt import PromptLoader
from analyzer import VideoAnalyzer
from audio_processor import AudioProcessor, AudioTranscript
from clients.ollama import OllamaClient
from clients.openrouter import OpenRouterClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_files(output_dir: Path):
    """Clean up temporary files and directories."""
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.info(f"Cleaned up frames directory: {frames_dir}")
            
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.info(f"Cleaned up audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def create_client(config: Config):
    """Create the appropriate client based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = get_client(config)
    
    if client_type == "ollama":
        return OllamaClient(client_config)
    elif client_type == "openrouter":
        return OpenRouterClient(client_config)
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def main():
    parser = argparse.ArgumentParser(description="Analyze video using Vision models")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--config", type=str, default="config",
                        help="Path to configuration directory")
    parser.add_argument("--output", type=str, help="Output directory for analysis results")
    parser.add_argument("--client", type=str, help="Client to use (ollama or openrouter)")
    parser.add_argument("--ollama-url", type=str, help="URL for the Ollama service")
    parser.add_argument("--openrouter-key", type=str, help="API key for OpenRouter service")
    parser.add_argument("--model", type=str, help="Name of the vision model to use")
    parser.add_argument("--duration", type=float, help="Duration in seconds to process")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames after analysis")
    parser.add_argument("--whisper-model", type=str, help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--start-stage", type=int, default=1, help="Stage to start processing from (1-3)")
    parser.add_argument("--max-frames", type=int, default=sys.maxsize, help="Maximum number of frames to process")
    args = parser.parse_args()

    # Load and update configuration
    config = Config(args.config)
    config.update_from_args(args)

    # Initialize components
    video_path = Path(args.video_path)
    output_dir = Path(config.get("output_dir"))
    client = create_client(config)
    model = get_model(config)
    prompt_loader = PromptLoader(config.get("prompt_dir"), config.get("prompts", []))
    
    try:
        transcript = None
        frames = []
        frame_analyses = []
        technical_description = None
        enhanced_narrative = None
        
        # Stage 1: Frame and Audio Processing
        if args.start_stage <= 1:
            # Initialize audio processor and extract transcript
            logger.info("Initializing audio processing...")
            audio_processor = AudioProcessor(model_size=config.get("audio", {}).get("whisper_model", "medium"))
            
            logger.info("Extracting audio from video...")
            audio_path = audio_processor.extract_audio(video_path, output_dir)
            
            logger.info("Transcribing audio...")
            transcript = audio_processor.transcribe(audio_path)
            if transcript is None:
                logger.warning("Could not generate reliable transcript. Proceeding with video analysis only.")
            
            logger.info(f"Extracting frames from video using model {model}...")
            processor = VideoProcessor(
                video_path, 
                output_dir / "frames", 
                model
            )
            frames = processor.extract_keyframes(
                frames_per_minute=config.get("frames", {}).get("per_minute", 60),
                duration=config.get("duration")
            )
            # Limit frames if max_frames specified
            frames = frames[:args.max_frames]
            
        # Stage 2: Frame Analysis
        if args.start_stage <= 2:
            logger.info("Analyzing frames...")
            analyzer = VideoAnalyzer(client, model, prompt_loader)
            frame_analyses = []
            for frame in frames:
                analysis = analyzer.analyze_frame(frame)
                frame_analyses.append(analysis)
                
            logger.info("Reconstructing video description...")
            technical_description = analyzer.reconstruct_video(
                frame_analyses, frames, transcript
            )

        # Stage 3: Narrative Enhancement
        if args.start_stage <= 3:
            logger.info("Enhancing narrative...")
            enhanced_narrative = analyzer.enhance_narrative(
                technical_description, transcript
            )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "metadata": {
                "client": config.get("clients", {}).get("default"),
                "model": model,
                "whisper_model": config.get("audio", {}).get("whisper_model"),
                "frames_per_minute": config.get("frames", {}).get("per_minute"),
                "duration_processed": config.get("duration"),
                "frames_extracted": len(frames),
                "frames_processed": min(len(frames), args.max_frames),
                "start_stage": args.start_stage,
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None
            } if transcript else None,
            "frame_analyses": frame_analyses,
            "technical_description": technical_description,
            "enhanced_narrative": enhanced_narrative
        }
        
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {output_dir / 'analysis.json'}")
        
        print("\nTranscript:")
        if transcript:
            print(transcript.text)
        else:
            print("No reliable transcript available")
            
        if enhanced_narrative:
            print("\nEnhanced Video Narrative:")
            print(enhanced_narrative.get("response", "No narrative generated"))
        
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
            
    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise

if __name__ == "__main__":
    main()
