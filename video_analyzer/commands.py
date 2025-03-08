import json
import logging
import shutil
from pathlib import Path

from .config import Config, get_client, get_model
from .frame import VideoProcessor
from .prompt import PromptLoader
from .analyzer import VideoAnalyzer
from .audio_processor import AudioProcessor
from .clients.ollama import OllamaClient
from .clients.generic_openai_api import GenericOpenAIAPIClient

def create_client(config: Config):
    """Create the appropriate client based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = get_client(config)
    
    if client_type == "ollama":
        return OllamaClient(client_config["url"])
    elif client_type == "openai_api":
        return GenericOpenAIAPIClient(client_config["api_key"], client_config["api_url"])
    else:
        raise ValueError(f"Unknown client type: {client_type}")

logger = logging.getLogger(__name__)

def show_prompts_command(args):
    """Handle the show-prompts command."""
    # Configure minimal logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    # Load configuration
    config = Config(args.config)
    prompt_loader = PromptLoader(config.get("prompt_dir"), config.get("prompts", []))
    
    # Get and display prompt locations
    prompt_locations = prompt_loader.list_prompt_locations()
    
    print("\nPrompt File Locations:")
    print("=====================")
    
    for prompt in prompt_locations:
        print(f"\n{prompt['name']}:")
        print(f"  {prompt['path']}")
    
    return 0  # Success exit code

def cleanup_files(output_dir: Path):
    """Clean up temporary files and directories."""
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.debug(f"Cleaned up frames directory: {frames_dir}")
            
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.debug(f"Cleaned up audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def analyze_video_command(args):
    """Handle the video analysis command."""
    from .cli import get_log_level  # Import here to avoid circular dependency
    
    # Set up logging with specified level
    log_level = get_log_level(args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    logger.setLevel(log_level)

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
        video_description = None
        
        # Stage 1: Frame and Audio Processing
        if args.start_stage <= 1:
            # Initialize audio processor and extract transcript
            logger.debug("Initializing audio processing...")
            audio_processor = AudioProcessor(language=config.get("audio", {}).get("language", ""), 
                                             model_size_or_path=config.get("audio", {}).get("whisper_model", "medium"),
                                             device=config.get("audio", {}).get("device", "cpu"))
            
            logger.info("Extracting audio from video...")
            audio_path = audio_processor.extract_audio(video_path, output_dir)
            
            if audio_path is None:
                logger.debug("No audio found in video - skipping transcription")
                transcript = None
            else:
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
                duration=config.get("duration"),
                max_frames=args.max_frames
            )
            
        # Stage 2: Frame Analysis
        if args.start_stage <= 2:
            logger.info("Analyzing frames...")
            analyzer = VideoAnalyzer(client, model, prompt_loader, config.get("prompt", ""))
            frame_analyses = []
            for frame in frames:
                analysis = analyzer.analyze_frame(frame)
                frame_analyses.append(analysis)
                
        # Stage 3: Video Reconstruction
        if args.start_stage <= 3:
            logger.info("Reconstructing video description...")
            video_description = analyzer.reconstruct_video(
                frame_analyses, frames, transcript
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
            "video_description": video_description
        }
        
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {output_dir / 'analysis.json'}")
        
        print("\nTranscript:")
        if transcript:
            print(transcript.text)
        else:
            print("No reliable transcript available")
            
        if video_description:
            print("\nVideo Description:")
            print(video_description.get("response", "No description generated"))
        
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
            
    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise
