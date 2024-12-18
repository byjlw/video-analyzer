from typing import List, Dict, Any, Optional
import logging
from clients.llm_client import LLMClient
from prompt import PromptLoader
from frame import Frame
from audio_processor import AudioTranscript

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, client: LLMClient, model: str, prompt_loader: PromptLoader):
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self._load_prompts()
        
    def _load_prompts(self):
        """Load prompts from files."""
        self.frame_prompt = self.prompt_loader.get_by_index(0)  # First prompt
        self.video_prompt = self.prompt_loader.get_by_index(1)  # Second prompt
        self.narrative_prompt = self.prompt_loader.get_by_index(2)  # Third prompt

    def analyze_frame(self, frame: Frame) -> Dict[str, Any]:
        """Analyze a single frame using the LLM."""
        prompt = f"{self.frame_prompt}\nThis is frame {frame.number} captured at {frame.timestamp:.2f} seconds."
        
        try:
            response = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                model=self.model,
                num_predict=300
            )
            logger.info(f"Successfully analyzed frame {frame.number}")
            return {k: v for k, v in response.items() if k != "context"}
        except Exception as e:
            logger.error(f"Error analyzing frame {frame.number}: {e}")
            return {"response": f"Error analyzing frame {frame.number}: {str(e)}"}

    def reconstruct_video(self, frame_analyses: List[Dict[str, Any]], frames: List[Frame], 
                         transcript: Optional[AudioTranscript] = None) -> Dict[str, Any]:
        """Reconstruct video description from frame analyses and transcript."""
        frame_notes = []
        for i, (frame, analysis) in enumerate(zip(frames, frame_analyses)):
            frame_note = (
                f"Frame {i} ({frame.timestamp:.2f}s):\n"
                f"{analysis.get('response', 'No analysis available')}"
            )
            frame_notes.append(frame_note)
        
        analysis_text = "\n\n".join(frame_notes)
        
        # Include transcript information if available and of good quality
        transcript_text = ""
        if transcript and transcript.text.strip():
            transcript_text = (
                f"\nAudio Transcript:\n{transcript.text}\n\n"
                f"Detailed segments:\n"
            )
            for segment in transcript.segments:
                transcript_text += (
                    f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: "
                    f"{segment['text']}\n"
                )
        
        prompt = (f"{self.video_prompt}\n\n"
                 f"Frame Notes:\n{analysis_text}\n\n"
                 f"{transcript_text}")
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                num_predict=1000
            )
            logger.info("Successfully reconstructed video description")
            return {k: v for k, v in response.items() if k != "context"}
        except Exception as e:
            logger.error(f"Error reconstructing video: {e}")
            return {"response": f"Error reconstructing video: {str(e)}"}

    def enhance_narrative(self, technical_description: Dict[str, Any], 
                         transcript: Optional[AudioTranscript] = None) -> Dict[str, Any]:
        """Transform the technical video description into an engaging narrative."""
        tech_desc = technical_description.get('response', '')
        
        # Include transcript context if available
        transcript_context = ""
        if transcript and transcript.text.strip():
            transcript_context = f"\n\nSpoken content context:\n{transcript.text}\n"
        
        prompt = (f"{self.narrative_prompt}\n\n"
                 f"Here's the technical description to transform:\n\n"
                 f"{tech_desc}"
                 f"{transcript_context}")
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                num_predict=500
            )
            logger.info("Successfully enhanced video narrative")
            return {k: v for k, v in response.items() if k != "context"}
        except Exception as e:
            logger.error(f"Error enhancing narrative: {e}")
            return {"response": f"Error enhancing narrative: {str(e)}"}
