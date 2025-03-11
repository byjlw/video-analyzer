from typing import List, Dict, Any, Optional
import logging
from .clients.llm_client import LLMClient
from .prompt import PromptLoader
from .frame import Frame
from .audio_processor import AudioTranscript

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, client: LLMClient, model: str, prompt_loader: PromptLoader, temperature: float, user_prompt: str = ""):
        """Initialize the VideoAnalyzer.
        
        Args:
            client: LLM client for making API calls
            model: Name of the model to use
            prompt_loader: Loader for prompt templates
            user_prompt: Optional user question about the video that will be injected into frame analysis
                        and video description prompts using the {prompt} token
        """
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.user_prompt = user_prompt  # Store user's question about the video
        self._load_prompts()
        self.previous_analyses = []
        
    def _format_user_prompt(self) -> str:
        """Format the user's prompt by adding prefix if not empty."""
        if self.user_prompt:
            return f"I want to know {self.user_prompt}"
        return ""
        
    def _load_prompts(self):
        """Load prompts from files."""
        self.frame_prompt = self.prompt_loader.get_by_index(0)  # Frame Analysis prompt
        self.video_prompt = self.prompt_loader.get_by_index(1)  # Video Reconstruction prompt

    def _format_previous_analyses(self) -> str:
        """Format previous frame analyses for inclusion in prompt."""
        if not self.previous_analyses:
            return ""
            
        formatted_analyses = []
        for i, analysis in enumerate(self.previous_analyses):
            formatted_analysis = (
                f"Frame {i}\n"
                f"{analysis.get('response', 'No analysis available')}\n"
            )
            formatted_analyses.append(formatted_analysis)
            
        return "\n".join(formatted_analyses)

    def analyze_frame(self, frame: Frame) -> Dict[str, Any]:
        """Analyze a single frame using the LLM."""
        # Replace {PREVIOUS_FRAMES} token with formatted previous analyses
        # Replace tokens in the prompt template
        prompt = self.frame_prompt.replace("{PREVIOUS_FRAMES}", self._format_previous_analyses())
        prompt = prompt.replace("{prompt}", self._format_user_prompt())
        prompt = f"{prompt}\nThis is frame {frame.number} captured at {frame.timestamp:.2f} seconds."
        
        try:
            response = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                model=self.model,
                temperature=self.temperature,
                num_predict=300
            )
            logger.debug(f"Successfully analyzed frame {frame.number}")
            
            # Store the analysis for future frames
            analysis_result = {k: v for k, v in response.items() if k != "context"}
            self.previous_analyses.append(analysis_result)
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error analyzing frame {frame.number}: {e}")
            error_result = {"response": f"Error analyzing frame {frame.number}: {str(e)}"}
            self.previous_analyses.append(error_result)
            return error_result

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
        
        # Get first frame analysis
        first_frame_text = ""
        if frame_analyses and len(frame_analyses) > 0:
            first_frame_text = frame_analyses[0].get('response', '')
        
        # Include transcript information if available
        transcript_text = ""
        if transcript and transcript.text.strip():
            transcript_text = transcript.text
        
        # Replace tokens in the prompt template
        prompt = self.video_prompt.replace("{prompt}", self._format_user_prompt())
        prompt = prompt.replace("{FRAME_NOTES}", analysis_text)
        prompt = prompt.replace("{FIRST_FRAME}", first_frame_text)
        prompt = prompt.replace("{TRANSCRIPT}", transcript_text)
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                num_predict=1000
            )
            logger.info("Successfully reconstructed video description")
            return {k: v for k, v in response.items() if k != "context"}
        except Exception as e:
            logger.error(f"Error reconstructing video: {e}")
            return {"response": f"Error reconstructing video: {str(e)}"}
