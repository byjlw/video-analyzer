import requests
import json
from typing import Optional, Dict, Any
from .llm_client import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256) -> Dict[Any, Any]:
        try:
            # Build the request data
            data = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict
                }
            }
            
            if image_path:
                # Use encode_image from parent LLMClient class
                data["images"] = [self.encode_image(image_path)]
                    
            response = requests.post(self.generate_url, json=data)
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")
            
    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'response' in json_response:
                        accumulated_response += json_response['response']
                except json.JSONDecodeError:
                    continue
                    
        return {"response": accumulated_response}
