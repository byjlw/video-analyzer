import requests
import json
import logging
from typing import Optional, Dict, Any
from .llm_client import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self._logger = logging.getLogger(__name__)

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256) -> Dict[Any, Any]:
        try:
            # Some models (e.g., qwen3-vl) behave better with chat API
            if str(model).lower().startswith("qwen"):
                return self._chat_request(prompt, image_path, stream, model, temperature, num_predict)
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
            
            # Try the generate endpoint first
            response = requests.post(self.generate_url, json=data)
            if response.status_code in (404, 405):
                # Fallback to chat if generate endpoint is not available/allowed
                return self._chat_request(prompt, image_path, stream, model, temperature, num_predict)
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                try:
                    j = response.json()
                except Exception:
                    j = None
                # If we sent an image but got an empty/absent response, try chat fallback
                if image_path and isinstance(j, dict) and not str(j.get("response", "")).strip():
                    return self._chat_request(prompt, image_path, stream, model, temperature, num_predict)
                return j if j is not None else {"response": response.text}
                
        except requests.exceptions.RequestException as e:
            # If generate failed due to route/method, retry using chat
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status in (400, 404, 405):
                return self._chat_request(prompt, image_path, stream, model, temperature, num_predict)
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")
            
    def _chat_request(self,
        prompt: str,
        image_path: Optional[str],
        stream: bool,
        model: str,
        temperature: float,
        num_predict: int) -> Dict[Any, Any]:
        # Build chat-style payload. Attach image both top-level and on the user message for compatibility.
        message: Dict[str, Any] = {"role": "user", "content": prompt}
        payload: Dict[str, Any] = {
            "model": str(model) if model is not None else "",
            "messages": [message],
            "stream": bool(stream)
        }
        if image_path:
            b64 = self.encode_image(image_path)
            payload["images"] = [b64]
            message["images"] = [b64]
            self._logger.debug(f"Chat request includes image bytes: {len(b64)} base64 chars from {image_path}")
        headers = {"Content-Type": "application/json"}
        resp = requests.post(self.chat_url, json=payload, headers=headers)
        if not resp.ok:
            # Surface server error text to caller for easier debugging
            try:
                detail = resp.text
            except Exception:
                detail = ""
            raise Exception(f"Chat API request failed: {resp.status_code} {detail}")
        if stream:
            return self._handle_streaming_response(resp)
        # Non-streaming chat responses return {"message": {"content": "..."}}
        try:
            j = resp.json()
        except Exception:
            return {"response": resp.text}
        if isinstance(j, dict) and "message" in j and isinstance(j["message"], dict) and "content" in j["message"]:
            return {"response": j["message"]["content"]}
        return j
            
    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    # generate stream frames: {"response":"..."}
                    if 'response' in json_response:
                        accumulated_response += json_response['response']
                    # chat stream frames: {"message":{"content":"..."}}
                    elif 'message' in json_response and isinstance(json_response['message'], dict):
                        content = json_response['message'].get('content')
                        if isinstance(content, str):
                            accumulated_response += content
                except json.JSONDecodeError:
                    continue
                    
        return {"response": accumulated_response}
