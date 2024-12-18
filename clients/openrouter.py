import requests
import json
from typing import Optional, Dict, Any
from .llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)

class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.generate_url = f"{self.base_url}/chat/completions"

    def _make_request(self,
        data: Dict[str, Any],
        headers: Dict[str, str]) -> requests.Response:
        """Make request to OpenRouter API with rate limit handling."""
        response = requests.post(self.generate_url, headers=headers, json=data)
        
        if response.status_code == 429:
            error_data = response.json()
            # Get detailed error message from provider if available
            provider_message = ""
            try:
                if 'metadata' in error_data and 'raw' in error_data['metadata']:
                    raw_error = json.loads(error_data['metadata']['raw'])
                    if 'error' in raw_error and 'message' in raw_error['error']:
                        provider_message = raw_error['error']['message']
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            
            error_message = provider_message or error_data.get('message', 'Rate limit exceeded')
            logger.warning(f"Rate limit hit: {error_message}")
            
            # Return the error response to be handled by generate()
            return response
            
        return response

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256) -> Dict[Any, Any]:
        try:
            if image_path:
                base64_image = self.encode_image(image_path)
                content = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            else:
                content = prompt

            messages = [{
                "role": "user",
                "content": content
            }]

            data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": num_predict
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/byjlw/video-analyzer",
                "X-Title": "Video Analyzer",
                "Content-Type": "application/json"
            }

            response = self._make_request(data, headers)
            
            if response.status_code == 429:
                # Return error response without retrying
                error_data = response.json()
                error_message = error_data.get('message', 'Rate limit exceeded')
                return {"response": f"Rate limit error: {error_message}"}
            
            response.raise_for_status()

            if stream:
                return self._handle_streaming_response(response)
            else:
                try:
                    json_response = response.json()
                    if 'error' in json_response:
                        raise Exception(f"API error: {json_response['error']}")
                    
                    if 'choices' not in json_response or not json_response['choices']:
                        raise Exception("No choices in response")
                        
                    message = json_response['choices'][0].get('message', {})
                    if not message or 'content' not in message:
                        raise Exception("No content in response message")
                        
                    return {"response": message['content']}
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response: {response.text}")

        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'text'):
                raise Exception(f"API request failed: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")

    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'choices' in json_response and len(json_response['choices']) > 0:
                        delta = json_response['choices'][0].get('delta', {})
                        if 'content' in delta:
                            accumulated_response += delta['content']
                except json.JSONDecodeError:
                    continue

        return {"response": accumulated_response}
