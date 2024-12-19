import requests
import json
import time
import re
from typing import Optional, Dict, Any, Tuple
from .llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_WAIT_TIME = 25  # seconds
MIN_WAIT_TIME = 1  # second
RATE_LIMIT_BUFFER = 1.1  # 10% buffer on rate limit calculations

class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str, max_retries: int = DEFAULT_MAX_RETRIES):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.generate_url = f"{self.base_url}/chat/completions"
        self.max_retries = max_retries

    def _extract_rate_limit_info(self, response: requests.Response) -> Tuple[str, int]:
        """Extract rate limit message and wait time from response.
        
        Args:
            response: The 429 response from the API
            
        Returns:
            Tuple of (error_message, wait_time_in_seconds)
        """
        wait_time = DEFAULT_WAIT_TIME
        error_message = "Rate limit exceeded"
        
        try:
            error_data = response.json()
            if 'metadata' in error_data and 'raw' in error_data['metadata']:
                raw_error = json.loads(error_data['metadata']['raw'])
                if 'error' in raw_error and 'message' in raw_error['error']:
                    error_message = raw_error['error']['message']
                    # Try to extract rate limit from message (e.g., "10 queries per minute")
                    if match := re.search(r'(\d+(?:\.\d+)?)\s*queries per minute', error_message):
                        rate_limit = float(match.group(1))
                        # Calculate wait time with buffer
                        wait_time = int((60 / rate_limit) * RATE_LIMIT_BUFFER)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error parsing rate limit response: {e}")
        
        # Check Retry-After header
        if 'Retry-After' in response.headers:
            try:
                header_wait_time = int(response.headers['Retry-After'])
                wait_time = max(header_wait_time, MIN_WAIT_TIME)
            except (ValueError, TypeError):
                pass
        
        return error_message, wait_time

    def _handle_rate_limit(self, response: requests.Response, retry_count: int) -> None:
        """Handle rate limit response by waiting appropriate time.
        
        Args:
            response: The 429 response from the API
            retry_count: Current retry attempt number
            
        Raises:
            requests.exceptions.HTTPError: If max retries exceeded
        """
        if retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded for rate limit handling")
            response.raise_for_status()
        
        error_message, wait_time = self._extract_rate_limit_info(response)
        logger.warning(
            f"Rate limit hit (attempt {retry_count + 1}/{self.max_retries}). "
            f"Waiting {wait_time} seconds before retry. Details: {error_message}"
        )
        time.sleep(wait_time)

    def _make_request(self,
        data: Dict[str, Any],
        headers: Dict[str, str],
        retry_count: int = 0) -> requests.Response:
        """Make request to OpenRouter API with rate limit handling."""
        try:
            response = requests.post(self.generate_url, headers=headers, json=data)
            
            if response.status_code == 429:
                self._handle_rate_limit(response, retry_count)
                return self._make_request(data, headers, retry_count + 1)
            
            return response
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                logger.warning(f"Request failed (attempt {retry_count + 1}/{self.max_retries}). Retrying...")
                time.sleep(MIN_WAIT_TIME)
                return self._make_request(data, headers, retry_count + 1)
            raise

    def _parse_response(self, response: requests.Response) -> Dict[str, str]:
        """Parse API response and extract content.
        
        Args:
            response: API response to parse
            
        Returns:
            Dict containing response content
            
        Raises:
            Exception: If response parsing fails
        """
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

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256) -> Dict[Any, Any]:
        """Generate response from OpenRouter API."""
        try:
            # Prepare request content
            if image_path:
                base64_image = self.encode_image(image_path)
                content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            else:
                content = prompt

            # Prepare request data
            data = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "stream": stream,
                "temperature": temperature,
                "max_tokens": num_predict
            }

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/byjlw/video-analyzer",
                "X-Title": "Video Analyzer",
                "Content-Type": "application/json"
            }

            # Make request with retry handling
            response = self._make_request(data, headers)
            response.raise_for_status()

            # Handle response
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._parse_response(response)

        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")

    def _handle_streaming_response(self, response: requests.Response) -> Dict[Any, Any]:
        """Handle streaming response from API.
        
        Args:
            response: Streaming response from API
            
        Returns:
            Dict containing accumulated response
        """
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
