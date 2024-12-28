import requests
import json
import time
from typing import Optional, Dict, Any
from .openai_compatible import OpenAICompatibleClient
import logging

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
RATE_LIMIT_WAIT_TIME = 25 # seconds
DEFAULT_WAIT_TIME = 25  # seconds

class OpenRouterClient(OpenAICompatibleClient):

    def __init__(self, api_key: str, max_retries: int = DEFAULT_MAX_RETRIES):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.generate_url = f"{self.base_url}/chat/completions"
        self.max_retries = max_retries
