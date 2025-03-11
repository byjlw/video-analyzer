import argparse
from pathlib import Path
import json
from typing import Any
import logging
import pkg_resources

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_dir: str = "config"):
        # Handle user-provided config directory
        self.config_dir = Path(config_dir)
        self.user_config = self.config_dir / "config.json"
        
        # First try to find default_config.json in the user-provided directory
        self.default_config = self.config_dir / "default_config.json"
        
        # If not found, fallback to package's default config
        if not self.default_config.exists():
            try:
                default_config_path = pkg_resources.resource_filename('video_analyzer', 'config/default_config.json')
                self.default_config = Path(default_config_path)
                logger.debug(f"Using packaged default config from {self.default_config}")
            except Exception as e:
                logger.error(f"Error finding default config: {e}")
                raise
            
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file with cascade:
        1. Try user config (config.json)
        2. Fall back to default config (default_config.json)
        """
        try:
            if self.user_config.exists():
                logger.debug(f"Loading user config from {self.user_config}")
                with open(self.user_config) as f:
                    self.config = json.load(f)
            else:
                logger.debug(f"No user config found, loading default config from {self.default_config}")
                with open(self.default_config) as f:
                    self.config = json.load(f)
                    
            # Ensure prompts is a list
            if not isinstance(self.config.get("prompts", []), list):
                logger.warning("Prompts in config is not a list, setting to empty list")
                self.config["prompts"] = []
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)

    def update_from_args(self, args: argparse.Namespace):
        """Update configuration with command line arguments."""
        for key, value in vars(args).items():
            if value is not None:  # Only update if argument was provided
                if key == "client":
                    self.config["clients"]["default"] = value
                elif key == "ollama_url":
                    self.config["clients"]["ollama"]["url"] = value
                elif key == "api_key":
                    self.config["clients"]["openai_api"]["api_key"] = value
                    # If key is provided but no client specified, use OpenAI API
                    if not args.client:
                        self.config["clients"]["default"] = "openai_api"
                elif key == "api_url":
                    self.config["clients"]["openai_api"]["api_url"] = value
                elif key == "model":
                    client = self.config["clients"]["default"]
                    self.config["clients"][client]["model"] = value
                elif key == "prompt":
                    self.config["prompt"] = value
                #overide audio config
                elif key == "whisper_model":
                    self.config["audio"]["whisper_model"] = value  # default is 'medium'
                elif key == "language":
                    if value is not None:
                        self.config["audio"]["language"] = value
                elif key == "device":
                    self.config["audio"]["device"] = value
                elif key == "temperature":
                    self.config["clients"]["temperature"] = value
                elif key not in ["start_stage", "max_frames"]:  # Ignore these as they're command-line only
                    self.config[key] = value

    def save_user_config(self):
        """Save current configuration to user config file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.user_config, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.debug(f"Saved user config to {self.user_config}")
        except Exception as e:
            logger.error(f"Error saving user config: {e}")
            raise

def get_client(config: Config) -> dict:
    """Get the appropriate client configuration based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    
    if client_type == "ollama":
        return {"url": client_config.get("url", "http://localhost:11434")}
    elif client_type == "openai_api":
        api_key = client_config.get("api_key")
        api_url = client_config.get("api_url")
        if not api_key:
            raise ValueError("API key is required when using OpenAI API client")
        if not api_url:
            raise ValueError("API URL is required when using OpenAI API client")
        return {
            "api_key": api_key,
            "api_url": api_url
        }
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def get_model(config: Config) -> str:
    """Get the appropriate model based on client type and configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get("clients", {}).get(client_type, {})
    return client_config.get("model", "llama3.2-vision")
