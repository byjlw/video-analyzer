from .config import Config
from .clients.ollama import OllamaClient
from .clients.openrouter import OpenRouterClient

def create_client(config: Config):
    """Create the appropriate client based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = config.get_client()
    
    if client_type == "ollama":
        return OllamaClient(client_config)
    elif client_type == "openrouter":
        return OpenRouterClient(client_config)
    else:
        raise ValueError(f"Unknown client type: {client_type}")