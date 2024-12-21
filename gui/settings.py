import json
import os
from pathlib import Path

class Settings:
    def __init__(self):
        self.config_dir = Path.home() / '.video-analyzer'
        self.config_file = self.config_dir / 'gui_settings.json'
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save(self, config):
        # Remove video_path before saving
        config = config.copy()
        config.pop('video_path', None)
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load(self):
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        return None