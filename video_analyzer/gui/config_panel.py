from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout,
                             QLineEdit, QSpinBox, QDoubleSpinBox, 
                             QCheckBox, QPushButton, QFileDialog, QGroupBox,
                             QComboBox)
from PyQt6.QtCore import Qt
from .settings import Settings

class ConfigPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Video Input Group
        video_group = QGroupBox("Input/Output")
        video_layout = QFormLayout()
        
        self.video_path = QLineEdit()
        browse_button = QPushButton("Browse Video")
        browse_button.clicked.connect(self.browse_video)
        video_layout.addRow("Video File:", self.video_path)
        video_layout.addRow("", browse_button)

        self.output_dir = QLineEdit()
        browse_output = QPushButton("Browse Output")
        browse_output.clicked.connect(self.browse_output)
        video_layout.addRow("Output Directory:", self.output_dir)
        video_layout.addRow("", browse_output)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Client Configuration Group
        client_group = QGroupBox("Client Configuration")
        client_layout = QFormLayout()
        
        self.client_type = QComboBox()
        self.client_type.addItems(["ollama", "openrouter"])
        client_layout.addRow("Client Type:", self.client_type)
        
        self.ollama_url = QLineEdit()
        self.ollama_url.setPlaceholderText("http://localhost:11434")
        client_layout.addRow("Ollama URL:", self.ollama_url)
        
        self.openrouter_key = QLineEdit()
        self.openrouter_key.setPlaceholderText("or-xxxxxxxxxxxx")
        client_layout.addRow("OpenRouter Key:", self.openrouter_key)

        client_group.setLayout(client_layout)
        layout.addWidget(client_group)
        
        # Analysis Options Group
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QFormLayout()
        
        self.model = QLineEdit()
        self.model.setPlaceholderText("llava")
        analysis_layout.addRow("Model:", self.model)
        
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0, 3600)
        self.duration.setValue(0)
        self.duration.setSingleStep(1)
        analysis_layout.addRow("Duration (0 for full):", self.duration)
        
        self.max_frames = QSpinBox()
        self.max_frames.setRange(1, 10000)
        self.max_frames.setValue(100)
        analysis_layout.addRow("Max Frames:", self.max_frames)

        self.whisper_model = QComboBox()
        self.whisper_model.addItems(["tiny", "base", "small", "medium", "large"])
        self.whisper_model.setCurrentText("medium")
        analysis_layout.addRow("Whisper Model:", self.whisper_model)
        
        self.keep_frames = QCheckBox()
        analysis_layout.addRow("Keep Frames:", self.keep_frames)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Prompt Group
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QFormLayout()

        self.prompt = QLineEdit()
        prompt_layout.addRow("Custom Prompt:", self.prompt)

        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        layout.addStretch()

    def browse_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)"
        )
        if file_name:
            self.video_path.setText(file_name)
            # Also set video in player
            self.parent().video_player.set_video(file_name)

    def browse_output(self):
        dir_name = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        if dir_name:
            self.output_dir.setText(dir_name)

    def get_config(self):
        config = {
            'video_path': self.video_path.text(),
            'output_dir': self.output_dir.text(),
            'client': self.client_type.currentText(),
            'ollama_url': self.ollama_url.text(),
            'openrouter_key': self.openrouter_key.text(),
            'model': self.model.text(),
            'duration': self.duration.value(),
            'max_frames': self.max_frames.value(),
            'whisper_model': self.whisper_model.currentText(),
            'keep_frames': self.keep_frames.isChecked(),
            'prompt': self.prompt.text()
        }
        self.settings.save(config)
        return config

    def load_settings(self):
        config = self.settings.load()
        if config:
            self.output_dir.setText(config.get('output_dir', ''))
            self.client_type.setCurrentText(config.get('client', 'ollama'))
            self.ollama_url.setText(config.get('ollama_url', ''))
            self.openrouter_key.setText(config.get('openrouter_key', ''))
            self.model.setText(config.get('model', ''))
            self.duration.setValue(config.get('duration', 0))
            self.max_frames.setValue(config.get('max_frames', 100))
            self.whisper_model.setCurrentText(config.get('whisper_model', 'medium'))
            self.keep_frames.setChecked(config.get('keep_frames', False))
            self.prompt.setText(config.get('prompt', ''))