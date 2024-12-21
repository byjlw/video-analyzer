from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout,
                             QLineEdit, QSpinBox, QDoubleSpinBox, 
                             QCheckBox, QPushButton, QFileDialog, QGroupBox)
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
        video_group = QGroupBox("Video Input")
        video_layout = QFormLayout()
        
        self.video_path = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_video)
        video_layout.addRow("Video File:", self.video_path)
        video_layout.addRow("", browse_button)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Frame Extraction Group
        frame_group = QGroupBox("Frame Extraction")
        frame_layout = QFormLayout()
        
        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setRange(0.1, 60.0)
        self.frame_interval.setValue(1.0)
        frame_layout.addRow("Frame Interval (s):", self.frame_interval)
        
        self.max_frames = QSpinBox()
        self.max_frames.setRange(1, 10000)
        self.max_frames.setValue(100)
        frame_layout.addRow("Max Frames:", self.max_frames)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # Audio Transcription Group
        audio_group = QGroupBox("Audio Transcription")
        audio_layout = QFormLayout()
        
        self.enable_transcription = QCheckBox()
        audio_layout.addRow("Enable Transcription:", self.enable_transcription)
        
        self.language = QLineEdit()
        self.language.setPlaceholderText("en")
        audio_layout.addRow("Language:", self.language)
        
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        # Analysis Options Group
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QFormLayout()
        
        self.enable_face_detection = QCheckBox()
        analysis_layout.addRow("Face Detection:", self.enable_face_detection)
        
        self.enable_object_detection = QCheckBox()
        analysis_layout.addRow("Object Detection:", self.enable_object_detection)
        
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.0, 1.0)
        self.confidence_threshold.setValue(0.5)
        self.confidence_threshold.setSingleStep(0.1)
        analysis_layout.addRow("Confidence Threshold:", self.confidence_threshold)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
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

    def get_config(self):
        config = {
            'video_path': self.video_path.text(),
            'frame_interval': self.frame_interval.value(),
            'max_frames': self.max_frames.value(),
            'enable_transcription': self.enable_transcription.isChecked(),
            'language': self.language.text() or 'en',
            'enable_face_detection': self.enable_face_detection.isChecked(),
            'enable_object_detection': self.enable_object_detection.isChecked(),
            'confidence_threshold': self.confidence_threshold.value()
        }
        self.settings.save(config)
        return config

    def load_settings(self):
        config = self.settings.load()
        if config:
            self.frame_interval.setValue(config.get('frame_interval', 1.0))
            self.max_frames.setValue(config.get('max_frames', 100))
            self.enable_transcription.setChecked(config.get('enable_transcription', False))
            self.language.setText(config.get('language', 'en'))
            self.enable_face_detection.setChecked(config.get('enable_face_detection', False))
            self.enable_object_detection.setChecked(config.get('enable_object_detection', False))
            self.confidence_threshold.setValue(config.get('confidence_threshold', 0.5))