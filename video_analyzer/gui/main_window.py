from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt
from .config_panel import ConfigPanel
from .video_player import VideoPlayer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create left panel for configuration
        self.config_panel = ConfigPanel()
        main_layout.addWidget(self.config_panel, stretch=1)
        
        # Create right panel for video player and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add video player
        self.video_player = VideoPlayer()
        right_layout.addWidget(self.video_player, stretch=4)
        
        # Add control buttons
        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze Video")
        self.analyze_button.clicked.connect(self.analyze_video)
        button_layout.addWidget(self.analyze_button)
        
        right_layout.addLayout(button_layout)
        main_layout.addWidget(right_panel, stretch=2)

    def analyze_video(self):
        # Get configuration from config panel
        config = self.config_panel.get_config()
        if not config.get('video_path'):
            return