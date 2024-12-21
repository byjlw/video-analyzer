from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog,
                             QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from .config_panel import ConfigPanel
from .video_player import VideoPlayer
from pathlib import Path
import subprocess
import sys

class AnalysisWorker(QThread):
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)  # status message

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            # Build command
            cmd = [sys.executable, '-m', 'video_analyzer.cli']
            
            # Add all configuration options
            cmd.extend([str(self.config['video_path'])])
            
            if self.config.get('output_dir'):
                cmd.extend(['--output', str(self.config['output_dir'])])
            if self.config.get('client'):
                cmd.extend(['--client', self.config['client']])
            if self.config.get('ollama_url'):
                cmd.extend(['--ollama-url', self.config['ollama_url']])
            if self.config.get('openrouter_key'):
                cmd.extend(['--openrouter-key', self.config['openrouter_key']])
            if self.config.get('model'):
                cmd.extend(['--model', self.config['model']])
            if self.config.get('duration'):
                cmd.extend(['--duration', str(self.config['duration'])])
            if self.config.get('whisper_model'):
                cmd.extend(['--whisper-model', self.config['whisper_model']])
            if self.config.get('max_frames'):
                cmd.extend(['--max-frames', str(self.config['max_frames'])])
            if self.config.get('keep_frames'):
                cmd.extend(['--keep-frames'])
            if self.config.get('prompt'):
                cmd.extend(['--prompt', self.config['prompt']])

            # Run the analysis
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Monitor output
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.progress.emit(output.strip())

            if process.returncode == 0:
                self.finished.emit(True, "Analysis completed successfully!")
            else:
                self.finished.emit(False, f"Analysis failed with return code {process.returncode}")

        except Exception as e:
            self.finished.emit(False, str(e))

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

        # Initialize progress dialog
        self.progress = QProgressDialog(self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.setAutoReset(True)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)

    def analyze_video(self):
        config = self.config_panel.get_config()
        
        # Validate required fields
        if not config.get('video_path'):
            QMessageBox.warning(self, "Error", "Please select a video file.")
            return

        # Create worker thread
        self.worker = AnalysisWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_complete)

        # Show progress dialog
        self.progress.setLabelText("Initializing analysis...")
        self.progress.setRange(0, 0)  # Indeterminate progress
        self.progress.show()

        # Disable analyze button
        self.analyze_button.setEnabled(False)

        # Start analysis
        self.worker.start()

    def update_progress(self, message):
        self.progress.setLabelText(message)

    def analysis_complete(self, success, message):
        self.progress.close()
        self.analyze_button.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)