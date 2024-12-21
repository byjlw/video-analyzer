from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog,
                             QMessageBox, QProgressDialog, QTextEdit, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from .config_panel import ConfigPanel
from .video_player import VideoPlayer
from pathlib import Path
import subprocess
import sys

class AnalysisWorker(QThread):
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)  # status message
    output = pyqtSignal(str)    # analysis output

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
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Monitor both stdout and stderr
            while True:
                # Check stdout
                output = process.stdout.readline()
                if output:
                    self.output.emit(output.strip())

                # Check stderr
                error = process.stderr.readline()
                if error:
                    self.progress.emit(error.strip())

                # Check if process has finished
                if output == '' and error == '' and process.poll() is not None:
                    break

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
        main_layout = QVBoxLayout(main_widget)
        
        # Create horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel container
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Add config panel to left panel
        self.config_panel = ConfigPanel()
        self.config_panel.video_selected.connect(self.on_video_selected)
        left_layout.addWidget(self.config_panel)
        
        # Add analyze button
        self.analyze_button = QPushButton("Analyze Video")
        self.analyze_button.clicked.connect(self.analyze_video)
        left_layout.addWidget(self.analyze_button)
        
        splitter.addWidget(left_panel)
        
        # Right panel with video preview and output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add video player
        self.video_player = VideoPlayer()
        right_layout.addWidget(self.video_player)
        
        # Add output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        right_layout.addWidget(self.output_text)
        
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (40% left, 60% right)
        splitter.setSizes([400, 600])

        # Initialize progress dialog
        self.progress = QProgressDialog(self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.setAutoReset(True)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)

    def on_video_selected(self, file_path):
        """Handle video selection from config panel"""
        self.video_player.set_video(file_path)

    def analyze_video(self):
        config = self.config_panel.get_config()
        
        # Validate required fields
        if not config.get('video_path'):
            QMessageBox.warning(self, "Error", "Please select a video file.")
            return

        # Clear previous output
        self.output_text.clear()

        # Create worker thread
        self.worker = AnalysisWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.output.connect(self.update_output)
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

    def update_output(self, text):
        self.output_text.append(text)

    def analysis_complete(self, success, message):
        self.progress.close()
        self.analyze_button.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)