from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog,
                             QMessageBox, QProgressDialog, QTextEdit, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess
from .config_panel import ConfigPanel
from .video_player import VideoPlayer
from pathlib import Path
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AnalysisWorker(QThread):
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)  # status message
    output = pyqtSignal(str)    # analysis output

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process = None

    def run(self):
        try:
            logger.debug("Starting analysis worker")
            # Build command arguments
            args = ['-m', 'video_analyzer.cli']
            
            # Add all configuration options
            args.append(str(self.config['video_path']))
            
            if self.config.get('output_dir'):
                args.extend(['--output', str(self.config['output_dir'])])
            if self.config.get('client'):
                args.extend(['--client', self.config['client']])
            if self.config.get('ollama_url'):
                args.extend(['--ollama-url', self.config['ollama_url']])
            if self.config.get('openrouter_key'):
                args.extend(['--openrouter-key', self.config['openrouter_key']])
            if self.config.get('model'):
                args.extend(['--model', self.config['model']])
            if self.config.get('duration'):
                args.extend(['--duration', str(self.config['duration'])])
            if self.config.get('whisper_model'):
                args.extend(['--whisper-model', self.config['whisper_model']])
            if self.config.get('max_frames'):
                args.extend(['--max-frames', str(self.config['max_frames'])])
            if self.config.get('keep_frames'):
                args.extend(['--keep-frames'])
            if self.config.get('prompt'):
                args.extend(['--prompt', self.config['prompt']])

            logger.debug(f"Running command: python {' '.join(args)}")
            
            # Create QProcess
            self.process = QProcess()
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            
            # Connect process signals
            self.process.readyReadStandardOutput.connect(self.handle_output)
            self.process.finished.connect(self.handle_finished)
            
            # Start the process
            self.process.start(sys.executable, args)
            
            # Wait for process to finish
            self.process.waitForFinished(-1)  # Wait indefinitely

        except Exception as e:
            logger.exception("Error in analysis worker")
            self.finished.emit(False, str(e))
    
    def handle_output(self):
        output = self.process.readAllStandardOutput().data().decode()
        for line in output.splitlines():
            if line:
                self.output.emit(line)
                if line.startswith("INFO") or line.startswith("WARNING") or line.startswith("ERROR"):
                    self.progress.emit(line)
    
    def handle_finished(self, exit_code, exit_status):
        if exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit:
            self.finished.emit(True, "Analysis completed successfully!")
        else:
            self.finished.emit(False, f"Analysis failed with exit code {exit_code}")

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
        
        # Left panel container (40% width)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add config panel to left panel
        self.config_panel = ConfigPanel()
        self.config_panel.video_selected.connect(self.on_video_selected)
        left_layout.addWidget(self.config_panel)
        
        # Add analyze button
        self.analyze_button = QPushButton("Analyze Video")
        self.analyze_button.clicked.connect(self.analyze_video)
        left_layout.addWidget(self.analyze_button)
        
        splitter.addWidget(left_panel)
        
        # Right panel with video preview and output (60% width)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add video player (40% height)
        self.video_player = VideoPlayer()
        right_layout.addWidget(self.video_player, stretch=2)
        
        # Add output text area (60% height)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        right_layout.addWidget(self.output_text, stretch=3)
        
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (40% left, 60% right)
        splitter.setSizes([400, 600])
        
        # Initialize progress dialog
        self.progress = QProgressDialog("Preparing analysis...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.setAutoReset(True)
        self.progress.setCancelButton(None)
        self.progress.setMinimumWidth(400)
        self.progress.setStyleSheet("QProgressDialog { min-width: 400px; }")

        logger.debug("Main window initialized")

    def on_video_selected(self, file_path):
        logger.debug(f"Video selected: {file_path}")
        self.video_player.set_video(file_path)

    def analyze_video(self):
        config = self.config_panel.get_config()
        logger.debug(f"Starting analysis with config: {config}")
        
        # Validate required fields
        if not config.get('video_path'):
            logger.warning("No video path selected")
            QMessageBox.warning(self, "Error", "Please select a video file.")
            return

        # Clear previous output
        self.output_text.clear()
        self.output_text.append("Starting analysis...")

        # Create worker thread
        self.worker = AnalysisWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.output.connect(self.update_output)
        self.worker.finished.connect(self.analysis_complete)

        # Show progress dialog
        self.progress.setLabelText("Initializing analysis...")
        self.progress.show()

        # Disable analyze button
        self.analyze_button.setEnabled(False)

        logger.debug("Starting analysis worker thread")
        self.worker.start()

    def update_progress(self, message):
        logger.debug(f"Progress update: {message}")
        self.progress.setLabelText(message)
        self.output_text.append(message)

    def update_output(self, text):
        logger.debug(f"Output update: {text}")
        self.output_text.append(text)

    def analysis_complete(self, success, message):
        logger.debug(f"Analysis complete - success: {success}, message: {message}")
        self.progress.close()
        self.analyze_button.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)