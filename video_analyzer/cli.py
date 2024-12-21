import sys
import argparse
from pathlib import Path
from .gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication

def main():
    # Check for 'gui' command before any other argument processing
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        app = QApplication([])  # Use empty list to avoid processing other args
        window = MainWindow()
        window.show()
        return app.exec()

    # If not gui, proceed with normal CLI argument parsing
    parser = argparse.ArgumentParser(description='Video Analyzer')
    parser.add_argument('video_path', type=Path, help='Path to the video file')
    # Add your other arguments here
    
    args = parser.parse_args()
    
    # Your existing video analysis code here
    video_path = args.video_path
    # ... rest of your analysis code ...

if __name__ == '__main__':
    main()