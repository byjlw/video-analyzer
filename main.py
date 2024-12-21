import sys
import argparse
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

def parse_args():
    parser = argparse.ArgumentParser(description='Video Analyzer')
    parser.add_argument('--gui', action='store_true', help='Start in GUI mode')
    # Add existing CLI arguments here
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.gui:
        # Start GUI mode
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    else:
        # Run in CLI mode with existing functionality
        pass

if __name__ == '__main__':
    main()