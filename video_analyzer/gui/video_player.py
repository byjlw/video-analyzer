import os
import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer

try:
    import mpv
except ImportError:
    print("Please install MPV first:")
    print("MacOS: brew install mpv")
    print("Ubuntu/Debian: sudo apt-get install mpv")
    print("Windows: choco install mpv")
    sys.exit(1)

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        
        try:
            # Create MPV player
            self.player = mpv.MPV(wid=str(int(self.winId())))
            # Configure player
            self.player.loop = False
            self.player.keep_open = True
        except Exception as e:
            print(f"Error initializing MPV: {e}")
            print("Please ensure MPV is installed:")
            print("MacOS: brew install mpv")
            print("Ubuntu/Debian: sudo apt-get install mpv")
            print("Windows: choco install mpv")
            sys.exit(1)
        
        # Create video widget layout
        layout = QVBoxLayout(self)
        
        # Create video frame
        self.video_frame = QWidget()
        self.video_frame.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_frame)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider)
        
        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)
        
        layout.addLayout(controls_layout)
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)
        
        self.is_playing = False
        self.duration = 0

    def set_video(self, video_path):
        if not video_path:
            return
        
        try:
            self.player.play(str(video_path))
            self.player.pause = True
            self.is_playing = False
            self.play_button.setText("Play")
            
            # Wait for duration to be available
            while self.player.duration is None:
                pass
            self.duration = self.player.duration
        except Exception as e:
            print(f"Error loading video: {e}")

    def toggle_play(self):
        if self.is_playing:
            self.player.pause = True
            self.play_button.setText("Play")
        else:
            self.player.pause = False
            self.play_button.setText("Pause")
        self.is_playing = not self.is_playing

    def set_position(self, position):
        if self.duration > 0:
            self.player.seek(self.duration * (position / 1000.0), reference="absolute")

    def format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update_ui(self):
        if hasattr(self.player, 'time_pos') and self.player.time_pos is not None:
            # Update slider position
            pos = int((self.player.time_pos / self.duration) * 1000) if self.duration > 0 else 0
            self.position_slider.setValue(pos)
            
            # Update time label
            current = self.format_time(self.player.time_pos)
            total = self.format_time(self.duration)
            self.time_label.setText(f"{current} / {total}")
            
            # Update play state
            if not self.player.pause and not self.is_playing:
                self.is_playing = True
                self.play_button.setText("Pause")
            elif self.player.pause and self.is_playing:
                self.is_playing = False
                self.play_button.setText("Play")