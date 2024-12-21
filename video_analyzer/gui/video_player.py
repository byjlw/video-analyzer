from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create video widget
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)
        
        # Create media player
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_widget)
        
        # Create controls
        controls = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)
        
        # Position Slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        controls.addWidget(self.position_slider)
        
        # Time Label
        self.time_label = QLabel("00:00 / 00:00")
        controls.addWidget(self.time_label)
        
        layout.addLayout(controls)
        
        # Connect signals
        self.player.playbackStateChanged.connect(self.update_play_button)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.errorOccurred.connect(self.handle_error)
        
    def set_video(self, file_path):
        if not file_path:
            return
            
        url = QUrl.fromLocalFile(file_path)
        self.player.setSource(url)
        self.play_button.setText("Play")
        self.player.stop()
        
    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()
            
    def update_play_button(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")
            
    def position_changed(self, position):
        self.position_slider.setValue(position)
        self.update_time_label()
        
    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)
        self.update_time_label()
        
    def set_position(self, position):
        self.player.setPosition(position)
        
    def format_time(self, ms):
        s = ms // 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
        
    def update_time_label(self):
        position = self.player.position()
        duration = self.player.duration()
        if duration > 0:
            self.time_label.setText(
                f"{self.format_time(position)} / {self.format_time(duration)}"
            )
        else:
            self.time_label.setText("00:00 / 00:00")
            
    def handle_error(self, error, error_string):
        print(f"Media Player Error: {error} - {error_string}")
