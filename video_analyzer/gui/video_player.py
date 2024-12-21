from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create video widget
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)
        
        # Create media player
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_widget)
        
        # Create audio output
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider)
        
        layout.addLayout(controls_layout)
        
        # Connect signals
        self.player.durationChanged.connect(self.duration_changed)
        self.player.positionChanged.connect(self.position_changed)
        self.player.playbackStateChanged.connect(self.state_changed)

    def set_video(self, video_path):
        if not video_path:
            return
            
        self.player.setSource(QUrl.fromLocalFile(str(video_path)))
        self.player.stop()
        self.play_button.setText("Play")

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def set_position(self, position):
        self.player.setPosition(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")