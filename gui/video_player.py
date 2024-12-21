import vlc
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
from PyQt6.QtCore import Qt, QTimer

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        
        # Create VLC instance and media player
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        
        # Create video widget
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
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider)
        
        layout.addLayout(controls_layout)
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)
        
        self.is_playing = False

    def set_video(self, video_path):
        if not video_path:
            return
            
        # Set the video media
        media = self.instance.media_new(video_path)
        self.player.set_media(media)
        
        # Set the video widget as the rendering surface
        if hasattr(self.player, 'set_hwnd'):  # Windows
            self.player.set_hwnd(int(self.video_frame.winId()))
        elif hasattr(self.player, 'set_nsobject'):  # macOS
            self.player.set_nsobject(int(self.video_frame.winId()))
        else:  # Linux
            self.player.set_xwindow(int(self.video_frame.winId()))

    def toggle_play(self):
        if self.is_playing:
            self.player.pause()
            self.play_button.setText("Play")
        else:
            self.player.play()
            self.play_button.setText("Pause")
        self.is_playing = not self.is_playing

    def set_position(self, position):
        self.player.set_position(position / 1000.0)

    def update_ui(self):
        # Update slider position
        media_pos = int(self.player.get_position() * 1000)
        self.position_slider.setValue(media_pos)
        
        # Update button text if playback ends
        if not self.player.is_playing() and self.is_playing:
            self.is_playing = False
            self.play_button.setText("Play")