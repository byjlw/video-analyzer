from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import cv2

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create preview label
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.preview_label)
        
    def set_video(self, file_path):
        if not file_path:
            return
            
        try:
            # Open video and get first frame
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame_rgb.shape
                
                # Create QPixmap from frame
                from PyQt6.QtGui import QImage
                image = QImage(frame_rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                
                # Scale pixmap to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(self.preview_label.size(), 
                                             Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
                
                self.preview_label.setPixmap(scaled_pixmap)
            
            cap.release()
            
        except Exception as e:
            print(f"Error loading video preview: {e}")
            self.preview_label.clear()
            self.preview_label.setText("Error loading video preview")
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update preview size if there's a pixmap
        if not self.preview_label.pixmap().isNull():
            scaled_pixmap = self.preview_label.pixmap().scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)