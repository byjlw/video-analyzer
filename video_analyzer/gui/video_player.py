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
        
        # Create preview label with minimum size
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black;")
        self.preview_label.setMinimumSize(480, 270)  # 16:9 ratio
        self.preview_label.setText("No video loaded")
        layout.addWidget(self.preview_label)
        
    def set_video(self, file_path):
        print(f"Loading video preview: {file_path}")
        if not file_path:
            return
            
        try:
            # Open video and get first frame
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            print(f"Frame read success: {ret}")
            
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame_rgb.shape
                print(f"Frame size: {width}x{height}")
                
                # Create QPixmap from frame
                from PyQt6.QtGui import QImage
                image = QImage(frame_rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                
                # Get preview label size
                label_size = self.preview_label.size()
                print(f"Label size: {label_size.width()}x{label_size.height()}")
                
                # Scale pixmap to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    label_size.width(),
                    label_size.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                print(f"Scaled pixmap size: {scaled_pixmap.width()}x{scaled_pixmap.height()}")
                self.preview_label.setPixmap(scaled_pixmap)
            
            cap.release()
            print("Video preview loaded successfully")
            
        except Exception as e:
            print(f"Error loading video preview: {e}")
            self.preview_label.clear()
            self.preview_label.setText("Error loading video preview")
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Get the current pixmap
        pixmap = self.preview_label.pixmap()
        if pixmap and not pixmap.isNull():
            # Scale to new size while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)