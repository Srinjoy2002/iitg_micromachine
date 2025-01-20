from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsOpacityEffect
from PyQt5.QtCore import QTimer, Qt, QPoint, QRect, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
import cv2
import threading
import os
import time

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 960
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 960, 30
DEVICE_INDEX = 1
NUM_CAPTURE_IMAGES = 10
IMAGE_DIR = "images"

# Ensure the images directory exists
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Threaded decorator for non-blocking operations
def threaded(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
    return wrapper

class DinoLiteGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.camera = None
        self.timer = QTimer()
        self.captured_images = []
        self.last_captured_pixmap = QPixmap()
        self.drawing = False
        self.last_mouse_pos = QPoint()
        self.shapes = []  # Stores drawn shapes
        self.is_video_feed = True  # Toggle between video feed and last captured image

        self.start_camera()

    def initUI(self):
        self.setWindowTitle("Dino-Lite Microscope Control")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet("background-color: #1E1E1E; color: #A9A9A9; font-family: Arial, sans-serif;")

        # Main layout
        main_layout = QVBoxLayout()
        top_button_layout = QHBoxLayout()
        content_layout = QHBoxLayout()  # Change this to QHBoxLayout

        # Buttons
        self.btn_capture = QPushButton("Image Capture", self)
        self.btn_toggle_view = QPushButton("Toggle View", self)

        # Connect buttons to their respective methods
        self.btn_capture.clicked.connect(self.capture_images)
        self.btn_toggle_view.clicked.connect(self.toggle_view)

        for btn in [self.btn_capture, self.btn_toggle_view]:
            btn.setStyleSheet("background-color: #333; color: #A9A9A9; font-size: 14px; border-radius: 5px; padding: 8px;")
            top_button_layout.addWidget(btn)

        # Heading for video feed and captured image
        self.video_label_heading = QLabel("Live Video Feed", self)
        self.video_label_heading.setStyleSheet("color: #C1C1C1; font-size: 16px; margin: 10px 0;")
        self.last_captured_label_heading = QLabel("Last Captured Image", self)
        self.last_captured_label_heading.setStyleSheet("color: #C1C1C1; font-size: 16px; margin: 10px 0;")

        # Video feed section
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 150)
        self.video_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #333; border-radius: 5px;")

        # Last captured image section
        self.last_captured_label = QLabel(self)
        self.last_captured_label.setFixedSize(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 150)
        self.last_captured_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #333; border-radius: 5px;")

        # Drawing menu as an overlay
        self.drawing_menu = QWidget(self)
        self.drawing_menu.setFixedSize(WINDOW_WIDTH // 2, 100)
        self.drawing_menu.setStyleSheet("background-color: rgba(0, 0, 0, 0.7); border-radius: 5px; padding: 10px;")
        self.drawing_menu.move(WINDOW_WIDTH // 2, 50)  # Position it on top of the captured image
        self.drawing_menu.setVisible(False)  # Initially hidden

        # Create layouts for each section
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label_heading)
        video_layout.addWidget(self.video_label)

        last_captured_layout = QVBoxLayout()
        last_captured_layout.addWidget(self.last_captured_label_heading)
        last_captured_layout.addWidget(self.last_captured_label)

        # Add the layouts to content_layout side by side
        content_layout.addLayout(video_layout)
        content_layout.addLayout(last_captured_layout)

        main_layout.addLayout(top_button_layout)
        main_layout.addLayout(content_layout)  # Adding content layout below buttons

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Animate button click effect
        self.animate_button(self.btn_capture)

    def animate_button(self, button):
        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(200)
        animation.setStartValue(button.geometry())
        animation.setEndValue(QRect(button.x() - 5, button.y() - 5, button.width() + 10, button.height() + 10))
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()

    def start_camera(self):
        self.camera = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.camera.isOpened():
            self.statusBar().showMessage("Error: Could not open camera.")
            return

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

    @threaded
    def capture_images(self, *args):
        self.captured_images.clear()
        for i in range(NUM_CAPTURE_IMAGES):
            ret, frame = self.camera.read()
            if ret:
                filename = os.path.join(IMAGE_DIR, f"capture_{i + 1}.png")
                self.captured_images.append(frame)
                # Save image as it is (no processing)
                cv2.imwrite(filename, frame)
                time.sleep(1)  # 1-second delay between captures

        if self.captured_images:
            last_frame = self.captured_images[-1]
            rgb_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.last_captured_pixmap = QPixmap.fromImage(qimg)
            self.update_last_captured_image()

    def update_last_captured_image(self):
        pixmap = self.last_captured_pixmap.copy()
        painter = QPainter(pixmap)
        pen = QPen(Qt.white, 2, Qt.SolidLine)
        painter.setPen(pen)

        for shape in self.shapes:
            painter.drawLine(shape["start"], shape["end"])

        painter.end()
        self.last_captured_label.setPixmap(pixmap)

    def toggle_view(self):
        """Toggle between live video feed and captured image."""
        self.is_video_feed = not self.is_video_feed
        if self.is_video_feed:
            self.video_label.setPixmap(self.video_label.pixmap())  # Display live feed
            self.drawing_menu.setVisible(False)  # Hide drawing menu
        else:
            self.last_captured_label.setPixmap(self.last_captured_pixmap)  # Display last captured image
            self.drawing_menu.setVisible(True)  # Show drawing menu

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.last_captured_label.underMouse():
            self.drawing = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_captured_label.underMouse():
            start = self.last_mouse_pos
            end = event.pos()
            self.shapes.append({"start": start, "end": end})
            self.last_mouse_pos = end
            self.update_last_captured_image()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.shapes.clear()
            self.update_last_captured_image()

    def closeEvent(self, event):
        self.timer.stop()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    gui = DinoLiteGUI()
    gui.show()
    app.exec()
