from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtCore import QTimer, Qt, QPoint
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

        self.start_camera()

    def initUI(self):
        self.setWindowTitle("Dino-Lite Microscope Control")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet("background-color: black; color: neon;")

        # Main layout
        main_layout = QVBoxLayout()
        top_button_layout = QHBoxLayout()
        content_layout = QHBoxLayout()

        # Buttons
        self.btn_calibrate = QPushButton("Calibrate", self)
        self.btn_capture = QPushButton("Image Capture", self)
        self.btn_open3d = QPushButton("Open 3D Viz", self)
        self.btn_parameters = QPushButton("Parameters", self)

        # Connect buttons to their respective methods
        self.btn_calibrate.clicked.connect(self.calibrate)
        self.btn_capture.clicked.connect(self.capture_images)
        self.btn_open3d.clicked.connect(self.open_3d_viz)
        self.btn_parameters.clicked.connect(self.show_parameters)

        for btn in [self.btn_calibrate, self.btn_capture, self.btn_open3d, self.btn_parameters]:
            btn.setStyleSheet("background-color: #222; color: #0f0; font-size: 14px;")
            top_button_layout.addWidget(btn)

        # Add headings
        self.video_label_heading = QLabel("Live Video Feed")
        self.video_label_heading.setStyleSheet("color: #0f0; font-size: 16px;")
        self.last_captured_label_heading = QLabel("Last Captured Image")
        self.last_captured_label_heading.setStyleSheet("color: #0f0; font-size: 16px;")

        # Video feed section
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 150)
        self.video_label.setStyleSheet("border: 1px solid #0f0;")

        # Last captured image section
        self.last_captured_label = QLabel(self)
        self.last_captured_label.setFixedSize(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 150)
        self.last_captured_label.setStyleSheet("border: 1px solid #0f0;")

        content_layout.addWidget(self.video_label)
        content_layout.addWidget(self.last_captured_label)

        main_layout.addLayout(top_button_layout)
        main_layout.addWidget(self.video_label_heading)
        main_layout.addWidget(self.last_captured_label_heading)
        main_layout.addLayout(content_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

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

    def calibrate(self):
        self.statusBar().showMessage("Calibrate button pressed (no functionality yet).")

    def open_3d_viz(self):
        self.statusBar().showMessage("Open 3D Viz button pressed (opens placeholder window).")

    def show_parameters(self):
        self.statusBar().showMessage("Parameters button pressed (opens placeholder window).")

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
