from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QComboBox
from PyQt5.QtCore import QTimer, Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
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
        self.current_tool = 'Line'  # Default tool is line

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

        # Connect buttons to their respective methods
        self.btn_capture.clicked.connect(self.capture_images)

        for btn in [self.btn_capture]:
            btn.setStyleSheet("background-color: #333; color: #A9A9A9; font-size: 14px; border-radius: 5px; padding: 8px;")
            top_button_layout.addWidget(btn)

        # Tool selection for drawing
        self.tool_selector = QComboBox(self)
        self.tool_selector.addItems(['Line', 'Rectangle', 'Circle'])
        self.tool_selector.currentTextChanged.connect(self.change_tool)
        self.tool_selector.setStyleSheet("background-color: #333; color: #A9A9A9; padding: 8px;")
        top_button_layout.addWidget(self.tool_selector)

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
        
        # Set a bold red line for drawing
        pen = QPen(QColor(255, 0, 0), 5, Qt.SolidLine)  # Bold red color, thicker line
        painter.setPen(pen)

        for shape in self.shapes:
            if shape["type"] == 'Line':
                painter.drawLine(shape["start"], shape["end"])
            elif shape["type"] == 'Rectangle':
                rect = QRect(shape["start"], shape["end"])
                painter.drawRect(rect)
            elif shape["type"] == 'Circle':
                radius = int(shape["start"].manhattanLength())
                painter.drawEllipse(shape["start"], radius, radius)

        painter.end()
        self.last_captured_label.setPixmap(pixmap)

    def change_tool(self, tool):
        """Change the current drawing tool."""
        self.current_tool = tool
        print(f"Changed tool to {tool}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.last_captured_label.underMouse():
            self.drawing = True
            self.last_mouse_pos = event.pos()
            print("Mouse pressed on captured image...")

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_captured_label.underMouse():
            start = self.last_mouse_pos
            end = event.pos()

            # Draw depending on the current tool
            self.shapes.append({"start": start, "end": end, "type": self.current_tool})
            self.last_mouse_pos = end
            self.update_last_captured_image()
            print("Mouse moved...")

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            print("Mouse released...")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.shapes.clear()
            self.update_last_captured_image()
            print("Cleared all shapes...")

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
