import sys
import time
import os
import cv2
import numpy as np
from dnx64 import DNX64
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QAction, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# Global variables
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 960, 30
DNX64_PATH = "D:\\dnx64_python\\dnx64_python\\DNX64.dll"
DEVICE_INDEX = 0
CAM_INDEX = 1
QUERY_TIME = 1
COMMAND_TIME = 1
last_captured_image = None  # Stores the last captured image filename
def threaded(func):
    """Decorator to run functions in a separate thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
    return wrapper

# --- Original Helper Functions ---
def clear_line(n=1):
    LINE_CLEAR = "\x1b[2K"
    for i in range(n):
        print("", end=LINE_CLEAR)

def custom_microtouch_function():
    """Executes when MicroTouch press event got detected"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clear_line(1)
    print(f"{timestamp} MicroTouch press detected!", end="\r")

def init_microscope(microscope):
    """Initialize the microscope (same as original code)."""
    microscope.SetVideoDeviceIndex(DEVICE_INDEX)
    time.sleep(0.1)
    microscope.EnableMicroTouch(True)
    time.sleep(0.1)
    microscope.SetEventCallback(custom_microtouch_function)
    time.sleep(0.1)
    return microscope


# --- PyQt5 GUI Implementation ---
class MicroscopeApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Microscope Viewer")
        self.setGeometry(100, 100, 2 * WINDOW_WIDTH + 100, WINDOW_HEIGHT + 150)

        # Create central widget and layouts
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.button_layout = QVBoxLayout()

        # Live feed label
        self.live_feed_label = QLabel(self)
        self.live_feed_label.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.live_feed_label.setStyleSheet("border: 2px solid black; background-color: #222;")
        self.image_layout.addWidget(self.live_feed_label)

        # Last captured image label
        self.captured_image_label = QLabel(self)
        self.captured_image_label.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.captured_image_label.setStyleSheet("border: 2px solid black; background-color: #444;")
        self.image_layout.addWidget(self.captured_image_label)

        # Button to capture image
        self.capture_button = QPushButton("Capture Image", self)
        self.capture_button.setFixedSize(200, 50)
        self.capture_button.setStyleSheet("font-size: 16px; background-color: #0078D7; color: white;")
        self.capture_button.clicked.connect(self.capture_image)
        self.button_layout.addWidget(self.capture_button)

        # Add layouts
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.button_layout)
        self.central_widget.setLayout(self.main_layout)

        # Timer for real-time video feed updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Initialize microscope and camera
        self.microscope = DNX64(DNX64_PATH)
        self.microscope = init_microscope(self.microscope)
        self.camera = self.initialize_camera()

        # Setup menu bar with help instructions
        self.create_menu_bar()

        # Start the video feed
        self.timer.start(10)  # Faster refresh for lower latency

    def create_menu_bar(self):
        """Create menu bar with Help option."""
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")

        help_action = QAction("Keyboard Shortcuts", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

    def show_help(self):
        """Display a message box with keyboard shortcuts."""
        msg = QMessageBox()
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setText(
            "0: LED Off\n"
            "1: Show AMR Magnification\n"
            "2: Flash LEDs\n"
            "c: List Configuration\n"
            "d: Show Device ID\n"
            "f: Show FOV\n"
            "r: Record Video / Stop Recording\n"
            "Esc: Quit"
        )
        msg.exec_()

    def initialize_camera(self):
        """Setup OpenCV camera."""
        camera = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return camera

    def update_frame(self):
        """Update the live feed with low latency."""
        ret, frame = self.camera.read()
        if ret:
            frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.live_feed_label.setPixmap(QPixmap.fromImage(q_img))

    def capture_image(self):
        """Capture image and update the GUI."""
        global last_captured_image
        ret, frame = self.camera.read()
        if ret:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
            cv2.imwrite(filename, frame)
            last_captured_image = filename
            clear_line(1)
            print(f"Saved image to {filename}", end="\r")
            self.update_captured_image()

    def update_captured_image(self):
        """Load and display last captured image."""
        if last_captured_image and os.path.exists(last_captured_image):
            image = cv2.imread(last_captured_image)
            image_resized = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.captured_image_label.setPixmap(QPixmap.fromImage(q_img))

    def print_amr(self):
        """Print AMR magnification to the terminal."""
        self.microscope = init_microscope(self.microscope)  # Refresh microscope
        config = self.microscope.GetConfig(DEVICE_INDEX)
        if (config & 0x40) == 0x40:
            amr = self.microscope.GetAMR(DEVICE_INDEX)
            amr = round(amr, 1)
            clear_line(1)
            print(f"{amr}x", end="\r")
            time.sleep(0.001)
        else:
            clear_line(1)
            print("It does not belong to the AMR series.", end="\r")
            time.sleep(0.001)
    def print_fov_mm(self):
        amr = self.microscope.GetAMR(DEVICE_INDEX)
        fov = self.microscope.FOVx(DEVICE_INDEX, amr)
        amr = round(amr, 1)
        fov = round(fov / 1000, 2)
        if fov == math.inf:
            fov = round(microscope.FOVx(DEVICE_INDEX, 50.0) / 1000.0, 2)
            clear_line(1)
            print("50x fov: ", fov, "mm", end="\r")
        else:
            clear_line(1)
            print(f"{amr}x fov: ", fov, "mm", end="\r")
        time.sleep(QUERY_TIME)
    @threaded
    def flash_leds(self):
        self.microscope.SetLEDState(0, 0)
        time.sleep(COMMAND_TIME)
        self.microscope.SetLEDState(0, 1)
        time.sleep(COMMAND_TIME)
        clear_line(1)
        print("flash_leds", end="\r")

    @threaded
    def led_off(self):
        self.microscope.SetLEDState(0, 0)
        time.sleep(COMMAND_TIME)
        clear_line(1)
        print("led off", end="\r")
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()

        if key == ord("0"):
            self.led_off()
        elif key == ord("1"):
            self.print_amr()
        elif key == ord("2"):
            self.flash_leds()
        elif key == ord("c"):
            print("List Configuration")
        elif key == ord("d"):
            print("Show Device ID")
        elif key == ord("f"):
            print_fov_mm(microscope)
        elif key == ord("r"):
            print("Record Video / Stop Recording")
        elif key == 27:  # ESC key
            self.close()

    def closeEvent(self, event):
        """Handle window close event."""
        self.camera.release()
        event.accept()


def run_gui():
    """Launch the PyQt5 GUI."""
    app = QApplication(sys.argv)
    window = MicroscopeApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()