
import sys
import cv2
import time
import os
import threading
from tkinter import Tk, Label, Button, Frame, Canvas, BooleanVar, Menu, Toplevel
from PIL import Image, ImageTk  # Pillow library to handle image formats

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1600, 900
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 960, 30
DEVICE_INDEX = 1  # Ensure this points to the Dino-Lite camera
IMAGE_SAVE_DIR = "images"

# Initialize microscope
from dnx64 import DNX64
microscope = DNX64('D:\\dnx64_python\\dnx64_python\\DNX64.dll')

class TechnicalPointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D Point Cloud Reconstruction Tool")
        self.master.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.master.configure(bg="#1A1A1A")

        # Variables for stage and drawing
        self.stage_position = 5.0  # Initial stage position
        self.upper_bound = 8.0
        self.lower_bound = 2.0
        self.manual_mode = BooleanVar(value=False)
        self.drawn_objects = []

        # AMR, FOV, Configurations, LED Status
        self.amr_value = 1.5  # Example AMR value
        self.fov_value = "5x5 mm"  # Example FOV value
        self.config_status = "Loaded"
        self.led_status = "ON"

        # Top Panel for Buttons
        self.top_panel = Frame(master, bg="#262626", height=80, relief="groove", bd=1)
        self.top_panel.pack(side="top", fill="x", pady=5)

        # Buttons
        self.calibrate_button = Button(
            self.top_panel, text="Calibrate", font=("Consolas", 12, "bold"),
            bg="#FF4500", fg="#1A1A1A", relief="flat", command=self.calibrate_action
        )
        self.calibrate_button.pack(side="left", padx=20, pady=10)

        self.capture_button = Button(
            self.top_panel, text="Image Capture", font=("Consolas", 12, "bold"),
            bg="#00BFFF", fg="#1A1A1A", relief="flat", command=self.capture_action
        )
        self.capture_button.pack(side="left", padx=20, pady=10)

        self.focus_stack_button = Button(
            self.top_panel, text="Focus Stack", font=("Consolas", 12, "bold"),
            bg="#FFD700", fg="#1A1A1A", relief="flat", command=self.focus_stack_action
        )
        self.focus_stack_button.pack(side="left", padx=20, pady=10)

        self.parameters_button = Button(
            self.top_panel, text="Parameters", font=("Consolas", 12, "bold"),
            bg="#8A2BE2", fg="#FFFFFF", relief="flat", command=self.show_parameters
        )
        self.parameters_button.pack(side="left", padx=20, pady=10)

        # Video Feed Panel
        self.left_frame = Frame(master, bg="#262626", width=800, height=700, borderwidth=2, relief="groove")
        self.left_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.live_feed_canvas = Canvas(self.left_frame, width=600, height=600, bg="#1A1A1A", relief="sunken")
        self.live_feed_canvas.pack(padx=5, pady=5)

        # Last Captured Image Section
        self.middle_frame = Frame(master, bg="#262626", width=600, height=700)
        self.middle_frame.pack(side="top", fill="both", padx=10, pady=10)

        self.last_image_canvas = Canvas(self.middle_frame, width=580, height=580, bg="#1A1A1A", relief="sunken")
        self.last_image_canvas.pack(pady=5)

        # Initialize Camera
        self.camera = self.initialize_camera()
        self.video_thread = threading.Thread(target=self.capture_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def initialize_camera(self):
        """Setup the Dino-Lite camera and return the camera object."""
        camera = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return camera

    def capture_video(self):
        """Capture video feed from the camera."""
        while True:
            ret, frame = self.camera.read()
            if ret:
                self.display_live_feed(frame)

    def display_live_feed(self, frame):
        """Display live video feed on the canvas."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image)

        self.live_feed_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.live_feed_canvas.image = photo

    def capture_action(self):
        """Capture an image, save it, and display it."""
        ret, frame = self.camera.read()
        if ret:
            self.save_image(frame)
            self.display_last_image(frame)

    def save_image(self, frame):
        """Save the captured image to disk."""
        if not os.path.exists(IMAGE_SAVE_DIR):
            os.makedirs(IMAGE_SAVE_DIR)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(IMAGE_SAVE_DIR, f"capture_{timestamp}.png")
        cv2.imwrite(filename, frame)

    def display_last_image(self, frame):
        """Display the last captured image on the canvas."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image)

        self.last_image_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.last_image_canvas.image = photo

    def focus_stack_action(self):
        """Handle focus stacking action."""
        print("Focus stacking initiated...")

    def calibrate_action(self):
        """Handle calibration action."""
        print("Calibration initiated...")

    def show_parameters(self):
        """Show the parameters window with AMR, FOV, and other values."""
        param_window = Toplevel(self.master)
        param_window.title("Microscope Parameters")
        param_window.geometry("400x300")
        param_window.configure(bg="#262626")

        Label(param_window, text="Microscope Parameters", font=("Consolas", 14, "bold"), fg="#FFD700", bg="#262626").pack(pady=10)
        Label(param_window, text=f"AMR: {self.amr_value}", font=("Consolas", 12), fg="white", bg="#262626").pack(pady=5)
        Label(param_window, text=f"FOV: {self.fov_value}", font=("Consolas", 12), fg="white", bg="#262626").pack(pady=5)
        Label(param_window, text=f"Configuration: {self.config_status}", font=("Consolas", 12), fg="white", bg="#262626").pack(pady=5)
        Label(param_window, text=f"LED Status: {self.led_status}", font=("Consolas", 12), fg="white", bg="#262626").pack(pady=5)

if __name__ == "__main__":
    root = Tk()
    app = TechnicalPointCloudApp(root)
    root.mainloop()


