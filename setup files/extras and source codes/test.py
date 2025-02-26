import sys
import cv2
import time
import os
import threading
from tkinter import Tk, Label, Button, Frame, Canvas, BooleanVar, Menu
from PIL import Image, ImageTk  # Pillow library to handle image formats

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 1600, 900
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 960, 30
DEVICE_INDEX = 1  # Ensure this points to the Dino-Lite camera
IMAGE_SAVE_DIR = "images"
CAPTURE_COUNT = 1  # Number of images to capture
CAPTURE_INTERVAL = 1  # Time delay (in seconds) between image captures

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
        self.drawing_mode = None  # Drawing mode
        self.start_x = None
        self.start_y = None
        self.current_item = None
        self.stage_position = 5.0  # Initial stage position
        self.upper_bound = 8.0
        self.lower_bound = 2.0
        self.manual_mode = BooleanVar(value=False)  # Toggle switch to switch between manual and auto mode
        self.drawn_objects = []  # Store drawn objects for removal

        # Menu Bar
        self.menu_bar = Menu(master)
        self.master.config(menu=self.menu_bar)
        
        # File menu for the menu bar
        file_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=master.quit)

        # Top Panel for Buttons and Logo
        self.top_panel = Frame(master, bg="#262626", height=80, relief="groove", bd=1)
        self.top_panel.pack(side="top", fill="x", pady=5)

        # Add Logo (Ensure path is correct)
        try:
            self.logo_image = Image.open("D:/iitg/iitg_micromachine/setup files/download.png")
            self.logo_image = self.logo_image.resize((150, 50), Image.Resampling.LANCZOS)  # Updated for Pillow 10+
            self.logo_photo = ImageTk.PhotoImage(self.logo_image)
            self.logo_label = Label(self.top_panel, image=self.logo_photo, bg="#262626")
            self.logo_label.pack(side="right", padx=20, pady=5)
        except Exception as e:
            print(f"Error loading logo image: {e}")

        # Buttons for calibrate, img capture, focus stack
        self.calibrate_button = Button(
            self.top_panel,
            text="Calibrate",
            font=("Consolas", 12, "bold"),
            bg="#FF4500",  
            fg="#1A1A1A",
            relief="flat",
            command=self.calibrate_action,
            activebackground="#E63E00",
        )
        self.calibrate_button.pack(side="left", padx=20, pady=10)

        self.capture_button = Button(
            self.top_panel,
            text="Image Capture",
            font=("Consolas", 12, "bold"),
            bg="#00BFFF",  
            fg="#1A1A1A",
            relief="flat",
            command=self.capture_action,
            activebackground="#009ACD",
        )
        self.capture_button.pack(side="left", padx=20, pady=10)

        self.focus_stack_button = Button(
            self.top_panel,
            text="Focus Stack",
            font=("Consolas", 12, "bold"),
            bg="#FFD700",  
            fg="#1A1A1A",
            relief="flat",
            command=self.focus_stack_action,
            activebackground="#FFC400",
        )
        self.focus_stack_button.pack(side="left", padx=20, pady=10)

        # Video Feed with Scale from the dino lite camera
        self.left_frame = Frame(master, bg="#262626", width=800, height=700, borderwidth=2, relief="groove")
        self.left_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.live_feed_label = Label(
            self.left_frame, text="Live Video Feed", font=("Consolas", 14, "bold"), fg="#00BFFF", bg="#262626"
        )
        self.live_feed_label.pack(pady=5)

        self.live_feed_canvas = Canvas(self.left_frame, width=600, height=600, bg="#1A1A1A", relief="sunken")
        self.live_feed_canvas.pack(side="left", padx=5, pady=5)

        # Box to view the last captured image in the center
        self.middle_frame = Frame(master, bg="#262626", width=600, height=700)
        self.middle_frame.pack(side="top", fill="both", padx=10, pady=10)

        self.last_image_label = Label(self.middle_frame, text="Last Captured Image", font=("Consolas", 14, "bold"), fg="#00BFFF", bg="#262626")
        self.last_image_label.pack(pady=5)

        self.last_image_canvas = Canvas(self.middle_frame, width=580, height=580, bg="#1A1A1A", relief="sunken")
        self.last_image_canvas.pack(pady=5)

        # Drawing options below the last captured image
        self.draw_buttons_frame = Frame(self.middle_frame, bg="#262626")
        self.draw_buttons_frame.pack(pady=10)

        # Create shape buttons using basic drawing
        self.draw_line_button = Button(self.draw_buttons_frame, text="Line", command=self.start_drawing_line, relief="flat", bg="#4CAF50", fg="white")
        self.draw_line_button.pack(side="left", padx=10)

        self.draw_circle_button = Button(self.draw_buttons_frame, text="Circle", command=self.start_drawing_circle, relief="flat", bg="#2196F3", fg="white")
        self.draw_circle_button.pack(side="left", padx=10)

        self.draw_rectangle_button = Button(self.draw_buttons_frame, text="Rectangle", command=self.start_drawing_rectangle, relief="flat", bg="#FF9800", fg="white")
        self.draw_rectangle_button.pack(side="left", padx=10)

        # Parameters Section moved to bottom-right corner
        self.right_frame = Frame(master, bg="#262626", width=600, height=250, borderwidth=2, relief="groove")
        self.right_frame.pack(side="right", fill="both", padx=10, pady=10)

        self.parameters_label = Label(self.right_frame, text="Parameters", font=("Consolas", 14, "bold"), fg="#00BFFF", bg="#262626")
        self.parameters_label.pack(pady=10)

        # Displaying Stage Position, Upper Bound, Lower Bound, and Last Capture Info
        self.stage_position_label = Label(self.right_frame, text=f"Stage Position: {self.stage_position} mm", font=("Consolas", 12), fg="#FFFFFF", bg="#262626")
        self.stage_position_label.pack(pady=10)

        self.upper_bound_label = Label(self.right_frame, text=f"Upper Bound: {self.upper_bound} mm", font=("Consolas", 12), fg="#FFFFFF", bg="#262626")
        self.upper_bound_label.pack(pady=10)

        self.lower_bound_label = Label(self.right_frame, text=f"Lower Bound: {self.lower_bound} mm", font=("Consolas", 12), fg="#FFFFFF", bg="#262626")
        self.lower_bound_label.pack(pady=10)

        self.last_capture_label = Label(self.right_frame, text="Last Capture: None", font=("Consolas", 12), fg="#FFFFFF", bg="#262626")
        self.last_capture_label.pack(pady=10)

        # Initialize Camera
        self.camera = self.initialize_camera()

        # Start video capturing in a separate thread
        self.video_thread = threading.Thread(target=self.capture_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Bind mouse events for drawing on the last image canvas
        self.last_image_canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.last_image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.last_image_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Bind spacebar to clear drawing
        self.master.bind("<space>", self.clear_drawing)

    def start_drawing_rectangle(self):
        """Start drawing a rectangle on the last captured image canvas."""
        self.drawing_mode = 'rectangle'

    def start_drawing_line(self):
        """Start drawing a line on the last captured image canvas."""
        self.drawing_mode = 'line'

    def start_drawing_circle(self):
        """Start drawing a circle on the last captured image canvas."""
        self.drawing_mode = 'circle'

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
        self.live_feed_canvas.image = photo  # Keep a reference to prevent garbage collection

    def capture_action(self):
        """Capture an image, show it, and save it to disk.""" 
        ret, frame = self.camera.read()
        if ret:
            self.save_image(frame)
            self.display_last_image(frame)

    def save_image(self, frame):
        """Save the captured image.""" 
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(IMAGE_SAVE_DIR, f"capture_{timestamp}.png")
        cv2.imwrite(filename, frame)

    def display_last_image(self, frame):
        """Display the last captured image in the canvas.""" 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image)

        self.last_image_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.last_image_canvas.image = photo  # Keep a reference to prevent garbage collection

    def focus_stack_action(self):
        """Handle focus stacking action.""" 
        print("Focus stacking initiated...")

    def calibrate_action(self):
        """Handle calibration action.""" 
        print("Calibration initiated...")

    def on_mouse_press(self, event):
        """Capture the start coordinates when mouse is pressed."""
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        """Update the drawn shape as the mouse moves."""
        if self.drawing_mode == 'rectangle':
            self.redraw_rectangle(event.x, event.y)
        elif self.drawing_mode == 'line':
            self.redraw_line(event.x, event.y)
        elif self.drawing_mode == 'circle':
            self.redraw_circle(event.x, event.y)

    def on_mouse_release(self, event):
        """Finalize the drawn shape when mouse is released."""
        if self.drawing_mode == 'rectangle':
            self.finalize_rectangle(event.x, event.y)
        elif self.drawing_mode == 'line':
            self.finalize_line(event.x, event.y)
        elif self.drawing_mode == 'circle':
            self.finalize_circle(event.x, event.y)

    def finalize_rectangle(self, end_x, end_y):
        """Finalize and add rectangle shape."""
        rect = self.last_image_canvas.create_rectangle(self.start_x, self.start_y, end_x, end_y, outline="yellow")
        self.drawn_objects.append(rect)

    def finalize_line(self, end_x, end_y):
        """Finalize and add line shape."""
        line = self.last_image_canvas.create_line(self.start_x, self.start_y, end_x, end_y, fill="yellow")
        self.drawn_objects.append(line)

    def finalize_circle(self, end_x, end_y):
        """Finalize and add circle shape."""
        r = max(abs(end_x - self.start_x), abs(end_y - self.start_y))
        circle = self.last_image_canvas.create_oval(self.start_x-r, self.start_y-r, self.start_x+r, self.start_y+r, outline="yellow")
        self.drawn_objects.append(circle)

    def redraw_rectangle(self, end_x, end_y):
        """Redraw the rectangle as we drag the mouse."""
        if self.current_item:
            self.last_image_canvas.delete(self.current_item)
        self.current_item = self.last_image_canvas.create_rectangle(self.start_x, self.start_y, end_x, end_y, outline="yellow")

    def redraw_line(self, end_x, end_y):
        """Redraw the line as we drag the mouse."""
        if self.current_item:
            self.last_image_canvas.delete(self.current_item)
        self.current_item = self.last_image_canvas.create_line(self.start_x, self.start_y, end_x, end_y, fill="yellow")

    def redraw_circle(self, end_x, end_y):
        """Redraw the circle as we drag the mouse."""
        if self.current_item:
            self.last_image_canvas.delete(self.current_item)
        r = max(abs(end_x - self.start_x), abs(end_y - self.start_y))
        self.current_item = self.last_image_canvas.create_oval(self.start_x-r, self.start_y-r, self.start_x+r, self.start_y+r, outline="yellow")

    def clear_drawing(self, event):
        """Clear all drawn shapes when spacebar is pressed."""
        for item in self.drawn_objects:
            self.last_image_canvas.delete(item)
        self.drawn_objects.clear()


if __name__ == "__main__":
    root = Tk()
    app = TechnicalPointCloudApp(root)
    root.mainloop()
