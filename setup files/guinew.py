import sys
from tkinter import Tk, Label, Button, Frame, Canvas, PhotoImage


class TechnicalPointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D Point Cloud Reconstruction Tool")
        self.master.configure(bg="#1A1A1A")
        self.master.state("zoomed")  # Open in maximized window

        self.drawing_mode = None
        self.start_x = None
        self.start_y = None
        self.current_item = None

        # Stage position for scale indicator
        self.stage_position = 0.0  # Initial position in mm

        # Top Panel for Buttons and Logo
        self.top_panel = Frame(master, bg="#262626", height=80, relief="groove", bd=1)
        self.top_panel.pack(side="top", fill="x", pady=5)

        # Add Logo
        self.logo_image = PhotoImage(file="D:/iitg/iitg_micromachine/setup files/download.png")  # Replace with your logo file path
        self.logo_label = Label(self.top_panel, image=self.logo_image, bg="#262626")
        self.logo_label.pack(side="right", padx=20, pady=5)

        # Buttons
        self.calibrate_button = Button(
            self.top_panel,
            text="Calibrate",
            font=("Consolas", 12, "bold"),
            bg="#FF4500",  # Neon red
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
            bg="#00BFFF",  # Neon blue
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
            bg="#FFD700",  # Neon yellow
            fg="#1A1A1A",
            relief="flat",
            command=self.focus_stack_action,
            activebackground="#FFC400",
        )
        self.focus_stack_button.pack(side="left", padx=20, pady=10)

        # Left Section: Live Video Feed with Scale
        self.left_frame = Frame(master, bg="#262626", width=450, height=600, borderwidth=2, relief="groove")
        self.left_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.live_feed_label = Label(
            self.left_frame, text="Live Video Feed", font=("Consolas", 14, "bold"), fg="#00BFFF", bg="#262626"
        )
        self.live_feed_label.pack(pady=5)

        self.live_feed_canvas = Canvas(self.left_frame, width=400, height=400, bg="#1A1A1A", relief="sunken")
        self.live_feed_canvas.pack(side="left", pady=5)

        # Scale and Jog Buttons
        self.scale_frame = Frame(self.left_frame, bg="#1A1A1A")
        self.scale_frame.pack(side="right", fill="y", padx=5)

        # Upper Bound Indicator
        self.upper_bound_label = Label(
            self.scale_frame, text="Upper Bound", font=("Consolas", 10), fg="#FFD700", bg="#1A1A1A"
        )
        self.upper_bound_label.pack(pady=5)

        # Scale Display
        self.scale_canvas = Canvas(self.scale_frame, width=50, height=400, bg="#262626", relief="flat")
        self.scale_canvas.pack(pady=5)
        self.update_scale()

        # Lower Bound Indicator
        self.lower_bound_label = Label(
            self.scale_frame, text="Lower Bound", font=("Consolas", 10), fg="#FFD700", bg="#1A1A1A"
        )
        self.lower_bound_label.pack(pady=5)

        # Jog Buttons
        self.jog_up_button = Button(
            self.scale_frame,
            text="▲",
            font=("Consolas", 12, "bold"),
            bg="#00BFFF",
            fg="#1A1A1A",
            relief="flat",
            command=self.jog_up,
            activebackground="#009ACD",
        )
        self.jog_up_button.pack(pady=5)

        self.jog_down_button = Button(
            self.scale_frame,
            text="▼",
            font=("Consolas", 12, "bold"),
            bg="#FF4500",
            fg="#1A1A1A",
            relief="flat",
            command=self.jog_down,
            activebackground="#E63E00",
        )
        self.jog_down_button.pack(pady=5)

        # Center Section: Last Captured Image
        self.center_frame = Frame(master, bg="#262626", width=450, height=600, borderwidth=2, relief="groove")
        self.center_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.captured_image_label = Label(
            self.center_frame, text="Last Captured Image", font=("Consolas", 14, "bold"), fg="#FFD700", bg="#262626"
        )
        self.captured_image_label.pack(pady=5)

        # Menu Bar for Drawing
        self.menu_bar = Frame(self.center_frame, bg="#1A1A1A")
        self.menu_bar.pack(fill="x", padx=5, pady=5)

        Button(
            self.menu_bar, text="Draw Line", bg="#FF4500", fg="white", font=("Consolas", 10), command=self.set_draw_line
        ).pack(side="left", padx=5, pady=5)
        Button(
            self.menu_bar, text="Draw Circle (Diameter)", bg="#00BFFF", fg="white", font=("Consolas", 10), command=self.set_draw_circle_diameter
        ).pack(side="left", padx=5, pady=5)
        Button(
            self.menu_bar, text="Draw Circle (Radius)", bg="#FFD700", fg="white", font=("Consolas", 10), command=self.set_draw_circle_radius
        ).pack(side="left", padx=5, pady=5)

        self.captured_image_canvas = Canvas(self.center_frame, width=400, height=400, bg="#1A1A1A", relief="sunken")
        self.captured_image_canvas.pack(pady=5)

        self.captured_image_canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.captured_image_canvas.bind("<B1-Motion>", self.update_drawing)
        self.captured_image_canvas.bind("<ButtonRelease-1>", self.finish_drawing)

        self.master.bind("<space>", self.clear_drawings)

        # Right Section: 3D Visualization
        self.right_frame = Frame(master, bg="#262626", width=450, height=600, borderwidth=2, relief="groove")
        self.right_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.visualization_label = Label(
            self.right_frame, text="3D Visualization", font=("Consolas", 14, "bold"), fg="#FF4500", bg="#262626"
        )
        self.visualization_label.pack(pady=5)

        self.visualization_canvas = Canvas(self.right_frame, width=400, height=400, bg="#1A1A1A", relief="sunken")
        self.visualization_canvas.pack(pady=5)

    def update_scale(self):
        """Update the scale indicator on the scale canvas."""
        self.scale_canvas.delete("all")
        self.scale_canvas.create_line(25, 0, 25, 400, fill="white", width=2)  # Main scale line
        for i in range(0, 401, 20):
            self.scale_canvas.create_line(20, i, 30, i, fill="white")  # Tick marks
        # Current position indicator
        pos = 400 - (self.stage_position * 40)  # Map stage position to scale
        self.scale_canvas.create_oval(15, pos - 5, 35, pos + 5, fill="#FFD700", outline="")

    def jog_up(self):
        """Move the stage up."""
        if self.stage_position < 10.0:  # Upper limit
            self.stage_position += 0.1
            self.update_scale()

    def jog_down(self):
        """Move the stage down."""
        if self.stage_position > 0.0:  # Lower limit
            self.stage_position -= 0.1
            self.update_scale()

    def set_draw_line(self):
        self.drawing_mode = "line"

    def set_draw_circle_diameter(self):
        self.drawing_mode = "circle_diameter"

    def set_draw_circle_radius(self):
        self.drawing_mode = "circle_radius"

    def start_drawing(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.drawing_mode == "line":
            self.current_item = self.captured_image_canvas.create_line(
                self.start_x, self.start_y, event.x, event.y, fill="#FF4500", width=2
            )
        elif self.drawing_mode in {"circle_diameter", "circle_radius"}:
            self.current_item = self.captured_image_canvas.create_oval(
                self.start_x, self.start_y, event.x, event.y, outline="#00BFFF", width=2
            )

    def update_drawing(self, event):
        if self.current_item and self.drawing_mode:
            self.captured_image_canvas.coords(self.current_item, self.start_x, self.start_y, event.x, event.y)

    def finish_drawing(self, event):
        self.start_x = self.start_y = self.current_item = None

    def clear_drawings(self, event=None):
        self.captured_image_canvas.delete("all")

    def calibrate_action(self):
        print("Calibrating...")

    def capture_action(self):
        print("Starting image capture...")

    def focus_stack_action(self):
        print("Performing focus stacking...")


if __name__ == "__main__":
    root = Tk()
    app = TechnicalPointCloudApp(root)
    root.mainloop()
