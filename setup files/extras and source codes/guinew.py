from tkinter import Tk, Label, Button, Frame, Canvas, PhotoImage, Toplevel, BooleanVar, Scrollbar

class TechnicalPointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D Point Cloud Reconstruction Tool")

        self.master.geometry("1600x900")
        self.master.configure(bg="#1A1A1A")

        # Variables for stage and drawing
        self.drawing_mode = None
        self.start_x = None
        self.start_y = None
        self.current_item = None
        self.stage_position = 5.0
        self.upper_bound = 8.0
        self.lower_bound = 2.0
        self.manual_mode = BooleanVar(value=False)  # toggle switch to switch b/w manual and auto mode

        # Top Panel for Buttons and Logo
        self.top_panel = Frame(master, bg="#262626", height=120, relief="groove", bd=1)
        self.top_panel.pack(side="top", fill="x", pady=5)

        # Add Logo (smaller size)
        self.logo_image = PhotoImage(file="download.png").subsample(3)  # Resize by a factor of 3
        self.logo_label = Label(self.top_panel, image=self.logo_image, bg="#262626")
        self.logo_label.pack(side="left", padx=20, pady=5)

        # Buttons for calibrate, img capture, focus stack
        self.calibrate_button = Button(
            self.top_panel,
            text="Calibrate",
            font=("Consolas", 10, "bold"),
            bg="#FF4500",  
            fg="#1A1A1A",
            relief="flat",
            command=self.calibrate_action,
            activebackground="#E63E00",
        )
        self.calibrate_button.pack(side="left", padx=10, pady=10)

        self.capture_button = Button(
            self.top_panel,
            text="Image Capture",
            font=("Consolas", 10, "bold"),
            bg="#00BFFF",  
            fg="#1A1A1A",
            relief="flat",
            command=self.capture_action,
            activebackground="#009ACD",
        )
        self.capture_button.pack(side="left", padx=10, pady=10)

        self.focus_stack_button = Button(
            self.top_panel,
            text="Focus Stack",
            font=("Consolas", 10, "bold"),
            bg="#FFD700",  
            fg="#1A1A1A",
            relief="flat",
            command=self.focus_stack_action,
            activebackground="#FFC400",
        )
        self.focus_stack_button.pack(side="left", padx=10, pady=10)

        # Button to open 3D Visualization in a diff window(full screen pop up)
        self.open_3d_button = Button(
            self.top_panel,
            text="Open 3D Viz",
            font=("Consolas", 10, "bold"),
            bg="#FF4500",  
            fg="#1A1A1A",
            relief="flat",
            command=self.open_3d_visualization,
            activebackground="#E63E00",
        )
        self.open_3d_button.pack(side="left", padx=10, pady=10)

        # Parameters Button - will show parameters in a popup window
        self.parameters_button = Button(
            self.top_panel,
            text="Parameters",
            font=("Consolas", 12, "bold"),
            bg="#FFD700",
            fg="#1A1A1A",
            relief="flat",
            command=self.show_parameters,
            activebackground="#FFC400",
        )
        self.parameters_button.pack(side="right", padx=20)

        # Adjusted size for left frame (live feed)
        self.left_frame = Frame(master, bg="#262626", width=600, height=600, borderwidth=2, relief="groove")
        self.left_frame.pack(side="left", fill="both", padx=10, pady=10, expand=True)

        self.live_feed_label = Label(
            self.left_frame, text="Live Video Feed", font=("Consolas", 14, "bold"), fg="#00BFFF", bg="#262626"
        )
        self.live_feed_label.pack(pady=5)

        self.live_feed_canvas = Canvas(self.left_frame, width=600, height=600, bg="#1A1A1A", relief="sunken")
        self.live_feed_canvas.pack(side="left", padx=5, pady=5)

        # Scale and Jog Buttons, for moving the linear stage up and down
        self.scale_frame = Frame(self.left_frame, bg="#1A1A1A")
        self.scale_frame.pack(side="right", fill="y", padx=5, pady=5)

        # Scrollable Scale to visualize the upper and lower bounds, in form of a slider dot
        self.scale_canvas = Canvas(self.scale_frame, width=50, height=600, bg="#262626", relief="flat")
        self.scale_canvas.pack(side="left", fill="y", pady=5)

        self.scrollbar = Scrollbar(self.scale_frame, orient="vertical", command=self.scale_canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.scale_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.update_scale()

        # Jog Buttons, to manually move the stage up and down
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

        # Manual/Automatic mode Toggle Button for calibration
        self.toggle_mode_button = Button(
            self.scale_frame,
            text="Manual Mode",
            font=("Consolas", 10),
            bg="#FFD700",
            fg="#1A1A1A",
            command=self.toggle_mode,
            relief="flat",
        )
        self.toggle_mode_button.pack(pady=5)

        # Fix Bound Buttons
        self.fix_upper_button = Button(
            self.scale_frame,
            text="Set Upper Bound",
            font=("Consolas", 10),
            bg="#00BFFF",
            fg="#1A1A1A",
            command=self.fix_upper_bound,
            relief="flat",
            state="disabled",
        )
        self.fix_upper_button.pack(pady=5)

        self.fix_lower_button = Button(
            self.scale_frame,
            text="Set Lower Bound",
            font=("Consolas", 10),
            bg="#FF4500",
            fg="#1A1A1A",
            command=self.fix_lower_bound,
            relief="flat",
            state="disabled",
        )
        self.fix_lower_button.pack(pady=5)

        # Last Captured Image with Drawing Features
        self.center_frame = Frame(master, bg="#262626", width=750, height=700, borderwidth=2, relief="groove")
        self.center_frame.pack(side="left", fill="both", padx=10, pady=10, expand=True)

        self.captured_image_label = Label(
            self.center_frame, text="Last Captured Image", font=("Consolas", 14, "bold"), fg="#FFD700", bg="#262626"
        )
        self.captured_image_label.pack(pady=5)

        self.captured_image_canvas = Canvas(self.center_frame, width=500, height=500, bg="#1A1A1A", relief="sunken")
        self.captured_image_canvas.pack(pady=5)

        self.captured_image_canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.captured_image_canvas.bind("<B1-Motion>", self.update_drawing)
        self.captured_image_canvas.bind("<ButtonRelease-1>", self.finish_drawing)

        self.master.bind("<space>", self.clear_drawings)

    def show_parameters(self):
        # Create a new top-level window to show the parameters
        params_window = Toplevel(self.master)
        params_window.title("Parameters")
        params_window.geometry("300x200")
        params_window.configure(bg="#1A1A1A")

        # Display parameter values
        Label(
            params_window,
            text=f"Stage Position: {self.stage_position:.2f}",
            font=("Consolas", 12),
            fg="#FFD700",
            bg="#1A1A1A",
        ).pack(pady=10)

        Label(
            params_window,
            text=f"Upper Bound: {self.upper_bound:.2f}",
            font=("Consolas", 12),
            fg="#FFD700",
            bg="#1A1A1A",
        ).pack(pady=10)

        Label(
            params_window,
            text=f"Lower Bound: {self.lower_bound:.2f}",
            font=("Consolas", 12),
            fg="#FFD700",
            bg="#1A1A1A",
        ).pack(pady=10)

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

    def update_scale(self):
        """Update the scale with indicators."""
        self.scale_canvas.delete("all")
        self.scale_canvas.create_line(25, 0, 25, 600, fill="white", width=2)
        for i in range(0, 601, 30):
            self.scale_canvas.create_line(20, i, 30, i, fill="white")
        upper_pos = 600 - (self.upper_bound * 60)
        lower_pos = 600 - (self.lower_bound * 60)
        pos = 600 - (self.stage_position * 60)
        self.scale_canvas.create_text(15, upper_pos, text="U", fill="#FFD700", anchor="e")
        self.scale_canvas.create_text(15, lower_pos, text="L", fill="#FFD700", anchor="e")
        self.scale_canvas.create_oval(15, pos - 5, 35, pos + 5, fill="#FFD700", outline="")

    def jog_up(self):
        if self.stage_position < 10.0:
            self.stage_position += 0.1
            self.update_scale()

    def jog_down(self):
        if self.stage_position > 0.0:
            self.stage_position -= 0.1
            self.update_scale()

    def toggle_mode(self):
        self.manual_mode.set(not self.manual_mode.get())
        if self.manual_mode.get():
            self.toggle_mode_button.config(text="Manual Mode", bg="#E63E00")
            self.fix_upper_button.config(state="normal")
            self.fix_lower_button.config(state="normal")
        else:
            self.toggle_mode_button.config(text="Automatic", bg="#FFD700")
            self.fix_upper_button.config(state="disabled")
            self.fix_lower_button.config(state="disabled")

    def fix_upper_bound(self):
        self.upper_bound = self.stage_position
        self.update_scale()

    def fix_lower_bound(self):
        self.lower_bound = self.stage_position
        self.update_scale()

    def open_3d_visualization(self):
        viz_window = Toplevel(self.master)
        viz_window.title("3D Visualization")
        viz_window.state("zoomed")
        viz_window.configure(bg="#1A1A1A")
        Label(viz_window, text="3D Visualization Placeholder", font=("Consolas", 16, "bold"), fg="#FFD700", bg="#1A1A1A").pack(pady=50)

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
