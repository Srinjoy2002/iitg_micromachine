import sys
from tkinter import Tk, Label, Button, Frame, Canvas, ttk, PhotoImage


class TechnicalPointCloudApp:
    def __init__(self, master):
        self.master = master
        self.master.title("3D Point Cloud Reconstruction Tool")
        self.master.configure(bg="#1E1E1E")
        self.master.state("zoomed")  # Open in full-screen mode

        # Top Panel for Buttons and Logo
        self.top_panel = Frame(master, bg="#1E1E1E", height=80)
        self.top_panel.pack(side="top", fill="x", pady=5)

        # Add Logo in the Top Panel
        self.logo_image = PhotoImage(file="D:\iitg\iitg_micromachine\setup files\download.png")  # Replace with your logo file path
        self.logo_label = Label(self.top_panel, image=self.logo_image, bg="#1E1E1E")
        self.logo_label.pack(side="right", padx=20, pady=5)

        # Buttons in Top Panel
        self.calibrate_button = Button(
            self.top_panel,
            text="Calibrate",
            font=("Consolas", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            relief="flat",
            command=self.calibrate_action,
            activebackground="#3E8E41",
        )
        self.calibrate_button.pack(side="left", padx=20, pady=10)

        self.capture_button = Button(
            self.top_panel,
            text="Start Image Capture",
            font=("Consolas", 12, "bold"),
            bg="#2196F3",
            fg="white",
            relief="flat",
            command=self.capture_action,
            activebackground="#1E88E5",
        )
        self.capture_button.pack(side="left", padx=20, pady=10)

        self.focus_stack_button = Button(
            self.top_panel,
            text="Focus Stacking",
            font=("Consolas", 12, "bold"),
            bg="#FF5722",
            fg="white",
            relief="flat",
            command=self.focus_stack_action,
            activebackground="#E64A19",
        )
        self.focus_stack_button.pack(side="left", padx=20, pady=10)

        # Left Section: Live Video Feed
        self.left_frame = Frame(master, bg="#262626", width=450, height=600, borderwidth=2, relief="groove")
        self.left_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.live_feed_label = Label(
            self.left_frame, text="Live Video Feed", font=("Consolas", 14, "bold"), fg="white", bg="#262626"
        )
        self.live_feed_label.pack(pady=5)

        self.live_feed_canvas = Canvas(self.left_frame, width=400, height=400, bg="black", relief="sunken")
        self.live_feed_canvas.pack(pady=5)

        # Center Section: Last Captured Image
        self.center_frame = Frame(master, bg="#262626", width=450, height=600, borderwidth=2, relief="groove")
        self.center_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.captured_image_label = Label(
            self.center_frame, text="Last Captured Image", font=("Consolas", 14, "bold"), fg="white", bg="#262626"
        )
        self.captured_image_label.pack(pady=5)

        self.captured_image_canvas = Canvas(self.center_frame, width=400, height=400, bg="grey", relief="sunken")
        self.captured_image_canvas.pack(pady=5)

        # Right Section: 3D Visualization
        self.right_frame = Frame(master, bg="#262626", width=450, height=600, borderwidth=2, relief="groove")
        self.right_frame.pack(side="left", fill="both", padx=10, pady=10)

        self.visualization_label = Label(
            self.right_frame, text="3D Visualization", font=("Consolas", 14, "bold"), fg="white", bg="#262626"
        )
        self.visualization_label.pack(pady=5)

        self.visualization_canvas = Canvas(self.right_frame, width=400, height=400, bg="grey", relief="sunken")
        self.visualization_canvas.pack(pady=5)

        # Bottom Section: Parameter Display
        self.bottom_frame = Frame(master, bg="#1E1E1E", height=100)
        self.bottom_frame.pack(side="bottom", fill="x", pady=10)

        self.parameters_label = Label(
            self.bottom_frame, text="Parameters", font=("Consolas", 14, "bold"), fg="white", bg="#1E1E1E"
        )
        self.parameters_label.pack(pady=5)

        self.parameter_text = ttk.Treeview(
            self.bottom_frame, columns=("Parameter", "Value"), show="headings", height=5
        )
        self.parameter_text.heading("Parameter", text="Parameter")
        self.parameter_text.heading("Value", text="Value")
        self.parameter_text.column("Parameter", width=150)
        self.parameter_text.column("Value", width=200)
        self.parameter_text.pack(fill="x", padx=10)

    # Button actions (placeholders for actual functionality)
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
