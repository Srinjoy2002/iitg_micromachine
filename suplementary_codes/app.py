# def run_focus_stack_script():
#     """
#     Launches the external open3dviznew.py script in a separate process/thread.
#     """
#     try:
#         subprocess.run(["python", "open3dviznew.py"], check=True)
#     except Exception as e:
#         messagebox.showerror("Focus Stack Error", f"Error running focus stack script: {e}")


# def focus_stack_handler():
#     # Spawn the external script without blocking the GUI
#     threading.Thread(target=run_focus_stack_script, daemon=True).start()
import os
import cv2
import time
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
import tkinter.ttk as ttk
import subprocess
import numpy as np
from PIL import Image, ImageTk
import SPiiPlusPython as sp
# from focus import load_images_from_folder, process_focus_stack

import open3d as o3d  # for 3D point cloud visualization

# Conversion constants
MM_TO_DEG = 72   # 1 mm = 72 degrees (since 360° = 5 mm)
DEG_TO_MM = 1/72

# ---------------------- GLOBAL CONFIGURATION ---------------------- #
CAMERA_INDEX = 0
from datetime import datetime

# 1. Grab the current date & time
now = datetime.now()

# 2. Format it as a string: yyyy-mm-dd_HH-MM-SS
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# 3. Build your folder name
IMAGE_FOLDER= f"captured_img_{timestamp}"


if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

current_frame = None
camera_running = True
in_multi_sequence = False

try:
    hc = sp.OpenCommEthernetTCP("192.168.0.40", 701)
    sp.SetVelocity(hc, 0, 500, sp.SYNCHRONOUS, True)
    sp.SetAcceleration(hc, 0, 50, sp.SYNCHRONOUS, True)
    sp.SetDeceleration(hc, 0, 50, sp.SYNCHRONOUS, True)
    sp.SetJerk(hc, 0, 500, sp.SYNCHRONOUS, True)
    sp.SetKillDeceleration(hc, 0, 10000, sp.SYNCHRONOUS, True)
    sp.SetFPosition(hc, sp.Axis.ACSC_AXIS_0, 0, failure_check=True)
except Exception as e:
    messagebox.showerror("Connection Error", f"Failed to open communication: {e}")
    hc = None

#---------------------- CAMERA FUNCTIONS ----------------------#
cap = cv2.VideoCapture(CAMERA_INDEX)

def camera_loop():
    global current_frame, cap, camera_running
    while camera_running:
        ret, frame = cap.read()
        if ret:
            current_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            live_feed.imgtk = imgtk  # keep a reference!
            live_feed.create_image(0, 0, anchor="nw", image=imgtk)
        time.sleep(0.1)

def capture_image(auto_capture=False):
    global current_frame, in_multi_sequence
    if auto_capture and not in_multi_sequence:
        return

    if current_frame is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(IMAGE_FOLDER, f"image_{timestamp}.jpg")
        cv2.imwrite(filename, current_frame)
        print(f"Image saved as {filename}")
        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image = pil_image.resize((300, 300))
        imgtk = ImageTk.PhotoImage(image=pil_image)
        last_image.imgtk = imgtk
        last_image.create_image(0, 0, anchor="nw", image=imgtk)
    else:
        print("No frame available to capture.")

# ---------------------- MOTION & CONTROL FUNCTIONS ---------------------- #

def enable():
    try:
        subprocess.run(["python", "E:\ACS motion Controller\motion_control_code\code\enable.py"], check=True)
        messagebox.showinfo("Info", "Enable command executed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error in enabling: {e}")

def disable():
    try:
        subprocess.run(["python", "E:\ACS motion Controller\motion_control_code\code\disable.py"], check=True)
        messagebox.showinfo("Info", "Disable command executed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error in disabling: {e}")
# ---------------------- GO TO HOME FUNCTION ---------------------- #
def go_to_home():
    try:
        sp.ToPoint(hc, 0, sp.Axis.ACSC_AXIS_0, 0, failure_check=True)
        messagebox.showinfo("Info", "Stage moved to home position (0 mm) successfully!")
    except Exception as e:
        messagebox.showerror("Motion Error", f"Error moving to home: {e}")

def run_sequence():
    """
    Moves the stage from the initial 0 mm position to Lower Bound (LB) without capturing.
    Then from LB to Upper Bound (UB), it moves in increments of 'step' mm,
    waits at each position, captures an image, and finally returns to 0 mm.
    All user inputs (LB, UB, Step) are in mm.
    """
    global in_multi_sequence

    try:
        lb_mm = float(lower_entry.get())
        ub_mm = float(upper_entry.get())
        step_mm = float(step_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for LB, UB, and Step Size.")
        return 

    if lb_mm > ub_mm:
        messagebox.showerror("Input Error", "Lower Bound must be <= Upper Bound.")
        return

    tolerance_mm = 0.1
    tolerance_deg = tolerance_mm * MM_TO_DEG

    # Convert mm values to degrees
    lb_deg = lb_mm * MM_TO_DEG
    ub_deg = ub_mm * MM_TO_DEG
    step_deg = step_mm * MM_TO_DEG

    # Move from current (0 deg, 0 mm) to LB
    try:
        sp.ToPoint(hc, 0, sp.Axis.ACSC_AXIS_0, lb_deg, failure_check=True)
        while abs(sp.GetFPosition(hc, sp.Axis.ACSC_AXIS_0, failure_check=True) - lb_deg) > tolerance_deg:
            time.sleep(0.1)
        print(f"Motor reached Lower Bound: {lb_mm} mm")
    except Exception as e:
        messagebox.showerror("Motion Error", f"Error moving to Lower Bound: {e}")
        return

    in_multi_sequence = True

    def sequence_thread():
        global in_multi_sequence
        current_target_mm = lb_mm
        try:
            while current_target_mm <= ub_mm:
                target_deg = current_target_mm * MM_TO_DEG
                sp.ToPoint(hc, 0, sp.Axis.ACSC_AXIS_0, target_deg, failure_check=True)
                print(f"Moving to position: {current_target_mm} mm")
                while abs(sp.GetFPosition(hc, sp.Axis.ACSC_AXIS_0, failure_check=True) - target_deg) > tolerance_deg:
                    time.sleep(0.05)
                print(f"Reached position: {current_target_mm} mm")
                time.sleep(1)  # dwell time
                capture_image(auto_capture=True)
                current_target_mm += step_mm
            # After traversal, return to 0 mm (0 deg)
            sp.ToPoint(hc, 0, sp.Axis.ACSC_AXIS_0, 0, failure_check=True)
            in_multi_sequence = False
            messagebox.showinfo("Info", "Motion sequence completed successfully!")
        except Exception as e:
            messagebox.showerror("Motion Error", f"Error during motion sequence: {e}")
            in_multi_sequence = False

    threading.Thread(target=sequence_thread, daemon=True).start()

def move_up():
    try:
        inc_mm = float(increment_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid increment value!")
        return
    try:
        current_deg = sp.GetFPosition(hc, sp.Axis.ACSC_AXIS_0, failure_check=True)
        current_mm = current_deg * DEG_TO_MM
        new_mm = current_mm + inc_mm
        new_deg = new_mm * MM_TO_DEG
        sp.ToPoint(hc, 0, sp.Axis.ACSC_AXIS_0, new_deg, failure_check=True)
        print(f"Moving UP: from {current_mm:.2f} mm to {new_mm:.2f} mm")
    except Exception as e:
        messagebox.showerror("Motion Error", f"Error in moving up: {e}")

def move_down():
    try:
        inc_mm = float(increment_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid increment value!")
        return
    try:
        current_deg = sp.GetFPosition(hc, sp.Axis.ACSC_AXIS_0, failure_check=True)
        current_mm = current_deg * DEG_TO_MM
        new_mm = current_mm - inc_mm
        new_deg = new_mm * MM_TO_DEG
        sp.ToPoint(hc, 0, sp.Axis.ACSC_AXIS_0, new_deg, failure_check=True)
        print(f"Moving DOWN: from {current_mm:.2f} mm to {new_mm:.2f} mm")
    except Exception as e:
        messagebox.showerror("Motion Error", f"Error in moving down: {e}")
def update_position():
    try:
        current_deg = sp.GetFPosition(hc, sp.Axis.ACSC_AXIS_0, failure_check=True)
        current_mm = current_deg * DEG_TO_MM
        position_label.config(text=f"{current_mm:.2f} mm")  # Update label text
    except Exception as e:
        position_label.config(text="Error")  # If an error occurs, display "Error"
    root.after(100, update_position)  # Schedule update every 100 ms

# ---------------------- FOCUS STACKING FUNCTIONS ---------------------- #

def update_progress(current):
    progress_bar["value"] = current
    root.update_idletasks()

def run_focus_stack_script():
    """
    Launches the external open3dviznew.py script in a separate process/thread.
    """
    try:
        subprocess.run(["python", "open3dviznew.py"], check=True)
    except Exception as e:
        messagebox.showerror("Focus Stack Error", f"Error running focus stack script: {e}")


def focus_stack_handler():
    # Spawn the external script without blocking the GUI
    threading.Thread(target=run_focus_stack_script, daemon=True).start()

def run_task():
    messagebox.showinfo("Run Task", "Run Task functionality is not implemented yet.")

def emergency_stop():
    
    sp.Kill(hc, 0, sp.SYNCHRONOUS, True)
    # messagebox.showwarning("Emergency Stop", "Emergency Stop triggered!")
    # Add actual emergency stop logic here

def view_directory():
    try:
        os.startfile(os.path.abspath(IMAGE_FOLDER))
    except Exception as e:
        messagebox.showerror("Error", f"Could not open directory: {e}")

# ---------------------- GUI LAYOUT ---------------------- #

root = tk.Tk()
root.title("Motion Control")
root.state('zoomed')  # Open maximized

root.columnconfigure(0, minsize=220)  # Left control panel
root.columnconfigure(1, weight=1)     # Right display panel
root.rowconfigure(1, weight=1)

title_frame = tk.Frame(root)
title_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")
tk.Label(title_frame, text="MOTION CONTROL", font=("Arial", 20, "bold")).pack(side=tk.LEFT, padx=10)
# Logo Image (Rightmost Corner)
logo_image = Image.open("E:\ACS motion Controller\motion_control_code\code\download.png")  # Adjust the path
logo_image = logo_image.resize((120, 120))  # Resize if necessary
logo_photo = ImageTk.PhotoImage(logo_image)

logo_label = tk.Label(title_frame, image=logo_photo)
logo_label.image = logo_photo  # Keep a reference
logo_label.pack(side=tk.RIGHT, padx=10)

control_frame = tk.LabelFrame(root, text="Control Panel", padx=5, pady=5)
control_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsw")
control_frame.columnconfigure(0, weight=1)
control_frame.columnconfigure(1, weight=1)

tk.Label(control_frame, text="Multi-Point Sequence").grid(row=0, column=0, columnspan=2, pady=(0,5))
tk.Label(control_frame, text="Lower Bound (mm):").grid(row=1, column=0, sticky="e")
lower_entry = tk.Entry(control_frame, width=10)
lower_entry.grid(row=1, column=1, padx=5, pady=2)
tk.Label(control_frame, text="Upper Bound (mm):").grid(row=2, column=0, sticky="e")
upper_entry = tk.Entry(control_frame, width=10)
upper_entry.grid(row=2, column=1, padx=5, pady=2)
tk.Label(control_frame, text="Step Size (mm):").grid(row=3, column=0, sticky="e")
step_entry = tk.Entry(control_frame, width=10)
step_entry.grid(row=3, column=1, padx=5, pady=2)
step_entry.insert(0, "10")
tk.Button(control_frame, text="Run Multi-Point Sequence", command=run_sequence).grid(row=4, column=0, columnspan=2, pady=5)

tk.Label(control_frame, text="Single-Point Motion").grid(row=5, column=0, columnspan=2, pady=(10,5))
tk.Label(control_frame, text="Increment (mm):").grid(row=6, column=0, sticky="e")
increment_entry = tk.Entry(control_frame, width=10)
increment_entry.grid(row=6, column=1, padx=5, pady=2)
increment_entry.insert(0, "100")
tk.Button(control_frame, text="▲ Up", width=10, command=move_up).grid(row=7, column=0, padx=5, pady=5)
tk.Button(control_frame, text="▼ Down", width=10, command=move_down).grid(row=7, column=1, padx=5, pady=5)

tk.Button(control_frame, text="Enable", width=10, bg="green", command=enable).grid(row=8, column=0, padx=5, pady=5)
tk.Button(control_frame, text="Disable", width=10, bg="red", command=disable).grid(row=8, column=1, padx=5, pady=5)

additional_frame = tk.LabelFrame(control_frame, text="Additional Controls", padx=5, pady=5)
additional_frame.grid(row=9, column=0, columnspan=2, pady=(10,0))
tk.Button(additional_frame, text="Go to Home", width=15, command=go_to_home).grid(row=0, column=0, padx=5, pady=2)
tk.Button(additional_frame, text="Focus Stack", width=15, command=run_focus_stack_script).grid(row=0, column=1, padx=5, pady=2)
tk.Button(additional_frame, text="View Directory", width=15, command=view_directory).grid(row=1, column=0, padx=5, pady=2)
tk.Button(additional_frame, text="Capture Image", width=15, command=capture_image).grid(row=1, column=1, padx=5, pady=2)
tk.Button(additional_frame, text="EMERGENCY STOP", width=15, bg="red", command=emergency_stop).grid(row=2, column=0, columnspan=2, padx=5, pady=2)
progress_bar = ttk.Progressbar(additional_frame, length=180, mode='determinate')
progress_bar.grid(row=3, column=0, columnspan=2, pady=5)
# Add a real-time position display label
position_frame = tk.Frame(additional_frame)
position_frame.grid(row=4, column=0, columnspan=2, pady=5)

tk.Label(position_frame, text="Current Position: ", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
position_label = tk.Label(position_frame, text="0.00 mm", font=("Arial", 12), fg="blue")
position_label.pack(side=tk.LEFT)

display_frame = tk.Frame(root, padx=5, pady=5)
display_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
display_frame.columnconfigure(0, weight=1)
display_frame.rowconfigure(0, weight=1)
display_frame.rowconfigure(1, weight=1)

live_feed_label = tk.Label(display_frame, text="LIVE FEED", font=("Arial", 14, "bold"))
live_feed_label.grid(row=0, column=0, sticky="n", pady=(0,5))
live_feed = tk.Canvas(display_frame, width=600, height=600, bg="black")
live_feed.grid(row=0, column=0, pady=(40,0))

last_image_label = tk.Label(display_frame, text="LAST IMAGE", font=("Arial", 12))
last_image_label.grid(row=1, column=0, sticky="n", pady=(20,5))
last_image = tk.Canvas(display_frame, width=300, height=300, bg="gray")
last_image.grid(row=1, column=0, sticky="n")

root.bind("<Up>", lambda event: move_up())
root.bind("<Down>", lambda event: move_down())

threading.Thread(target=camera_loop, daemon=True).start()

# Ensure proper cleanup on exit
def on_closing():
    go_to_home()
    
    global camera_running
    camera_running = False
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.after(100, update_position)
root.mainloop()
