import tkinter as tk
from tkinter import messagebox
import subprocess
import SPiiPlusPython as sp
from focu import load_images_from_folder, process_focus_stack

# Global communication handle (created once for the session)
try:
    hc = sp.OpenCommEthernetTCP("192.168.0.40", 701)
except Exception as e:
    messagebox.showerror("Connection Error", f"Failed to open communication: {e}")
    hc = None

def enable():
    try:
        subprocess.run(["python", "enable.py"], check=True)
        messagebox.showinfo("Info", "Enable command executed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error in enabling: {e}")

def disable():
    try:
        subprocess.run(["python", "disable.py"], check=True)
        messagebox.showinfo("Info", "Disable command executed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error in disabling: {e}")

def run_sequence():
    # Retrieve user inputs for multi-point motion
    try:
        lower = float(lower_entry.get())
        upper = float(upper_entry.get())
        step = float(step_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for lower bound, upper bound, and step size.")
        return

    if lower > upper:
        messagebox.showerror("Input Error", "Lower bound must be less than or equal to Upper bound.")
        return

    # Setup motion parameters
    try:
        sp.SetVelocity(hc, 0, 500, sp.SYNCHRONOUS, True)
        sp.SetAcceleration(hc, 0, 9, sp.SYNCHRONOUS, True)
        sp.SetDeceleration(hc, 0, 90, sp.SYNCHRONOUS, True)
        sp.SetJerk(hc, 0, 100, sp.SYNCHRONOUS, True)
        sp.SetKillDeceleration(hc, 0, 1000, sp.SYNCHRONOUS, True)
        sp.SetFPosition(hc, sp.Axis.ACSC_AXIS_0, 0, failure_check=True)
    except Exception as e:
        messagebox.showerror("Motion Setup Error", f"Failed to set motion parameters: {e}")
        return

    # Start multi-point motion sequence with 1000 ms dwell time at each point
    try:
        sp.MultiPoint(
            hc,            # Communication handle
            0,             # Create the multi-point motion with default velocity
            sp.Axis.ACSC_AXIS_0,
            1000,          # Dwell time in ms
            failure_check=True
        )
    except Exception as e:
        messagebox.showerror("Error", f"Error in MultiPoint setup: {e}")
        return

    pos = lower
    try:
        # Loop from lower to upper bound (inclusive) using the specified step size
        while pos <= upper:
            sp.AddPoint(hc, sp.Axis.ACSC_AXIS_0, pos)
            # Optionally, get and print the feedback position
            fpos = sp.GetFPosition(hc, sp.Axis.ACSC_AXIS_0, failure_check=True)
            print("Feedback Position:", fpos)
            pos += step

        # Finish the multi-point sequence
        sp.EndSequence(hc, sp.Axis.ACSC_AXIS_0)
    except Exception as e:
        messagebox.showerror("Motion Error", f"Error during motion sequence: {e}")
        return

    messagebox.showinfo("Info", "Motion sequence completed successfully!")

def move_up():
    """Move the axis upward by the desired increment."""
    try:
        inc = float(increment_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid increment value!")
        return

    try:
        # Get current position and add the increment
        current_pos = sp.GetFPosition(hc, sp.Axis.ACSC_AXIS_0, failure_check=True)
        new_target = current_pos + inc
        sp.ToPoint(
            hc,                 # Communication handle
            0,                  # Start motion immediately
            sp.Axis.ACSC_AXIS_0,
            new_target,         # New absolute target position
            failure_check=True
        )
        print(f"Moving UP: from {current_pos} to {new_target}")
    except Exception as e:
        messagebox.showerror("Motion Error", f"Error in moving up: {e}")

def move_down():
    """Move the axis downward by the desired decrement."""
    try:
        inc = float(increment_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid increment value!")
        return

    try:
        # Get current position and subtract the increment
        current_pos = sp.GetFPosition(hc, sp.Axis.ACSC_AXIS_0, failure_check=True)
        new_target = current_pos - inc
        sp.ToPoint(
            hc,                 # Communication handle
            0,                  # Start motion immediately
            sp.Axis.ACSC_AXIS_0,
            new_target,         # New absolute target position
            failure_check=True
        )
        print(f"Moving DOWN: from {current_pos} to {new_target}")
    except Exception as e:
        messagebox.showerror("Motion Error", f"Error in moving down: {e}")

# Create the main window
root = tk.Tk()
root.title("Motion Sequence Control")

# --- Multi-point motion controls ---
tk.Label(root, text="Lower Bound:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
lower_entry = tk.Entry(root)
lower_entry.grid(row=0, column=1, padx=10, pady=5)
lower_entry.insert(0, "30")  # Default value

tk.Label(root, text="Upper Bound:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
upper_entry = tk.Entry(root)
upper_entry.grid(row=1, column=1, padx=10, pady=5)
upper_entry.insert(0, "200")  # Default value

tk.Label(root, text="Step Size:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
step_entry = tk.Entry(root)
step_entry.grid(row=2, column=1, padx=10, pady=5)
step_entry.insert(0, "10")  # Default value

run_button = tk.Button(root, text="Run Multi-Point Sequence", width=25, command=run_sequence)
run_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# --- Enable/Disable controls ---
enable_button = tk.Button(root, text="Enable", width=15, command=enable)
enable_button.grid(row=4, column=0, padx=10, pady=5)

disable_button = tk.Button(root, text="Disable", width=15, command=disable)
disable_button.grid(row=4, column=1, padx=10, pady=5)

# --- Up/Down motion controls ---
tk.Label(root, text="Increment Value:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
increment_entry = tk.Entry(root)
increment_entry.grid(row=5, column=1, padx=10, pady=5)
increment_entry.insert(0, "100")  # Default increment value

up_button = tk.Button(root, text="▲ Up", width=15, command=move_up)
up_button.grid(row=6, column=0, padx=10, pady=5)

down_button = tk.Button(root, text="▼ Down", width=15, command=move_down)
down_button.grid(row=6, column=1, padx=10, pady=5)
root.bind("<Up>", lambda event: move_up())    # Keyboard Up Arrow → Move Up
root.bind("<Down>", lambda event: move_down())  # Keyboard Down Arrow → Move Down
# Start the GUI event loop
root.mainloop()
