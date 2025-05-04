import SPiiPlusPython as sp
import tkinter as tk
import global_vars

# frame object
ptp_frame = tk.Frame()
axis_var = tk.StringVar()
fpos_var = tk.StringVar()
target_pos_var = tk.StringVar()
enable_button = tk.Button()
disable_button = tk.Button()
ptp_button = tk.Button()


# fpos textbox update
def fpos_update():
    global axis_var
    global fpos_var
    axis = int(axis_var.get())
    with global_vars.hc_lock:
        if global_vars.hc >= 0:
            fpos = sp.GetFPosition(global_vars.hc, axis, sp.SYNCHRONOUS, True)
            fpos_var.set(fpos)


# update status frame
def update_axis_status_frame():
    global canvas, enable_id0, moving_id0
    global ptp_frame
    global enable_button
    global disable_button
    global axis_var

    # create status frame
    axis = int(axis_var.get())
    # check connection
    with global_vars.hc_lock:
        if global_vars.hc >= 0:
            mst = sp.GetMotorState(global_vars.hc, axis, sp.SYNCHRONOUS, True)
            # check #ENABLED bit
            if bin(mst)[-1] == '1':
                canvas.itemconfig(enable_id0, fill='#56e393')
                enable_button.place_forget()
                disable_button.place(x=10, y=50)
                ptp_button.config(state='active')
            else:
                canvas.itemconfig(enable_id0, fill='#cf2155')
                disable_button.place_forget()
                enable_button.place(x=10, y=50)
                ptp_button.config(state='disabled')

            # check #MOVE bit
            if bin(mst)[-6] == '1':
                canvas.itemconfig(moving_id0, fill="#56e393")
            else:
                canvas.itemconfig(moving_id0, fill="#cf2155")


# ptp button action
def ptp_action():
    global axis_var
    global target_pos_var
    with global_vars.hc_lock:
        if global_vars.hc >= 0:
            axis = int(axis_var.get())
            target_pos = int(target_pos_var.get())
            sp.ToPoint(global_vars.hc, 0, axis, target_pos, sp.SYNCHRONOUS, True)
            print(f"Axis {axis} started movement to point {target_pos}")


# Enable button action
def enable_action():
    global enable_button
    global disable_button
    global axis_var
    with global_vars.hc_lock:
        if global_vars.hc >= 0:
            axis = int(axis_var.get())
            sp.Enable(global_vars.hc, axis, sp.SYNCHRONOUS, True)
            enable_button.place_forget()
            disable_button.place(x=10, y=50)
            print(f"Axis {axis} has been enabled")


# Disable button action
def disable_action():
    global enable_button
    global disable_button
    global axis_var
    with global_vars.hc_lock:
        if global_vars.hc >= 0:
            axis = int(axis_var.get())
            sp.Disable(global_vars.hc, axis, sp.SYNCHRONOUS, True)
            disable_button.place_forget()
            enable_button.place(x=10, y=50)
            print(f"Axis {axis} has been disabled")


# After controller connection, this updates ptp frame in every constant time cycle
def update_canvas():
    global ptp_frame
    global axis_var
    global fpos_var
    if global_vars.connection_bool:
        update_axis_status_frame()
        fpos_update()

    global_vars.app.after(200, update_canvas)


def init_ptp_frame():
    global canvas, enable_id0, moving_id0
    global ptp_frame
    global axis_var
    global fpos_var
    global enable_button
    global disable_button
    global ptp_button
    global target_pos_var
    # create and place ptp frame in main window
    ptp_frame = tk.LabelFrame(global_vars.app, text="Point to point motion", width=400, height=150, padx=3, pady=3)
    ptp_frame.place(x=30, y=140)

    status_frame = tk.LabelFrame(ptp_frame, text="Status", width=110, height=80, padx=3, pady=3)
    status_frame.place(x=120, y=35)

    # Add enabled disabled labels
    enabled_label = tk.Label(status_frame, text="Enabled")
    enabled_label.place(x=0, y=5)

    disabled_label = tk.Label(status_frame, text="Moving")
    disabled_label.place(x=0, y=30)

    # create canvas to signal lights
    canvas = tk.Canvas(status_frame, width=30, height=50)
    canvas.place(x=60, y=0)

    enable_id0 = canvas.create_oval(5, 5, 25, 25, fill="gray")
    moving_id0 = canvas.create_oval(5, 30, 25, 50, fill="gray")

    # create dropdown to select axis
    options = ["0", "1", "2", '3', '4', '5', '6', '7']
    axis_var = tk.StringVar(ptp_frame)
    dropdown = tk.OptionMenu(ptp_frame, axis_var, *options)
    dropdown_label = tk.Label(ptp_frame, text="Axes")

    # target ptp label and textbox
    target_ptp_label = tk.Label(ptp_frame, text="Target point", wraplength=55)
    target_pos_var = tk.StringVar(value="0")
    remote_entry = tk.Entry(ptp_frame, font=('Lato', 10), justify='center', width=7, bd=3, textvariable=target_pos_var)

    # fpos label and textbox
    fpos_label = tk.Label(ptp_frame, text="Feedback position", wraplength=55)
    fpos_var = tk.StringVar(value="0")
    fpos_entry = tk.Entry(ptp_frame, font=('Lato', 10), justify='center', width=7, bd=3, textvariable=fpos_var,
                          state='disabled')

    # connect, disconnect and ptp buttons
    enable_button = tk.Button(ptp_frame, text="Enable", width=10)
    disable_button = tk.Button(ptp_frame, text="Disable", width=10)
    ptp_button = tk.Button(ptp_frame, text="Start PTP", width=10)

    # set action command for buttons
    enable_button.config(command=enable_action)
    disable_button.config(command=disable_action)
    ptp_button.config(command=ptp_action)

    # set default for axis dropdown
    axis_var.set(options[0])

    # place object in the frame
    dropdown.place(x=200, y=0)
    dropdown_label.place(x=160, y=5)
    target_ptp_label.place(x=250, y=40)
    remote_entry.place(x=250, y=80)
    fpos_label.place(x=320, y=40)
    fpos_entry.place(x=320, y=80)
    enable_button.place(x=10, y=50)
    ptp_button.place(x=10, y=85)

    update_axis_status_frame()
    update_canvas()

