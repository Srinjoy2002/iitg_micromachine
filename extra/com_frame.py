import SPiiPlusPython as sp
import tkinter as tk
import global_vars
from functools import partial


# connection variable
ip = None
canvas = tk.Canvas()


# controller connection function
def connect(remote_entry, dropdown_var, connect_button, disconnect_button, canvas, version_text):
    global ip

    ip = remote_entry.get()
    with global_vars.hc_lock:
        # Attempting to establish a TCP connection
        if dropdown_var.get() == "TCP":
            global_vars.hc = sp.OpenCommEthernetTCP(ip, 701)

        # Attempting to establish a UDP connection
        elif dropdown_var.get() == "UDP":
            global_vars.hc = sp.OpenCommEthernetUDP(ip, 700)

        # Attempting to establish a Simulator connection
        elif dropdown_var.get() == "Simulator":
            global_vars.hc = sp.OpenCommSimulator()

        if global_vars.hc != -1:
            global_vars.connection_bool = True
            connect_button.place_forget()
            disconnect_button.place(x=60, y=60)

            version_hex = sp.GetLibraryVersion()
            major = (version_hex >> 24) & 0xFF
            minor = (version_hex >> 16) & 0xFF
            build = (version_hex >> 8) & 0xFF
            revision = version_hex & 0xFF
            version_text.set(f"{major}.{minor}.{build}.{revision}")

            print(f"Successfully connected to ip: {ip}")

        else:
            print(f"Failed to connect to ip: {ip}")

    check_connection()


# controller disconnection function
def disconnect(disconnect_button, connect_button, canvas, version_text):
    global ip
    global_vars.connection_bool = False
    with global_vars.hc_lock:
        if global_vars.hc >= 0:
            sp.CloseComm(global_vars.hc, True)
            global_vars.hc = -1

    try:
        version_text.set(f"")
        draw_connection_circle(canvas)
        disconnect_button.place_forget()
        connect_button.place(x=60, y=60)
        print(f"disconnected from ip: {ip}")
    except Exception as e:
        print(f"Error occurred when trying to disconnect handle")
        print(e)


# Function to show text input when "Network" is selected
def show_entry(option, remote_entry, address_label):

    if option == "TCP" or option == "UDP":
        remote_entry.place(x=280, y=5)
        address_label.place(x=210, y=0)
    else:
        address_label.place_forget()
        remote_entry.place_forget()


# draw connection circle in green if the connection to controller is on, otherwise in red
def draw_connection_circle(canvas):
    canvas.delete("all")
    canvas.update()
    x1 = 4
    y1 = 4
    x2 = canvas.winfo_width()-4
    y2 = canvas.winfo_height()-4

    with global_vars.hc_lock:
        if global_vars.hc >= 0:
            connection_info = sp.GetConnectionInfo(global_vars.hc, True)
            if connection_info.EthernetIP == '' or connection_info.EthernetPort == 0 or\
                    connection_info.EthernetProtocol == 0:
                # Draw the light green circle
                canvas.create_oval(x1, y1, x2, y2, fill="#cf2155")
                # Write "On" inside the circle
                canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="Off", justify='center', fill="black",
                                   font=("Arial", 10, "bold"))
            else:
                # Draw the light green circle
                canvas.create_oval(x1, y1, x2, y2, fill="#56e393")
                # Write "On" inside the circle
                canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="On", justify='center', fill="black",
                                   font=("Arial", 10, "bold"))
        else:
            # Draw the light green circle
            canvas.create_oval(x1, y1, x2, y2, fill="#cf2155")
            # Write "On" inside the circle
            canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="Off", justify='center', fill="black",
                               font=("Arial", 10, "bold"))


# separate thread that will check connection every 2 seconds
def check_connection():
    global canvas
    if global_vars.connection_bool:
        draw_connection_circle(canvas)
    global_vars.app.after(200, check_connection)


def init_com_frame():
    global canvas
    com_frame = tk.LabelFrame(global_vars.app, text="Communication", width=400, height=120, padx=3, pady=3)
    com_frame.place(x=30, y=10)

    # Text input box for ip
    remote_text = tk.StringVar(value="10.0.0.100")
    remote_entry = tk.Entry(com_frame, font=('Lato', 10), justify='center',  width=12, bd=5, textvariable=remote_text)
    address_label = tk.Label(com_frame, text="Controller Address", wraplength=55)

    # Connect button
    connect_button = tk.Button(com_frame, text="Connect")
    connect_button.place(x=60, y=60)

    # Disconnect button
    disconnect_button = tk.Button(com_frame, text="Disconnect")

    canvas = tk.Canvas(com_frame, width=40, height=40)
    canvas.place(x=150, y=50)

    # Draw the circle initially
    draw_connection_circle(canvas)

    # Dropdown menu options
    options = ["TCP", "UDP", "Simulator"]
    # Variable to hold selected option
    dropdown_var = tk.StringVar(com_frame)
    # Dropdown menu
    dropdown = tk.OptionMenu(com_frame, dropdown_var, *options, command=partial(
        show_entry, remote_entry=remote_entry, address_label=address_label))
    # Dropdown label
    dropdown_label = tk.Label(com_frame, text="Connection Type")

    dropdown.place(x=120, y=0)
    dropdown_label.place(x=20, y=5)

    version_label = tk.Label(com_frame, text="C-Library Version", wraplength=55)
    version_text = tk.StringVar(value="")
    version_label_output = tk.Entry(com_frame, font=('Lato', 10), justify='center', width=12, bd=5,
                                    textvariable=version_text, state='disabled')
    version_label.place(x=210, y=55)
    version_label_output.place(x=280, y=60)

    connect_button.config(command=partial(connect, remote_entry, dropdown_var, connect_button, disconnect_button,
                                          canvas, version_text))
    disconnect_button.config(command=partial(disconnect, disconnect_button, connect_button, canvas, version_text))

    # default value for dropdown
    dropdown_var.set(options[0])
    show_entry(options[0], remote_entry, address_label)
