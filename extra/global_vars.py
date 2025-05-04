import tkinter as tk
import threading


# window variable
app = tk.Tk()
# handle variable
hc = -1
# Show whether the connection has been established
connection_bool = False
open_threads = []

hc_lock = threading.Lock()


def config_app():
    global app
    app.geometry("450x550")
    app.title("SPiiPlusPython Library Example")

