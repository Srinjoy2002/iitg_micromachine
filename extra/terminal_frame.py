from functools import partial
import tkinter as tk
import tkinter.scrolledtext as tk_scroll
import global_vars
import SPiiPlusPython as sp


def send_terminal_command(command: tk.StringVar, terminal_output: tk.Text):
    with global_vars.hc_lock:
        if global_vars.hc >= 0:
            try:
                command_text = command.get()
                reply = sp.Transaction(global_vars.hc, command_text+'\r', len(command_text)+1, 1024, sp.SYNCHRONOUS, True)
                terminal_output.config(state="normal")
                terminal_output.insert('end', reply+'\n')
                terminal_output.see(tk.END)
                terminal_output.config(state="disabled")

            except:
                pass


def init_terminal_frame():
    # terminal frame creation
    terminal_frame = tk.LabelFrame(global_vars.app, text="Communication Terminal", width=400, height=200, padx=3,
                                   pady=3)
    terminal_frame.place(x=30, y=300)

    # terminal input textbox
    terminal_input_text = tk.StringVar(value="")
    terminal_input_entry = tk.Entry(terminal_frame, font=('Lato', 10), justify='left',  width=45, bd=3,
                                    textvariable=terminal_input_text)
    terminal_input_entry.place(x=0, y=3)

    # terminal output
    terminal_output = tk_scroll.ScrolledText(terminal_frame, state='disabled', width=45, height=8, bd=3)
    terminal_output.place(x=0, y=32)

    # send button
    send_button = tk.Button(terminal_frame, text="Send", width=7)
    send_button.place(x=325, y=0)
    send_button.config(command=partial(send_terminal_command, terminal_input_text, terminal_output))

