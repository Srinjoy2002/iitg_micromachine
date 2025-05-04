import global_vars
from com_frame import init_com_frame, check_connection
from ptp_frame import init_ptp_frame, update_canvas
from terminal_frame import init_terminal_frame


global_vars.config_app()
init_com_frame()
init_ptp_frame()
init_terminal_frame()

# update gui after constant time
global_vars.app.after(200, check_connection)
global_vars.app.after(200, update_canvas)

# Start the GUI event loop
global_vars.app.mainloop()

