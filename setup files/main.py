import importlib
import math
import threading
import time
from dnx64 import DNX64
import cv2
import numpy as np
import os

# Global variables
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480  # Increased window size
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 960, 30
DNX64_PATH = "D:\\dnx64_python\\dnx64_python\\DNX64.dll"
DEVICE_INDEX = 0
CAM_INDEX = 1  # Camera index, please change it if you have more than one camera
QUERY_TIME = 1
COMMAND_TIME = 1

# Global variable for last captured image path
last_captured_image = None


def clear_line(n=1):
    LINE_CLEAR = "\x1b[2K"
    for i in range(n):
        print("", end=LINE_CLEAR)


def threaded(func):
    """Wrapper to run a function in a separate thread with @threaded decorator"""

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()

    return wrapper


def custom_microtouch_function():
    """Executes when MicroTouch press event got detected"""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clear_line(1)
    print(f"{timestamp} MicroTouch press detected!", end="\r")


def capture_image(frame):
    """Capture an image and save it in the current working directory."""

    global last_captured_image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    cv2.imwrite(filename, frame)
    last_captured_image = filename  # Save the path of the last captured image
    clear_line(1)
    print(f"Saved image to {filename}", end="\r")


def start_recording(frame_width, frame_height, fps):
    """Start recording video and return the video writer object."""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"video_{timestamp}.avi"
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
    clear_line(1)
    print(f"Video recording started: {filename}. Press r to stop.", end="\r")
    return video_writer


def stop_recording(video_writer):
    """Stop recording video and release the video writer object."""

    video_writer.release()
    clear_line(1)
    print("Video recording stopped", end="\r")


def initialize_camera():
    camera = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("m", "j", "p", "g"))
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return camera


def process_frame(frame):
    return cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))


def init_microscope(microscope):
    microscope.SetVideoDeviceIndex(DEVICE_INDEX)
    time.sleep(0.1)
    microscope.EnableMicroTouch(True)
    time.sleep(0.1)
    microscope.SetEventCallback(custom_microtouch_function)
    time.sleep(0.1)

    return microscope


def print_keymaps():
    print(
        "Press the key below prompts to continue \n \
        0:Led off \n \
        1:AMR \n \
        2:Flash_leds and On \n \
        c:List config \n \
        d:Show device id \n \
        f:Show fov \n \
        r:Record video or Stop Record video \n \
        s:Capture image \n \
        6:Set EFLC Quadrant 1 to flash \n \
        7:Set EFLC Quadrant 2 to flash \n \
        8:Set EFLC Quadrant 3 to flash \n \
        9:Set EFLC Quadrant 4 to flash \n \
        Esc:Quit \
        "
    )


def config_keymaps(microscope, frame):
    key = cv2.waitKey(1) & 0xFF

    if key == ord("0"):
        led_off(microscope)
    if key == ord("1"):
        print_amr(microscope)
    if key == ord("2"):
        flash_leds(microscope)
    if key == ord("c"):
        print_config(microscope)
    if key == ord("d"):
        print_deviceid(microscope)
    if key == ord("f"):
        print_fov_mm(microscope)
    if key == ord("s"):
        capture_image(frame)
    if key == ord("6"):
        microscope.SetEFLC(DEVICE_INDEX, 1, 32)
        time.sleep(0.1)
        microscope.SetEFLC(DEVICE_INDEX, 1, 31)
    if key == ord("7"):
        microscope.SetEFLC(DEVICE_INDEX, 2, 32)
        time.sleep(0.1)
        microscope.SetEFLC(DEVICE_INDEX, 2, 15)
    if key == ord("8"):
        microscope.SetEFLC(DEVICE_INDEX, 3, 32)
        time.sleep(0.1)
        microscope.SetEFLC(DEVICE_INDEX, 3, 15)
    if key == ord("9"):
        microscope.SetEFLC(DEVICE_INDEX, 4, 32)
        time.sleep(0.1)
        microscope.SetEFLC(DEVICE_INDEX, 4, 31)

    return key


def start_camera(microscope):
    """Starts camera, initializes variables for video preview, and listens for shortcut keys."""

    camera = initialize_camera()

    if not camera.isOpened():
        print("Error opening the camera device.")
        return

    recording = False
    video_writer = None
    inits = True

    print_keymaps()

    while True:
        ret, frame = camera.read()
        if ret:
            resized_frame = process_frame(frame)

            # Show live feed and last captured image side by side
            if last_captured_image:
                last_img = cv2.imread(last_captured_image)
                last_img_resized = cv2.resize(last_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
                # Concatenate the live feed and last captured image horizontally
                combined_frame = cv2.hconcat([resized_frame, last_img_resized])
            else:
                combined_frame = resized_frame

            # Show the combined frame
            cv2.imshow("Live Feed and Last Captured Image", combined_frame)

            if recording:
                video_writer.write(frame)
            if inits:
                microscope = init_microscope(microscope)
                inits = False

        key = config_keymaps(microscope, frame)

        if key == ord("r") and not recording:
            recording = True
            video_writer = start_recording(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)

        elif key == ord("r") and recording:
            recording = False
            stop_recording(video_writer)

        if key == 27:
            clear_line(1)
            break

    if video_writer is not None:
        video_writer.release()
    camera.release()
    cv2.destroyAllWindows()


def run_usb():
    # Initialize microscope
    micro_scope = DNX64(DNX64_PATH)
    start_camera(micro_scope)


if __name__ == "__main__":
    run_usb()
