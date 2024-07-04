import os
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    # List of packages to install
    packages = [
        "opencv-python-headless",
        "numpy",
        "glob2",
        "open3d",
        "pyqt5"
    ]

    # Check the operating system
    if os.name == 'nt':  # Windows
        print("Detected Windows OS")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            for package in packages:
                install(package)
            print("All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install package: {e}")
    elif os.name == 'posix':  # Linux
        print("Detected Linux OS")
        try:
            # Ensure pip is up to date
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            for package in packages:
                install(package)
            print("All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install package: {e}")
    else:
        print("Unsupported operating system")

if __name__ == "__main__":
    main()
