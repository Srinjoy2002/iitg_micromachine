import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def main():
    with open('requirements.txt') as f:
        packages = f.readlines()
    
    # Remove whitespace characters and empty lines
    packages = [pkg.strip() for pkg in packages if pkg.strip()]
    
    for package in packages:
        try:
            install(package)
            print(f'Successfully installed {package}')
        except Exception as e:
            print(f'Error installing {package}: {e}')

if __name__ == "__main__":
    main()
