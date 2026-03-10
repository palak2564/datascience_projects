"""
Test Script - Verify Installation
This script checks if all required libraries are installed correctly
"""

import sys

def check_library(library_name, import_name=None):
    """Check if a library is installed"""
    if import_name is None:
        import_name = library_name
    
    try:
        __import__(import_name)
        print(f"✓ {library_name} is installed")
        return True
    except ImportError:
        print(f"✗ {library_name} is NOT installed")
        return False

def main():
    """Check all required libraries"""
    print("=" * 50)
    print("Checking Required Libraries")
    print("=" * 50)
    
    libraries = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pillow", "PIL"),
        ("Tkinter", "tkinter")
    ]
    
    all_installed = True
    
    for lib_name, import_name in libraries:
        if not check_library(lib_name, import_name):
            all_installed = False
    
    print("=" * 50)
    
    if all_installed:
        print("✓ All libraries are installed!")
        print("You can now run: python main_app.py")
    else:
        print("✗ Some libraries are missing")
        print("Please run: pip install -r requirements.txt")
    
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"\nPython Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major >= 3 and python_version.minor >= 7:
        print("✓ Python version is compatible")
    else:
        print("✗ Please use Python 3.7 or higher")

if __name__ == "__main__":
    main()
