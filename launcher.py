"""
launcher.py - EXE Launcher for UHF-ECG Activation Mapper
This script is compiled to EXE via PyInstaller.
It starts the Streamlit server and opens the browser.
"""
import subprocess
import sys
import os
import webbrowser
import time
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def main():
    # Get the directory where this exe/script is located
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    app_path = os.path.join(base_dir, 'app.py')
    
    if not os.path.exists(app_path):
        print(f"ERROR: app.py not found at {app_path}")
        input("Press Enter to exit...")
        return
    
    port = 8501
    url = f"http://localhost:{port}"
    
    print("=" * 60)
    print("  UHF-ECG Ventricular Activation Mapper")
    print("  Research & Education Tool")
    print("=" * 60)
    print()
    print(f"  Starting server on {url}")
    print("  The browser will open automatically.")
    print()
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)
    
    # Find streamlit executable
    streamlit_cmd = 'streamlit'
    
    # Try to find streamlit in PATH or Scripts
    python_dir = os.path.dirname(sys.executable)
    scripts_dir = os.path.join(python_dir, 'Scripts')
    if os.path.exists(os.path.join(scripts_dir, 'streamlit.exe')):
        streamlit_cmd = os.path.join(scripts_dir, 'streamlit.exe')
    
    # Open browser after a delay
    def open_browser():
        time.sleep(3)
        webbrowser.open(url)
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start streamlit
    try:
        process = subprocess.run(
            [streamlit_cmd, 'run', app_path,
             '--server.headless', 'true',
             '--server.port', str(port),
             '--server.address', 'localhost',
             '--browser.gatherUsageStats', 'false',
             '--theme.base', 'dark',
             '--theme.primaryColor', '#FF6B6B',
             '--theme.backgroundColor', '#0a0a2e',
             '--theme.secondaryBackgroundColor', '#12122e',
             '--theme.textColor', '#e0e0ff'],
            cwd=base_dir
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except FileNotFoundError:
        print(f"\nERROR: Streamlit not found. Please install it:")
        print(f"  pip install streamlit")
        input("Press Enter to exit...")

if __name__ == '__main__':
    main()
