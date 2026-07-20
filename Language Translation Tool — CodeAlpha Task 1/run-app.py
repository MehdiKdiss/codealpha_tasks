"""
Double-click this file to launch the Language Translation Tool.
A browser tab opens automatically. Close the console window (or Ctrl+C
inside it) when you're done to stop the app.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "app.py"


def main():
    if not APP_PATH.exists():
        print(f"ERROR: could not find {APP_PATH}")
        input("\nPress Enter to exit...")
        return

    print("Starting Language Translation Tool...")
    print("A browser tab should open automatically in a few seconds.")
    print("Keep this window open while using the app. Close it to stop.\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(APP_PATH)],
            cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError:
        print("ERROR: streamlit doesn't seem to be installed.")
        print("Run this once first: pip install -r requirements.txt")
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()