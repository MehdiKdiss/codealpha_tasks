"""
Double-click this file to launch the AI Piano Music Generator.

A browser tab will open automatically after a few seconds. A console
window will stay open while the app is running — that's normal, it's
just showing Streamlit's log output. Close that window (or press Ctrl+C
inside it) when you're done to stop the app.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "src" / "app.py"


def main():
    if not APP_PATH.exists():
        print(f"ERROR: could not find {APP_PATH}")
        print("Make sure this file is sitting in your project's root folder,")
        print("next to the 'src' folder.")
        input("\nPress Enter to exit...")
        return

    print("Starting AI Piano Music Generator...")
    print("A browser tab should open automatically in a few seconds.")
    print("Keep this window open while using the app. Close it to stop.\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(APP_PATH)],
            cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError:
        print("ERROR: streamlit doesn't seem to be installed.")
        print("Run this once first: pip install streamlit")
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()