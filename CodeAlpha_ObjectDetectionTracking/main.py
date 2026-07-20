"""
CodeAlpha Task 4 — Object Detection and Tracking

Double-click this file to run it. You'll be asked to choose your webcam
or a video file, then a live window opens showing real-time object
detection with bounding boxes, class labels, and tracking IDs.

Press 'q' in that window, or just close it, to stop.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
# Ultralytics can produce either path depending on version/settings quirks —
# check both, use whichever actually exists.
FINE_TUNED_CANDIDATES = [
    PROJECT_ROOT / "runs" / "detect" / "ppe_finetune" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "ppe_finetune" / "weights" / "best.pt",
]


def choose_source():
    root = tk.Tk()
    root.withdraw()  # hide the empty main tkinter window, we only want the dialogs

    use_webcam = messagebox.askyesno(
        "Choose video source",
        "Use your webcam?\n\nYes = webcam\nNo = pick a video file from your computer",
    )

    if use_webcam:
        root.destroy()
        return 0  # default webcam device index

    file_path = filedialog.askopenfilename(
        title="Choose a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )
    root.destroy()

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    return file_path


def main():
    import cv2

    fine_tuned_path = next((p for p in FINE_TUNED_CANDIDATES if p.exists()), None)

    if fine_tuned_path is not None:
        print(f"Loading fine-tuned PPE detection model: {fine_tuned_path}")
        model = YOLO(str(fine_tuned_path))
    else:
        print("No fine-tuned model found — using stock YOLOv8n (COCO classes).")
        print(f"(Run fine_tune.py first to enable PPE-specific detection. Checked: {FINE_TUNED_CANDIDATES})")
        model = YOLO("yolov8n.pt")

    source = choose_source()

    # Only flip webcam output — video files should play in their original orientation.
    is_webcam = source == 0

    print(f"Starting detection + tracking on: {source}")
    if is_webcam:
        print("Webcam mode: output will be horizontally flipped (mirror correction).")
    print("A window will open showing the live feed. Press 'q' in that window, or close it, to stop.\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source}")
        sys.exit(1)

    window_name = "Object Detection & Tracking - press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fix mirrored webcam: flip horizontally so left/right match reality.
        if is_webcam:
            frame = cv2.flip(frame, 1)

        # Run detection + tracking on this single frame.
        # persist=True keeps the same tracking IDs across frames within this session.
        # show=False prevents YOLO from opening its own second window.
        results = model.track(
            source=frame,
            tracker="botsort.yaml",  # BoT-SORT: Kalman filter + Hungarian assignment,
                                      # the same core association strategy as the original
                                      # SORT algorithm, plus camera-motion compensation on top
            persist=True,             # keeps the same tracking ID for the same object across frames
            conf=0.4,                 # confidence threshold — filters out low-confidence junk detections
            show=False,               # we render manually below — prevents a duplicate window
            verbose=False,            # suppress per-frame console output
        )

        # Render bounding boxes, labels, and tracking IDs onto the frame.
        annotated = results[0].plot()

        cv2.imshow(window_name, annotated)

        # Exit on 'q' keypress or window close.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        input("\nPress Enter to exit...")