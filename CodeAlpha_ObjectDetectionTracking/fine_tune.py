"""
CodeAlpha Task 4 — Fine-tune YOLOv8 on a custom medical/biomedical PPE dataset.

Before running this:
1. Download a medical PPE detection dataset from Roboflow Universe in
   YOLOv8 format (see README.md for exact steps).
2. Extract it into a folder named `dataset/` in this project, so it looks like:
       dataset/
         data.yaml
         train/images, train/labels
         valid/images, valid/labels
         test/images,  test/labels   (optional)

This fine-tunes a COCO-pretrained YOLOv8n on that dataset, starting from
general object knowledge rather than from scratch, and saves the result to
runs/detect/ppe_finetune/weights/best.pt.

Double-click this file to run it, or `python fine_tune.py`.
"""

from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"


def main():
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found.")
        print("Download a dataset from Roboflow Universe first — see README.md.")
        input("\nPress Enter to exit...")
        return

    print(f"Fine-tuning YOLOv8n on: {DATA_YAML}")
    print("This starts from COCO-pretrained weights (general object knowledge)")
    print("rather than training from scratch, which needs far less data and time.\n")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,          # stop early if validation performance plateaus
        project=str(PROJECT_ROOT / "runs" / "detect"),  # absolute path avoids
                                                          # a duplicated-folder bug seen
                                                          # with relative paths in some
                                                          # Ultralytics versions
        name="ppe_finetune",
        exist_ok=True,
    )

    best_weights = PROJECT_ROOT / "runs" / "detect" / "ppe_finetune" / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"Best weights saved to: {best_weights}")
    print("main.py will automatically use these weights the next time you run it.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        input("\nPress Enter to exit...")