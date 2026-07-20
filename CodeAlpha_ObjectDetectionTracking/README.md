# 🎯 Medical PPE Detection and Tracking — CodeAlpha Task 4

Real-time detection and tracking of medical Personal Protective Equipment (masks, gloves, coveralls, goggles, face shields) from a webcam or video feed — a YOLOv8 model fine-tuned specifically for healthcare-setting PPE compliance monitoring, built as part of the CodeAlpha Artificial Intelligence Internship.

## Why this over generic object detection

The base task only requires detecting *some* pretrained object classes with tracking. Rather than running stock YOLOv8 on generic COCO classes (people, cars, etc.), this project fine-tunes YOLOv8 on a medical PPE dataset — directly relevant to a biomedical engineering background, and a genuinely useful real-world application: automated PPE compliance monitoring in a clinical or lab setting, where *tracking* (not just single-frame detection) matters because compliance needs to be monitored continuously as people move through a space.

## How it works

- **Detection:** YOLOv8n, fine-tuned from COCO-pretrained weights on a medical PPE dataset (classes such as Mask, Gloves, Coverall, Goggles, Face Shield — exact classes depend on the dataset used, see `dataset/data.yaml` after download).
- **Tracking:** [ByteTrack](https://github.com/ifzhang/ByteTrack), via `ultralytics`' built-in `.track()` API — assigns and maintains a consistent ID per detected person/item across frames.
- **Display:** a live OpenCV window shows bounding boxes, class labels, confidence scores, and tracking IDs in real time.
- **Fallback behavior:** if fine-tuning hasn't been run yet, `main.py` automatically falls back to the stock COCO-pretrained model, so the detection/tracking pipeline is always testable even before the custom dataset step is done.

## Setup

```bash
pip install -r requirements.txt
```

### Getting the dataset

1. Go to [universe.roboflow.com](https://universe.roboflow.com) and search **"medical ppe detection"**.
2. Pick a dataset with classes like Mask, Gloves, Coverall, Goggles, Face Shield.
3. Download in **YOLOv8** format, as a plain zip file (no API key needed).
4. Extract it into a folder named `dataset/` in this project root (see structure below).

```
CodeAlpha_ObjectDetectionTracking/
├── dataset/
│   ├── data.yaml
│   ├── train/images, train/labels
│   ├── valid/images, valid/labels
│   └── test/images, test/labels   (if included)
├── main.py
├── fine_tune.py
```

### Fine-tuning

```bash
python fine_tune.py
```
Fine-tunes from COCO-pretrained weights (50 epochs, early stopping patience 10) rather than training from scratch — needs far less data and time to reach good accuracy, since the model already understands general object shapes/edges/textures before ever seeing a PPE image.

## Usage

Double-click `main.py` (or run `python main.py`). A dialog asks whether to use your webcam or a video file, then a live window shows the annotated detection + tracking feed. Press `q` or close the window to stop.

If `fine_tune.py` hasn't been run yet, `main.py` automatically uses the stock COCO model instead — useful for testing the pipeline mechanics before the dataset/training step.

## Design notes

- Fine-tuning from pretrained COCO weights (transfer learning) rather than training from scratch — appropriate given typical PPE dataset sizes (hundreds to low thousands of images, versus the millions needed to train a detector from random initialization).
- `persist=True` in tracking keeps the same ID assigned to a person/object across frames rather than reassigning a new ID every frame.
- `conf=0.4` filters low-confidence detections; adjustable in `main.py`.

## Results

Fine-tuned for 50 epochs (~12 minutes on an RTX 4060) from COCO-pretrained YOLOv8n weights, on 1,788 training / 224 validation images:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Coverall | 0.908 | 0.877 | 0.922 | 0.655 |
| Gloves | 0.828 | 0.556 | 0.711 | 0.466 |
| Goggles | 0.824 | 0.832 | 0.838 | 0.586 |
| Mask | 0.803 | 0.774 | 0.800 | 0.467 |
| **Overall** | **0.841** | **0.760** | **0.818** | **0.544** |

An overall mAP50 of 0.818 is a solid result for a nano-sized model (3M parameters, the smallest/fastest in the YOLOv8 family), a modest dataset size, and ~12 minutes of training.

**Known limitation — "Gloves" is the weakest class.** Recall of 0.556 means the model misses roughly 45% of real gloves in validation images, clearly below the other three classes. Likely causes: gloves are small, frequently only partially visible, and sometimes close in color to bare skin or the background — combined with the dataset containing many near-duplicate frames sampled from the same few source videos (see note below), which reduces the effective visual diversity the model actually learned from despite the raw image count. This is an honest limitation worth stating rather than hiding, in the same spirit as the MAESTRO timing-quantization finding documented for the music generation task.

**Dataset composition note:** a meaningful portion of the training images are consecutive frames extracted from a small number of source videos (same person, same setting, same lighting), rather than fully independent photos. This likely biases the model toward those specific conditions to some degree — worth validating against genuinely independent test footage rather than assuming the raw image count reflects true diversity.

## Task Requirement Compliance

| CodeAlpha requirement | How it's satisfied |
|---|---|
| Real-time video input via webcam or file (OpenCV) | `model.track(source=...)`, where `source` is either webcam index `0` or a user-picked file path; Ultralytics reads frames via OpenCV internally |
| Pretrained model (YOLO or Faster R-CNN) | YOLOv8n, fine-tuned from COCO-pretrained weights on the medical PPE dataset |
| Per-frame processing + bounding boxes | Built into `.track(show=True)` — boxes drawn automatically every frame |
| Tracking algorithm "like SORT or Deep SORT" | **BoT-SORT** — same core association strategy as SORT (Kalman filter motion prediction + Hungarian algorithm assignment), plus camera-motion compensation; a direct technical descendant of SORT, not an unrelated algorithm |
| Labels + tracking IDs shown live | Built into `.track(show=True)` — class label, confidence, and a persistent tracking ID are overlaid on every detected object in the live window |

---

## Credits

Built as part of the [CodeAlpha](https://www.codealpha.tech) Artificial Intelligence Internship — Task 4: Object Detection and Tracking.