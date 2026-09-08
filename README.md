# Staff Detection

Custom YOLO26n staff detection and ByteTrack tracking for overhead video. The prototype identifies frames with returned staff tracks and records bounding-box centre coordinates.

## Documentation

[Read the two-page solution](output/pdf/staff_detection_solution.pdf) for the approach, training setup, annotated example, validation results and limitations.

## Setup and inference

Run commands from the repository root. A compatible Python/PyTorch environment is required; the scripts currently select NVIDIA GPU device 0.

```powershell
python -m venv yolo
.\yolo\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Before running:

1. Place your test video at `input_video/sample.mp4` (create the directory if needed).
2. Update the hardcoded project, model and video paths in the script you want to run. The current paths refer to `D:\Learning\Project\Yolo`.
3. If CUDA is unavailable, change `device=0` to `device="cpu"` in the relevant script.

The trained checkpoint is included at `yolov26n_staff_2026-09-09_04-32-58_e50_b4/weights/best.pt`.

```powershell
python tracking.py
```

This writes annotated images to `frame/` and positive-frame records to `frame/staff_frame.txt`. It overwrites the log and matching image filenames. Frame numbering starts at 1. XY coordinates are integer bounding-box centres in pixels, measured from the top-left of the image. Frames with no returned tracking IDs are omitted from the log. A tracking ID is not independent proof that a badge is present.

For an annotated prediction video:

```powershell
python test_inteference.py
```

The filename above matches the current script. Ultralytics saves the video under `runs/detect/predict*`.

## Project files

| File | Purpose |
| --- | --- |
| `staff_detector.py` | Fine-tune pretrained YOLO26n |
| `tracking.py` | Track staff and save frame/ID/XY output |
| `custom_tracker.yaml` | ByteTrack settings |
| `test_inteference.py` | Predict and save annotated video |
| `model_export.py` | Export best.pt to ONNX |
| `dataset_create.py` | Extract every fifth video frame |
| `dataset_split.py` | Randomly split source images 80/20 |
| `parse_ndjson.py` | Convert split NDJSON annotations to YOLO labels |
| `data.yaml` | Single Staff class and dataset paths |

The video, labelled dataset, environment, original evaluation brief and generated frame/video outputs are not included. Obtain or prepare your own authorised inputs.

## Training

The saved run used 50 epochs, batch size 4 and image size 640. Prepare `dataset/train/images`, `dataset/train/labels`, `dataset/val/images` and `dataset/val/labels`, then update `data.yaml` and paths in `staff_detector.py`. The pretrained `yolo26n.pt` base model is not bundled; provide it locally or allow Ultralytics to download it.

```powershell
python staff_detector.py
```

Dataset scripts are separate preparation utilities, not an automatic end-to-end pipeline. The splitter moves images only; keep labels paired with images or generate labels from matching split NDJSON exports. Avoid randomly splitting neighbouring frames when assessing generalization.

## Recorded validation results

Final epoch of the included `results.csv` (not a separate best.pt evaluation):

| Metric | Value |
| --- | ---: |
| Precision | 99.41% |
| Recall | 90.91% |
| mAP50 | 90.50% |
| mAP50-95 | 47.10% |

The dataset contains 215 training images (48 staff boxes) and 54 validation images (11 staff boxes). Of the validation images, 52 have an immediately adjacent sampled image in training. These single-video metrics may be optimistic and do not establish unseen-video accuracy.

## Limitations

- The detector learns a Staff class from examples; it does not explicitly verify or read name tags.
- Similar clothing/background, small badges, occlusion, rotation and lighting can cause errors.
- Tracking can fragment or miss frames; there is no offline backfill or explicit negative-frame export.
- Paths and GPU selection are hardcoded; video-open and image-write failures are not explicitly checked.
- Frame-level accuracy, tracking continuity, runtime FPS and exported-model parity need separate evaluation.

The repository preparation preserves the existing Python implementation. Dependencies are minimum-version requirements rather than a locked environment.
