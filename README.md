# Staff Detection

Track staff in overhead video using this project's trained YOLO26n `best.pt` checkpoint and the existing ByteTrack configuration. The staff tag defines the Staff class. The model predicts whole-person Staff boxes; it does not separately detect or read the tag.

[Read the two-page solution documentation](output/pdf/staff_detection_solution.pdf).

## Quick start

Run from the repository root. Place your source video at `input_video/sample.mp4`.

```powershell
python -m venv yolo
.\yolo\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tracking.py --save-video
```

If the environment already exists, activate it rather than recreating it. Tracking automatically selects GPU 0 when CUDA is available, otherwise CPU. It loads the included checkpoint:

```text
yolov26n_staff_2026-09-09_04-32-58_e50_b4/weights/best.pt
```

The command uses `custom_tracker.yaml` (ByteTrack). No retraining or alternate detector is needed. To track another video:

```powershell
python tracking.py --source "D:\Videos\test.mp4" --save-video --device cpu
```

Optional JPEG output or a short diagnostic run:

```powershell
python tracking.py --save-images --max-frames 100
python tracking.py --help
```

Tracking paths default relative to the project directory. Use `--model` to specify your checkpoint explicitly if its location changes.

## Outputs

Each run creates `runs/tracking_<timestamp>/`. `--output PATH` chooses an unused directory; existing directories are rejected to prevent mixing runs.

| File | Contents |
| --- | --- |
| `annotated.mp4` | With `--save-video`: boxes, IDs, confidence, centre XY and trajectory trails |
| `frames.jsonl` | One JSON record per decoded frame, including frames with no tracks |
| `summary.json` | Model/settings, source metadata, completeness and timing |
| `images/frame_XXXX.jpg` | Annotated JPEGs, only with `--save-images` |

Visual output is optional. Thin grey boxes show detector output; coloured boxes and trails show returned tracks. Trails restart after missing frames. The MP4 uses the `mp4v` codec, whose browser playback support varies.

Frame records contain one-based frame number, timestamp `(frame-1)/FPS`, detections and tracks with confidence, class, bounding box and floating-point centre XY. Raw detections mean postprocessed detector output before tracker mutation, not network logits. An ID is included for each returned track. Coordinates are pixels from the top-left, x rightward and y downward: `((x1+x2)/2, (y1+y2)/2)`.

`staff_present` means at least one returned track; this is a prediction, not verified ground truth. The diagnostic categories are `no_detections`, `detections_without_tracks` and `tracks_returned`. A missing detection is only an error when staff should actually be present.

Tracking-call FPS excludes the first call and includes raw-box capture and GPU synchronization. Pipeline FPS includes model setup, warm-up, decoding, logging, enabled rendering/writing and resource release; summary serialization is excluded. Both measurements are saved in `summary.json`.

## Original dataset labels

Use the original YOLO labels in `dataset/train/labels` and `dataset/val/labels`, paired with their respective images. This workflow does not require a new manual labelling tool.

`dataset_create.py` samples every fifth video frame. Image `frame_NNNN.jpg` maps to one-based video frame `5 * NNNN + 1`; for example, `frame_0063.jpg` maps to frame 316. Intermediate frames are unlabelled, not automatically staff-absent. Do not treat interpolated boxes as verified ground truth.

There are 215 training images (48 Staff boxes) and 54 validation images (11 Staff boxes). Empty label files represent background examples according to the original annotations. These class-and-box labels contain no person identities, so they cannot establish ID-switch accuracy. Staff identity is established through the tag; hiding the tag temporarily does not necessarily make an identifiable staff member absent.

## Recorded results

Final-epoch box-detection metrics from the included training `results.csv`:

| Metric | Value |
| --- | ---: |
| Precision | 99.41% |
| Recall | 90.91% |
| mAP50 | 90.50% |
| mAP50-95 | 47.10% |

These are not tracking accuracy or a separate evaluation of `best.pt`. The dataset comes from one video; 52 of 54 validation images have an immediately adjacent sampled training image. Use an unseen video or untouched temporal split before claiming generalization.

A recorded full tracking run on 10 September 2026 processed all 1,341 frames at 960 x 720 and 25 source FPS. It used the bundled model, custom ByteTrack, confidence 0.1, image size 640, GPU 0 and Ultralytics 8.4.144. Tracks were returned on 208 frames, detections without tracks on 99 frames, and no detections on 1,034 frames. These counts are diagnostics, not accuracy measurements.

That run measured approximately 64.5 tracking calls/sec and 43.3 pipeline frames/sec with MP4 saving enabled. Performance depends on hardware and settings. The callback order used to capture detector output was verified with Ultralytics 8.4.144.

## Files and optional training

| File | Purpose |
| --- | --- |
| `tracking.py` | Run the trained model, diagnostics, trails and optional MP4/JPEG output |
| `custom_tracker.yaml` | Existing ByteTrack settings |
| `train.py` | Optional training entry point |
| `yolo26n.yaml` | Dataset paths and Staff class, not a model architecture YAML |
| `dataset_create.py` | Extract sampled images |
| `parse_ndjson.py` | Convert original split NDJSON annotations to YOLO labels |
| `test_inteference.py` | Legacy prediction-only video script |
| `model_export.py` | Export the trained checkpoint to ONNX |

To retrain, prepare the original dataset layout and update the hardcoded paths in `train.py` and `data.yaml`, then run `python train.py`. It uses 50 epochs, batch size 4, image size 640, two workers and GPU 0. Provide the base `yolo26n.pt` or allow Ultralytics to download it. Retraining is optional and is not part of the MP4 tracking command.

Preparation, training, legacy prediction and export scripts retain local path/device assumptions. Tracking has portable CLI paths and checks video metadata and image-write results. Dependencies specify minimum versions rather than a locked environment.

## Limitations and repository contents

The model can learn clothing/background cues as well as the tag. Tracking can miss frames or change IDs; there is no offline backfill or explicit badge verification. Exported-model parity is untested. The original labels cover only sampled frames and cannot establish continuous identity accuracy.

The trained checkpoint and solution PDF are included. The local Python environment, source video, dataset, evaluation brief and generated run outputs are excluded from GitHub.
