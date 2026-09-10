"""Track staff with per-frame diagnostics and optional visual output."""
import argparse
from collections import defaultdict, deque
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'yolov26n_staff_2026-09-09_04-32-58_e50_b4/weights/best.pt'


def box_records(boxes, tracked=False):
    if boxes is None or (tracked and boxes.id is None):
        return []
    ids = boxes.id.int().cpu().tolist() if tracked else [None] * len(boxes)
    return [dict(track_id=tid, confidence=float(conf), class_id=int(cls),
                 bbox_xyxy=[float(v) for v in xyxy],
                 centre_xy=[float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2)])
            for xyxy, conf, cls, tid in zip(boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), ids)]


class Trails:
    def __init__(self, length=45):
        self.length = length
        self.history = defaultdict(lambda: deque(maxlen=length))
        self.last_seen = {}

    def draw(self, frame, tracks, frame_number):
        for t in tracks:
            tid = t['track_id']
            # Never draw a straight line across an unseen interval.
            if self.last_seen.get(tid) != frame_number - 1:
                self.history[tid].clear()
            point = tuple(round(v) for v in t['centre_xy'])
            self.history[tid].append(point)
            self.last_seen[tid] = frame_number
            color = tuple(int(v) for v in np.random.default_rng(tid).integers(70, 256, 3))
            points = np.array(self.history[tid], np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [points], False, color, 2)
            x1, y1, x2, y2 = map(round, t['bbox_xyxy'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, point, 4, color, -1)
            cv2.putText(frame, f"ID {tid} {t['confidence']:.2f} {point}",
                        (max(0, x1), max(18, y1 - 7)), 0, .5, color, 1)
        for tid in list(self.last_seen):
            if frame_number - self.last_seen[tid] > self.length:
                del self.last_seen[tid]
                del self.history[tid]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'input_video/sample.mp4')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--tracker', default=str(ROOT / 'custom_tracker.yaml'))
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--conf', type=float, default=.1)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--save-images', action='store_true')
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--max-frames', type=int, help='Limit a diagnostic run; omit for full video')
    p.add_argument('--trail-length', type=int, default=45)
    a = p.parse_args()
    if not 0 < a.conf <= 1 or a.imgsz <= 0 or a.trail_length < 1 or (a.max_frames is not None and a.max_frames < 1):
        p.error('Require 0 < conf <= 1 and positive sizes/frame limits')
    if not a.source.is_file() or not a.model.is_file():
        p.error('Source video and model must exist')
    cap = cv2.VideoCapture(str(a.source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {a.source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        cap.release()
        raise RuntimeError('Invalid video metadata; timestamps require a valid FPS')
    output = a.output or ROOT / 'runs' / ('tracking_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    # Refuse reuse so partial/new runs cannot silently mix with old results.
    output.mkdir(parents=True, exist_ok=False)
    writer = None
    count = 0
    elapsed = 0.0
    warm_seconds = 0.0
    raw_snapshot = None
    complete = False
    pipeline_start = time.perf_counter()
    try:
        model = YOLO(str(a.model))
        def capture_raw(predictor):
            nonlocal raw_snapshot
            if any(r.boxes is not None and r.boxes.id is not None for r in predictor.results):
                raise RuntimeError('Raw-box callback ran after tracking; check Ultralytics callback order')
            raw_snapshot = [box_records(r.boxes) for r in predictor.results]
        model.track(np.zeros((64, 64, 3), dtype=np.uint8), persist=True)
        # Register BEFORE model.track registers its postprocessing callback.
        model.callbacks['on_predict_postprocess_end'].insert(0, capture_raw)
        trails = Trails(a.trail_length)
        if a.save_images:
            (output / 'images').mkdir()
        if a.save_video:
            writer = cv2.VideoWriter(str(output / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError('Cannot initialize video writer')
        def sync():
            if torch.cuda.is_available() and a.device not in ('cpu', 'mps'):
                torch.cuda.synchronize(torch.device('cuda:' + a.device) if a.device.isdigit() else a.device)
        with (output / 'frames.jsonl').open('w', encoding='utf-8') as log:
            while a.max_frames is None or count < a.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                raw_snapshot = None
                sync()
                start = time.perf_counter()
                result = model.track(frame, persist=True, tracker=a.tracker, conf=a.conf,
                                     imgsz=a.imgsz, classes=[0], device=a.device, verbose=False)[0]
                sync()
                duration = time.perf_counter() - start
                if raw_snapshot is None or len(raw_snapshot) != 1:
                    raise RuntimeError('Expected one raw detector snapshot per frame')
                if count == 0:
                    warm_seconds = duration
                else:
                    elapsed += duration
                count += 1
                detections = raw_snapshot[0]
                tracks = box_records(result.boxes, tracked=True)
                status = 'tracks_returned' if tracks else ('detections_without_tracks' if detections else 'no_detections')
                record = dict(frame=count, timestamp_sec=(count - 1) / fps,
                              staff_present=bool(tracks), diagnostic=status,
                              detections=detections, tracks=tracks, track_call_ms=duration * 1000)
                log.write(json.dumps(record, allow_nan=False) + '\n')
                if a.save_images or writer is not None:
                    # Raw detections: thin grey boxes; returned tracks: coloured boxes and trails.
                    for detection in detections:
                        x1, y1, x2, y2 = map(round, detection['bbox_xyxy'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 170, 170), 1)
                    trails.draw(frame, tracks, count)
                    cv2.putText(frame, f'{count} | {status} | raw {len(detections)} tracks {len(tracks)}',
                                (12, 24), 0, .55, (0, 255, 255), 1)
                    if a.save_images and not cv2.imwrite(str(output / 'images' / f'frame_{count:04d}.jpg'), frame):
                        raise RuntimeError(f'Failed saving image {count}')
                    if writer is not None:
                        writer.write(frame)
                if count % 100 == 0:
                    print(f'Processed {count}/{total} frames')
        complete = count > 0 and (count == total if total > 0 else a.max_frames is None)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        wall = time.perf_counter() - pipeline_start
        summary = dict(source=str(a.source.resolve()), source_fps=fps, width=width, height=height,
                       source_frame_count=total, processed_frames=count, complete_video=complete,
                       model=str(a.model.resolve()), tracker=a.tracker, conf=a.conf, imgsz=a.imgsz,
                       device=a.device, ultralytics_version=ultralytics_version,
                       save_images=a.save_images, save_video=a.save_video,
                       first_call_seconds=warm_seconds, timed_frames=max(count - 1, 0),
                       track_call_fps=(count - 1) / elapsed if elapsed > 0 else None,
                       pipeline_fps=count / wall if wall > 0 else None, pipeline_seconds=wall,
                       timing_note='Track-call FPS excludes first call, includes raw-box capture and GPU sync. Pipeline includes model setup, first call, decoding, logging, optional drawing/writing and resource release; excludes summary serialization.')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not count:
        raise RuntimeError('No frames decoded')
    print(f'Output: {output}\nTrack-call FPS: {summary["track_call_fps"]}\nPipeline FPS: {summary["pipeline_fps"]}')


if __name__ == '__main__':
    main()
