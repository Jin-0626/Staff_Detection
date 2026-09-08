import os
import cv2
from ultralytics import YOLO

def main():
    project_dir = r"D:\Learning\Project\Yolo"
    model_dir = r"D:\Learning\Project\Yolo\yolov26n_staff_2026-09-09_04-32-58_e50_b4"
    model_path = os.path.join(project_dir, model_dir, "weights", "best.pt")
    video_path = os.path.join(project_dir, "input_video", "sample.mp4")
    frame_dir = os.path.join(project_dir, "frame")
    
    os.makedirs(frame_dir, exist_ok=True)
    txt_path = os.path.join(frame_dir, "staff_frame.txt")
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_path)
    count = 0
    
    print(f"tracking initiated.")
    
    with open(txt_path, "w") as f:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            count += 1
            results = model.track(frame, persist=True, tracker = "custom_tracker.yaml", device = 0, verbose = False)
            
            result = results[0]
            
            frame = result.plot()
            
            
            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                staff_coordinates = result.boxes.xyxy.cpu().numpy()
                tracked_count = len(track_ids)
                f.write(f"Frame {count}: {tracked_count} validated staff tracked.\n")
                for idx, coords in enumerate(staff_coordinates):
                    x = int((coords[0]+ coords[2])/2)
                    y = int((coords[1]+ coords[3])/2)
                    
                    tid = track_ids[idx] if track_ids is not None else "N/A"
                    f.write(f" Staff IDL {tid}: Coordinates: ({x}, {y})\n")
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    label = f"({x}, {y})"
                    if track_ids is not None:
                        label = f"ID:{track_ids[idx]} {label}"
                    
                    cv2.putText(frame, f"({x}, {y})", (x + 10, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
         
            cv2.imwrite(os.path.join(frame_dir, f"frame_{count:04d}.jpg"), frame)
            
            
            if count % 10 == 0:
                print(f"Processed {count} frames...")
               
        video.release()
       
        print(f"Tracking run fully complete.")
        
        
if __name__ == '__main__':
     main()