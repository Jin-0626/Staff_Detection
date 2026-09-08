import cv2
import os

# load the video using path of video
model_path = r"D:\Learning\Project\Yolo"
video_path = os.path.join(model_path,"input_video","sample.mp4")
dataset_path = os.path.join(model_path, "dataset")
video = cv2.VideoCapture(video_path)
frame_rate = 5
count=0
saved_count=0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    if count % frame_rate == 0:
        cv2.imwrite(f"dataset/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1
 
        
    count += 1
        
video.release()
print(f"Extracted {saved_count} frames.")