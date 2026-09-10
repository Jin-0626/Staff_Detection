import os
from ultralytics import YOLO

def main():
    # Load the trained yolo26n
    model_path = r"D:\Learning\Project\Yolo\yolov26n_staff_2026-09-09_04-32-58_e50_b4\weights\best.pt"
    video_path = r"D:\Learning\Project\Yolo\input_video\sample.mp4"
    model = YOLO(model_path)
        
    print(f"Loading custom model weights from: {model_path}")
    print("Running inference and saving output video...")
    
    
    results = model.predict(
        source=video_path,
        save=True,
        conf=0.5,
        device=0 
    )
    
    print("\nInference Complete!")
    print("Your annotated video has been successfully saved to your 'runs/detect/predict' folder.")

if __name__ == '__main__':
    main()