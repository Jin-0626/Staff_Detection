import os
from datetime import datetime
from ultralytics import YOLO

def main():
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_epoch = 50
    model_batch = 4
    model_name = f"yolov26n_staff_{current_datetime}_e{model_epoch}_b{model_batch}"
    project_dir = r"D:\Learning\Project\Yolo"
    dataset_yaml = r"D:\Learning\Project\Yolo\yolo26n.yaml"
    model = YOLO("yolo26n.pt")
    print("--- Starting Ultralytics YOLOv8 Local Training Run ---")
    
    results = model.train(
        data=dataset_yaml,
        epochs=model_epoch, 
        batch=model_batch,
        project=project_dir,  
        name=model_name,              
        imgsz=640,
        device=0,         # 0 uses your NVIDIA GPU. Use 'cpu' if no GPU is available.
        workers=2         # Windows safely handles 2 data workers. Set to 0 if it freezes.
    )
    print(f"\nTraining Complete! Finished weights and validation graphs are at:")
    print(os.path.join(project_dir, model_name))
    
    
if __name__ == '__main__':
    main()