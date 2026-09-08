import os
from ultralytics import YOLO


def main():
    project_dir = r"D:\Learning\Project\Yolo"
    model_trained= os.path.join(project_dir, "yolov26n_staff_2026-09-09_04-32-58_e50_b4", "weights", "best.pt")
    model = YOLO(model_trained)
    
    model.export(format="onnx")
    
if __name__ == '__main__':
    main()
    