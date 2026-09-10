from ultralytics import YOLO



def main():
# Load a model
    model = YOLO(r"D:\Learning\Project\Yolo\yolov26n_staff_2026-09-09_04-32-58_e50_b4\weights\best.pt")  # load a custom model

    # Validate the model
    metrics = model.val(data=r"D:\Learning\Project\Yolo\data.yaml") 
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list containing mAP50-95 for each category
    metrics.box.image_metrics  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN


if __name__ == "__main__":
    main()