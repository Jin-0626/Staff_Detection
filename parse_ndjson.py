import os
import json

# Absolute path to your dataset root
dataset_base = r"D:\Learning\Project\Yolo\dataset"

# Define the configurations for both splits since their layouts are identical
splits_config = [
    {
        "name": "train",
        "ndjson": os.path.join(dataset_base, "train", "train.ndjson"),
        "target_labels": os.path.join(dataset_base, "train", "labels")
    },
    {
        "name": "val",
        "ndjson": os.path.join(dataset_base, "val", "val.ndjson"),
        "target_labels": os.path.join(dataset_base, "val", "labels")
    }
]

for config in splits_config:
    ndjson_path = config["ndjson"]
    labels_dir = config["target_labels"]
    
    if not os.path.exists(ndjson_path):
        print(f"Warning: Could not find {config['name']}.ndjson at {ndjson_path}. Skipping...")
        continue
        
    # Automatically build the respective labels directory if it's missing
    os.makedirs(labels_dir, exist_ok=True)
    print(f"Processing {config['name']} labels...")
    
    parsed_count = 0
    with open(ndjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Skip the platform metadata header entry
            if data.get("type") == "dataset":
                continue
                
            if data.get("type") == "image":
                file_name = data["file"]
                
                # frame_0001.jpg -> frame_0001.txt
                txt_name = os.path.splitext(file_name)[0] + ".txt"
                txt_path = os.path.join(labels_dir, txt_name)
                
                with open(txt_path, 'w') as txt_file:
                    annotations = data.get("annotations", {})
                    boxes = annotations.get("boxes", [])
                    
                    for box in boxes:
                        # Box structure: [class_id, x_center, y_center, width, height]
                        class_id = int(box[0])
                        x_center = box[1]
                        y_center = box[2]
                        width = box[3]
                        height = box[4]
                        
                        # Format floats to 6 decimal places for YOLO compliance
                        txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                parsed_count += 1

print("\nSuccess! Custom YOLO text labels generated for both splits.")