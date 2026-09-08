import os
import random
import shutil

# The Ultralytics Platform comes with auto splitting train and val dataset in one ndjson file,
# but here we are doing it manually for better control and understanding.


base_path = r"D:\Learning\Project\Yolo\dataset"
source_images_dir = base_path  


train_img_dir = os.path.join(base_path, "train", "images")
val_img_dir = os.path.join(base_path, "val", "images")


os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(os.path.join(base_path, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(base_path, "val", "labels"), exist_ok=True)


images = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg')]


random.seed(42)
random.shuffle(images)

# Calculate 80/20 split 
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

# Move images to respective folders
for img in train_images:
    shutil.move(os.path.join(source_images_dir, img), os.path.join(train_img_dir, img))

for img in val_images:
    shutil.move(os.path.join(source_images_dir, img), os.path.join(val_img_dir, img))

print(f"Dataset split complete! Train: {len(train_images)} images, Val: {len(val_images)} images.")