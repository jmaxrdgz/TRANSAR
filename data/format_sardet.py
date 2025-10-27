import os
import json
from tqdm import tqdm
from PIL import Image

INPUT_DIR = "dataset/sardet-100k/SARDet_100K"
output_dir = "dataset/SARDet_YOLO"

annotations_dir = os.path.join(INPUT_DIR, "Annotations")
images_dir = os.path.join(INPUT_DIR, "JPEGImages", "train")

print("Annotations:", os.listdir(annotations_dir))
print("Sample training images:", os.listdir(images_dir)[:5])

os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

# Helper function: Convert COCO -> YOLO
def convert_coco_to_yolo(json_path, split):
    with open(json_path) as f:
        data = json.load(f)

    images_info = {img['id']: img for img in data['images']}
    annotations = data['annotations']

    for ann in tqdm(annotations, desc=f"Converting {split} annotations"):
        image_info = images_info[ann['image_id']]
        img_filename = image_info['file_name']

        # Get bbox [x, y, width, height]
        x, y, w, h = ann['bbox']
        img_path = os.path.join(INPUT_DIR, "JPEGImages", split, img_filename)
        label_path = os.path.join(output_dir, "labels", split, img_filename.replace(".png", ".txt"))

        # Normalize bbox
        img = Image.open(img_path)
        img_w, img_h = img.size
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w /= img_w
        h /= img_h

        # Class ID (SARDet has only one class: ship)
        cls_id = ann['category_id'] - 1  # Make 0-indexed

        # Write label file
        with open(label_path, "a") as f_out:
            f_out.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Run for train and val
convert_coco_to_yolo(os.path.join(INPUT_DIR, "Annotations", "train.json"), "train")
convert_coco_to_yolo(os.path.join(INPUT_DIR, "Annotations", "val.json"), "val")

print("Conversion complete! Labels saved in:", os.path.join(output_dir, "labels"))