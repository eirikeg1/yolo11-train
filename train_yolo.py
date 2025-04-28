#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import json
import shutil
import random

from ultralytics import YOLO

################################################################################
# USER PARAMETERS
################################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--out_path', default="/cluster/work/projects/ec12/ec-eirikeg/yolo_out")
parser.add_argument('--data_dir', default="/cluster/work/projects/ec12/ec-eirikeg/data/soccernet",
                    help='Path to the folder containing train/, valid/, test/')
parser.add_argument('--weights', default=None)
parser.add_argument('--finetune_class',
                    default="ball",
                    choices=["player", "ball", "all"],
                    help='Which class(es) to keep: player, ball or both (all)')
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    help='Number of epochs to train for')
parser.add_argument('--iou',
                    type=float,
                    default=0.5,
                    help='IoU threshold passed to model.train()')
parser.add_argument('--imagesz',
                    default="640,1280",
                    help='Training image size(s). Either a single int or a comma-separated list (e.g. "640,1280")')


args = parser.parse_args()

# Parse --imagesz -> int | list[int]
if "," in args.imagesz:
    imgsz = [int(sz) for sz in args.imagesz.split(",") if sz.strip()]
else:
    imgsz = int(args.imagesz)
    
data_dir = args.data_dir # parent folder containing train/, valid/, test/, test/
output_dir = args.out_path
selected_class = args.finetune_class
train_yolo = True # if True, train YOLO at the end
train_ratio = 0.85 # 0.9 = 90% train and 10% val
weights = args.weights # path to YOLO weights, e.g. 'yolo11m.pt'
epochs = args.epochs # number of epochs to train for


# Label filtering: set to 1 for player, 0 for ball.
# Note: even though in the original mapping player was 0 and ball 1, for filtering we use:
#       1 --> player, 0 --> ball.
FILTER_LABEL = {
    "player": 1,
    "ball":   0,
    "all":    None,
}[selected_class]

################################################################################
# HELPER FUNCTIONS
################################################################################

def parse_labels_json(json_path: str, images_dir: str):
    """
    Loads one JSON (matching the Rust 'Labels' structure) and returns a list of:
      (img_path, bboxes, width, height)

    where bboxes is: [(class_id, x, y, w, h), ...] and the class_id is remapped to 0
    (because we are only keeping one label type).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # For quick lookups
    images_info = {img['image_id']: img for img in data['images']}
    category_map = {cat['id']: cat['name'] for cat in data['categories']}

    image_to_bboxes = {}
    label_to_id = {"player": 0, "ball": 1}

    for ann in data['annotations']:
        bbox_img = ann.get('bbox_image')
        if not bbox_img:
            continue  # skip annotation if no image-based bbox

        image_id = ann['image_id']
        cat_id = ann['category_id']
        cat_name = category_map.get(cat_id, "").lower()
        
        if FILTER_LABEL is not None:
            allowed_label = "player" if FILTER_LABEL == 1 else "ball"
            if cat_name != allowed_label:
                continue
            class_id = 0
        else:
            if cat_name not in label_to_id:
                continue
            class_id = label_to_id[cat_name]

        x = bbox_img['x']
        y = bbox_img['y']
        w = bbox_img['w']
        h = bbox_img['h']

        if image_id not in image_to_bboxes:
            image_to_bboxes[image_id] = []
        image_to_bboxes[image_id].append((class_id, x, y, w, h))

    # Build a list of (img_path, [bboxes], width, height)
    dataset_list = []
    for image_id, bboxes in image_to_bboxes.items():
        img_info = images_info[image_id]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        # full image path
        img_full_path = os.path.join(images_dir, file_name)
        dataset_list.append((img_full_path, bboxes, width, height))

    return dataset_list

def gather_split_data():
    """
    Gathers data from data_dir's subfolders: train/, valid/, test/.
    Ignores the test/ folder.

    Returns a combined list of:
      (img_path, bboxes, width, height, scene_name).
    """
    splits_to_gather = ["train", "valid", "test"]
    all_items = []

    for split_name in splits_to_gather:
        split_path = os.path.join(data_dir, split_name)
        print(f"Scanning {split_path}", flush=True)
        if not os.path.isdir(split_path):
            print(f"Warning: subfolder '{split_path}' not found, skipping.", flush=True)
            continue

        # Inside e.g. 'train', we might have SNGS-060, SNGS-061, etc.
        for scene_dir in os.listdir(split_path):
            scene_path = os.path.join(split_path, scene_dir)
            if not os.path.isdir(scene_path):
                print("Is not a directory:", scene_path)
                continue

            json_file = os.path.join(scene_path, 'Labels-GameState.json')
            img_folder = os.path.join(scene_path, 'img1')

            if not os.path.exists(json_file) or not os.path.isdir(img_folder):
                print(f"Warning: {json_file} or {img_folder} not found, skipping.")
                continue

            # Parse data from this scene
            scene_data = parse_labels_json(json_file, img_folder)

            # Append the scene name so we can rename images uniquely
            updated_scene_data = []
            for (img_path, bboxes, width, height) in scene_data:
                # store 5-tuple including the scene_dir
                updated_scene_data.append((img_path, bboxes, width, height, scene_dir))

            all_items.extend(updated_scene_data)

    return all_items

def create_yolo_directory_structure(base_dir: str):
    """
    Creates the YOLO folder structure:
      base_dir/images/train
      base_dir/images/val
      base_dir/labels/train
      base_dir/labels/val
    """
    subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for sd in subdirs:
        path = Path(base_dir) / sd
        path.mkdir(parents=True, exist_ok=True)

def split_dataset(dataset_list, ratio=0.95):
    """
    Splits dataset_list into train/val by the given ratio.
    """
    random.shuffle(dataset_list)
    train_size = int(len(dataset_list) * ratio)
    train_data = dataset_list[:train_size]
    val_data = dataset_list[train_size:]
    return train_data, val_data

def save_yolo_labels(dataset_split, split_type: str, base_dir: str):
    """
    For each (img_path, bboxes, width, height, scene_dir), copy image and
    create a .txt label file with YOLO 'class x_center y_center w h' format (normalized).

    We rename each image to: {scene_dir}_{original_filename}
    to avoid overwriting if multiple scenes have 000001.jpg, etc.
    """
    for (img_path, bboxes, width, height, scene_name) in dataset_split:
        if not os.path.exists(img_path):
            print(f"Warning: image not found: {img_path}")
            continue

        original_filename = os.path.basename(img_path)
        # e.g. "SNGS-060_000001.jpg"
        new_image_filename = f"{scene_name}_{original_filename}"

        # Copy image to the YOLO folder
        img_dest = os.path.join(base_dir, 'images', split_type, new_image_filename)
        shutil.copy2(img_path, img_dest)

        # Create label with the same prefix
        label_filename = os.path.splitext(new_image_filename)[0] + '.txt'
        label_dest = os.path.join(base_dir, 'labels', split_type, label_filename)

        lines = []
        for (cls_id, x, y, w, h) in bboxes:
            x_center = x + w/2.0
            y_center = y + h/2.0
            x_center_norm = x_center / width
            y_center_norm = y_center / height
            w_norm = w / width
            h_norm = h / height

            lines.append(f"{cls_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(label_dest, 'w') as f:
            f.write('\n'.join(lines))



# 1. Gather data from train/, valid/, challenge/
all_data = gather_split_data()
print(f"Found {len(all_data)} annotated images in total (train+valid+challenge).", flush=True)

if not all_data:
    print("No data found. Exiting.")
    exit(1)

# 2. Create YOLO folder structure
create_yolo_directory_structure(output_dir)

# 3. Split dataset (e.g., 90% train, 10% val)
train_data, val_data = split_dataset(all_data, train_ratio)
print(f"Training images: {len(train_data)}", flush=True)
print(f"Validation images: {len(val_data)}", flush=True)

# 4. Convert to YOLO format
save_yolo_labels(train_data, 'train', output_dir)
save_yolo_labels(val_data, 'val', output_dir)

# 5. Write data.yaml
# When filtering, we only have one class; update the names accordingly.
data_yaml_path = os.path.join(output_dir, "data.yaml")
if selected_class == "all":
    class_block = "names:\n  0: player\n  1: ball\n"
else:
    allowed_label_name = "player" if FILTER_LABEL == 1 else "ball"
    class_block = f"names:\n  0: {allowed_label_name}\n"

data_yaml = f"""# YOLO dataset config
train: {output_dir}/images/train
val:   {output_dir}/images/val

{class_block}"""
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml)

print(f"\nDataset prepared in '{output_dir}'. data.yaml at '{data_yaml_path}'.", flush=True)


# %%
# 6. Train


if train_yolo:
    print("\nStarting YOLO training...", flush=True)

    # YOLOv8 automatically prints a progress bar, saves the best model, etc.
    # 'save_period=1' means it will save an extra checkpoint each epoch. 
    model = YOLO(weights) if weights else YOLO('yolo11m.pt')  # or 'yolo11m.pt', etc.
    results = model.train(
        iou=0.5,
        data=data_yaml_path,
        epochs=epochs,
        imgsz=1280,
        batch=0.9,        # auto-batch
        # pretrained=True, # use pretrained backbone
        scale=0.3,       # randomly scales images by 0.3-1.3
        workers=10,       # Increase data loading speed
        device=0,        # Ensure it's running on GPU
        amp=True,        # Enable mixed precision training (saves VRAM, speeds up training)
        name=f"{selected_class}detection",
        save_period=10,   # checkpoints
        verbose=True     # progress bar
    )
    print("\nTraining finished. Check 'runs/detect/player-ball-detection' for results.", flush=True)
    print("Best model is saved to 'weights/best.pt' inside that folder.", flush=True)
