import os
import cv2
import yaml
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

# Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def augment_image_and_boxes(image, boxes, transform_params):
    """
    Apply transformations to image and corresponding bounding boxes.
    """
    h, w = image.shape[:2]
    
    # 1. Flip
    if transform_params['flip']:
        image = cv2.flip(image, 1)
        new_boxes = []
        for cls_id, xc, yc, bw, bh in boxes:
            new_boxes.append([cls_id, 1.0 - xc, yc, bw, bh])
        boxes = new_boxes

    # 2. Brightness and Contrast
    alpha = transform_params['contrast']
    beta = transform_params['brightness']
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 3. Affine (Rotate, Scale, Shift)
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, transform_params['angle'], transform_params['scale'])
    rot_mat[0, 2] += transform_params['tx'] * w
    rot_mat[1, 2] += transform_params['ty'] * h
    
    image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Transform boxes
    augmented_boxes = []
    for cls_id, xc, yc, bw, bh in boxes:
        # 4 corners
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
        corners = np.array([
            [x1, y1, 1], [x2, y1, 1],
            [x1, y2, 1], [x2, y2, 1]
        ])
        transformed_corners = corners @ rot_mat.T
        
        new_x1 = np.min(transformed_corners[:, 0])
        new_x2 = np.max(transformed_corners[:, 0])
        new_y1 = np.min(transformed_corners[:, 1])
        new_y2 = np.max(transformed_corners[:, 1])

        # Clip and normalize
        new_x1, new_x2 = np.clip([new_x1, new_x2], 0, w - 1)
        new_y1, new_y2 = np.clip([new_y1, new_y2], 0, h - 1)
        
        new_bw = (new_x2 - new_x1) / w
        new_bh = (new_y2 - new_y1) / h
        if new_bw > 0.001 and new_bh > 0.001:
            augmented_boxes.append([cls_id, (new_x1 + new_x2) / (2 * w), (new_y1 + new_y2) / (2 * h), new_bw, new_bh])

    # 4. Noise (Gaussian)
    if transform_params['noise'] > 0:
        noise = np.random.normal(0, transform_params['noise'], image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image, augmented_boxes

def run_augmentation(target_ratio=0.6):
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "data" / "processed"
    dst_dir = project_root / "data" / "augmented"
    
    train_img_dir = src_dir / "images" / "train"
    train_lbl_dir = src_dir / "labels" / "train"
    
    if not train_img_dir.exists():
        print(f"Error: {train_img_dir} not found.")
        return

    # 1. Count instances
    print("Counting instances...")
    class_counts = Counter()
    image_to_classes = {}
    
    label_files = list(train_lbl_dir.glob("*.txt"))
    for lbl_file in label_files:
        with open(lbl_file, 'r') as f:
            classes = [int(line.split()[0]) for line in f.readlines() if line.strip()]
            class_counts.update(classes)
            image_to_classes[lbl_file.stem] = classes

    print(f"Original counts: {dict(class_counts)}")
    
    target_count = int(class_counts[0] * target_ratio)
    print(f"Target count for minority classes (1, 2): {target_count}")

    # 2. Prepare output directories
    for split in ["train", "val"]:
        (dst_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 3. Copy Val set directly
    print("Copying validation set...")
    import shutil
    for f in (src_dir / "images" / "val").glob("*.*"):
        shutil.copy(f, dst_dir / "images" / "val" / f.name)
    for f in (src_dir / "labels" / "val").glob("*.txt"):
        shutil.copy(f, dst_dir / "labels" / "val" / f.name)

    # 4. Copy all original training data
    print("Copying original training data...")
    for f in train_img_dir.glob("*.*"):
        shutil.copy(f, dst_dir / "images" / "train" / f.name)
    for f in train_lbl_dir.glob("*.txt"):
        shutil.copy(f, dst_dir / "labels" / "train" / f.name)

    # 5. Over-sample and Augment
    for cls_id in [1, 2]:
        needed = target_count - class_counts[cls_id]
        if needed <= 0:
            print(f"Class {cls_id} already has enough instances ({class_counts[cls_id]}).")
            continue
        
        print(f"Augmenting class {cls_id}, need ~{needed} more instances.")
        candidate_stems = [stem for stem, classes in image_to_classes.items() if cls_id in classes]
        
        if not candidate_stems:
            print(f"Warning: No samples found for class {cls_id}")
            continue

        pbar = tqdm(total=needed)
        generated_count = 0
        while generated_count < needed:
            stem = random.choice(candidate_stems)
            
            # Find image
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                if (train_img_dir / f"{stem}{ext}").exists():
                    img_path = train_img_dir / f"{stem}{ext}"
                    break
            if not img_path: continue
            
            try:
                pil_img = Image.open(str(img_path)).convert("RGB")
                image = np.array(pil_img)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception:
                continue  # 跳过损坏的图片
            with open(train_lbl_dir / f"{stem}.txt", 'r') as f:
                boxes = [list(map(float, line.split())) for line in f.readlines() if line.strip()]
                # convert class to int
                for b in boxes: b[0] = int(b[0])

            # Random params
            params = {
                'flip': random.random() > 0.5,
                'angle': random.uniform(-15, 15),
                'scale': random.uniform(0.9, 1.1), # scale_limit=0.1 -> 0.9 to 1.1
                'tx': random.uniform(-0.05, 0.05), # shift_limit=0.05
                'ty': random.uniform(-0.05, 0.05),
                'brightness': random.uniform(-30, 30),
                'contrast': random.uniform(0.8, 1.2),
                'noise': random.uniform(0, 10) if random.random() > 0.5 else 0
            }

            aug_img, aug_boxes = augment_image_and_boxes(image, boxes, params)
            
            if aug_boxes:
                new_stem = f"{stem}_aug_{cls_id}_{generated_count}"
                cv2.imwrite(str(dst_dir / "images" / "train" / f"{new_stem}.jpg"), aug_img)
                with open(dst_dir / "labels" / "train" / f"{new_stem}.txt", 'w') as f:
                    for b in aug_boxes:
                        f.write(f"{int(b[0])} {' '.join(f'{x:.6f}' for x in b[1:])}\n")
                
                # Count how many of OUR target class we added
                added = sum(1 for b in aug_boxes if b[0] == cls_id)
                generated_count += added
                pbar.update(added)
        pbar.close()

    # 6. Generate configs/dataset_aug.yaml
    config = {
        'path': '../data/augmented',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}
    }
    
    with open(project_root / "configs" / "dataset_aug.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Generated configs/dataset_aug.yaml")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ratio", type=float, default=0.6)
    args = parser.parse_args()
    run_augmentation(args.target_ratio)
