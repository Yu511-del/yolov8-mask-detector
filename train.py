import os
import sys
import subprocess
from pathlib import Path
from ultralytics import YOLO

def run_command(command, description):
    print(f"\n>>> Running: {description}")
    print(f"Command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        sys.exit(1)

def main():
    project_root = Path(__file__).resolve().parent
    exp_dir = project_root / "experiments"
    exp_dir.mkdir(exist_ok=True)

    # ==========================================
    # Phase 0: Baseline
    # ==========================================
    baseline_dir = exp_dir / "baseline"
    if not (baseline_dir / "weights" / "best.pt").exists():
        print("\n" + "="*60)
        print("STAGE 0: Starting Baseline Experiment")
        print("="*60)
        
        model = YOLO("yolov8n.pt")
        model.train(
            data="configs/dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            mosaic=0.0,
            mixup=0.0,
            project="experiments",
            name="baseline",
            exist_ok=True
        )
    else:
        print("\n[INFO] Baseline experiment already completed. Skipping.")

    # ==========================================
    # Phase 1: Data Generation
    # ==========================================
    aug_data_dir = project_root / "data" / "augmented"
    aug_config = project_root / "configs" / "dataset_aug.yaml"
    
    if not aug_data_dir.exists() or not aug_config.exists():
        print("\n" + "="*60)
        print("STAGE 1: Generating Augmented Dataset")
        print("="*60)
        run_command(f"{sys.executable} data/augment.py --target_ratio 0.6", "Data Augmentation")
    else:
        print("\n[INFO] Augmented dataset already exists. Skipping generation.")

    # ==========================================
    # Phase 2: Ablation Study
    # ==========================================
    ablation_configs = [
        {"name": "abl1", "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0},
        {"name": "abl2", "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0},
        {"name": "abl3", "mosaic": 1.0, "mixup": 0.5, "copy_paste": 0.0},
        {"name": "abl4", "mosaic": 1.0, "mixup": 0.5, "copy_paste": 0.5},
    ]

    print("\n" + "="*60)
    print("STAGE 2: Starting Ablation Study (4 Experiments)")
    print("="*60)

    for cfg in ablation_configs:
        cfg_name = cfg["name"]
        cfg_dir = exp_dir / "ablation" / cfg_name
        
        if (cfg_dir / "weights" / "best.pt").exists():
            print(f"\n[INFO] Experiment {cfg_name} already completed. Skipping.")
            continue

        print(f"\n>>> Running Ablation: {cfg_name} (mosaic={cfg['mosaic']}, mixup={cfg['mixup']}, copy_paste={cfg['copy_paste']})")
        
        model = YOLO("yolov8n.pt")
        model.train(
            data="configs/dataset_aug.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            mosaic=cfg["mosaic"],
            mixup=cfg["mixup"],
            copy_paste=cfg["copy_paste"],
            project="experiments/ablation",
            name=cfg_name,
            exist_ok=True
        )

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results are stored in: {exp_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
