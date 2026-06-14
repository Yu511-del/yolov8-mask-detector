<<<<<<< HEAD
from training.trainers.yolo_trainer import CustomYoloTrainer

def run_comparison_experiments():
    """
    运行对比实验：
    1. YOLOv8n / YOLOv8s / YOLOv8m 对比实验
    2. 基于相同数据与增强配置，比较不同模型规模的精度与速度权衡

    """

    # 旧版本实验流程（保留作参考，不再执行）
    # --- 实验 1: Baseline (使用原始处理后的数据，无增强) ---
    # print("\n" + "="*60)
    # print("STEP 1: Starting Baseline Experiment (Original Data, No Augmentation)")
    # print("="*60)
    
    # baseline_trainer = CustomYoloTrainer(
    #     model_variant='yolov8s.pt',
    #     project_name='face_mask_detection',
    #     experiment_name='baseline_original'
    # )
    
    # baseline_trainer.train(
    #     data_config='configs/dataset.yaml',   # 指向原始数据 (data/processed)
    #     epochs=100,
    #     imgsz=640,
    #     batch=16,
    #     patience=20,
    #     mosaic=0.0,
    #     mixup=0.0
    # )
    
    # # --- 实验 2: Improved (使用原始数据 但开启 YOLO 内置增强) ---
    # print("\n" + "="*60)
    # print("STEP 2: Starting Improved Experiment (Original Data + YOLO Built-in Augmentation)")
    # print("="*60)
    
    # improved_trainer = CustomYoloTrainer(
    #     model_variant='yolov8s.pt',
    #     project_name='face_mask_detection',
    #     experiment_name='improved_augmented'
    # )
    
    # improved_trainer.train(
    #     data_config='configs/dataset.yaml',   # 仍然指向原始数据
    #     epochs=100,
    #     imgsz=640,
    #     batch=16,
    #     patience=20,
    #     mosaic=1.0,
    #     mixup=0.1
    # )

    experiment_configs = [
        ("YOLOv8n", "yolov8n.pt", "improved_nano"),
        ("YOLOv8s", "yolov8s.pt", "iroved_small"),
        ("YOLOv8m", "yolov8m.pt", "improved_medium"),
    ]

    for index, (model_label, model_variant, experiment_name) in enumerate(experiment_configs, start=1):
        print("\n" + "=" * 60)
        print(f"STEP {index}: Starting {model_label} Experiment (Original Data + YOLO Built-in Augmentation)")
        print("=" * 60)

        trainer = CustomYoloTrainer(
            model_variant=model_variant,
            project_name='face_mask_detection',
            experiment_name=experiment_name,
        )

        trainer.train(
            data_config='configs/dataset.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            patience=20,
            mosaic=1.0,
            mixup=0.1,
        )

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("Results are stored in: experiments/face_mask_detection/")
    print("="*60)

if __name__ == "__main__":
    run_comparison_experiments()
=======
from training.trainers.yolo_trainer import CustomYoloTrainer
import os
import sys

def run_comparison_experiments():
    """
    运行对比实验：
    1. Baseline (原始处理后的数据)
    2. Improved (经过数据增强后的数据)

    该脚本设计用于在云端(Kaggle)一键启动完整实验流程。
    """

    # --- 实验 1: Baseline (使用原始处理后的数据) ---
    print("\n" + "="*60)
    print("STEP 1: Starting Baseline Experiment (Original Data)")
    print("="*60)

    baseline_trainer = CustomYoloTrainer(
        model_variant='yolov8s.pt', 
        project_name='face_mask_detection', 
        experiment_name='baseline_original'
    )

    # 默认使用 configs/dataset.yaml (指向 data/processed)
    # 建议生产环境设为 100+ epochs，此处为演示设为 50
    baseline_trainer.train(
        data_config='configs/dataset.yaml', 
        epochs=100, 
        imgsz=640, 
        batch=16,
        patience=20
    )

    # --- 实验 2: 数据增强 (如果尚未生成) ---
    if not os.path.exists('data/augmented/images'):
        print("\n" + "-"*60)
        print("[INFO] Augmented data not found. Running data/augment.py...")
        print("-"*60)

        # 确保父目录存在
        os.makedirs('data/augmented', exist_ok=True)

        # 执行增强脚本
        exit_code = os.system(f"{sys.executable} data/augment.py")
        if exit_code != 0:
            print("[ERROR] Data augmentation failed. Please check data/augment.py")
            return
    else:
        print("\n[INFO] Augmented data already exists. Skipping augmentation step.")

    # --- 实验 3: Improved (使用增强后的数据) ---
    print("\n" + "="*60)
    print("STEP 2: Starting Improved Experiment (Augmented Data)")
    print("="*60)

    improved_trainer = CustomYoloTrainer(
        model_variant='yolov8s.pt', 
        project_name='face_mask_detection', 
        experiment_name='improved_augmented'
    )

    # 使用 configs/dataset_aug.yaml (指向 data/augmented)
    improved_trainer.train(
        data_config='configs/dataset_aug.yaml', 
        epochs=100, 
        imgsz=640, 
        batch=16,
        patience=20
    )

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("Results are stored in: experiments/face_mask_detection/")
    print("="*60)

if __name__ == "__main__":
    # 注意：在 Kaggle 上运行前，请确保已安装 requirements.txt 中的依赖
    run_comparison_experiments()

>>>>>>> 09478e0fb37821ea5f0745798a58211fdd17090d
