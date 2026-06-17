"""阶段0：原始数据基线实验。
在原始不平衡数据（无任何增强）上训练 yolov8n，作为最终对照。

Usage:
    python run_baseline.py
"""

from __future__ import annotations

from pathlib import Path

from training.trainers.yolo_trainer import CustomYoloTrainer


def main() -> None:
    exp_dir = Path("experiments") / "baseline" / "weights" / "best.pt"

    if exp_dir.exists():
        print(f"[SKIP] Phase 0 Baseline 已完成 → experiments/baseline/")
        return

    print(f"\n{'=' * 60}")
    print("  Phase 0: Baseline（原始不平衡数据，关闭所有增强）")
    print(f"{'=' * 60}")

    trainer = CustomYoloTrainer(
        model_variant="yolov8n.pt",
        experiment_name="baseline",
        stage="",  # 直接输出到 experiments/baseline/
    )
    trainer.train(
        data_config="configs/dataset.yaml",  # 原始不平衡数据
        epochs=100,
        batch=16,
        imgsz=640,
        mosaic=0.0,
        mixup=0.0,
    )


if __name__ == "__main__":
    main()
