"""阶段3：模型尺度实验独立运行脚本。

在 BestAug（Mosaic=1.0, MixUp=0.5）下对比 yolov8n/s/m 三个尺度。

Usage:
    python run_scale.py
    python run_scale.py --resume   # 断点续跑（默认行为）
"""

from __future__ import annotations

from pathlib import Path

from training.trainers.yolo_trainer import CustomYoloTrainer

# BestAug 参数（来自消融实验 abl5）
BEST_AUG = {
    "mosaic": 1.0,
    "mixup": 0.5,
    "copy_paste": 0.0,
    "close_mosaic": 0,
}

SCALE_CONFIGS = [
    # (实验名称, 模型权重, batch,  备注)
    # RTX 4090 24G 绰绰有余，无需降 batch
    ("scale_n", "yolov8n.pt", 32, "nano — 消融实验中 abl5 已完成"),
    ("scale_s", "yolov8s.pt", 24, "small — 性价比首选"),
    ("scale_m", "yolov8m.pt", 16, "medium — 精度优先"),
]


def main() -> None:
    errors: list[str] = []

    for name, model_variant, batch, note in SCALE_CONFIGS:
        exp_dir = Path("experiments") / "scale" / name / "weights" / "best.pt"

        if exp_dir.exists():
            print(f"[SKIP] {name} ({note}) — 已完成")
            continue

        print(f"\n{'=' * 60}")
        print(f"  阶段3：模型尺度实验 — {name}")
        print(f"  模型: {model_variant}   batch: {batch}")
        print(f"  增强: Mosaic=1.0  MixUp=0.5  Copy-Paste=0")
        print(f"  备注: {note}")
        print(f"{'=' * 60}")

        try:
            trainer = CustomYoloTrainer(
                model_variant=model_variant,
                experiment_name=name,
                stage="scale",
            )
            trainer.train(
                data_config="configs/dataset_balanced.yaml",
                epochs=100,
                batch=batch,
                imgsz=640,
                patience=50,
                **BEST_AUG,
            )
        except KeyboardInterrupt:
            print(f"\n[中断] {name} 未完成，下次运行将从此继续。")
            break
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"[ERROR] {name}: {exc}")

    # 汇总
    completed = sum(
        1 for n, *_ in SCALE_CONFIGS
        if (Path("experiments") / "scale" / n / "weights" / "best.pt").exists()
    )
    print(f"\n{'=' * 60}")
    print(f"  完成: {completed}/{len(SCALE_CONFIGS)}")
    if errors:
        print(f"  错误: {len(errors)}")
        for e in errors:
            print(f"    ✗ {e}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
