"""消融实验独立运行脚本。

支持断点续跑：已完成的实验自动跳过，Ctrl+C 中断后重新执行即可继续。

Usage:
    python run_ablation.py
"""

from __future__ import annotations

from pathlib import Path

from training.trainers.yolo_trainer import CustomYoloTrainer

ABLATION_CONFIGS = [
    # (实验名称,         mosaic, mixup, copy_paste, close_mosaic)
    ("abl1_baseline",       0.0, 0.0, 0.0, 0),
    ("abl2_mosaic",         1.0, 0.0, 0.0, 0),
    ("abl3_mixup",          0.0, 0.5, 0.0, 0),
    ("abl4_copy_paste",     0.0, 0.0, 0.5, 0),
    ("abl5_mosaic_mixup",   1.0, 0.5, 0.0, 0),
    ("abl6_mosaic_cp",      1.0, 0.0, 0.5, 0),
    ("abl7_mixup_cp",       0.0, 0.5, 0.5, 0),
    ("abl8_all",            1.0, 0.5, 0.5, 0),
]


def main() -> None:
    errors: list[str] = []

    for name, mosaic, mixup, copy_paste, close_mosaic in ABLATION_CONFIGS:
        exp_dir = Path("experiments") / "ablation" / name / "weights" / "best.pt"

        if exp_dir.exists():
            print(f"[SKIP] {name} — 已完成")
            continue

        print(f"\n{'=' * 60}")
        print(f"  消融实验: {name}")
        print(f"  mosaic={mosaic}  mixup={mixup}  copy_paste={copy_paste}")
        print(f"{'=' * 60}")

        try:
            trainer = CustomYoloTrainer(
                model_variant="yolov8n.pt",
                experiment_name=name,
                stage="ablation",
            )
            trainer.train(
                data_config="configs/dataset_balanced.yaml",
                epochs=100,
                batch=16,
                imgsz=640,
                augment=True,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                close_mosaic=close_mosaic,
            )
        except KeyboardInterrupt:
            print(f"\n[中断] {name} 未完成，下次运行将从此继续。")
            break
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"[ERROR] {name}: {exc}")

    # 汇总
    completed = sum(
        1 for n, *_ in ABLATION_CONFIGS
        if (Path("experiments") / "ablation" / n / "weights" / "best.pt").exists()
    )
    print(f"\n{'=' * 60}")
    print(f"  完成: {completed}/{len(ABLATION_CONFIGS)}")
    if errors:
        print(f"  错误: {len(errors)}")
        for e in errors:
            print(f"    ✗ {e}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
