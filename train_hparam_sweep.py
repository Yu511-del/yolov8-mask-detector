from __future__ import annotations

import csv
from pathlib import Path

from training.trainers.yolo_trainer import CustomYoloTrainer


REPO_ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = REPO_ROOT / "runs" / "detect" / "experiments" / "face_mask_detection" / "hparam_sweep_summary.csv"


EXPERIMENTS = [
    {
        "name": "hparam_baseline",
        "label": "Baseline",
        "params": {
            "model_variant": "yolov8m.pt",
            "data_config": "configs/dataset.yaml",
            "epochs": 80,
            "imgsz": 640,
            "batch": 16,
            "patience": 20,
            "amp": True,
            "mosaic": 1.0,
            "mixup": 0.1,
            "lr0": 0.001,
            "lrf": 0.01,
            "optimizer": "SGD",
        },
    },
    {
        "name": "hparam_lr0_0005",
        "label": "Lower LR",
        "params": {
            "model_variant": "yolov8m.pt",
            "data_config": "configs/dataset.yaml",
            "epochs": 80,
            "imgsz": 640,
            "batch": 16,
            "patience": 20,
            "amp": True,
            "mosaic": 1.0,
            "mixup": 0.1,
            "lr0": 0.0005,
            "lrf": 0.01,
            "optimizer": "SGD",
        },
    },
    {
        "name": "hparam_lr0_0005_adamw",
        "label": "Lower LR + AdamW",
        "params": {
            "model_variant": "yolov8m.pt",
            "data_config": "configs/dataset.yaml",
            "epochs": 80,
            "imgsz": 640,
            "batch": 16,
            "patience": 20,
            "amp": True,
            "mosaic": 1.0,
            "mixup": 0.1,
            "lr0": 0.0005,
            "lrf": 0.01,
            "optimizer": "AdamW",
        },
    },
    {
        "name": "hparam_batch32",
        "label": "Larger Batch",
        "params": {
            "model_variant": "yolov8m.pt",
            "data_config": "configs/dataset.yaml",
            "epochs": 80,
            "imgsz": 640,
            "batch": 32,
            "patience": 20,
            "amp": True,
            "mosaic": 1.0,
            "mixup": 0.1,
            "lr0": 0.0005,
            "lrf": 0.01,
            "optimizer": "AdamW",
        },
    },
    {
        "name": "hparam_patience15",
        "label": "Early Stop Earlier",
        "params": {
            "model_variant": "yolov8m.pt",
            "data_config": "configs/dataset.yaml",
            "epochs": 80,
            "imgsz": 640,
            "batch": 16,
            "patience": 15,
            "amp": True,
            "mosaic": 0.5,
            "mixup": 0.2,
            "lr0": 0.0005,
            "lrf": 0.05,
            "optimizer": "AdamW",
            "degrees": 10,
            "scale": 0.5,
        },
    },
]


def _read_latest_metrics(results_csv_path: Path) -> dict[str, str]:
    with results_csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return rows[-1]


def _collect_run_metrics(run_name: str) -> dict[str, str]:
    results_csv = REPO_ROOT / "runs" / "detect" / "experiments" / "face_mask_detection" / run_name / "results.csv"
    if not results_csv.exists():
        return {
            "run_name": run_name,
            "status": "missing_results",
        }

    metrics = _read_latest_metrics(results_csv)
    return {
        "run_name": run_name,
        "status": "ok",
        "epoch": metrics.get("epoch", ""),
        "precision": metrics.get("metrics/precision(B)", ""),
        "recall": metrics.get("metrics/recall(B)", ""),
        "mAP50": metrics.get("metrics/mAP50(B)", ""),
        "mAP50_95": metrics.get("metrics/mAP50-95(B)", ""),
        "train_time": metrics.get("time", ""),
    }


def run_hparam_sweep() -> None:
    summary_rows: list[dict[str, str]] = []

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        print("\n" + "=" * 70)
        print(f"STEP {index}: {experiment['label']} -> {experiment['name']}")
        print("=" * 70)

        params = dict(experiment["params"])
        model_variant = params.pop("model_variant")
        data_config = params.pop("data_config")

        trainer = CustomYoloTrainer(
            model_variant=model_variant,
            project_name="face_mask_detection",
            experiment_name=experiment["name"],
        )

        trainer.train(data_config=data_config, **params)
        summary_rows.append(_collect_run_metrics(experiment["name"]))

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["run_name", "status", "epoch", "precision", "recall", "mAP50", "mAP50_95", "train_time"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print("Hyperparameter sweep completed")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_hparam_sweep()
