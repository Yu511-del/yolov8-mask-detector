"""YOLOv8 训练器模块 —— 面向对象封装，集成自动评估与指标记录。

所有训练结果统一存放在 experiments/ 目录下，按训练阶段组织：
    experiments/
    ├── baseline/               # Phase 0: 基准实验
    ├── ablation/               # Phase 2: 消融实验
    │   ├── abl1_baseline/
    │   ├── abl2_mosaic/
    │   └── ...
    └── hparam_sweep/           # 超参数搜索
"""

from __future__ import annotations

import atexit
import json
import os
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

# ── 临时文件管理 ──────────────────────────────────────────────
# _resolve_data_config() 会创建临时 YAML 文件来解决相对路径问题，
# 所有临时文件在此追踪，进程退出时自动清理。

_TEMP_FILES: list[str] = []


def _cleanup_temp_files() -> None:
    """清理所有由 _resolve_data_config 创建的临时 YAML 文件。"""
    for path in _TEMP_FILES:
        try:
            os.unlink(path)
        except OSError:
            pass
    _TEMP_FILES.clear()


atexit.register(_cleanup_temp_files)


def _resolve_data_config(data_config: str | os.PathLike) -> str:
    """将 YOLO 数据集 YAML 配置中的相对 path 转换为绝对路径。

    YOLO 的 data.yaml 中 path 字段如果是相对路径，会相对于「运行时的工作目录」
    而非「配置文件所在目录」来解析。当从不同位置运行训练脚本时，这会导致
    数据集路径找不到。此函数将相对路径展开为绝对路径，写入临时文件后返回。

    Args:
        data_config: 数据集 YAML 配置文件路径。

    Returns:
        临时 YAML 文件的绝对路径（进程退出时自动清理），
        如果不需要路径转换则返回原路径。
    """
    if not isinstance(data_config, (str, os.PathLike)):
        return data_config

    config_path = Path(data_config).expanduser().resolve()

    # 非 YAML 文件直接返回
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        return str(config_path)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    dataset_root = config.get("path")
    if dataset_root:
        dataset_root = Path(dataset_root)
        if not dataset_root.is_absolute():
            # 以配置文件所在仓库根目录为基准展开相对路径
            repo_root = Path(__file__).resolve().parents[2]
            dataset_root = (repo_root / dataset_root).resolve()
        config["path"] = str(dataset_root)

    # 写入临时文件，进程退出时自动删除
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=config_path.suffix,
        delete=False,
        encoding="utf-8",
    )
    with temp_file as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    _TEMP_FILES.append(temp_file.name)
    return temp_file.name


# ── 训练器类 ───────────────────────────────────────────────────


class CustomYoloTrainer:
    """面向对象的 YOLOv8 训练器。

    特性：
    - 延迟加载模型（仅在首次访问时下载权重）
    - 自动将相对数据集路径转换为绝对路径
    - 训练完成后自动保存指标摘要 JSON
    - 统一输出到 experiments/ 目录下

    Usage:
        # 基础用法
        trainer = CustomYoloTrainer("yolov8n.pt", experiment_name="baseline")
        trainer.train(data_config="configs/dataset.yaml")

        # 消融实验（结果存入 experiments/ablation/xxx）
        trainer = CustomYoloTrainer(
            "yolov8n.pt",
            experiment_name="abl2_mosaic",
            stage="ablation",
        )
        trainer.train(data_config="configs/dataset_balanced.yaml", mosaic=1.0)
    """

    def __init__(
        self,
        model_variant: str = "yolov8n.pt",
        experiment_name: str = "baseline",
        stage: str = "",
    ) -> None:
        """初始化训练器。

        Args:
            model_variant: YOLOv8 模型变体，如 'yolov8n.pt', 'yolov8s.pt' 等。
            experiment_name: 实验名称，作为 experiments/ 下的子目录名。
            stage: 可选，训练阶段子目录（如 'ablation', 'hparam_sweep'）。
                   结果将存入 experiments/{stage}/{experiment_name}/。
                   不传则直接存入 experiments/{experiment_name}/。
        """
        self.model_variant = model_variant
        self.experiment_name = experiment_name
        self.stage = stage

        # 确定输出根目录
        repo_root = Path(__file__).resolve().parents[2]
        if stage:
            self.project_path = repo_root / "experiments" / stage
        else:
            self.project_path = repo_root / "experiments"

        self._model: YOLO | None = None
        self._train_results = None

    # ── 属性 ───────────────────────────────────────────────

    @property
    def model(self) -> YOLO:
        """延迟加载 YOLO 模型（首次访问时才下载/加载权重文件）。"""
        if self._model is None:
            print(f"  加载模型: {self.model_variant}")
            self._model = YOLO(self.model_variant)
        return self._model

    @property
    def output_dir(self) -> Path:
        """当前实验的输出目录。"""
        return self.project_path / self.experiment_name

    # ── 核心方法 ───────────────────────────────────────────

    def train(
        self,
        data_config: str = "configs/dataset.yaml",
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 50,
        amp: bool = True,
        mosaic: float = 1.0,
        mixup: float = 0.0,
        **extra_train_args,
    ):
        """执行模型训练，完成后自动保存指标摘要。

        Args:
            data_config: 数据集 YAML 配置文件路径。
            epochs: 最大训练轮数。
            imgsz: 输入图像尺寸。
            batch: 批处理大小。
            patience: 早停容忍轮数（0 表示关闭早停）。
            amp: 是否开启混合精度训练（AMP）。
            mosaic: Mosaic 数据增强概率（0.0 = 关闭, 1.0 = 全程开启）。
            mixup: MixUp 数据增强概率。
            **extra_train_args: 其他 ultralytics 训练参数
                （如 copy_paste, close_mosaic, lr0, optimizer 等）。

        Returns:
            ultralytics 训练结果对象。
        """
        print(f"\n{'=' * 60}")
        print(f"  实验名称 : {self.experiment_name}")
        print(f"  模型      : {self.model_variant}")
        print(f"  AMP      : {amp}, 早停 Patience: {patience}")
        print(f"  Mosaic   : {mosaic}, MixUp: {mixup}")
        print(f"  输出目录  : {self.output_dir}")
        print(f"{'=' * 60}\n")

        self._train_results = self.model.train(
            data=_resolve_data_config(data_config),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            amp=amp,
            project=str(self.project_path),
            name=self.experiment_name,
            mosaic=mosaic,
            mixup=mixup,
            exist_ok=True,
            **extra_train_args,
        )

        # 训练完成后自动保存指标摘要
        self._save_metrics_summary()

        return self._train_results

    def validate(
        self,
        data_config: str = "configs/dataset.yaml",
        split: str = "val",
    ):
        """在指定的数据集划分上评估当前模型。

        Args:
            data_config: 数据集 YAML 配置文件路径。
            split: 数据集划分，可选 'train', 'val', 'test'。

        Returns:
            ultralytics 验证指标对象（包含 mAP, Precision, Recall 等）。
        """
        print(f"\n  验证模型于 {split} 集 (配置: {data_config}) ...")
        metrics = self.model.val(
            data=_resolve_data_config(data_config),
            split=split,
            project=str(self.project_path),
            name=self.experiment_name,
            exist_ok=True,
        )
        return metrics

    # ── 内部方法 ───────────────────────────────────────────

    def _save_metrics_summary(self) -> None:
        """将训练结果的关键指标保存为 metrics_summary.json。

        便于后续在不同实验之间进行横向对比分析。
        """
        if self._train_results is None:
            return

        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        summary: dict = {
            "experiment": self.experiment_name,
            "stage": self.stage or "root",
            "model": self.model_variant,
        }

        # 从训练结果的 results_dict 中提取最终 epoch 指标
        if hasattr(self._train_results, "results_dict"):
            rd = self._train_results.results_dict
            summary["metrics"] = {
                "mAP50": round(float(rd.get("metrics/mAP50(B)", 0)), 4),
                "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0)), 4),
                "precision": round(float(rd.get("metrics/precision(B)", 0)), 4),
                "recall": round(float(rd.get("metrics/recall(B)", 0)), 4),
            }

        summary_path = output_dir / "metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"  指标摘要已保存: {summary_path}")
