"""模型评估脚本 —— 计算 mAP, Precision, Recall 及推理延迟 (FPS)。"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def evaluate_model(
    model_path: str,
    data_config: str = "configs/dataset.yaml",
    project: str = "experiments",
    name: str = "evaluation",
    split: str = "test",
) -> dict:
    """对训练好的模型在测试集上进行全面评估。

    Args:
        model_path: 训练好的模型权重路径。
        data_config: 数据集 YAML 配置文件路径。
        project: 结果输出项目目录。
        name: 本次评估的名称。

    Returns:
        包含 mAP, Precision, Recall, FPS 的指标字典。
    """
    print(f"==> 加载模型: {model_path}")
    model = YOLO(model_path)

    # 1. 在测试集上运行验证
    start_time = time.time()
    results = model.val(
        data=data_config,
        split=split,
        project=project,
        name=name,
        exist_ok=True,
    )
    end_time = time.time()

    # 2. 提取核心指标
    mAP50 = results.box.map50
    mAP95 = results.box.map
    precision = results.box.mp
    recall = results.box.mr

    # 3. 计算推理速度
    speed_dict = results.speed
    preprocess = speed_dict.get("preprocess", 0)
    inference = speed_dict.get("inference", 0)
    postprocess = speed_dict.get("postprocess", 0)
    total_time_per_img = preprocess + inference + postprocess
    fps = 1000 / total_time_per_img if total_time_per_img > 0 else 0

    print("\n" + "=" * 40)
    print(f"  评估结果: {model_path}")
    print(f"  mAP@0.5:       {mAP50:.4f}")
    print(f"  mAP@0.5:0.95:  {mAP95:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  速度:          {total_time_per_img:.2f} ms/img (FPS: {fps:.1f})")
    print("=" * 40 + "\n")

    # 4. 保存为 JSON 报告
    report = {
        "model": model_path,
        "mAP50": float(mAP50),
        "mAP95": float(mAP95),
        "precision": float(precision),
        "recall": float(recall),
        "fps": float(fps),
        "ms_per_img": float(total_time_per_img),
    }

    report_path = Path(project) / name / "metrics_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"  指标报告已保存至: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 模型评估脚本")
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/baseline/weights/best.pt",
        help="训练好的模型权重路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/dataset.yaml",
        help="数据集 YAML 配置文件路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="评估数据集划分（默认 test）",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="test_set_evaluation",
        help="评估结果存放目录名",
    )

    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: 权重文件不存在: {args.weights}")
        print("请确保已完成训练，或通过 --weights 参数指定正确的路径。")
    else:
        evaluate_model(args.weights, data_config=args.data, name=args.name, split=args.split)
