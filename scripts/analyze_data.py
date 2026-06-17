"""分析 Face Mask Detection 数据集的长尾分布，生成可视化图表。"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def count_classes_in_split(label_dir: Path) -> Counter:
    """统计一个划分中各类别的实例数量。

    Args:
        label_dir: YOLO 格式标注文件目录。

    Returns:
        {class_id: count} 的 Counter 对象。
    """
    counter = Counter()
    for txt_file in Path(label_dir).glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    counter[cls_id] += 1
    return counter


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    labels_base = project_root / "data" / "processed" / "labels"

    splits = ["train", "val", "test"]
    all_counts = Counter()
    split_counts = {}

    for split in splits:
        label_dir = labels_base / split
        if label_dir.exists():
            c = count_classes_in_split(label_dir)
            split_counts[split] = c
            all_counts.update(c)

    class_names = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}

    # 准备绘图数据
    categories = [class_names[i] for i in sorted(all_counts.keys())]
    counts = [all_counts[i] for i in sorted(all_counts.keys())]

    # 绘制柱状图
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=["#2ecc71", "#e74c3c", "#f39c12"])

    # 在柱子上方标注数值
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.title("Long-Tail Distribution of Mask Wearing Classes", fontsize=14)
    plt.ylabel("Number of Instances", fontsize=12)
    plt.xlabel("Class", fontsize=12)

    # 标注不平衡比例
    max_count = max(counts)
    min_count = min(counts)
    ratio = max_count / min_count
    plt.figtext(
        0.15, 0.85,
        f"Imbalance ratio: {ratio:.1f}:1",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # 保存图表
    output_dir = project_root / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "longtail_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "longtail_distribution.pdf", bbox_inches="tight")
    print(f"图表已保存至: {output_dir / 'longtail_distribution.png'}")

    # 打印统计信息
    print("\n各类别分布（全部划分合计）:")
    for cls_id in sorted(all_counts.keys()):
        print(f"  {class_names[cls_id]:25s}: {all_counts[cls_id]:5d} 个实例")
    print(f"\n总实例数: {sum(all_counts.values())}")
    print(f"不平衡比例 (最大/最小): {ratio:.2f}")


if __name__ == "__main__":
    main()
