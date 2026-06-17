"""将 PASCAL VOC XML 标注转换为 YOLOv8 TXT 格式，并进行分层 8:1:1 划分。"""

import glob
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

CLASS_MAP = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
RANDOM_SEED = 42
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / val / test


def parse_xml(xml_path: str) -> tuple[str, list]:
    """解析 PASCAL VOC XML 文件。

    Args:
        xml_path: XML 文件路径。

    Returns:
        (文件名, [ (cls_id, x_center, y_center, box_width, box_height), ... ])
        坐标均已归一化到 [0, 1]。
        标记为 <difficult>1</difficult> 的目标会被跳过。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue

        name = obj.find("name").text
        cls_id = CLASS_MAP[name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 归一化 YOLO 格式: [class_id, x_center, y_center, width, height]
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        boxes.append((cls_id, x_center, y_center, box_width, box_height))

    return filename, boxes


def stratified_split(records: list, ratios: tuple, seed: int) -> list:
    """按主类别进行分层划分。

    每张图片按其出现最多的类别分组，然后在组内按比例随机划分，
    确保 train/val/test 三个集合的类别分布一致。

    Args:
        records: [(filename, boxes), ...] 格式的记录列表。
        ratios: (train_ratio, val_ratio, test_ratio)。
        seed: 随机种子。

    Returns:
        [train_records, val_records, test_records]。
    """
    random.seed(seed)

    # 按主类别分组
    groups = defaultdict(list)
    for rec in records:
        boxes = rec[1]
        if not boxes:
            continue
        class_counts = Counter(b[0] for b in boxes)
        primary_class = class_counts.most_common(1)[0][0]
        groups[primary_class].append(rec)

    splits = [[], [], []]
    for group in groups.values():
        random.shuffle(group)
        n = len(group)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        splits[0].extend(group[:n_train])
        splits[1].extend(group[n_train:n_train + n_val])
        splits[2].extend(group[n_train + n_val:])

    return splits


def write_yolo_label(boxes: list, txt_path: Path) -> None:
    """写入 YOLO TXT 标注文件（6 位小数精度）。

    Args:
        boxes: [(cls_id, xc, yc, bw, bh), ...]。
        txt_path: 输出文件路径。
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, bw, bh in boxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_images = project_root / "data" / "raw" / "images"
    src_annotations = project_root / "data" / "raw" / "annotations"
    dst_images = project_root / "data" / "processed" / "images"
    dst_labels = project_root / "data" / "processed" / "labels"

    # 解析所有 XML
    xml_paths = sorted(glob.glob(str(src_annotations / "*.xml")))
    print(f"找到 {len(xml_paths)} 个标注文件。")

    records = []
    parse_errors = 0
    for xml_path in xml_paths:
        try:
            records.append(parse_xml(xml_path))
        except Exception as e:
            print(f"  跳过 {Path(xml_path).name}: {e}")
            parse_errors += 1

    print(f"成功解析 {len(records)} 个标注 ({parse_errors} 个错误)。")

    # 分层划分
    splits = stratified_split(records, SPLIT_RATIOS, RANDOM_SEED)
    split_names = ["train", "val", "test"]

    # 创建输出目录
    for name in split_names:
        (dst_images / name).mkdir(parents=True, exist_ok=True)
        (dst_labels / name).mkdir(parents=True, exist_ok=True)

    # 写入标注 + 复制图片
    split_stats = {}
    all_class_counts = Counter()
    src_images_map = {p.stem: p for p in Path(src_images).iterdir() if p.is_file()}

    for name, split_records in zip(split_names, splits):
        class_counts = Counter()
        for filename, boxes in split_records:
            stem = Path(filename).stem
            txt_path = dst_labels / name / f"{stem}.txt"
            write_yolo_label(boxes, txt_path)

            # 查找匹配的图片（保留原始扩展名）
            ext = None
            for candidate in src_images_map.values():
                if candidate.stem == stem:
                    ext = candidate.suffix
                    break
            if ext is None:
                print(f"  Warning: 未找到图片 {filename}")
                continue
            img_src = src_images / f"{stem}{ext}"
            img_dst = dst_images / name / f"{stem}{ext}"
            shutil.copy2(img_src, img_dst)

            class_counts.update(b[0] for b in boxes)

        split_stats[name] = {"images": len(split_records), "instances": class_counts}
        all_class_counts.update(class_counts)

    # 打印统计信息
    cls_names = {v: k for k, v in CLASS_MAP.items()}
    print("\n" + "=" * 50)
    print("数据集转换摘要")
    print("=" * 50)
    total_images = sum(s["images"] for s in split_stats.values())
    print(f"总图片数:          {total_images}")
    for name in split_names:
        s = split_stats[name]
        print(f"  {name:6s}:        {s['images']:4d} 张图片  "
              f"{sum(s['instances'].values()):4d} 个实例")
    print(f"\n总实例数:          {sum(all_class_counts.values())}")
    print("各类别分布:")
    for cls_id, count in sorted(all_class_counts.items()):
        print(f"  {cls_names[cls_id]:30s} (id={cls_id}): {count:5d}")
    print("=" * 50)
    print("\n输出结构:")
    for name in split_names:
        print(f"  data/processed/images/{name}/")
        print(f"  data/processed/labels/{name}/")


if __name__ == "__main__":
    main()
