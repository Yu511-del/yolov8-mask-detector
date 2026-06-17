# 基于 YOLOv8 的实时口罩佩戴检测系统

> **课程**：数据分析与数据挖掘 ｜ **任务**：端到端目标检测系统开发
> **数据集**：[Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) (Kaggle, 853 张, 3 类别)

---

## 1. 项目概述

构建一个端到端目标检测系统，识别图像中人脸是否佩戴口罩。涵盖数据预处理、格式转换、数据增强、模型训练、消融实验、模型评估及推理加速全流程。

**三类检测目标**：

| ID | 类别 | 含义 |
|:--:|:---|:---|
| 0 | `with_mask` | 正确佩戴口罩 |
| 1 | `without_mask` | 未佩戴口罩 |
| 2 | `mask_weared_incorrect` | 不规范佩戴 |

**实验结论摘要**：

- **最佳增强策略**：Mosaic + MixUp（协同比 Baseline 提升 6.9% mAP@50）
- **最佳模型**：YOLOv8m（mAP@50 = 0.8621）; 性价比首选 YOLOv8n（仅低 0.23%, 参数少 8 倍）
- **Copy-Paste 无效**：4 组对照实验指标完全一致
- 详细实验数据与分析见 [`experiment_results.md`](experiment_results.md)

---

## 2. 环境配置

| 组件 | 版本/规格 |
|:---|:---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0.0 (CUDA) |
| ultralytics | ≥ 8.0.0 |
| GPU | NVIDIA RTX 4090 / 24GB（实验使用） |

```bash
pip install -r requirements.txt
```

---

## 3. 快速开始

### 3.1 数据准备

```bash
python scripts/download_dataset.py          # 下载原始数据集
python scripts/voc2yolo.py                  # VOC XML → YOLO txt，含分层 8:1:1 划分
python scripts/analyze_data.py              # （可选）类别分布分析
```

### 3.2 生成均衡数据集

```bash
python data/augment.py --target_ratio 0.6
```

输出至 `data/augmented/`，配置文件 `configs/dataset_balanced.yaml`。

### 3.3 阶段0：原始数据基线

```bash
python run_baseline.py
```

在原始不平衡数据上训练 YOLOv8n，关闭所有在线增强。输出至 `experiments/baseline/`。

### 3.4 阶段2：增强策略消融

```bash
python run_ablation.py
```

8 组实验，支持断点续跑（已完成的实验会自动跳过）。

### 3.5 阶段3：模型尺度对比

```bash
python run_scale.py
```

### 3.6 模型评估

```bash
# 评估 BestModel（mAP + FPS）
python evaluate.py --weights experiments/scale/scale_m/weights/best.pt \
    --data configs/dataset_balanced.yaml --split val

# 评估其他模型
python evaluate.py --weights experiments/ablation/abl5_mosaic_mixup/weights/best.pt \
    --data configs/dataset_balanced.yaml --split val
```

---

## 4. 项目结构

```
├── configs/
│   ├── dataset.yaml                    # 原始数据集配置
│   └── dataset_balanced.yaml           # 均衡数据集配置
│
├── scripts/                            # 数据管道
│   ├── download_dataset.py             # Kaggle 数据集下载
│   ├── voc2yolo.py                     # VOC XML → YOLO txt 转换
│   └── analyze_data.py                 # 类别分布 EDA
│
├── data/
│   ├── augment.py                      # 离线数据增强
│   ├── processed/                      # 原始数据 (gitignored)
│   └── augmented/                      # 均衡数据 (gitignored)
│
├── training/trainers/
│   └── yolo_trainer.py                 # CustomYoloTrainer 封装类
│
├── experiments/                        # 训练产出
│   ├── baseline/                       #   阶段0 原始基线
│   ├── ablation/                       #   阶段2 8组消融
│   └── scale/                          #   阶段3 n/s/m 尺度对比
│
├── run_baseline.py                     # 阶段0 执行入口
├── run_ablation.py                     # 阶段2 执行入口
├── run_scale.py                        # 阶段3 执行入口
├── evaluate.py                         # 模型评估（mAP / FPS）
├── experiment_results.md               # 完整实验结果与分析
└── requirements.txt
```

---

## 5. 学术诚信声明

本项目除引用 Ultralytics YOLOv8 开源框架外，核心实验设计、数据增强策略、消融分析及代码实现均为原创工作。

---

## License

MIT. See [LICENSE](LICENSE).
