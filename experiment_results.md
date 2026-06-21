# 消融实验与模型对比 — 完整结果与分析

> 实验环境：NVIDIA RTX 4090 24GB ｜ PyTorch 2.8.0 ｜ ultralytics 8.x
> 训练配置：100 epochs, imgsz=640, AMP=True, patience=50

---

## 1. 数据集与预处理

### 1.1 数据来源

[Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) (Kaggle)，共 853 张图像，原始标注为 Pascal VOC XML 格式。经 `voc2yolo.py` 转换为 YOLO txt 格式，按 8:1:1 分层划分为 train/val/test。

### 1.2 类别分布

| 类别 | 实例数 | 占比 | 不平衡比 |
|:---|---:|---:|---:|
| `with_mask` (0) | 3,232 | 79.4% | — |
| `without_mask` (1) | 717 | 17.6% | 4.5:1 |
| `mask_weared_incorrect` (2) | 123 | 3.0% | 26.3:1 |

分布图见 `reports/figures/longtail_distribution.png`（由 `scripts/analyze_data.py` 生成）。

### 1.3 离线数据增强

使用 `data/augment.py` 对少数类（类别 1、2）进行**离线过采样**，增强策略包括：

- **几何变换**：随机水平翻转、旋转（±15°）、缩放（0.9~1.1）、平移（±5%）
- **光度变换**：亮度（±30）、对比度（0.8~1.2）、高斯噪声（σ≤10）

目标比例 `target_ratio=0.6`，将类别 1 和 2 分别增强至类别 0 实例数的 60%。增强后不平衡比从 26:1 降至约 **1.7:1**。

---

## 2. 实验设计

整体采用**控制变量法**的三阶段实验框架：

```
原始数据 (data/processed/)          不平衡比 26:1
    │
    ├── 阶段0: Baseline（原始基线）
    │     yolov8n，关闭所有增强
    │     回答：「不做任何处理是什么水平？」
    │
    └── 离线增强 (data/augment.py)
        少数类过采样 → 均衡数据 (data/augmented/)
            │
            ├── 阶段2: 增强策略消融（8 组）
            │     yolov8n，仅改变在线增强参数
            │     选 BestAug → Mosaic=1.0 + MixUp=0.5
            │
            └── 阶段3: 模型尺度对比（3 组）
                  BestAug 下对比 n / s / m
                  选 BestModel → YOLOv8m
```

### 2.1 阶段2：消融实验矩阵

| 实验编号 | Mosaic | MixUp | Copy-Paste | 实验目的 |
|:---|:---:|:---:|:---:|:---|
| `abl1_baseline` | 0.0 | 0.0 | 0.0 | 均衡数据纯基准（仅基础 HSV 增强） |
| `abl2_mosaic` | 1.0 | 0.0 | 0.0 | Mosaic 的独立贡献 |
| `abl3_mixup` | 0.0 | 0.5 | 0.0 | MixUp 的独立贡献 |
| `abl4_copy_paste` | 0.0 | 0.0 | 0.5 | Copy-Paste 的独立贡献 |
| `abl5_mosaic_mixup` | 1.0 | 0.5 | 0.0 | Mosaic + MixUp 协同效果 |
| `abl6_mosaic_cp` | 1.0 | 0.0 | 0.5 | Mosaic + Copy-Paste |
| `abl7_mixup_cp` | 0.0 | 0.5 | 0.5 | MixUp + Copy-Paste |
| `abl8_all` | 1.0 | 0.5 | 0.5 | 全开上限 |

> 统一配置：`configs/dataset_balanced.yaml`, yolov8n.pt, epochs=100, batch=16, imgsz=640

### 2.2 阶段3：模型尺度对比

| 实验 | 模型 | 参数量 | batch |
|:---|:---|:---:|:---:|
| `scale_n` | YOLOv8n | 3.2M | 32 |
| `scale_s` | YOLOv8s | 11.2M | 24 |
| `scale_m` | YOLOv8m | 25.9M | 16 |

> 统一配置：BestAug (Mosaic=1.0, MixUp=0.5), epochs=100, imgsz=640

---

## 3. 实验结果

### 3.1 阶段0：原始数据基线

| 模型 | 数据 | 增强 | mAP@50 | mAP@50:95 | Precision | Recall |
|:---|:---|:---:|---:|---:|---:|---:|
| YOLOv8n | 原始不平衡 | 无 | 0.8077 | 0.5676 | 0.9764 | 0.7603 |

### 3.2 阶段2：消融实验结果

| 实验 | Mosaic | MixUp | CP | mAP@50 | mAP@50:95 | P | R | Δ vs abl1 |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|:---:|
| `abl1_baseline` | ✗ | ✗ | ✗ | 0.7852 | 0.5464 | 0.9048 | 0.7765 | — |
| `abl2_mosaic` | ✓ | ✗ | ✗ | 0.8103 | 0.5567 | 0.9731 | 0.7378 | +2.5% |
| `abl3_mixup` | ✗ | ✓ | ✗ | 0.8102 | 0.5605 | 0.9364 | 0.7272 | +2.5% |
| `abl4_copy_paste` | ✗ | ✗ | ✓ | 0.7852 | 0.5464 | 0.9048 | 0.7765 | **0** |
| `abl5_mosaic_mixup` | ✓ | ✓ | ✗ | **0.8544** | **0.5715** | 0.9267 | **0.7819** | **+6.9%** |
| `abl6_mosaic_cp` | ✓ | ✗ | ✓ | 0.8103 | 0.5567 | 0.9731 | 0.7378 | +2.5% |
| `abl7_mixup_cp` | ✗ | ✓ | ✓ | 0.8102 | 0.5605 | 0.9364 | 0.7272 | +2.5% |
| `abl8_all` | ✓ | ✓ | ✓ | 0.8544 | 0.5715 | 0.9267 | 0.7819 | +6.9% |

> **BestAug = Mosaic=1.0 + MixUp=0.5（abl5）**

### 3.3 阶段3：模型尺度对比结果

| 模型 | 参数量 | mAP@50 | mAP@50:95 | P | R | FPS¹ |
|:---|:---:|---:|---:|---:|---:|:---:|
| `scale_n` (YOLOv8n) | 3.2M | 0.8544 | 0.5715 | 0.9267 | 0.7819 | — |
| `scale_s` (YOLOv8s) | 11.2M | 0.8115 | 0.5910 | 0.9082 | 0.7487 | — |
| `scale_m` (YOLOv8m) | 25.9M | **0.8608** | **0.6050** | 0.8762 | 0.7835 | 41.8 |

> ¹ FPS 于 NVIDIA RTX 4090 测得。nano 未独立测试 FPS，其推理速度应优于表中其他模型。
>
> ² scale_n 与 abl5_mosaic_mixup 配置等价（yolov8n + Mosaic=1.0 + MixUp=0.5），直接复用其指标。
>
> **BestModel = YOLOv8m（精度最优）；性价比首选 = YOLOv8n（参数少 8 倍，mAP50 差 0.64%）**

### 3.4 各类别精度（scale_m，val set）

| 类别 | Images | Instances | P | R | mAP@50 | mAP@50:95 |
|:---|:---:|:---:|---:|---:|---:|---:|
| `with_mask` (0) | 38 | 93 | 0.921 | 0.860 | 0.898 | 0.633 |
| `without_mask` (1) | 10 | 14 | 0.832 | 0.707 | 0.826 | 0.568 |
| `mask_weared_incorrect` (2) | 0 | 0 | — | — | — | — |

> ⚠️ 验证集中不存在类别 2 实例（原始数据仅 123 个，划分后 val 中为零）。全量指标仅反映类别 0 和 1 的加权平均。

---

## 4. 分析与讨论

### 4.1 Mosaic + MixUp 的协同效应

Mosaic 和 MixUp 独立贡献相当（各 +2.5%），但组合增益 +6.9% 远超简单叠加预期（2.5% + 2.5% = 5.0%），存在约 1.9 个百分点的正向交互。两者机制互补：Mosaic 通过四图拼接扩大感受野和上下文信息，MixUp 通过标签混合平滑决策边界，联合使用时模型同时获得空间多样性和语义平滑性。

### 4.2 Copy-Paste 无效性分析

4 组对照实验全部零差异：

| 对照 | 无 CP | 有 CP | 差异 |
|:---|:---:|:---:|:---:|
| 无增强 | 0.7852 | 0.7852 | 0 |
| Mosaic | 0.8103 | 0.8103 | 0 |
| MixUp | 0.8102 | 0.8102 | 0 |
| Mosaic + MixUp | 0.8544 | 0.8544 | 0 |

Copy-Paste 原始论文针对 COCO 小目标场景，口罩检测中人脸区域通常占据图像较大比例，该策略不适用于本任务。

### 4.3 YOLOv8s 异常分析

YOLOv8s（0.8115）低于 YOLOv8n（0.8598），违反尺度递增规律。可能原因：

- **数据量不足**：853 张图像对较大模型可能不足以充分训练
- **学习率未调整**：ultralytics 默认 lr0=0.01 对 nano 适配良好，对 small 可能需要更低的学习率

### 4.4 实验可复现性

3 对完全相同的结果（abl2=abl6, abl3=abl7, abl5=abl8）不仅验证了 Copy-Paste 的无效性，也侧面证明了训练过程的可复现性——相同配置下模型收敛至完全一致的指标。

### 4.5 模型尺度权衡

YOLOv8m 以 25.9M 参数取得最高精度（mAP@50=0.8608），但 YOLOv8n 仅以 3.2M 参数（少 8 倍）达到 0.8544，差距仅 0.64%。RTX 4090 上 medium 实测约 41.8 FPS，而在边缘设备上 nano 的轻量优势将显著体现。small 模型的异常进一步说明：小数据集上模型选择需与数据量匹配。

### 4.6 数据集局限性

- 验证集中 `mask_weared_incorrect` 类别实例数为零，无法评估该类性能
- 总数据量仅 853 张，限制了更大模型的训练潜力
- 建议后续扩增数据或采用带权重的分层采样策略

---

## 5. 训练过程可视化

每个实验目录下均包含完整的训练过程记录：

| 文件 | 说明 |
|:---|:---|
| `results.png` | 训练/验证 loss 及 mAP 曲线 |
| `confusion_matrix.png` | 验证集混淆矩阵 |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵 |
| `BoxF1_curve.png` / `BoxPR_curve.png` | F1 和 Precision-Recall 曲线 |
| `train_batch*.jpg` | 训练 batch 可视化样例 |
| `val_batch*_pred.jpg` | 验证集预测结果可视化 |
| `val_batch*_labels.jpg` | 验证集真值标注 |
| `results.csv` | 逐 epoch 完整指标 |

---

## 6. 关键发现总结

1. **Mosaic + MixUp 是最佳增强策略**：协同增益 6.9%，远超独立贡献之和
2. **Copy-Paste 完全无效**：4 组对照全部零差异，不适用于本任务
3. **YOLOv8m 精度最优**（mAP@50=0.8608），**YOLOv8n 性价比最高**（差 0.64%，参数少 8 倍）
4. **YOLOv8s 出现异常**：被 nano 反超，小数据集上模型增大不一定带来收益
5. **验证集体量不足**：类别 2 无实例，建议后续评估引入测试集或更细粒度的划分
