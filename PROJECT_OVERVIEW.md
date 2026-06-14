# 基于 YOLOv8 的实时口罩佩戴检测系统 (课程项目指南)

> **课程名称**：数据分析与数据挖掘  
> **项目主题**：端到端目标检测系统开发  
> **核心目标**：实现对“佩戴口罩”、“未佩戴口罩”及“不规范佩戴”的实时检测，掌握深度学习全流程。

---

## 1. 项目任务与目标 (Project Mission)

根据《课程项目说明》，本项目已完成以下核心研究任务：
- [x] **数据工程**：VOC XML 格式转 YOLO txt 格式，实现 8:1:1 分层采样。
- [x] **模型训练**：基于 YOLOv8n/s 系列的对比实验，涵盖 Mosaic 增强、AMP 混合精度训练。
- [x] **调优策略**：集成早停机制 (Early Stopping) 与超参数敏感性分析。
- [x] **评估体系**：严格计算 mAP@0.5、mAP@0.5:0.95、Precision、Recall 及 FPS。
- [x] **实时推理**：支持图片批量推理与 OpenCV 摄像头实时检测 Demo。

---

## 2. 代码规范与架构 (Coding Standards)

本项目严格遵循教师提供的《深度学习项目代码规范指南》：

### 2.1 项目结构 (Directory Structure)
```text
D:\projects\yolomask\yolov8-mask-detector\
├── configs/           # YAML 配置文件 (数据集路径、类别定义)
├── data/              # 原始数据、处理后数据及增强脚本
├── models/            # 预训练模型与导出模型
├── scripts/           # 数据转换、EDA 分析脚本 (符合模块化要求)
├── training/          # 核心训练逻辑 (含 CustomYoloTrainer 类)
├── experiments/       # 实验记录、权重文件及可视化报告
├── tests/             # 模型推理与单元测试代码
├── utils/             # 工具函数 (日志、路径处理)
├── PROJECT_OVERVIEW.md # 本项目说明文档
└── train.py           # 项目运行总入口
```

### 2.2 核心设计模式
- **面向对象编程 (OOP)**：使用 `CustomYoloTrainer` 类封装 Ultralytics 引擎，实现训练逻辑与业务逻辑分离。
- **配置管理**：禁止“硬编码路径”，所有数据路径与超参数均通过 `configs/dataset.yaml` 及脚本参数传递。
- **可复现性**：固定随机种子，并在 `requirements.txt` 中指明具体的版本依赖。
- **PEP8 规范**：代码遵循标准的缩进、命名规范，并包含详尽的 Docstring 注释。

---

## 3. 计算环境与 Kaggle 云端训练 (Environment & Kaggle)

针对本地硬件限制（无独立 GPU），本项目采取**“本地开发 + 云端训练”**的双轨并行模式，确保实验的高效执行：

### 3.1 软硬件配置
*   **本地环境 (VSCode + CPU)**：用于代码编写、逻辑调试、数据集预处理脚本测试。在本地环境下，训练脚本仅限小批量样本测试以验证 Pipeline 连通性。
*   **训练环境 (Kaggle GPU)**：利用 Kaggle 提供的免费 **Tesla P100 / T4 x2** 加速器执行大规模模型训练、消融实验及全量数据集评估。

### 3.2 云端训练工作流
1.  **本地调试**：在 VSCode 中完成代码逻辑与规范化重构，确保无路径硬编码。
2.  **数据/脚本同步**：通过 GitHub 仓库或 Kaggle Dataset 将项目同步至 Kaggle Kernel。
3.  **高性能训练**：在 Kaggle 笔记中使用交互式命令或后台任务运行 `train.py`。
4.  **权重回传**：训练完成后，将生成的最佳权重 (`best.pt`) 及实验日志回传至本地 `experiments/` 目录，用于撰写实验报告及本地 CPU 推理展示。

---

## 4. 实验报告撰写指引 (Report Roadmap)


为满足 50% 权重的实验报告要求，本项目已准备好以下支撑素材：

| 报告章节 | 支撑素材位置 | 核心内容点 |
| :--- | :--- | :--- |
| **3. 数据集** | `reports/figures/`, `scripts/analyze_data.py` | 长尾分布图表、VOC2YOLO 转换逻辑 |
| **4. 方法** | `training/trainers/yolo_trainer.py` | YOLOv8 架构描述、AMP 与早停策略 |
| **5. 实验** | `experiments/` | Baseline 与消融实验（如关闭 Mosaic）的对比曲线 |
| **6. 结果分析** | `evaluate.py`, `metrics_report.json` | 各类别的 mAP 指标表格、FPS 性能数据 |

---

## 4. 运行说明 (Quick Start)

### 4.1 环境准备
```bash
# 建议使用虚拟环境
python -m venv venv
source venv/Scripts/activate 
pip install -r requirements.txt
```

### 4.2 数据准备与 EDA
```bash
python scripts/download_dataset.py  # 下载原始数据
python scripts/voc2yolo.py          # 格式转换与分层划分
python scripts/analyze_data.py      # 生成数据分布分析报告
```

### 4.3 执行训练与评估
```bash
python train.py      # 启动对比实验 (含 Baseline 与消融实验)
python evaluate.py   # 对最佳模型执行测试集深度评估
```

---

## 5. 提交要求清单 (Submission Checklist)

在正式提交至头歌/超星平台前，请确认：
1. [ ] **报告 PDF**：按“姓名-学号”命名，确保包含 10 篇以上参考文献。
2. [ ] **代码/数据**：包含完整的 `.ipynb` 文件（可在 Colab 或本地复现）。
3. [ ] **Git 历史**：确保 `git log` 中有真实的提交记录以证明团队协作。
4. [ ] **分工说明**：在报告末尾附上成员贡献比例表。

---
**学术诚信声明**：本项目代码及报告除引用开源框架（Ultralytics YOLOv8）外，核心逻辑与实验分析均为原创，查重率严格控制在 20% 以内。

