# 基于四通道输入的 ResNet 微表情识别（4-Channel Micro-Expression Recognition）

> 本科毕业论文研究项目 ｜ 上海理工大学（Upper / University of Shanghai for Science and Technology）
> 数据集：CASME II · SMIC-HS · DSME（自采） ｜ 网络：ResNet18（+SPP / +CBAM / +SE / 迁移学习）
> 本仓库仅包含**代码与结果可视化**；模型权重与原始视频数据未上传（见下文说明）。

---

## 一、项目简介

微表情（Micro-Expression）是持续时间极短（1/25~1/5 秒）的自发面部表情，难以伪装，在测谎、心理分析、安防等领域有重要价值。针对单通道（RGB）输入信息不足、识别精度有限的问题，本项目提出**四通道输入**的 ResNet 微表情识别方法：

1. **人脸检测与对齐**：YOLO 检测 + 关键点对齐（`facenet-pytorch`）；
2. **四通道构造**：RGB 三通道 + 运动信息通道（**动态图像 Dynamic Image / MHI 运动历史图 / 光流**）；
3. **特征提取与分类**：ResNet18 骨干，对比 **SPP 空间金字塔池化、CBAM、SE、非对称卷积**等增强结构；
4. **迁移学习**：在 JAFFE 宏观表情库预训练后迁移；
5. **实时演示**：Flask + YOLO 的 Web 实时识别 GUI（`app.py` + `templates/`）。

**核心消融结果（CASME II，三分类）**：

| 方法 Method | Accuracy | F1 | UAR | UF1 |
|------|:---:|:---:|:---:|:---:|
| ResNet18 (RGB only) | 75.43% | 68.18% | 67.67% | 68.18% |
| ResNet18 (Dynamic Image only) | 87.28% | 85.63% | 87.72% | 85.63% |
| ResNet18 (RGB + Dynamic Fusion) | 88.21% | 85.97% | 85.97% | 85.97% |
| ResNet18 + SPP | **91.76%** | 90.29% | 87.84% | 90.29% |
| ResNet18 + SPP + Transfer (JAFFE) | **92.41%** | **90.54%** | **90.22%** | **90.54%** |

## 二、数据集统计

| 数据集 Dataset | 人数 | 分辨率 | 帧率 | 积极 | 消极 | 惊讶 | 总计 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CASME II | 25 | 640×480 | 200 | 32 | 99 | 25 | 156 |
| SMIC (HS) | 16 | 640×480 | 100 | 51 | 70 | 43 | 164 |
| DSME（自采） | 9 | 1920×1080 | 25→75 | 32 | 84 | 28 | 144 |

> ⚠️ **数据说明**：原始视频与帧数据（`data/`，约 16GB）**未上传**。CASME II 与 SMIC 为公开学术数据集，请向原数据集作者申请获取；DSME 为本项目自采数据集，如需要请邮件联系作者。

## 三、目录结构

```
├── app.py / templates/          # Flask + YOLO 实时识别 GUI
├── requirements.txt             # 依赖（清华源）
├── *.py / *.csv / *.png         # 数据统计、消融表格、论文图表脚本
├── paper_visualization/         # 论文图表与可视化脚本
├── grad-cam-lunwen/             # Grad-CAM 可视化
├── lunwen-jiance/               # 论文相关检测/实验
├── models/                      # 模型定义
├── casme2_* / samm_* / smic_* / DSME_*
│                                # 各数据集/各消融实验的训练目录
│   ├── fusion_train*            #   融合训练（2/3/5/7 分类、SPP、CBAM、SE…）
│   ├── single_rgb_backbone_compare  # 单通道骨干对比
│   ├── single_dynamic_backbone_compare
│   └── fusion_result*           #   实验结果（表格/图表，权重未上传）
├── cross_db_fusion_train*       # 跨库泛化实验
└── 20260518毕业论文大纲.docx    # 毕业论文大纲
```

## 四、环境安装

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

主要依赖：PyTorch ≥1.9、torchvision、opencv-python、facenet-pytorch、ultralytics（YOLO）、pandas、scikit-learn、Flask。

## 五、训练与推理

```bash
# 训练（以 CASME II 融合实验为例，具体参数见各实验目录内脚本）
python casme2_fusion_train2_3class/train.py

# 实时识别 GUI
python app.py          # 浏览器打开 http://127.0.0.1:5000
```

> 各训练目录内保留完整脚本与配置，权重（`.pth/.pt`）因体积未上传；按目录内脚本重新训练即可复现。

## 六、权重与数据说明

- 模型权重（126 个 `.pth/.pt`，约 15GB）**保留在本地**，未上传；如需模型文件请邮件联系作者；
- 全部结果可视化图（混淆矩阵、曲线、Grad-CAM 热力图、论文图）已随仓库上传；
- 本仓库公开用于学术交流与复试展示，引用数据集请遵循原数据集许可。

---

# 4-Channel Micro-Expression Recognition with ResNet

> Undergraduate thesis research project.
> Datasets: CASME II · SMIC-HS · DSME (self-collected) ｜ Backbone: ResNet18 (+SPP / +CBAM / +SE / transfer learning)
> This repository contains **code and result visualizations only**; model weights and raw videos are not uploaded.

## Overview

Micro-expressions are spontaneous facial expressions lasting 1/25–1/5 s. To overcome the limited information of RGB-only inputs, this project proposes a **4-channel ResNet**:

1. Face detection & alignment (YOLO + `facenet-pytorch`);
2. 4-channel construction: RGB + motion channel (Dynamic Image / MHI / optical flow);
3. ResNet18 backbone with **SPP / CBAM / SE / asymmetric convolution** variants;
4. Transfer learning from JAFFE (macro-expression);
5. Real-time Flask + YOLO web GUI (`app.py`).

**Ablation on CASME II (3-class)**: RGB only 75.43% → Dynamic Image 87.28% → RGB+Dynamic fusion 88.21% → +SPP **91.76%** → +SPP+transfer **92.41%** (Accuracy).

## Datasets

| Dataset | Subjects | Resolution | FPS | Pos | Neg | Surprise | Total |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CASME II | 25 | 640×480 | 200 | 32 | 99 | 25 | 156 |
| SMIC (HS) | 16 | 640×480 | 100 | 51 | 70 | 43 | 164 |
| DSME (self-collected) | 9 | 1920×1080 | 25→75 | 32 | 84 | 28 | 144 |

> Raw data (≈16 GB) is not included. CASME II / SMIC must be requested from their original authors; DSME is self-collected — contact the author if needed.

## Repository Layout

- `app.py` + `templates/` — real-time recognition GUI
- `casme2_* / samm_* / smic_* / DSME_*` — per-dataset training & ablation directories
- `cross_db_fusion_train*` — cross-database generalization
- `paper_visualization/`, `grad-cam-lunwen/` — figures & Grad-CAM
- `20260518毕业论文大纲.docx` — thesis outline

## Setup

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

```bash
python casme2_fusion_train2_3class/train.py   # example training script
python app.py                                 # real-time GUI at http://127.0.0.1:5000
```

## Note on Weights

126 checkpoint files (≈15 GB) remain local. All scripts needed to reproduce training are included; contact the author for weights if necessary. Figures (confusion matrices, curves, Grad-CAM heatmaps, paper figures) are fully included.
