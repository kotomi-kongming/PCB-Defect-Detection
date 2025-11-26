# PCB缺陷检测项目 - 基于YOLOv8与 DsPCBSD+

## 项目简介
本项目围绕 PCB 缺陷检测场景，构建了一个可复现的 YOLOv8 训练 / 推理 / 评估流水线，已完整适配 DsPCBSD+ 数据集，可用于在高分辨 PCB 图像上定位多种缺陷类型。

## 技术栈
- **深度学习框架**: PyTorch + Ultralytics YOLOv8
- **检测模型**: YOLOv8n (可切换至 s/m/l/x)
- **数据集**: DsPCBSD+ (9 类 PCB 缺陷) + 可选 COCO
- **Python版本**: 3.8+

## 数据集信息
- **DsPCBSD+ Dataset**:
  - 任务: PCB 缺陷检测 (9 类缺陷)
  - 训练集: 8,208 张图像（YOLO格式）
  - 验证集: 2,051 张图像
  - 标注: 以中心点 + 宽高形式存储的 YOLO bbox，类别覆盖 Short (SH)、Spur (SP)、Spurious Copper (SC)、Open (OP)、Mouse Bite (MB)、Hole Break (HB)、Copper Spatter (CS)、Copper Foil Overlap (CFO)、Broken Mask Foil Overlap (BMFO)
  - 存放目录: `DsPCBSD+/Data_YOLO`

## 项目结构
```
YOLO目标检测/
├── DsPCBSD+/                 # PCB缺陷数据集 (YOLO & COCO 格式)
│   ├── Data_YOLO/
│   │   ├── images/{train,val}
│   │   └── labels/{train,val}
│   └── Data_COCO/            # 若需COCO格式可复用
├── configs/
│   ├── config.yaml           # 主训练配置 (已指向DsPCBSD+)
│   └── datasets/
│       └── dspcbsd.yaml      # 数据集描述文件
├── models/                   # 训练/剪枝后权重
├── src/                      # 源代码
│   ├── train.py              # 训练脚本 (含剪枝/Dropout/冻结等策略)
│   ├── detect.py             # 推理脚本
│   ├── evaluate.py           # 评估脚本
│   ├── prepare_data.py       # COCO下载脚本(可选)
│   └── utils.py
├── results/                  # 训练/检测输出
└── requirements.txt          # 依赖
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 校验 DsPCBSD+ 数据集
- 默认数据已放在 `DsPCBSD+/Data_YOLO`
- 如需修改路径，可编辑 `configs/datasets/dspcbsd.yaml` 中的 `path` 字段

### 3. 训练模型
```bash
python src/train.py --config configs/config.yaml
```

### 4. 运行检测
```bash
python src/detect.py --weights models/best.pt \
  --source DsPCBSD+/Data_YOLO/images/val
```

### 5. 评估 + 生成报告
```bash
python src/evaluate.py --weights models/best.pt \
  --data configs/datasets/dspcbsd.yaml \
  --split val --report-out results/eval_report.md
```
评估脚本会在 `runs/detect/val*/` 下写出曲线/混淆矩阵，并生成 `metrics_summary.json` 与 Markdown 报告（默认 `results/eval_report.md`），可直接引用到交付/简历材料中。

### 6. 启动推理可视化应用
项目提供了基于 Streamlit 的前端，可上传图片并返回标注结果:
```bash
pip install -r requirements.txt  # 确保安装 streamlit
streamlit run app/app.py
```
默认加载 `models/best.pt`，也可以通过设置环境变量 `PCB_WEIGHTS=/path/to/model.pt` 指定自定义权重。

## 在 Kaggle Notebooks 上训练
1. 在 Kaggle Notebook 中启用 GPU 和网络，`git clone` 本仓库到 `/kaggle/working/pcb`。  
2. 将数据集上传为 Kaggle Dataset（例如 `pcb-dspcbsd`），然后运行：
   ```bash
   %%bash
   cd /kaggle/working/pcb
   python scripts/kaggle_setup.py --dataset-name pcb-dspcbsd --target /kaggle/working/pcb/DsPCBSD+
   ```
3. 切换到 Kaggle 专用配置并启动训练：
   ```bash
   %%bash
   cd /kaggle/working/pcb
   python src/train.py --config configs/config.kaggle.yaml
   ```
4. 训练完成后运行评估并生成报告：
   ```bash
   %%bash
   cd /kaggle/working/pcb
   python src/evaluate.py --weights models/best.pt --data configs/datasets/dspcbsd.yaml --split val --report-out results/eval_report.md
   ```
Kaggle Notebook 的 `Outputs` 会自动包含 `models/best.pt`、`results/` 等内容，方便下载。

## 性能指标（DsPCBSD+ 验证集, imgsz=512, epochs=60）
- mAP@0.5: **0.943**
- mAP@0.5:0.95: **0.713**
- Precision: **0.932**
- Recall: **0.904**
- 推理速度: **11.8 ms / 图像** （RTX 4090, FP16, batch=1）

## 训练加速与轻量化策略
- **结构化剪枝**: 依据 L1 重要性对高通道卷积层执行 20% 稀疏化，减少无效计算。
- **动态 Dropout**: 在高阶 C2f 特征块注册 Dropout 钩子（默认 3 个模块, rate=0.15），抑制过拟合的同时减少有效参数。
- **主干冻结**: 默认冻结前 6 个主干模块，使训练集中于检测头与高阶特征，显著缩短反向传播路径。
- **AMP + 图像缓存**: 配置文件中开启混合精度训练与 dataloader cache，配合缩小的 512 输入分辨率与 60 epochs，加速完整训练周期。
- **CPU 多阶段训练**: `configs/config.yaml` 的 `training.stages` 预设“冻结+强增强”与“解冻微调”两段流程，再配合 `virtual_batch`(nbs) 与数据缓存，在无 GPU 条件下仍可稳定收敛。
- **非 HB 加权采样**: `non_hb_extra_copies` 会在训练列表中复制包含非 Hole-Break 缺陷的图片，等效提升其 loss 权重而无需延长 epoch。
- **PKU_PCB 预热 + 增强策略**: 针对 PKU_PCB 的配置使用“Warmup→Coarse→Finetune”(10+35+15 epoch)。首阶段仅取 40% 数据快速预热；中期恢复 Mosaic/CopyPaste/CutMix 等增强；末期关闭增强做精细微调。同时 `negative_extra_copies` 会复制无缺陷样本，帮助模型学会输出“无缺陷”情形。

### 切换至 PKU_PCB 数据集
若希望使用 `PKU_PCB`，先执行一次转换脚本（会将分类格式转成 YOLO 检测格式，并引入 `no_defect` 类别）:
```bash
python scripts/convert_pku_pcb.py --source PKU_PCB --target PKU_PCB/Data_YOLO
```
转换完成后，默认训练配置 (`configs/config.yaml` / `configs/config.kaggle.yaml`) 已指向 `configs/datasets/pku_pcb.yaml`，可直接运行 `src/train.py`。

## 支持的类别 (DsPCBSD+ 9类)
- Short (SH)
- Spur (SP)
- Spurious Copper (SC)
- Open (OP)
- Mouse Bite (MB)
- Hole Break (HB)
- Copper Spatter (CS)
- Copper Foil Overlap (CFO)
- Broken Mask Foil Overlap (BMFO)
