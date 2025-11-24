# 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速体验 (使用预训练模型)

```bash
python quick_start.py
```

这将使用YOLOv8预训练模型在示例图像上进行检测。

### 3. 校验 DsPCBSD+ 数据集
- 默认数据位于 `DsPCBSD+/Data_YOLO`
- 如需放到其他磁盘，请修改 `configs/datasets/dspcbsd.yaml -> path`
- 若仍需 COCO 数据，可运行 `python src/prepare_data.py` 自动下载

### 4. 训练模型

```bash
# 使用默认配置训练
python src/train.py

# 使用自定义配置
python src/train.py --config configs/config.yaml
```

### 5. 检测图像

```bash
# 检测单张图像
python src/detect.py --source path/to/image.jpg

# 检测验证集中所有图像
python src/detect.py --source DsPCBSD+/Data_YOLO/images/val

# 使用摄像头实时检测
python src/detect.py --source 0

# 自定义参数
python src/detect.py --source data/samples/ --conf 0.5 --iou 0.45
```

### 6. 评估模型

```bash
python src/evaluate.py --weights models/best.pt
```

## 配置说明

### 模型选择

在 `configs/config.yaml` 中可以选择不同的YOLOv8模型:

- `yolov8n.pt` - Nano (最快, 最小)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium  
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (最准确, 最慢)

### 训练参数调整

主要参数在 `configs/config.yaml` 中:

- `epochs`: 训练轮数 (默认 60)
- `batch_size`: 批次大小 (默认 24，可按显存调整)
- `img_size`: 图像大小 (默认 512)
- `dropout_rate` / `dropout_layers`: 控制注入 Dropout 的强度与模块数量
- `pruning_amount`: 剪枝比例 (0~1)
- `freeze_layers`: 冻结主干层数
- `lr0`: 初始学习率 (默认 0.01)

### 数据增强

可以在配置文件中调整数据增强参数:

- `mosaic`: Mosaic增强
- `fliplr`: 左右翻转
- `hsv_h/s/v`: 颜色增强
- `scale`: 缩放范围

## 常见问题

### 1. GPU内存不足

减小 `batch_size` 或使用更小的模型 (如 yolov8n)。

### 2. 训练速度慢

- 使用GPU训练
- 减小图像大小
- 使用更小的模型

### 3. 检测精度不高

- 增加训练轮数
- 使用更大的模型
- 调整置信度阈值 `--conf`

## 高级用法

### 自定义数据集

1. 准备YOLO格式的数据集
2. 创建数据集配置文件 (参考 `coco.yaml`)
3. 修改 `configs/config.yaml` 中的数据集路径
4. 开始训练

### 导出模型

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')

# 导出为ONNX格式
model.export(format='onnx')

# 导出为TensorRT
model.export(format='engine')
```

## 性能优化

### 训练优化

- 使用混合精度训练 (自动启用)
- 多GPU训练: 设置 `device: [0,1,2,3]`
- 调整workers数量以充分利用CPU

### 推理优化

- 使用TensorRT进行加速
- 批量推理多张图像
- 使用较小的图像尺寸

## 项目结构

```
YOLO目标检测/
├── DsPCBSD+/                 # PCB缺陷数据
│   └── Data_YOLO/
│       ├── images/{train,val}
│       └── labels/{train,val}
├── configs/
│   ├── config.yaml
│   └── datasets/dspcbsd.yaml
├── models/
├── results/
├── src/
│   ├── train.py
│   ├── detect.py
│   ├── evaluate.py
│   ├── prepare_data.py
│   └── utils.py
└── quick_start.py
```

## 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [COCO数据集](https://cocodataset.org/)
- [PyTorch 文档](https://pytorch.org/docs/)
