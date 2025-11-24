# 目标检测数据集推荐

## 0. DsPCBSD+ (PCB缺陷检测，项目默认) ⭐当前使用

### 简介
专注于 PCB 缺陷检测的公开数据集，共 9 类典型缺陷，包含高分辨率图像并已转换为 YOLO 格式。

### 特点
- **类别数量**: 9（Short、Spur、Spurious Copper、Open、Mouse Bite、Hole Break、Copper Spatter、Copper Foil Overlap、Broken Mask Foil Overlap）
- **图像数量**:
  - 训练集: 8,208
  - 验证集: 2,051
- **标注类型**: YOLO bbox
- **目录**: `DsPCBSD+/Data_YOLO`
- **配置文件**: `configs/datasets/dspcbsd.yaml`

### 优点
✅ 聚焦工业质检场景，贴近真实 PCB 缺陷形态  
✅ 数据量适中，易于快速迭代  
✅ 同时提供 COCO 与 YOLO 两种格式，便于互转  

### 使用方式
1. 确认 `DsPCBSD+/Data_YOLO` 已就绪
2. 如需放到其它路径，修改 `dspcbsd.yaml` 中的 `path`
3. 运行 `python src/train.py` 即可直接训练

---

## 1. COCO (Common Objects in Context) ⭐推荐

### 简介
COCO是目标检测领域最常用的大规模数据集，由微软发布。

### 特点
- **类别数量**: 80个常见物体类别
- **图像数量**: 
  - 训练集: 118,287张
  - 验证集: 5,000张
  - 测试集: 40,670张
- **标注类型**: 边界框、分割掩码、关键点
- **场景**: 日常生活场景中的常见物体

### 类别示例
- 人物类: person
- 交通工具: car, bus, truck, motorcycle, bicycle, airplane, train, boat
- 动物: cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- 日用品: bottle, cup, fork, knife, spoon, bowl, chair, couch, bed, table
- 电子产品: laptop, mouse, keyboard, cell phone, tv, remote

### 下载
```bash
# 使用本项目脚本自动下载
python src/prepare_data.py

# 或手动下载
# 训练集: http://images.cocodataset.org/zips/train2017.zip
# 验证集: http://images.cocodataset.org/zips/val2017.zip
# 标注: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

### 优点
✅ 数据量大、质量高
✅ 标注详细、准确
✅ 社区支持好、资源多
✅ 竞赛标准数据集

### 缺点
❌ 数据集较大(25GB+)
❌ 下载时间较长

---

## 2. PASCAL VOC

### 简介
经典的目标检测数据集，适合入门学习。

### 特点
- **类别数量**: 20个类别
- **图像数量**: 
  - VOC2007: 9,963张
  - VOC2012: 11,540张
- **标注类型**: 边界框、分割掩码
- **场景**: 日常场景

### 类别
- 人物: person
- 动物: bird, cat, cow, dog, horse, sheep
- 交通工具: aeroplane, bicycle, boat, bus, car, motorbike, train
- 室内物品: bottle, chair, dining table, potted plant, sofa, tv/monitor

### 下载
```bash
# VOC2007
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# VOC2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

### 优点
✅ 数据量适中、易于上手
✅ 标注质量高
✅ 适合学习和实验

### 缺点
❌ 类别数量较少
❌ 数据集较老

---

## 3. Open Images Dataset

### 简介
Google发布的超大规模数据集。

### 特点
- **类别数量**: 600个类别
- **图像数量**: 1,900,000张
- **标注**: 边界框、分割、视觉关系
- **场景**: 非常广泛

### 下载
- [官网](https://storage.googleapis.com/openimages/web/index.html)

### 优点
✅ 数据量极大
✅ 类别非常丰富
✅ 标注详细

### 缺点
❌ 数据集超大(500GB+)
❌ 训练成本高

---

## 4. KITTI

### 简介
自动驾驶场景下的目标检测数据集。

### 特点
- **类别数量**: 8个类别
- **图像数量**: 7,481张训练图像
- **标注**: 2D/3D边界框
- **场景**: 城市道路驾驶

### 类别
- Car, Van, Truck
- Pedestrian, Person_sitting
- Cyclist
- Tram, Misc

### 下载
- [官网](http://www.cvlibs.net/datasets/kitti/)

### 优点
✅ 专注自动驾驶场景
✅ 3D标注信息
✅ 真实路况数据

### 缺点
❌ 场景单一
❌ 类别较少

---

## 5. Objects365

### 简介
阿里巴巴发布的大规模数据集。

### 特点
- **类别数量**: 365个类别
- **图像数量**: 2,000,000张
- **标注**: 边界框
- **场景**: 日常生活场景

### 优点
✅ 数据量大
✅ 类别丰富
✅ 质量高

### 缺点
❌ 数据集很大
❌ 下载困难

---

## 数据集选择建议

### 初学者
推荐: **PASCAL VOC**
- 数据量适中
- 容易上手
- 训练速度快

### 常规项目
推荐: **COCO** ⭐
- 行业标准
- 平衡性好
- 资源丰富

### 自动驾驶
推荐: **KITTI**
- 专业场景
- 3D信息
- 行业认可

### 研究项目
推荐: **Open Images** 或 **Objects365**
- 数据量大
- 类别丰富
- 挑战性高

---

## 数据集使用技巧

### 1. 从小开始
先用小数据集验证代码，再用大数据集训练。

### 2. 数据子集
COCO可以只使用部分类别进行训练。

### 3. 数据增强
通过数据增强扩充训练集。

### 4. 迁移学习
使用预训练模型，减少训练时间。

### 5. 混合数据集
可以结合多个数据集训练。

---

## 本项目支持的数据集

当前项目默认指向 **DsPCBSD+**，但也兼容其它 YOLO 数据集:

1. 下载/准备数据并转换为 YOLO TXT
2. 在 `configs/datasets/` 下新增数据集 yaml (可参考 `dspcbsd.yaml` 或 `data/coco/coco.yaml`)
3. 在 `configs/config.yaml -> data.dataset` 中指向新的 yaml
4. 运行 `python src/train.py --config configs/config.yaml`

### YOLO格式说明

每张图像对应一个txt文件，格式:
```
<class_id> <x_center> <y_center> <width> <height>
```

坐标为归一化值 (0-1)。
