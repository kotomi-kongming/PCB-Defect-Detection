"""
工具函数
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes


def visualize_detection(image_path, boxes, labels, scores, class_names, save_path=None):
    """
    可视化检测结果
    
    Args:
        image_path: 图像路径
        boxes: 边界框 [[x1, y1, x2, y2], ...]
        labels: 类别标签 [cls1, cls2, ...]
        scores: 置信度分数 [score1, score2, ...]
        class_names: 类别名称字典
        save_path: 保存路径
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    fig, ax = plt.subplots(1, figsize=(12, 8))  # type: ignore
    ax.imshow(image)  # type: ignore
    
    # 绘制边界框
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))  # type: ignore
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # 边界框
        color = colors[label]
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)  # type: ignore
        
        # 标签
        label_text = f"{class_names[label]}: {score:.2f}"
        ax.text(  # type: ignore
            x1, y1 - 5,
            label_text,
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=10,
            color='white'
        )
    
    ax.axis('off')  # type: ignore
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU值
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 交集区域
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 并集区域
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制
    
    Args:
        boxes: 边界框列表 [[x1, y1, x2, y2], ...]
        scores: 置信度分数列表
        iou_threshold: IoU阈值
    
    Returns:
        保留的边界框索引
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 按分数排序
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # 计算IoU
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # 保留IoU小于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def plot_training_curves(results_dir):
    """
    绘制训练曲线
    
    Args:
        results_dir: 训练结果目录
    """
    results_dir = Path(results_dir)
    results_csv = results_dir / 'results.csv'
    
    if not results_csv.exists():
        print(f"结果文件不存在: {results_csv}")
        return
    
    import pandas as pd
    
    # 读取结果
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲线
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
    axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # mAP曲线
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].set_title('Validation mAP')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision & Recall
    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 学习率
    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR pg0')
    axes[1, 1].plot(df['epoch'], df['lr/pg1'], label='LR pg1')
    axes[1, 1].plot(df['epoch'], df['lr/pg2'], label='LR pg2')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存
    save_path = results_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    # 测试函数
    print("工具函数模块")
