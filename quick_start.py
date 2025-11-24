"""
快速开始示例 - 使用预训练模型进行检测
"""
from pathlib import Path
from ultralytics import YOLO  # type: ignore
import cv2


def quick_start_demo():
    """使用预训练模型快速体验目标检测"""
    
    print("=" * 60)
    print("目标检测快速开始示例")
    print("=" * 60)
    
    # 下载预训练模型
    print("\n1. 加载预训练模型 (YOLOv8n - 最轻量级版本)...")
    model = YOLO('yolov8n.pt')  # 会自动下载
    print("✓ 模型加载完成")
    
    # 显示模型信息
    print(f"\n模型信息:")
    print(f"  - 类别数量: {len(model.names)}")
    print(f"  - 支持的类别: {', '.join(list(model.names.values())[:10])}...")
    
    # 示例图像URL
    sample_images = [
        'https://ultralytics.com/images/bus.jpg',
        'https://ultralytics.com/images/zidane.jpg',
    ]
    
    print(f"\n2. 在线示例图像检测...")
    
    # 检测示例图像
    results = model(sample_images[0], save=True, conf=0.25)
    
    print("✓ 检测完成")
    
    # 显示检测结果
    for result in results:
        boxes = result.boxes
        print(f"\n检测结果:")
        print(f"  - 检测到 {len(boxes)} 个目标")
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"    • {cls_name}: {conf:.2f}")
    
    print(f"\n结果已保存到: runs/detect/predict")
    
    print("\n" + "=" * 60)
    print("快速开始完成!")
    print("=" * 60)
    
    print("\n下一步:")
    print("1. 校验DsPCBSD+数据集路径: configs/datasets/dspcbsd.yaml")
    print("2. 训练自己的模型: python src/train.py")
    print("3. 使用训练的模型检测: python src/detect.py --weights models/best.pt")


if __name__ == '__main__':
    try:
        quick_start_demo()
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示: 请确保已安装所需依赖:")
        print("  pip install -r requirements.txt")
