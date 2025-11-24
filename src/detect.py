"""
YOLOv8目标检测推理脚本
"""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO  # type: ignore
import torch


def detect(weights, source, conf_thres=0.25, iou_thres=0.45, save_dir='results/detect'):
    """
    运行目标检测
    
    Args:
        weights: 模型权重路径
        source: 输入源 (图像/视频/文件夹/摄像头)
        conf_thres: 置信度阈值
        iou_thres: NMS的IoU阈值
        save_dir: 结果保存目录
    """
    print("=" * 60)
    print("YOLOv8 目标检测")
    print("=" * 60)
    
    # 检查模型文件
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"错误: 模型文件不存在: {weights_path}")
        print("\n请先训练模型或下载预训练模型:")
        print("  python src/train.py")
        return
    
    # 检查GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU推理")
    
    # 加载模型
    print(f"\n加载模型: {weights_path}")
    model = YOLO(str(weights_path))
    
    # 输入源
    print(f"输入源: {source}")
    
    # 检测参数
    print(f"\n检测参数:")
    print(f"  - 置信度阈值: {conf_thres}")
    print(f"  - IoU阈值: {iou_thres}")
    print(f"  - 保存目录: {save_dir}")
    
    print("\n开始检测...")
    print("-" * 60)
    
    # 运行推理
    results = model.predict(
        source=source,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(Path(save_dir).parent),
        name=Path(save_dir).name,
        exist_ok=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    print("\n" + "=" * 60)
    print("检测完成!")
    print("=" * 60)
    
    # 显示结果统计
    total_detections = 0
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        total_detections += len(boxes)
        
        if len(boxes) > 0:
            print(f"\n图像: {Path(result.path).name}")
            print(f"检测到 {len(boxes)} 个目标:")
            
            # 统计每个类别的数量
            class_counts = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                
                if cls_name not in class_counts:
                    class_counts[cls_name] = 0
                class_counts[cls_name] += 1
            
            for cls_name, count in sorted(class_counts.items()):
                print(f"  - {cls_name}: {count}")
    
    print(f"\n总计检测到: {total_detections} 个目标")
    print(f"结果保存在: {save_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8目标检测推理')
    parser.add_argument('--weights', type=str, 
                       default='models/best.pt',
                       help='模型权重路径')
    parser.add_argument('--source', type=str, 
                       default='DsPCBSD+/Data_YOLO/images/val',
                       help='输入源 (图像/视频/文件夹/0表示摄像头)')
    parser.add_argument('--conf', type=float, 
                       default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, 
                       default=0.45,
                       help='NMS的IoU阈值')
    parser.add_argument('--save-dir', type=str, 
                       default='results/detect',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    root_dir = Path(__file__).parent.parent
    weights = root_dir / args.weights
    
    source_arg = args.source
    if not source_arg.isdigit() and not source_arg.startswith(('http://', 'https://', 'rtsp://')):
        source_path = Path(source_arg)
        if not source_path.is_absolute():
            source_arg = str((root_dir / source_path).resolve())
        else:
            source_arg = str(source_path)
    
    # 运行检测
    detect(
        weights=str(weights),
        source=source_arg,
        conf_thres=args.conf,
        iou_thres=args.iou,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
