"""
YOLOv8模型评估脚本
"""
import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from ultralytics import YOLO  # type: ignore

from report import generate_report


def evaluate(
    weights,
    data_yaml,
    split='val',
    conf_thres=0.001,
    iou_thres=0.7,
    report_out: Optional[Path] = None,
    report_images: Optional[List[str]] = None,
):
    """
    评估模型性能
    
    Args:
        weights: 模型权重路径
        data_yaml: 数据集配置文件
        split: 评估数据集 ('train', 'val', 'test')
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
    """
    print("=" * 60)
    print("YOLOv8 模型评估")
    print("=" * 60)
    
    # 检查模型文件
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"错误: 模型文件不存在: {weights_path}")
        return
    
    # 检查数据集配置
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"错误: 数据集配置不存在: {data_path}")
        return
    
    # 检查GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU评估")
    
    # 加载模型
    print(f"\n加载模型: {weights_path}")
    model = YOLO(str(weights_path))
    
    print(f"数据集: {data_path}")
    print(f"评估集: {split}")
    
    print("\n评估参数:")
    print(f"  - 置信度阈值: {conf_thres}")
    print(f"  - IoU阈值: {iou_thres}")
    
    print("\n开始评估...")
    print("-" * 60)
    
    # 运行验证
    metrics = model.val(
        data=str(data_path),
        split=split,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        verbose=True,
        plots=True,
        save_json=True,
    )
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    
    # 显示评估指标
    print("\n性能指标:")
    print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  - mAP@0.75: {metrics.box.map75:.4f}")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")
    
    # 各类别性能
    print("\n各类别mAP@0.5:")
    class_names = model.names
    for i, map_val in enumerate(metrics.box.maps):
        if i < len(class_names):
            print(f"  - {class_names[i]}: {map_val:.4f}")
    
    # 保存结构化指标
    per_class_stats = []
    names = model.names
    class_counts = getattr(metrics, 'nt_per_class', None)
    counts_list = []
    if class_counts is not None:
        counts_list = [int(x) for x in class_counts]

    for cls_id, cls_name in names.items():
        precision = float(metrics.box.p[cls_id]) if cls_id < len(metrics.box.p) else None
        recall = float(metrics.box.r[cls_id]) if cls_id < len(metrics.box.r) else None
        f1_score = float(metrics.box.f1[cls_id]) if cls_id < len(metrics.box.f1) else None
        map50 = float(metrics.box.maps[cls_id]) if cls_id < len(metrics.box.maps) else None
        instances = counts_list[cls_id] if counts_list and cls_id < len(counts_list) else None

        per_class_stats.append({
            'id': int(cls_id),
            'name': cls_name,
            'precision': precision,
            'recall': recall,
            'f1': f1_score,
            'map50': map50,
            'instances': instances,
        })

    summary = {
        'meta': {
            'weights': str(weights_path),
            'data_yaml': str(data_path),
            'split': split,
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'device': device,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'platform': platform.platform(),
            'torch': torch.__version__,
        },
        'overall': {
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'map50': float(metrics.box.map50),
            'map5095': float(metrics.box.map),
            'map75': float(metrics.box.map75),
        },
        'speed_ms': metrics.speed,
        'per_class': per_class_stats,
        'files': {
            'eval_dir': str(metrics.save_dir),
            'predictions': str(Path(metrics.save_dir) / 'predictions.json'),
        },
    }

    summary_path = Path(metrics.save_dir) / 'metrics_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n✓ 指标摘要已保存: {summary_path}")

    if report_out:
        report_path = report_out
        if not isinstance(report_path, Path):
            report_path = Path(report_path)
        eval_dir = Path(metrics.save_dir)
        generate_report(
            summary_path=summary_path,
            output_path=report_path,
            eval_dir=eval_dir,
            extra_images=report_images or [],
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='YOLOv8模型评估')
    parser.add_argument('--weights', type=str, 
                       default='models/best.pt',
                       help='模型权重路径')
    parser.add_argument('--data', type=str, 
                       default='configs/datasets/dspcbsd.yaml',
                       help='数据集配置文件')
    parser.add_argument('--split', type=str, 
                       default='val',
                       choices=['train', 'val', 'test'],
                       help='评估数据集')
    parser.add_argument('--conf', type=float, 
                       default=0.001,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, 
                       default=0.7,
                       help='IoU阈值')
    parser.add_argument('--report-out', type=str,
                       default=None,
                       help='评估报告输出路径 (留空则跳过)')
    parser.add_argument('--no-report', action='store_true',
                       help='仅打印指标，不生成报告')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    root_dir = Path(__file__).parent.parent
    def _resolve(path_str: str) -> Path:
        candidate = Path(path_str)
        if candidate.is_absolute():
            return candidate
        return (root_dir / candidate).resolve()

    weights = _resolve(args.weights)
    data_yaml = _resolve(args.data)

    report_images: List[str] = [
        'BoxPR_curve.png',
        'BoxP_curve.png',
        'BoxR_curve.png',
        'BoxF1_curve.png',
        'confusion_matrix.png',
    ]
    report_out = None if args.no_report else args.report_out

    config_yaml = root_dir / 'configs' / 'config.yaml'
    if config_yaml.exists():
        try:
            with open(config_yaml, 'r', encoding='utf-8') as cfg_file:
                cfg_data = yaml.safe_load(cfg_file)
            report_cfg = cfg_data.get('report', {})
            if report_out is None:
                report_out = report_cfg.get('output')
            if report_cfg.get('include_images'):
                report_images = report_cfg['include_images']
        except Exception:
            pass

    if report_out is not None and not str(report_out).strip():
        report_out = None
    if report_out is None and not args.no_report:
        report_out = 'results/eval_report.md'

    report_path = None
    if report_out:
        report_path = Path(report_out)
        if not report_path.is_absolute():
            report_path = (root_dir / report_path).resolve()
    
    # 运行评估
    evaluate(
        weights=str(weights),
        data_yaml=str(data_yaml),
        split=args.split,
        conf_thres=args.conf,
        iou_thres=args.iou,
        report_out=report_path,
        report_images=report_images,
    )


if __name__ == '__main__':
    main()
