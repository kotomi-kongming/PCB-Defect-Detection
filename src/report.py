"""
评估报告生成工具
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

__all__ = ['generate_report']


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return '-'
    return f"{value:.3f}"


def _rel_path(target: Path, base: Path) -> str:
    target = target.resolve()
    base = base.resolve()
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        return target.as_posix()


def _image_title(filename: str) -> str:
    stems = {
        'BoxPR_curve.png': 'Precision-Recall 曲线',
        'BoxP_curve.png': 'Precision-Confidence',
        'BoxR_curve.png': 'Recall-Confidence',
        'BoxF1_curve.png': 'F1-Confidence',
        'confusion_matrix.png': '混淆矩阵',
        'confusion_matrix_normalized.png': '归一化混淆矩阵',
    }
    return stems.get(filename, filename)


def generate_report(
    summary_path: Path,
    output_path: Path,
    eval_dir: Path,
    extra_images: Optional[Iterable[str]] = None,
) -> Path:
    """
    根据评估摘要生成 Markdown 报告
    """
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = summary.get('meta', {})
    overall = summary.get('overall', {})
    speed = summary.get('speed_ms', {})
    per_class = summary.get('per_class', [])

    lines: List[str] = []
    lines.append("# PCB 缺陷检测评估报告")
    lines.append("")
    lines.append("## 1. 实验配置")
    lines.append(f"- 权重: `{meta.get('weights', '')}`")
    lines.append(f"- 数据集: `{meta.get('data_yaml', '')}` / split=`{meta.get('split', '')}`")
    lines.append(f"- 运行设备: `{meta.get('device', 'cpu')}` (PyTorch {meta.get('torch', '')})")
    lines.append(f"- 评估时间: {meta.get('timestamp', '-')}")
    lines.append(f"- 置信度/IoU: {meta.get('conf_thres', 0)} / {meta.get('iou_thres', 0)}")
    lines.append("")

    lines.append("## 2. 全局指标")
    lines.append("| 指标 | 数值 | 含义 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Precision | {_fmt(overall.get('precision'))} | 预测为缺陷的准确度，越高误检越少 |")
    lines.append(f"| Recall | {_fmt(overall.get('recall'))} | 实际缺陷被检出的比例，越高漏检越少 |")
    lines.append(f"| mAP@0.5 | {_fmt(overall.get('map50'))} | IoU≥0.5 时的平均精度 |")
    lines.append(f"| mAP@0.5:0.95 | {_fmt(overall.get('map5095'))} | 覆盖 IoU 0.5~0.95 的严格平均精度 |")
    lines.append(f"| mAP@0.75 | {_fmt(overall.get('map75'))} | IoU≥0.75，更强调定位精度 |")
    lines.append("")

    lines.append("## 3. 速度表现")
    lines.append(f"- 预处理: {_fmt(speed.get('preprocess'))} ms/图")
    lines.append(f"- 推理: {_fmt(speed.get('inference'))} ms/图")
    lines.append(f"- 后处理: {_fmt(speed.get('postprocess'))} ms/图")
    lines.append("")

    lines.append("## 4. 各类别表现")
    lines.append("| 类别 | Precision | Recall | F1 | mAP@0.5 | 样本数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for cls in per_class:
        lines.append(
            f"| {cls.get('name')} | {_fmt(cls.get('precision'))} | "
            f"{_fmt(cls.get('recall'))} | {_fmt(cls.get('f1'))} | "
            f"{_fmt(cls.get('map50'))} | {cls.get('instances', '-')} |"
        )
    lines.append("")

    if per_class:
        sorted_by_map = sorted(
            [c for c in per_class if c.get('map50') is not None],
            key=lambda c: c.get('map50', 0.0)
        )
        if sorted_by_map:
            weakest = ", ".join(f"{c['name']}({c['map50']:.2f})" for c in sorted_by_map[:2])
            strongest = ", ".join(f"{c['name']}({c['map50']:.2f})" for c in sorted_by_map[-2:])
            lines.append("**类别洞察**")
            lines.append(f"- 弱项: {weakest}")
            lines.append(f"- 优势: {strongest}")
            lines.append("")

    lines.append("## 5. 曲线与可视化")
    displayed = False
    for img_name in extra_images or []:
        img_path = eval_dir / img_name
        if not img_path.exists():
            continue
        rel = _rel_path(img_path, output_path.parent)
        lines.append(f"![{_image_title(img_name)}]({rel})")
        displayed = True
    if not displayed:
        lines.append("（未找到曲线图，可检查评估输出目录）")
    lines.append("")

    lines.append("## 6. 建议")
    lines.append("- mAP 与 Recall 受弱类拖累时，可增加该类样本或定向增强。")
    lines.append("- 若需要更快推理，可进一步减小输入尺寸或导出 ONNX/INT8 模型。")
    lines.append("- 结合混淆矩阵定位主要误检对，制定人工复核策略。")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"✓ 评估报告已生成: {output_path}")
    return output_path


