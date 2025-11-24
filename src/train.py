"""
YOLOv8目标检测模型训练脚本
"""
import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.nn.utils import prune
from ultralytics import YOLO  # type: ignore

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(root_dir: Path, path_like: str) -> Path:
    """根据项目根目录解析相对路径"""
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (root_dir / candidate).resolve()


def infer_label_path(image_path: Path) -> Path:
    """根据图像路径推断标签路径（images -> labels, 后缀换为txt）"""
    path_str = str(image_path)
    replacements = [
        (f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"),
        ("/images/", "/labels/"),
        ("\\images/", "\\labels/"),
        ("/images\\", "/labels\\"),
    ]
    replaced = path_str
    for old, new in replacements:
        if old in replaced:
            replaced = replaced.replace(old, new, 1)
            break
    label_path = Path(replaced).with_suffix('.txt')
    return label_path


def contains_non_hb(label_path: Path, hb_index: int) -> bool:
    """判断标签文件中是否存在非HB类别"""
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cls_id = int(float(line.split()[0]))
        except ValueError:
            continue
        if cls_id != hb_index:
            return True
    return False


def prepare_weighted_dataset_yaml(base_yaml: Path, training_cfg: dict) -> Tuple[Path, Optional[Path]]:
    """生成带有非HB样本加权的临时数据集yaml"""
    extra_copies = int(training_cfg.get('non_hb_extra_copies', 0) or 0)
    if extra_copies <= 0:
        return base_yaml, None

    dataset_cfg = yaml.safe_load(base_yaml.read_text(encoding='utf-8'))
    dataset_root = Path(dataset_cfg.get('path', base_yaml.parent)).resolve()
    train_entry = dataset_cfg.get('train', 'images/train')
    train_path = Path(train_entry)
    if not train_path.is_absolute():
        train_path = (dataset_root / train_entry).resolve()

    if train_path.suffix.lower() == '.txt':
        image_paths: List[Path] = []
        for line in train_path.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s:
                continue
            img_path = Path(s)
            if not img_path.is_absolute():
                img_path = (dataset_root / img_path).resolve()
            image_paths.append(img_path)
    else:
        image_paths = sorted(
            [p for p in train_path.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS]
        )

    names = dataset_cfg.get('names', [])
    hb_index = training_cfg.get('hb_class_index')
    hb_keyword = str(training_cfg.get('hb_class_keyword', 'Hole-Break')).lower()
    if hb_index is None:
        for idx, name in enumerate(names):
            if hb_keyword in str(name).lower():
                hb_index = idx
                break
    if hb_index is None:
        raise ValueError("无法定位 HB 类别索引，请在配置中提供 hb_class_index 或 hb_class_keyword。")

    weighted_entries: List[str] = []
    for img_path in image_paths:
        img_path = img_path.resolve()
        path_str = str(img_path)
        weighted_entries.append(path_str)
        if contains_non_hb(infer_label_path(img_path), hb_index):
            weighted_entries.extend([path_str] * extra_copies)

    weighted_txt = dataset_root / f'weighted_train_nonhb_x{extra_copies}.txt'
    weighted_txt.parent.mkdir(parents=True, exist_ok=True)
    weighted_txt.write_text("\n".join(weighted_entries), encoding='utf-8')

    dataset_cfg['train'] = str(weighted_txt)
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml', encoding='utf-8') as tmp:
        yaml.safe_dump(dataset_cfg, tmp, allow_unicode=True)
        weighted_yaml = Path(tmp.name)

    print(f"\n✓ 非HB类别样本权重增强已启用: {weighted_txt.name} (每个样本额外复制 {extra_copies} 次)")
    return weighted_yaml, weighted_yaml


def apply_structured_pruning(model: YOLO, amount: float, min_channels: int = 64) -> None:
    """使用L1准则对卷积层进行非结构化剪枝以稀疏化权重"""
    if amount <= 0:
        return

    print(f"\n启用剪枝: 对通道数≥{min_channels}的卷积层按 {amount * 100:.1f}% 比例置零")
    pruned = 0
    for module in model.model.modules():  # type: ignore[attr-defined]
        target_conv = None
        if hasattr(module, 'conv') and isinstance(module.conv, nn.Conv2d):
            target_conv = module.conv  # type: ignore[assignment]
        elif isinstance(module, nn.Conv2d):
            target_conv = module

        if target_conv is None or target_conv.out_channels < min_channels:
            continue

        prune.l1_unstructured(target_conv, name='weight', amount=amount)
        prune.remove(target_conv, 'weight')
        pruned += 1

    print(f"  -> 已处理 {pruned} 个卷积层")


def attach_dropout_hooks(model: YOLO, rate: float, max_blocks: int) -> List[torch.utils.hooks.RemovableHandle]:
    """在若干高层特征块输出上注册Dropout钩子"""
    if rate <= 0 or max_blocks <= 0:
        return []

    c2f_blocks = [m for m in model.model.modules() if m.__class__.__name__ == 'C2f']  # type: ignore[attr-defined]
    if not c2f_blocks:
        print("⚠ 未找到C2f模块, 跳过Dropout注入")
        return []

    target_blocks = c2f_blocks[-max_blocks:]
    print(f"\n启用Dropout: 对 {len(target_blocks)} 个C2f块应用 rate={rate}")
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for block in target_blocks:
        dropout = nn.Dropout2d(rate)

        def hook_fn(_module, _inputs, output, dr=dropout):
            if _module.training:
                return dr(output)
            return output

        handles.append(block.register_forward_hook(hook_fn))

    return handles


def remove_dropout_hooks(model: YOLO) -> None:
    """移除之前注册的Dropout钩子以避免重复注入"""
    handles: List[torch.utils.hooks.RemovableHandle] = getattr(model, '_custom_dropout_hooks', [])
    if not handles:
        return
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
    model._custom_dropout_hooks = []


def freeze_backbone_layers(model: YOLO, num_layers: int) -> None:
    """冻结主干前若干层以缩短训练时间"""
    if num_layers <= 0:
        return

    backbone = getattr(model.model, 'model', None)  # type: ignore[attr-defined]
    if backbone is None:
        return

    frozen = 0
    for layer in list(backbone)[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False
        frozen += 1

    print(f"\n冻结前 {frozen} 个主干模块以减少反向传播开销")


def stage_value(stage_cfg: dict, base_cfg: dict, key: str, default=None):
    """优先返回阶段覆盖的配置，否则回退到基础配置"""
    if stage_cfg and key in stage_cfg:
        return stage_cfg[key]
    return base_cfg.get(key, default)


def prepare_model_for_stage(model: YOLO, training_cfg: dict, stage_cfg: dict) -> None:
    """根据阶段配置应用冻结/剪枝/Dropout"""
    freeze_layers = stage_value(stage_cfg, training_cfg, 'freeze_layers', 0)
    pruning_amount = stage_value(stage_cfg, training_cfg, 'pruning_amount', 0.0)
    dropout_rate = stage_value(stage_cfg, training_cfg, 'dropout_rate', 0.0)
    dropout_layers = stage_value(stage_cfg, training_cfg, 'dropout_layers', 0)

    freeze_backbone_layers(model, freeze_layers)
    apply_structured_pruning(model, pruning_amount)
    remove_dropout_hooks(model)

    dropout_hooks = attach_dropout_hooks(model, dropout_rate, dropout_layers)
    model._custom_dropout_hooks = dropout_hooks  # type: ignore[attr-defined]


def build_train_args(
    config: dict,
    stage_cfg: dict,
    data_yaml: Path,
    device_arg: str,
    root_dir: Path,
    stage_name: str,
) -> dict:
    """组合 Ultralytics 接口所需的训练参数"""
    data_cfg = config['data']
    training_cfg = config['training']
    validation_cfg = config['validation']

    def val(scope_cfg, key, default=None, scope_override: str = None):
        if scope_override == 'stage' and key in stage_cfg:
            return stage_cfg[key]
        if key in stage_cfg:
            return stage_cfg[key]
        return scope_cfg.get(key, default)

    run_name = stage_cfg.get('run_name', f"train_{stage_name}")

    return {
        'data': str(data_yaml),
        'epochs': val(training_cfg, 'epochs', 50),
        'patience': val(training_cfg, 'patience', 10),
        'batch': val(data_cfg, 'batch_size', 16),
        'imgsz': val(data_cfg, 'img_size', 320),
        'device': device_arg,
        'workers': val(data_cfg, 'workers', 4),
        'cache': val(data_cfg, 'cache_images', False),
        'close_mosaic': val(data_cfg, 'close_mosaic', 10),
        'optimizer': val(training_cfg, 'optimizer', 'SGD'),
        'lr0': val(training_cfg, 'lr0', 0.01),
        'lrf': val(training_cfg, 'lrf', 0.01),
        'momentum': val(training_cfg, 'momentum', 0.937),
        'weight_decay': val(training_cfg, 'weight_decay', 0.0005),
        'hsv_h': val(training_cfg, 'hsv_h', 0.015),
        'hsv_s': val(training_cfg, 'hsv_s', 0.7),
        'hsv_v': val(training_cfg, 'hsv_v', 0.4),
        'degrees': val(training_cfg, 'degrees', 0.0),
        'translate': val(training_cfg, 'translate', 0.1),
        'scale': val(training_cfg, 'scale', 0.4),
        'shear': val(training_cfg, 'shear', 0.0),
        'perspective': val(training_cfg, 'perspective', 0.0),
        'flipud': val(training_cfg, 'flipud', 0.0),
        'fliplr': val(training_cfg, 'fliplr', 0.5),
        'mosaic': val(training_cfg, 'mosaic', 0.0),
        'copy_paste': val(training_cfg, 'copy_paste', 0.0),
        'mixup': val(training_cfg, 'mixup', 0.0),
        'cutmix': val(training_cfg, 'cutmix', 0.0),
        'erasing': val(training_cfg, 'erasing', 0.0),
        'auto_augment': val(training_cfg, 'auto_augment', 'randaugment'),
        'val': True,
        'save': True,
        'save_period': validation_cfg['save_period'],
        'conf': val(validation_cfg, 'conf_thres', 0.001),
        'iou': val(validation_cfg, 'iou_thres', 0.65),
        'amp': val(training_cfg, 'amp', True),
        'cos_lr': val(training_cfg, 'cosine_lr', False),
        'project': str(root_dir / config['paths']['results']),
        'name': run_name,
        'exist_ok': True,
        'verbose': True,
        'plots': True,
        'box': val(training_cfg, 'box', 1.0),
        'cls': val(training_cfg, 'cls', 0.5),
        'dfl': val(training_cfg, 'dfl', 1.0),
        'nbs': val(training_cfg, 'virtual_batch', 64),
    }

def train(config_path):
    """训练目标检测模型（支持多阶段 CPU 策略）"""
    config = load_config(config_path)
    training_cfg = config['training']

    print("=" * 60)
    print("YOLOv8 目标检测训练")
    print("=" * 60)

    device = training_cfg['device']
    if device != 'cpu':
        if torch.cuda.is_available():
            print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ GPU不可用, 将使用CPU训练")
            device = 'cpu'
    else:
        print("使用CPU训练")

    root_dir = Path(__file__).parent.parent
    data_yaml = resolve_path(root_dir, config['data']['dataset'])

    if not data_yaml.exists():
        print(f"\n错误: 数据集配置文件不存在: {data_yaml}")
        return

    print(f"数据集配置: {data_yaml}")

    cleanup_paths: List[Path] = []
    data_yaml_use, temp_yaml = prepare_weighted_dataset_yaml(data_yaml, training_cfg)
    if temp_yaml:
        cleanup_paths.append(temp_yaml)

    device_arg = 'cpu' if str(device).lower() == 'cpu' else str(device)
    stages = training_cfg.get('stages')
    if not stages:
        stages = [{'name': 'default'}]

    current_weights = config['model']['name']
    final_results = None
    stage_records: List[dict] = []

    for idx, raw_stage in enumerate(stages):
        stage_cfg = raw_stage or {}
        stage_name = stage_cfg.get('name', f'stage_{idx + 1}')
        stage_weights = stage_cfg.get('weights', current_weights)

        print("\n" + "-" * 60)
        print(f"阶段 {idx + 1}/{len(stages)} -> {stage_name}")
        print("-" * 60)
        print(f"加载权重: {stage_weights}")

        model = YOLO(stage_weights)
        prepare_model_for_stage(model, training_cfg, stage_cfg)

        train_args = build_train_args(
            config=config,
            stage_cfg=stage_cfg,
            data_yaml=data_yaml_use,
            device_arg=device_arg,
            root_dir=root_dir,
            stage_name=stage_name
        )

        print("\n训练参数:")
        print(f"  - Epochs: {train_args['epochs']}")
        print(f"  - Batch Size: {train_args['batch']}")
        print(f"  - Image Size: {train_args['imgsz']}")
        print(f"  - Learning Rate: {train_args['lr0']}")
        print(f"  - Optimizer: {train_args['optimizer']}")
        print(f"  - Mosaic: {train_args['mosaic']}")
        print(f"  - Close Mosaic: {train_args['close_mosaic']}")

        print("\n开始训练...")
        print("-" * 60)

        stage_result = model.train(**train_args)
        final_results = stage_result

        run_dir = Path(train_args['project']) / train_args['name']
        best_model_src = run_dir / 'weights' / 'best.pt'

        if best_model_src.exists():
            current_weights = str(best_model_src)
            stage_records.append({
                'stage': stage_name,
                'run_dir': str(run_dir),
                'best_model': str(best_model_src)
            })
            print(f"\n✓ 阶段 {stage_name} 最佳模型: {best_model_src}")
        else:
            print(f"\n⚠ 阶段 {stage_name} 未找到 best.pt (run: {run_dir})")

        remove_dropout_hooks(model)

    if not stage_records:
        print("错误: 训练阶段未产生可用模型")
        return final_results

    import shutil

    model_dir = root_dir / config['paths']['weights']
    model_dir.mkdir(exist_ok=True, parents=True)

    final_best_src = Path(stage_records[-1]['best_model'])
    best_model_dst = model_dir / 'best.pt'

    if final_best_src.exists():
        shutil.copy(final_best_src, best_model_dst)
        print(f"\n✓ 最终最佳模型已保存到: {best_model_dst}")

    print("\n阶段摘要:")
    for record in stage_records:
        print(f"  - {record['stage']}: {record['run_dir']}")

    print(f"\n最终结果目录: {stage_records[-1]['run_dir']}")

    for tmp_path in cleanup_paths:
        tmp_path.unlink(missing_ok=True)

    return final_results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8目标检测训练')
    parser.add_argument('--config', type=str, 
                       default='configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--dataset', type=str,
                       default=None,
                       help='数据集配置文件覆盖路径')
    parser.add_argument('--epochs', type=int,
                       default=None,
                       help='训练轮次覆盖')
    parser.add_argument('--batch', type=int,
                       default=None,
                       help='批大小覆盖')
    parser.add_argument('--imgsz', type=int,
                       default=None,
                       help='输入图像尺寸覆盖')
    parser.add_argument('--device', type=str,
                       default=None,
                       help='设备覆盖 (如 0 或 cpu)')
    parser.add_argument('--weights', type=str,
                       default=None,
                       help='预训练权重覆盖 (如 yolov8n.pt)')
    
    args = parser.parse_args()
    
    # 配置文件路径
    root_dir = Path(__file__).parent.parent
    config_path = root_dir / args.config
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    # 加载并应用覆盖参数
    config = load_config(config_path)
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch is not None:
        config['data']['batch_size'] = args.batch
    if args.imgsz is not None:
        config['data']['img_size'] = args.imgsz
    if args.device is not None:
        config['training']['device'] = args.device
    if args.weights is not None:
        config['model']['name'] = args.weights

    # 将可能修改后的配置写入临时文件以复用现有流程
    import tempfile
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml', encoding='utf-8') as tmp:
        yaml.safe_dump(config, tmp, allow_unicode=True)
        tmp_path = tmp.name

    try:
        train(Path(tmp_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
