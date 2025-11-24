"""
将 PKU_PCB 数据集转换为 YOLO 检测格式。

原始目录结构示例:
PKU_PCB/
  train_cnn/missing_hole/*.jpg
  val_cnn/short/*.jpg
  test_cnn/spur/*.jpg
  ... 以及 no_defect/normal 等无缺陷目录

运行方式:
    python scripts/convert_pku_pcb.py --source PKU_PCB --target PKU_PCB/Data_YOLO

说明:
- 含有缺陷的图片会生成覆盖整图的 bbox(因为源数据是裁剪后的缺陷 patch)。
- `no_defect`/`normal` 目录会被视为负样本, 仅复制图片并生成空 label 文件。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable

from PIL import Image
from tqdm import tqdm


DEFECT_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

NEGATIVE_DIRS = {"no_defect", "normal"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG"}


def list_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.glob("*")):
        if path.suffix in IMAGE_EXTS and path.is_file():
            yield path


def convert_split(split_name: str, source_root: Path, target_root: Path) -> None:
    split_src = source_root / f"{split_name}_cnn"
    if not split_src.exists():
        raise FileNotFoundError(f"未找到 {split_src}")

    images_dir = target_root / "images" / split_name
    labels_dir = target_root / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for cls_dir in sorted(split_src.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name.lower()
        is_negative = any(token in cls_name for token in NEGATIVE_DIRS)
        cls_id = DEFECT_CLASSES.index(cls_name) if cls_name in DEFECT_CLASSES else None

        if cls_id is None and not is_negative:
            print(f"[warning] 忽略未注册类别目录: {cls_dir}")
            continue

        for img_path in tqdm(list_images(cls_dir), desc=f"{split_name}:{cls_dir.name}"):
            dest_img = images_dir / img_path.name
            shutil.copy2(img_path, dest_img)

            label_path = labels_dir / f"{img_path.stem}.txt"
            if is_negative:
                label_path.touch(exist_ok=True)
                continue

            with Image.open(img_path) as im:
                width, height = im.size

            x_center = 0.5
            y_center = 0.5
            w_norm = 1.0
            h_norm = 1.0
            content = f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            label_path.write_text(content, encoding="utf-8")


def convert_dataset(source: Path, target: Path) -> None:
    for split in ("train", "val", "test"):
        convert_split(split, source, target)

    print(f"\nPKU_PCB 已转换为 YOLO 格式 -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 PKU_PCB 转换为 YOLO 数据集")
    parser.add_argument("--source", type=Path, default=Path("PKU_PCB"),
                        help="PKU_PCB 原始目录 (包含 train_cnn/val_cnn/test_cnn)")
    parser.add_argument("--target", type=Path, default=Path("PKU_PCB/Data_YOLO"),
                        help="输出 YOLO 数据目录")
    args = parser.parse_args()

    convert_dataset(args.source.resolve(), args.target.resolve())


if __name__ == "__main__":
    main()

