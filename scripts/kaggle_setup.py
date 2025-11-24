"""
Kaggle Notebooks 环境准备脚本
用法（Notebook cell）:

```python
import subprocess, sys
!python scripts/kaggle_setup.py --dataset-name pcb-dspcbsd --target /kaggle/working/pcb/DsPCBSD+
```

脚本功能:
- 检测 Kaggle 环境 (通过 KAGGLE_KERNEL_RUN_TYPE)
- 将 `/kaggle/input/<dataset-name>` 下的数据软链接到工作目录
- 若 Kaggle Dataset 已经包含 `DsPCBSD+` 结构，可自动创建所需软链接
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def is_kaggle_env() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.is_symlink() or dst.is_dir():
            print(f"[skip] {dst} 已存在")
            return
        raise FileExistsError(f"目标 {dst} 已存在且不是目录/软链接，请手动清理")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
        print(f"[link] {dst} -> {src}")
    except OSError:
        # 部分 Kaggle 环境禁用 symlink，退化为复制
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"[copy] {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Notebook 环境准备")
    parser.add_argument("--dataset-name", required=True,
                        help="Kaggle Dataset 名称 (位于 /kaggle/input/<name>)")
    parser.add_argument("--target", default="/kaggle/working/pcb/DsPCBSD+",
                        help="希望软链接/复制到的路径")
    parser.add_argument("--source-subdir", default="DsPCBSD+",
                        help="Dataset 中具体子目录名称")
    args = parser.parse_args()

    if not is_kaggle_env():
        print("⚠️ 当前不在 Kaggle 环境，跳过处理")
        return

    source_root = Path("/kaggle/input") / args.dataset_name / args.source_subdir
    if not source_root.exists():
        raise FileNotFoundError(f"未在 {source_root} 找到数据，请确认 dataset 名称/结构")

    target_root = Path(args.target)
    ensure_symlink(source_root, target_root)
    print("✓ Kaggle 数据集链接/复制完成")


if __name__ == "__main__":
    main()

