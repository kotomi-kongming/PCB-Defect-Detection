"""
数据集准备脚本 - 自动下载和配置COCO数据集
"""
import os
import sys
from pathlib import Path
import yaml
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """下载文件并显示进度条"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_coco_dataset():
    """准备COCO数据集"""
    print("=" * 60)
    print("COCO数据集准备工具")
    print("=" * 60)
    
    # 项目根目录
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / 'data'
    coco_dir = data_dir / 'coco'
    
    # 创建目录
    data_dir.mkdir(exist_ok=True)
    coco_dir.mkdir(exist_ok=True)
    
    print(f"\n数据集将保存到: {coco_dir}")
    
    # COCO数据集URL
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    print("\n提示: COCO数据集较大 (约25GB), 下载可能需要较长时间")
    print("如果您已经下载过COCO数据集, 可以将其复制到以下目录:")
    print(f"  - 训练图像: {coco_dir / 'images' / 'train2017'}")
    print(f"  - 验证图像: {coco_dir / 'images' / 'val2017'}")
    print(f"  - 标注文件: {coco_dir / 'annotations'}")
    
    choice = input("\n是否继续自动下载? (y/n): ").lower()
    
    if choice == 'y':
        # 下载数据集
        for name, url in urls.items():
            zip_path = coco_dir / f"{name}.zip"
            
            if not zip_path.exists():
                print(f"\n正在下载 {name}...")
                try:
                    download_url(url, zip_path)
                    print(f"✓ {name} 下载完成")
                except Exception as e:
                    print(f"✗ 下载失败: {e}")
                    print(f"  请手动下载: {url}")
                    continue
            else:
                print(f"\n✓ {name} 已存在, 跳过下载")
            
            # 解压
            print(f"正在解压 {name}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(coco_dir)
                print(f"✓ {name} 解压完成")
                
                # 删除压缩包以节省空间
                zip_path.unlink()
            except Exception as e:
                print(f"✗ 解压失败: {e}")
    
    # 创建YOLOv8格式的数据集配置文件
    create_coco_yaml(coco_dir)
    
    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print("=" * 60)


def create_coco_yaml(coco_dir):
    """创建COCO数据集的YAML配置文件"""
    
    # COCO类别名称
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    coco_yaml = {
        'path': str(coco_dir.absolute()),
        'train': 'images/train2017',
        'val': 'images/val2017',
        'nc': 80,  # 类别数量
        'names': coco_names
    }
    
    yaml_path = coco_dir / 'coco.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(coco_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✓ 已创建数据集配置: {yaml_path}")


def create_sample_images():
    """创建样例图像目录"""
    root_dir = Path(__file__).parent.parent
    sample_dir = root_dir / 'data' / 'samples'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n样例图像目录: {sample_dir}")
    print("您可以将测试图像放入此目录")


if __name__ == '__main__':
    try:
        prepare_coco_dataset()
        create_sample_images()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)
