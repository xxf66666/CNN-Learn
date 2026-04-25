"""
Week 2 demo 配套: 把 CIFAR-10 测试集每类抽 10 张导出成 PNG 入仓库

意图:
  - 解决 "数据集是 .tar.gz 二进制" 的可读性问题 - 别人 clone 后直接能浏览
  - 给 demo (app.py) 提供"测试集浏览"模式的样本素材
  - 100 张 32×32 PNG 总共 ~100 KB, 入仓库零负担

输出:
  assets/week2/samples/
    airplane/
      0.png, 1.png, ..., 9.png    (每类 10 张)
    automobile/
      ...
    horse/
      ...

运行:
  python code/week2/export_cifar_samples.py
"""

import os
from pathlib import Path

import numpy as np
import torchvision

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.normpath(os.path.join(HERE, '..', '..'))
DATA_DIR    = os.path.join(ROOT, 'data', 'cifar10')
SAMPLES_DIR = Path(ROOT) / 'assets' / 'week2' / 'samples'

SAMPLES_PER_CLASS = 10


def main():
    print(f'加载 CIFAR-10 测试集 (root={DATA_DIR}) ...')
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=None,
    )
    classes = testset.classes
    print(f'  类别 ({len(classes)}): {classes}')

    # 按类别分组所有索引
    by_class = {c: [] for c in range(len(classes))}
    for idx, (_, label) in enumerate(testset):
        by_class[label].append(idx)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f'\n导出每类 {SAMPLES_PER_CLASS} 张到 {SAMPLES_DIR} ...')
    total = 0
    for class_idx, indices in by_class.items():
        class_name = classes[class_idx]
        out_dir = SAMPLES_DIR / class_name
        out_dir.mkdir(exist_ok=True)

        # 取每类前 N 张
        chosen = indices[:SAMPLES_PER_CLASS]
        for i, idx in enumerate(chosen):
            img, _ = testset[idx]    # img 是 PIL.Image (32x32 RGB)
            out_path = out_dir / f'{i:02d}.png'
            img.save(out_path)
            total += 1
        print(f'  {class_name:12s}: {SAMPLES_PER_CLASS} 张')

    # 总大小统计
    total_size = sum(p.stat().st_size for p in SAMPLES_DIR.rglob('*.png'))
    print(f'\n完成: 共 {total} 张 PNG, 总大小 {total_size / 1024:.1f} KB')
    print(f'路径: {SAMPLES_DIR.relative_to(Path(ROOT))}/')


if __name__ == '__main__':
    main()
