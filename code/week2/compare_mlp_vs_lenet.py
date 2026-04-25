"""
Week 2 T9 拓展实验: MLP vs LeNet 在 CIFAR-10 + 平移过的 CIFAR-10 上对比

实验目的:
  把"为什么需要 CNN"从口号变成肉眼可见的数据.
  T1 §3 说"MLP 没有平移不变性, 同一只猫在不同位置要重学". 这里直接验证.

实验设计:
  - 两个模型, 同样 lr/optimizer/epochs:
      MLP    : 3072 → 512 → 256 → 10  (拉平喂全连接)
      LeNet  : Week 2 T9 的 LeNet-5
  - 在 4 个评估集上各自给准确率:
      1) 原 CIFAR-10 测试集
      2) 测试集每张图随机平移 4 像素 (模拟物体出现位置变化)
  - 出 4 个准确率 + 对比柱图

预期 (T1 §3 的预言):
  - MLP-原 ≈ 50%        ← 训练分布上还行
  - MLP-平移 ≈ 25%      ← 平移立刻崩, 因为权重和位置硬绑定
  - LeNet-原 ≈ 60%      ← 跟 lenet_pytorch.py 跑出来对得上
  - LeNet-平移 ≈ 55%    ← 卷积权重共享带来的平移鲁棒性

运行:
  python code/week2/compare_mlp_vs_lenet.py
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from lenet_pytorch import (
    LeNet, load_cifar10, pick_device, count_params,
    OUTPUT_DIR, CIFAR_CLASSES,
)

plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#444444',
    'axes.labelcolor':  '#cccccc',
    'xtick.color':      '#aaaaaa',
    'ytick.color':      '#aaaaaa',
    'text.color':       'white',
    'axes.titlecolor':  'white',
    'font.family':      ['Arial Unicode MS', 'sans-serif'],
})

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'cifar10')


# ─────────────────────────────────────────────
# MLP (跟 Week 1 风格一致, 但适配 CIFAR-10 的 3072 输入)
# ─────────────────────────────────────────────
class MLP(nn.Module):
    """
    3072 → 512 → 256 → 10
    跟 Week 1 mlp_numpy.py 同样架构思路, 只是输入从 784 (MNIST) 换成 3072 (CIFAR).
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)            # (N, 3072)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ─────────────────────────────────────────────
# 平移过的 CIFAR-10 (用 transform 在加载时平移)
# ─────────────────────────────────────────────
class RandomShift:
    """每张图按均匀分布在 [-max_shift, +max_shift] 随机水平+垂直平移, 0 填充."""
    def __init__(self, max_shift=4, seed=None):
        self.max_shift = max_shift
        self.rng = np.random.RandomState(seed)

    def __call__(self, img_tensor):
        # img_tensor: (C, H, W) Tensor
        C, H, W = img_tensor.shape
        dy = int(self.rng.randint(-self.max_shift, self.max_shift + 1))
        dx = int(self.rng.randint(-self.max_shift, self.max_shift + 1))
        out = torch.zeros_like(img_tensor)
        # 计算源/目标区间
        sy0, sy1 = max(0, -dy), min(H, H - dy)
        sx0, sx1 = max(0, -dx), min(W, W - dx)
        ty0      = max(0, dy)
        tx0      = max(0, dx)
        ty1      = ty0 + (sy1 - sy0)
        tx1      = tx0 + (sx1 - sx0)
        out[:, ty0:ty1, tx0:tx1] = img_tensor[:, sy0:sy1, sx0:sx1]
        return out


def load_shifted_cifar10_test(batch_size=128, max_shift=4, seed=0):
    """返回平移过的 CIFAR-10 测试集 dataloader. 平移确定性的 (seed 固定)."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        RandomShift(max_shift=max_shift, seed=seed),
    ])
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform,
    )
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# ─────────────────────────────────────────────
# 训练 / 评估 (从 lenet_pytorch.py 简化复制, 不依赖 history 写文件)
# ─────────────────────────────────────────────
def train_model(model, trainloader, device, epochs=10, lr=0.01, momentum=0.9):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f'  epoch {epoch+1:2d}/{epochs}  loss={running/len(trainloader):.4f}  '
              f'({time.time()-t0:.1f}s)')
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return correct / total


# ─────────────────────────────────────────────
# 可视化对比
# ─────────────────────────────────────────────
def plot_comparison(results, save_path):
    """results: dict, 4 项 (model, dataset) → accuracy"""
    models   = ['MLP', 'LeNet']
    datasets = ['原测试集', '平移 ±4 像素']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f1117')

    x = np.arange(len(models))
    width = 0.35

    accs_orig  = [results[(m, '原测试集')]    for m in models]
    accs_shift = [results[(m, '平移 ±4 像素')] for m in models]

    bars1 = ax.bar(x - width/2, accs_orig,  width, color='#4fc3f7',
                   label='原 CIFAR-10 测试集')
    bars2 = ax.bar(x + width/2, accs_shift, width, color='#ff8a65',
                   label='平移 ±4 像素后')

    for bar, acc in zip(bars1, accs_orig):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', color='white', fontsize=11)
    for bar, acc in zip(bars2, accs_shift):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', color='white', fontsize=11)

    # 在每个模型组上标注下降幅度
    for i, m in enumerate(models):
        drop = accs_orig[i] - accs_shift[i]
        ax.annotate(f'↓ {drop:.1%}',
                    xy=(x[i], (accs_orig[i] + accs_shift[i]) / 2),
                    xytext=(x[i], (accs_orig[i] + accs_shift[i]) / 2),
                    ha='center', va='center',
                    color='#ef5350' if drop > 0.05 else '#66bb6a',
                    fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.set_ylim(0, max(max(accs_orig), max(accs_shift)) * 1.15)
    ax.set_ylabel('CIFAR-10 测试准确率', color='#aaaaaa')
    ax.set_title('MLP vs LeNet: 同样训练设置, 在原测试集 vs 平移测试集上的准确率\n'
                 '↓ 表示平移后准确率下降幅度 (越小越说明对位移鲁棒)',
                 color='white', fontsize=13)
    ax.legend(facecolor='#1a1d27', edgecolor='#555', labelcolor='white',
              loc='upper right')
    ax.grid(axis='y', alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f'\n对比柱图已保存: {save_path}')


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 62)
    print('Week 2 T9 拓展: MLP vs LeNet 在原 / 平移 CIFAR-10 上对比')
    print('=' * 62)

    device = pick_device()
    print(f'\n[device] {device}')

    # 准备数据 — 训练集和"原测试集"用标准 transform; "平移测试集"另起一个
    print('\n[data] 加载训练集 + 两个测试集...')
    trainloader, testloader_orig = load_cifar10(batch_size=128)
    testloader_shifted = load_shifted_cifar10_test(batch_size=128, max_shift=4, seed=0)

    # ── 训练 MLP ─────────────────────────────────
    print('\n[train MLP] 3072 → 512 → 256 → 10')
    mlp = MLP()
    print(f'  参数量: {count_params(mlp):,}')
    train_model(mlp, trainloader, device, epochs=10, lr=0.01, momentum=0.9)

    # ── 训练 LeNet ───────────────────────────────
    print('\n[train LeNet] (从头再训一次, 不复用之前的 weights, 保证公平)')
    lenet = LeNet()
    print(f'  参数量: {count_params(lenet):,}')
    train_model(lenet, trainloader, device, epochs=10, lr=0.01, momentum=0.9)

    # ── 在 4 个组合上评估 ────────────────────────
    print('\n[eval] 4 个组合的准确率...')
    results = {}
    for name, model in [('MLP', mlp), ('LeNet', lenet)]:
        for ds_name, ds_loader in [('原测试集', testloader_orig),
                                    ('平移 ±4 像素', testloader_shifted)]:
            acc = evaluate(model, ds_loader, device)
            results[(name, ds_name)] = acc
            print(f'  {name:>5} on {ds_name:>14}: {acc:.2%}')

    # 输出小结
    print('\n[summary]')
    print(f'{"":>8} | {"原测试集":>10} | {"平移 ±4 像素":>14} | {"下降幅度":>10}')
    print('-' * 55)
    for m in ['MLP', 'LeNet']:
        a, b = results[(m, '原测试集')], results[(m, '平移 ±4 像素')]
        print(f'{m:>8} | {a:>10.2%} | {b:>14.2%} | {a-b:>10.2%}')

    # 出图
    save_path = os.path.join(OUTPUT_DIR, 'mlp_vs_lenet_comparison.png')
    plot_comparison(results, save_path)
