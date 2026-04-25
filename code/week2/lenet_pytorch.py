"""
Week 2 T9: PyTorch 实现 LeNet-5, 在 CIFAR-10 上训练动物分类

理论对应:
  - 网络结构      docs/week2/09_pytorch_intro.md §10
  - PyTorch API   docs/week2/09_pytorch_intro.md §3-§7

设计:
  - 复用我们 Week 2 numpy 验证过的卷积/池化数学 (PyTorch 内置同样实现)
  - 用 nn.Module 把 forward 和参数管理打包成类
  - 用 DataLoader 自动 batch + shuffle
  - 用 SGD + momentum (LeNet 时代经典配置)
  - 自动 device 选择 (MPS / CUDA / CPU)
  - 训练曲线 + 测试集准确率写到 assets/week2/outputs/

运行:
  python code/week2/lenet_pytorch.py
"""

import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 主题 + 路径
# ─────────────────────────────────────────────
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

HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.normpath(os.path.join(HERE, '..', '..'))
DATA_DIR   = os.path.join(ROOT, 'data', 'cifar10')
OUTPUT_DIR = os.path.join(ROOT, 'assets', 'week2', 'outputs')

CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]
ANIMAL_CLASSES = {2, 3, 4, 5, 6, 7}  # bird, cat, deer, dog, frog, horse


# ─────────────────────────────────────────────
# Device 选择
# ─────────────────────────────────────────────
def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────
def load_cifar10(batch_size=128, num_workers=2):
    """
    返回 (trainloader, testloader)
    标准 CIFAR-10 预处理: ToTensor + Normalize 到约 [-1, 1]
    """
    transform = T.Compose([
        T.ToTensor(),                                   # PIL → (C, H, W) in [0, 1]
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 平移到 [-1, 1]
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform,
    )

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return trainloader, testloader


# ─────────────────────────────────────────────
# LeNet-5 模型 (适配 CIFAR-10 32×32 RGB)
# ─────────────────────────────────────────────
class LeNet(nn.Module):
    """
    LeCun 1998 原版是 28×28 灰度 + 5×5 valid conv. 这里改 RGB + same padding 不动,
    保持原版 5x5 + valid conv. CIFAR-10 32×32 → 经过两次 (conv + pool) 变 16×5×5.

    Input  (3, 32, 32)
      ↓ Conv 5×5, 6 filters,  ReLU       → (6, 28, 28)
      ↓ MaxPool 2×2, s=2                 → (6, 14, 14)
      ↓ Conv 5×5, 16 filters, ReLU       → (16, 10, 10)
      ↓ MaxPool 2×2, s=2                 → (16, 5, 5)
      ↓ Flatten                          → (400,)
      ↓ FC 120, ReLU
      ↓ FC 84,  ReLU
      ↓ FC 10                            → logits
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6,  kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))     # (6, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))     # (16, 5, 5)
        x = x.flatten(start_dim=1)                   # (400,)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)                           # logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────
# 训练 + 评估
# ─────────────────────────────────────────────
def evaluate(model, loader, device, restrict_to=None):
    """
    返回 (accuracy, total) 和 (per-class correct, per-class total).
    restrict_to: 若给 set, 只在那些类别上算准确率
    """
    model.eval()
    correct = total = 0
    per_cls_corr = np.zeros(10, dtype=np.int64)
    per_cls_tot  = np.zeros(10, dtype=np.int64)
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            for c in range(10):
                mask = (y == c)
                per_cls_tot[c]  += mask.sum().item()
                per_cls_corr[c] += (pred[mask] == c).sum().item()
            if restrict_to is None:
                correct += (pred == y).sum().item()
                total   += y.size(0)
            else:
                mask = torch.tensor([yi.item() in restrict_to for yi in y],
                                    dtype=torch.bool, device=device)
                correct += ((pred == y) & mask).sum().item()
                total   += mask.sum().item()
    return correct / max(total, 1), (per_cls_corr, per_cls_tot)


def train(model, trainloader, testloader, device,
          epochs=10, lr=0.01, momentum=0.9, log_every=100):
    """
    标准 SGD + momentum 训练. 返回训练历史 dict.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': [],
               'animal_test_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = total = 0
        t0 = time.time()

        for i, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc  = correct / total
        test_acc, _    = evaluate(model, testloader, device)
        animal_acc, _  = evaluate(model, testloader, device, restrict_to=ANIMAL_CLASSES)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['animal_test_acc'].append(animal_acc)

        elapsed = time.time() - t0
        print(f'Epoch {epoch+1:2d}/{epochs} | '
              f'Loss {train_loss:.4f} | '
              f'Train {train_acc:.1%} | '
              f'Test {test_acc:.1%} | '
              f'Animals {animal_acc:.1%} | '
              f'{elapsed:.1f}s')

    return history


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────
def plot_training(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], color='#4fc3f7', lw=2.5,
                 marker='o', ms=5)
    axes[0].set_title('训练损失曲线 (LeNet on CIFAR-10)', color='white', fontsize=13)
    axes[0].set_xlabel('Epoch', color='#aaaaaa')
    axes[0].set_ylabel('Cross-Entropy Loss', color='#aaaaaa')
    axes[0].grid(alpha=0.15)

    axes[1].plot(epochs, history['train_acc'],       color='#66bb6a', lw=2.5,
                 marker='o', ms=5, label='训练集 (10 类)')
    axes[1].plot(epochs, history['test_acc'],        color='#ff8a65', lw=2.5,
                 marker='s', ms=5, label='测试集 (10 类)')
    axes[1].plot(epochs, history['animal_test_acc'], color='#ce93d8', lw=2.5,
                 marker='^', ms=5, label='测试集 (仅 6 个动物类)')
    axes[1].set_title('准确率曲线', color='white', fontsize=13)
    axes[1].set_xlabel('Epoch', color='#aaaaaa')
    axes[1].set_ylabel('Accuracy', color='#aaaaaa')
    axes[1].legend(facecolor='#1a1d27', edgecolor='#555', labelcolor='white')
    axes[1].grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f'训练曲线已保存: {save_path}')


def plot_per_class(per_cls_corr, per_cls_tot, save_path):
    accs = per_cls_corr / np.maximum(per_cls_tot, 1)
    is_animal = [c in ANIMAL_CLASSES for c in range(10)]
    colors = ['#ff8a65' if a else '#4fc3f7' for a in is_animal]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1117')
    bars = ax.bar(CIFAR_CLASSES, accs, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy', color='#aaaaaa')
    ax.set_title('LeNet 在 CIFAR-10 各类别上的准确率\n橙色 = 动物 (我们关心的目标)',
                 color='white', fontsize=13)
    ax.grid(axis='y', alpha=0.15)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.0%}', ha='center', va='bottom', color='white', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f'分类柱图已保存: {save_path}')


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 62)
    print('Week 2 T9: LeNet-5 PyTorch 在 CIFAR-10 上训练')
    print('=' * 62)

    device = pick_device()
    print(f'\n[device] {device}')

    print('\n[data] 加载 CIFAR-10 ...')
    trainloader, testloader = load_cifar10(batch_size=128)
    print(f'  训练集: {len(trainloader.dataset)} 张')
    print(f'  测试集: {len(testloader.dataset)} 张')

    model = LeNet()
    print(f'\n[model] LeNet-5')
    print(f'  参数量: {count_params(model):,}')

    print('\n[train] 开始训练 (10 epoch, SGD + momentum) ...')
    history = train(model, trainloader, testloader, device,
                    epochs=10, lr=0.01, momentum=0.9)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存权重
    weight_path = os.path.join(OUTPUT_DIR, 'lenet_weights.pth')
    torch.save(model.state_dict(), weight_path)
    print(f'\n权重已保存: {weight_path}')

    # 训练曲线
    plot_training(history, os.path.join(OUTPUT_DIR, 'lenet_training_curve.png'))

    # 各类别准确率
    _, (per_cls_corr, per_cls_tot) = evaluate(model, testloader, device)
    plot_per_class(per_cls_corr, per_cls_tot,
                   os.path.join(OUTPUT_DIR, 'lenet_per_class_acc.png'))

    # 保存训练历史 (供对比实验复用)
    history_path = os.path.join(OUTPUT_DIR, 'lenet_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'训练历史已保存: {history_path}')

    print(f'\n最终测试准确率: {history["test_acc"][-1]:.2%}')
    print(f'其中动物类准确率: {history["animal_test_acc"][-1]:.2%}')
