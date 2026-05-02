"""
Week 2 文档插图生成脚本

一次性生成 docs/week2/ 02-05 章节里的 6 张可视化图，全部输出到
  assets/week2/figures/<章节>/

包含一次性 CIFAR-10 下载（~170 MB），后续训练也会用到这份数据。

运行:
  MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week2/figures.py
"""

import os
import urllib.request
import tarfile
import pickle

import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide='ignore', over='ignore', invalid='ignore')

# ─────────────────────────────────────────────
# 主题（与 Week 1 保持一致）
# ─────────────────────────────────────────────
DARK_BG   = '#0f1117'
PANEL_BG  = '#1a1d27'
TEXT_LT   = 'white'
TEXT_MD   = '#cccccc'
TEXT_DIM  = '#aaaaaa'
ACC_BLUE  = '#4fc3f7'
ACC_GREEN = '#66bb6a'
ACC_ORANGE = '#ff8a65'
ACC_RED   = '#ef5350'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   PANEL_BG,
    'axes.edgecolor':   '#444444',
    'axes.labelcolor':  TEXT_MD,
    'xtick.color':      TEXT_DIM,
    'ytick.color':      TEXT_DIM,
    'text.color':       TEXT_LT,
    'axes.titlecolor':  TEXT_LT,
    'font.family':      ['Arial Unicode MS', 'sans-serif'],
    'axes.unicode_minus': False,
})

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.normpath(os.path.join(HERE, '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'cifar10')
FIG_BASE = os.path.join(ROOT, 'assets', 'week2', 'figures')


# ─────────────────────────────────────────────
# CIFAR-10 数据加载
# ─────────────────────────────────────────────
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def download_cifar10():
    """首次跑会下载 CIFAR-10 (~170 MB)，之后幂等跳过。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    extracted = os.path.join(DATA_DIR, 'cifar-10-batches-py')
    if os.path.exists(extracted):
        return extracted
    targz = os.path.join(DATA_DIR, 'cifar-10-python.tar.gz')
    if not os.path.exists(targz):
        print(f'下载 CIFAR-10 (~170 MB) → {targz}')
        urllib.request.urlretrieve(CIFAR_URL, targz)
    print('解压 cifar-10-python.tar.gz ...')
    with tarfile.open(targz, 'r:gz') as f:
        f.extractall(DATA_DIR)
    return extracted


def load_cifar10_test():
    """返回 (images: uint8 (N,3,32,32), labels: int32 (N,))"""
    batch_path = os.path.join(download_cifar10(), 'test_batch')
    with open(batch_path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    images = d[b'data'].reshape(-1, 3, 32, 32).astype(np.uint8)
    labels = np.array(d[b'labels'], dtype=np.int32)
    return images, labels


def get_demo_image():
    """从 CIFAR-10 测试集挑一张"颜色丰富、边缘强"的图作为所有 demo 的演示用图。
    优先级: horse → cat → frog → 第 0 张"""
    images, labels = load_cifar10_test()
    for target in (7, 3, 6):  # horse, cat, frog
        idxs = np.where(labels == target)[0]
        if len(idxs) > 0:
            idx = int(idxs[0])
            return images[idx], CIFAR_CLASSES[target]
    return images[0], CIFAR_CLASSES[int(labels[0])]


# ─────────────────────────────────────────────
# MNIST 数据加载（仅 01 章 pixel-shuffle 实验用）
# ─────────────────────────────────────────────
import struct, gzip

MNIST_DIR = os.path.join(ROOT, 'data', 'mnist')
MNIST_URLS = {
    'train-images-idx3-ubyte.gz': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
}


def _mnist_load(name, is_image):
    path = os.path.join(MNIST_DIR, name)
    if not os.path.exists(path):
        os.makedirs(MNIST_DIR, exist_ok=True)
        print(f'  下载 {name} ...')
        urllib.request.urlretrieve(MNIST_URLS[name], path)
    with gzip.open(path, 'rb') as f:
        if is_image:
            _, n, h, w = struct.unpack('>4I', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h * w)
            return data.astype(np.float32) / 255.0
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)


def load_mnist():
    return (_mnist_load('train-images-idx3-ubyte.gz', True),
            _mnist_load('train-labels-idx1-ubyte.gz', False),
            _mnist_load('t10k-images-idx3-ubyte.gz',  True),
            _mnist_load('t10k-labels-idx1-ubyte.gz',  False))


# ─────────────────────────────────────────────
# 卷积工具（朴素实现，仅用于 demo）
# ─────────────────────────────────────────────
def correlate2d(X, W):
    """2D 互相关（valid，no padding，stride 1）。"""
    H, Win = X.shape
    k = W.shape[0]
    Hout, Wout = H - k + 1, Win - k + 1
    Y = np.zeros((Hout, Wout), dtype=np.float32)
    for i in range(Hout):
        for j in range(Wout):
            Y[i, j] = (X[i:i+k, j:j+k] * W).sum()
    return Y


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.sum()).astype(np.float32)


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
LAPLACE = np.array([[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]], dtype=np.float32)
GAUSS5  = gaussian_kernel(5, 1.0)


def to_grayscale(img_chw_uint8):
    """(3,H,W) uint8 → (H,W) float32 in [0,1] 灰度"""
    img = img_chw_uint8.astype(np.float32) / 255.0
    return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]


def style_imshow_axes(ax, title=''):
    ax.set_title(title, color=TEXT_LT, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#444')


def save_fig(fig, chapter, filename):
    out_dir = os.path.join(FIG_BASE, chapter)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  ✓ {os.path.relpath(out_path, ROOT)}')


# ─────────────────────────────────────────────
# 图 1：T2 §5 真实图片的边缘检测
# ─────────────────────────────────────────────
def fig_edge_detection(img_chw, label):
    img_rgb = img_chw.transpose(1, 2, 0)            # (32,32,3)
    gray    = to_grayscale(img_chw)                 # (32,32)
    sx      = correlate2d(gray, SOBEL_X)
    sy      = correlate2d(gray, SOBEL_Y)
    mag     = np.sqrt(sx**2 + sy**2)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(f'真实图片上的边缘检测  (CIFAR-10 第一张 {label}, 32×32)',
                 color=TEXT_LT, fontsize=13, y=1.02)

    axes[0].imshow(img_rgb, interpolation='nearest')
    style_imshow_axes(axes[0], '原图（RGB）')

    axes[1].imshow(gray, cmap='gray', interpolation='nearest')
    style_imshow_axes(axes[1], '灰度（卷积输入）')

    vlim = max(abs(sx.min()), abs(sx.max()))
    axes[2].imshow(sx, cmap='RdBu_r', vmin=-vlim, vmax=vlim, interpolation='nearest')
    style_imshow_axes(axes[2], 'Sobel-x 输出\n(垂直边缘: 红=右亮 蓝=左亮)')

    axes[3].imshow(mag, cmap='gray', interpolation='nearest')
    style_imshow_axes(axes[3], '梯度幅值 |∇|\n(全部边缘强度)')

    plt.tight_layout()
    save_fig(fig, '02_convolution', 'edge_detection_real.png')


# ─────────────────────────────────────────────
# 图 2：T2 §5.2 多个经典 filter 在同一张图上的对比
# ─────────────────────────────────────────────
def fig_classic_filters(img_chw, label):
    gray = to_grayscale(img_chw)

    outs = {
        '原图（灰度）': gray,
        'Sobel-x\n(垂直边缘)': correlate2d(gray, SOBEL_X),
        'Sobel-y\n(水平边缘)': correlate2d(gray, SOBEL_Y),
        'Laplace\n(全方向边缘 / 锐化)': correlate2d(gray, LAPLACE),
        'Gaussian 5×5\n(平滑去噪)': correlate2d(gray, GAUSS5),
    }

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    fig.suptitle(f'同一张图被 5 种经典 filter 处理后的对比  ({label})',
                 color=TEXT_LT, fontsize=13, y=1.03)

    for ax, (title, arr) in zip(axes, outs.items()):
        if 'Sobel' in title or 'Laplace' in title:
            vlim = max(abs(arr.min()), abs(arr.max())) + 1e-6
            ax.imshow(arr, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                      interpolation='nearest')
        else:
            ax.imshow(arr, cmap='gray', interpolation='nearest')
        style_imshow_axes(ax, title)

    plt.tight_layout()
    save_fig(fig, '02_convolution', 'classic_filters_grid.png')


# ─────────────────────────────────────────────
# 图 3：T4 §1.1 RGB 通道分解
# ─────────────────────────────────────────────
def fig_rgb_channels(img_chw, label):
    img_rgb = img_chw.transpose(1, 2, 0)            # (H,W,3)
    R, G, B = img_chw[0], img_chw[1], img_chw[2]    # 各 (H,W)

    # 通道单色显示：把单通道放回 RGB 的对应位置，其它通道清零
    R_only = np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=-1)
    G_only = np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=-1)
    B_only = np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=-1)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f'RGB 三个通道分解  (CIFAR-10 {label}, 32×32×3)',
                 color=TEXT_LT, fontsize=13, y=1.00)

    # 第一行：原图 + 三通道单独的灰度
    axes[0, 0].imshow(img_rgb, interpolation='nearest')
    style_imshow_axes(axes[0, 0], f'原图（RGB 三通道叠加）\nshape (3, 32, 32)')

    for ax, ch, name, color in zip(
        axes[0, 1:],
        (R, G, B),
        ('R', 'G', 'B'),
        (ACC_RED, ACC_GREEN, ACC_BLUE),
    ):
        ax.imshow(ch, cmap='gray', interpolation='nearest')
        style_imshow_axes(ax, f'{name} 通道（灰度显示）\nshape (32, 32)')
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(2)

    # 第二行：原图重复 + 三通道单色显示（看清每个通道贡献的颜色）
    axes[1, 0].imshow(img_rgb, interpolation='nearest')
    style_imshow_axes(axes[1, 0], '原图（参考）')

    for ax, ch_img, name, color in zip(
        axes[1, 1:],
        (R_only, G_only, B_only),
        ('R only', 'G only', 'B only'),
        (ACC_RED, ACC_GREEN, ACC_BLUE),
    ):
        ax.imshow(ch_img, interpolation='nearest')
        style_imshow_axes(ax, f'{name}\n(其它通道清零, 看 {name[0]} 贡献的颜色)')
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(2)

    plt.tight_layout()
    save_fig(fig, '04_multi_channel', 'rgb_channel_split.png')


# ─────────────────────────────────────────────
# 图 4：T3 §1.0 padding 覆盖热图
# ─────────────────────────────────────────────
def touch_count_map(H, W, k, padding=0):
    """每个原输入像素出现在多少个 k×k filter 窗口里。"""
    Hp, Wp = H + 2 * padding, W + 2 * padding
    counts = np.zeros((Hp, Wp), dtype=np.int32)
    for i in range(Hp - k + 1):
        for j in range(Wp - k + 1):
            counts[i:i+k, j:j+k] += 1
    if padding > 0:
        counts = counts[padding:padding+H, padding:padding+W]
    return counts


def fig_backprop_full_example():
    """完整反向传播算例：4×4 X + 2×2 W + b → Y → δ → ∇W, ∇X, ∇b 全部展开。"""
    X = np.array([[1, 2, 3, 1],
                  [4, 5, 1, 2],
                  [2, 3, 4, 5],
                  [1, 2, 0, 3]], dtype=float)
    W = np.array([[1, 0],
                  [-1, 1]], dtype=float)
    b = 0.0
    delta = np.ones((3, 3), dtype=float)  # 假设下游传回的 δ 全为 1

    # 前向 Y = correlate2d(X, W) + b
    Y = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            Y[i, j] = (X[i:i+2, j:j+2] * W).sum() + b

    # ∇W = correlate2d(X, delta)
    grad_W = np.zeros_like(W)
    for m in range(2):
        for n in range(2):
            grad_W[m, n] = (X[m:m+3, n:n+3] * delta).sum()

    # ∇X = correlate2d(pad(delta, 1), flip(W))
    delta_padded = np.pad(delta, 1, mode='constant')
    W_flip = np.flip(W, axis=(0, 1))
    grad_X = np.zeros_like(X)
    for a in range(4):
        for b_ in range(4):
            grad_X[a, b_] = (delta_padded[a:a+2, b_:b_+2] * W_flip).sum()

    grad_b = delta.sum()

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('§3.5 完整反向传播算例：用同一组 (X, W, δ) 同时验证 ∇W, ∇X, ∇b 三个量',
                 color=TEXT_LT, fontsize=14, y=0.995)
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 0.06, 1.1],
                          width_ratios=[1, 1, 1.1, 1, 1.1, 1.1],
                          hspace=0.45, wspace=0.35)

    # 第一行：前向 X ⊛ W = Y, 然后 δ
    ax = fig.add_subplot(gs[0, 0])
    _draw_grid_2d(ax, X, cmap='Blues', fontsize=14)
    ax.set_title('输入 X (4×4)', color=ACC_BLUE, fontsize=11)
    for sp in ax.spines.values(): sp.set_color(ACC_BLUE); sp.set_linewidth(1.5)

    ax = fig.add_subplot(gs[0, 1])
    _draw_grid_2d(ax, W, cmap='RdBu_r', vmin=-1.5, vmax=1.5, fontsize=18)
    ax.set_title('filter W (2×2)\nb = 0', color=ACC_ORANGE, fontsize=11)
    for sp in ax.spines.values(): sp.set_color(ACC_ORANGE); sp.set_linewidth(1.5)

    ax = fig.add_subplot(gs[0, 2])
    _draw_grid_2d(ax, Y, cmap='RdBu_r', vmin=-7, vmax=7, fontsize=14)
    ax.set_title('前向 Y = X ⋆ W + b\n(3×3)', color=TEXT_LT, fontsize=11)

    ax = fig.add_subplot(gs[0, 3])
    _draw_grid_2d(ax, delta, cmap='Reds', vmin=0, vmax=2, fontsize=14)
    ax.set_title('下游传回 δ = ∂L/∂Y\n(3×3, 这里取全 1)',
                 color=ACC_RED, fontsize=11)
    for sp in ax.spines.values(): sp.set_color(ACC_RED); sp.set_linewidth(1.5)

    # 中间分隔（标注：以下三块是反向）
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    ax.text(0.5, 0.5, '↓ ↓ ↓   反向：用 X 和 δ 算出三个梯度   ↓ ↓ ↓',
            color=ACC_GREEN, fontsize=14, ha='center', va='center',
            fontweight='bold')

    # 第二行 (gs[2, :]): ∇W / ∇X / ∇b
    # ∇W
    ax_w = fig.add_subplot(gs[2, 0:2])
    _draw_grid_2d(ax_w, grad_W, cmap='Greens', vmin=0, vmax=30, fontsize=22)
    ax_w.set_title('∇W = correlate(X, δ)  (2×2)\n'
                   '把 δ 当 filter 在 X 上滑\n'
                   f'∇W = [[{int(grad_W[0,0])}, {int(grad_W[0,1])}], '
                   f'[{int(grad_W[1,0])}, {int(grad_W[1,1])}]]',
                   color=ACC_GREEN, fontsize=11)
    for sp in ax_w.spines.values(): sp.set_color(ACC_GREEN); sp.set_linewidth(2)

    # ∇X
    ax_x = fig.add_subplot(gs[2, 2:4])
    _draw_grid_2d(ax_x, grad_X, cmap='Greens', vmin=-2, vmax=2, fontsize=14)
    ax_x.set_title('∇X = correlate(pad(δ, 1), flip(W))  (4×4)\n'
                   'pad δ 一圈 0，再用 flip(W) 卷过去\n'
                   '4 个角值 = flip(W) 的 4 个角',
                   color=ACC_GREEN, fontsize=11)
    for sp in ax_x.spines.values(): sp.set_color(ACC_GREEN); sp.set_linewidth(2)

    # ∇b（标量）
    ax_b = fig.add_subplot(gs[2, 4:6])
    ax_b.axis('off')
    ax_b.add_patch(plt.Rectangle((0.15, 0.2), 0.7, 0.55,
                                  facecolor=ACC_GREEN, alpha=0.18,
                                  edgecolor=ACC_GREEN, linewidth=2,
                                  transform=ax_b.transAxes))
    ax_b.text(0.5, 0.7, '∇b = sum(δ)', color=ACC_GREEN, fontsize=14,
              ha='center', va='center', fontweight='bold',
              transform=ax_b.transAxes)
    ax_b.text(0.5, 0.48, f'= {int(grad_b)}', color=TEXT_LT, fontsize=36,
              ha='center', va='center', fontweight='bold',
              transform=ax_b.transAxes)
    ax_b.text(0.5, 0.28, 'δ 全 1, 9 格相加 = 9',
              color=TEXT_DIM, fontsize=11, ha='center', va='center',
              style='italic', transform=ax_b.transAxes)
    ax_b.text(0.5, 0.05,
              '多 batch 时正确写法：grad_b = delta.sum(axis=(0, 2, 3))',
              color=ACC_RED, fontsize=10, ha='center', va='center',
              transform=ax_b.transAxes, fontweight='bold')

    save_fig(fig, '06_conv_backprop', 'backprop_full_example.png')


def fig_gradient_aggregation():
    """X[a,b] 在前向流向多个 Y, 反向把多个 δ·W 项汇聚回 X[a,b]。"""
    fig = plt.figure(figsize=(13, 5.5))
    fig.suptitle('§4.5 反向梯度怎么"汇聚"：一个 X[a,b] 通过多个 Y[i,j] 路径回流',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(1, 5,
                          width_ratios=[1, 0.6, 1, 0.6, 1],
                          wspace=0.15)

    # 左：4×4 X 高亮 X[1,1]
    ax_x = fig.add_subplot(gs[0, 0])
    X_disp = np.zeros((4, 4))
    X_disp[1, 1] = 1
    ax_x.imshow(X_disp, cmap='Blues', vmin=0, vmax=1)
    for r in range(4):
        for c in range(4):
            v = '★' if (r, c) == (1, 1) else '·'
            ax_x.text(c, r, v, ha='center', va='center',
                      color='white' if (r, c) == (1, 1) else '#666',
                      fontsize=18 if v == '★' else 14, fontweight='bold')
    ax_x.set_title('X (4×4)\n关注 X[1, 1]（★）',
                   color=ACC_BLUE, fontsize=11)
    ax_x.set_xticks([]); ax_x.set_yticks([])
    for sp in ax_x.spines.values(): sp.set_color('#444')

    # 中左：箭头 + "前向：X[1,1] → 多个 Y"
    ax_a1 = fig.add_subplot(gs[0, 1]); ax_a1.axis('off')
    ax_a1.text(0.5, 0.7, '前向\nX[1,1] 影响 4 个 Y',
               color=ACC_ORANGE, fontsize=11, ha='center',
               va='center', fontweight='bold')
    for k in range(4):
        ax_a1.annotate('', xy=(1.0, 0.55 - k*0.1),
                       xytext=(0.0, 0.55 - k*0.1),
                       arrowprops=dict(arrowstyle='->', color=ACC_ORANGE, lw=1.3,
                                       alpha=0.85))

    # 中：3×3 Y 高亮 (0,0), (0,1), (1,0), (1,1)
    ax_y = fig.add_subplot(gs[0, 2])
    Y_disp = np.zeros((3, 3))
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        Y_disp[r, c] = 1
    ax_y.imshow(Y_disp, cmap='Oranges', vmin=0, vmax=1.5)
    labels = {(0, 0): 'W[1,1]', (0, 1): 'W[1,0]',
              (1, 0): 'W[0,1]', (1, 1): 'W[0,0]'}
    for r in range(3):
        for c in range(3):
            if (r, c) in labels:
                ax_y.text(c, r, f'Y[{r},{c}]\n×{labels[(r,c)]}',
                          ha='center', va='center', color='black',
                          fontsize=9, fontweight='bold')
            else:
                ax_y.text(c, r, '·', ha='center', va='center',
                          color='#666', fontsize=14)
    ax_y.set_title('Y (3×3)\n4 个 Y 用了 X[1,1]\n配的 W 索引是反序',
                   color=ACC_ORANGE, fontsize=11)
    ax_y.set_xticks([]); ax_y.set_yticks([])
    for sp in ax_y.spines.values(): sp.set_color('#444')

    # 中右：箭头 + "反向：4 个 δ·W 汇聚"
    ax_a2 = fig.add_subplot(gs[0, 3]); ax_a2.axis('off')
    ax_a2.text(0.5, 0.7, '反向\n4 个 δ[i,j]·W 汇聚',
               color=ACC_GREEN, fontsize=11, ha='center',
               va='center', fontweight='bold')
    for k in range(4):
        ax_a2.annotate('', xy=(1.0, 0.55 - k*0.1),
                       xytext=(0.0, 0.55 - k*0.1),
                       arrowprops=dict(arrowstyle='->', color=ACC_GREEN, lw=1.3,
                                       alpha=0.85))

    # 右：∇X[1,1] 表达式
    ax_g = fig.add_subplot(gs[0, 4]); ax_g.axis('off')
    ax_g.add_patch(plt.Rectangle((0.05, 0.25), 0.9, 0.55,
                                  facecolor=ACC_GREEN, alpha=0.15,
                                  edgecolor=ACC_GREEN, linewidth=2,
                                  transform=ax_g.transAxes))
    ax_g.text(0.5, 0.74, '∇X[1, 1] =', color=ACC_GREEN, fontsize=14,
              ha='center', fontweight='bold', transform=ax_g.transAxes)
    ax_g.text(0.5, 0.6,
              'δ[0,0]·W[1,1] +\n'
              'δ[0,1]·W[1,0] +\n'
              'δ[1,0]·W[0,1] +\n'
              'δ[1,1]·W[0,0]',
              color=TEXT_LT, fontsize=11, ha='center', va='center',
              family='monospace', transform=ax_g.transAxes)
    ax_g.text(0.5, 0.16,
              '= δ[0:2, 0:2] · flip(W)',
              color=ACC_RED, fontsize=11, ha='center',
              fontweight='bold', transform=ax_g.transAxes)
    ax_g.text(0.5, 0.05,
              'W 自然反序 → "翻转 W"',
              color=TEXT_DIM, fontsize=10, ha='center',
              style='italic', transform=ax_g.transAxes)

    save_fig(fig, '06_conv_backprop', 'gradient_aggregation.png')


def fig_stride2_backward():
    """stride=2 卷积的反向：在 δ 元素之间插 0 后再做标准 ∇X 流程。"""
    # 5×5 X, 3×3 W, stride=2 → 2×2 Y
    delta = np.array([[1, 2],
                      [3, 4]], dtype=int)
    # 在 δ 元素之间插入 stride-1 = 1 个 0
    delta_dilated = np.zeros((3, 3), dtype=int)
    delta_dilated[0::2, 0::2] = delta
    # pad k-1 = 2 圈 0
    delta_padded = np.pad(delta_dilated, 2, mode='constant')

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('§4.6 stride > 1 反向：先在 δ 元素之间插 (s − 1) 个 0，再做标准的 ∇X 流程',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.4, 1.1, 0.4, 1.6],
                          wspace=0.25)

    # 左：原始 δ (2×2)
    ax = fig.add_subplot(gs[0, 0])
    _draw_grid_2d(ax, delta, cmap='Reds', vmin=0, vmax=5, fontsize=22)
    ax.set_title('δ (2×2)\nstride=2 卷积的输出梯度',
                 color=ACC_RED, fontsize=11)

    # 箭头 1
    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
    ax.text(0.5, 0.55, '插\n(s − 1) = 1\n个 0',
            color=ACC_ORANGE, fontsize=11, ha='center', va='center',
            fontweight='bold')
    ax.annotate('', xy=(1.0, 0.3), xytext=(0.0, 0.3),
                arrowprops=dict(arrowstyle='->', color=ACC_ORANGE, lw=2))

    # 中：dilated δ (3×3)
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(delta_dilated, cmap='Reds', vmin=0, vmax=5)
    for r in range(3):
        for c in range(3):
            v = delta_dilated[r, c]
            color = 'white' if v >= 3 else 'black'
            ax.text(c, r, str(v),
                    ha='center', va='center', color=color,
                    fontsize=18, fontweight='bold')
            # 给"插入的 0"加虚线边框
            if r % 2 == 1 or c % 2 == 1:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                           facecolor='none',
                                           edgecolor=ACC_ORANGE,
                                           linewidth=1.5,
                                           linestyle='--'))
    ax.set_title('dilated δ (3×3)\n橙虚框 = 插入的 0\n现在 δ 形状像 stride=1 的产出',
                 color=ACC_ORANGE, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(ACC_ORANGE); sp.set_linewidth(1.5)

    # 箭头 2
    ax = fig.add_subplot(gs[0, 3]); ax.axis('off')
    ax.text(0.5, 0.6,
            'pad k−1=2\n+ flip(W)\n+ correlate',
            color=ACC_GREEN, fontsize=11, ha='center', va='center',
            fontweight='bold')
    ax.annotate('', xy=(1.0, 0.3), xytext=(0.0, 0.3),
                arrowprops=dict(arrowstyle='->', color=ACC_GREEN, lw=2))

    # 右：dilated δ + 公式说明
    ax = fig.add_subplot(gs[0, 4]); ax.axis('off')
    ax.add_patch(plt.Rectangle((0.02, 0.10), 0.96, 0.78,
                                facecolor=ACC_GREEN, alpha=0.10,
                                edgecolor=ACC_GREEN, linewidth=2,
                                transform=ax.transAxes))
    ax.text(0.5, 0.82,
            '为什么这样做？',
            color=ACC_GREEN, fontsize=12, ha='center',
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.65,
            'stride>1 的前向只在 X 上每 s 个像素采一次样，\n'
            '所以 δ[i, j] 只对应 X 上 (i·s, j·s) 那一格。\n'
            '反向时把 δ 之间填 0，相当于"还原"出\n'
            'stride=1 卷积本来会有的输出梯度。',
            color=TEXT_LT, fontsize=10, ha='center', va='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.32,
            '剩下的步骤和 §4.4 一字不差：\n'
            'pad (k−1) 圈 0 + 用 flip(W) 做 correlate\n'
            '→ 输出 ∇X 形状 = (5×5)，对得上原始 X',
            color=ACC_BLUE, fontsize=10, ha='center', va='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.13,
            'numpy: dilated = np.zeros(...); dilated[::s, ::s] = δ',
            color=TEXT_DIM, fontsize=10, ha='center',
            family='monospace', transform=ax.transAxes)

    save_fig(fig, '06_conv_backprop', 'stride2_backward.png')


def fig_pool_numerical():
    """4×4 输入分 4 个 2×2 块 → MaxPool 输出 vs AvgPool 输出对照。"""
    X = np.array([[1, 3, 2, 4],
                  [5, 6, 7, 8],
                  [9, 2, 1, 5],
                  [3, 4, 8, 9]], dtype=int)
    # 4 个 2×2 块各取 max / mean
    blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
    block_colors = [ACC_BLUE, ACC_GREEN, ACC_ORANGE, '#ba68c8']
    Y_max = np.array([[max(X[i:i+2, j:j+2].flatten()) for j in (0, 2)]
                      for i in (0, 2)])
    Y_avg = np.array([[X[i:i+2, j:j+2].mean() for j in (0, 2)]
                      for i in (0, 2)])

    fig = plt.figure(figsize=(14, 5.5))
    fig.suptitle('§2/§3 池化数值算例：同一个 4×4 输入分 4 个 2×2 块，MaxPool 取 max / AvgPool 取 mean',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.2, 0.3, 1.0, 0.05, 1.0],
                          wspace=0.25)

    # 左：输入 X，4 个块用边框上色
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(X, cmap='Blues', vmin=0, vmax=12)
    for r in range(4):
        for c in range(4):
            ax.text(c, r, str(X[r, c]),
                    ha='center', va='center', color='black',
                    fontsize=14, fontweight='bold')
    for k_idx, (bi, bj) in enumerate(blocks):
        ax.add_patch(plt.Rectangle((bj-0.5, bi-0.5), 2, 2,
                                    facecolor='none',
                                    edgecolor=block_colors[k_idx],
                                    linewidth=3))
    ax.set_title('输入 X (4×4)\n4 个不重叠 2×2 子块（彩色框）',
                 color=TEXT_LT, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#444')

    # 中：箭头
    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
    ax.text(0.5, 0.65, '上路\nMaxPool', color=ACC_RED, fontsize=11,
            ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(1.0, 0.65), xytext=(0.0, 0.65),
                arrowprops=dict(arrowstyle='->', color=ACC_RED, lw=2))
    ax.text(0.5, 0.35, '下路\nAvgPool', color=ACC_GREEN, fontsize=11,
            ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(1.0, 0.35), xytext=(0.0, 0.35),
                arrowprops=dict(arrowstyle='->', color=ACC_GREEN, lw=2))

    # MaxPool 输出
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(Y_max, cmap='RdBu_r', vmin=0, vmax=12)
    for r in range(2):
        for c in range(2):
            k_idx = r * 2 + c
            ax.text(c, r, str(int(Y_max[r, c])),
                    ha='center', va='center', color='black',
                    fontsize=22, fontweight='bold')
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                       facecolor='none',
                                       edgecolor=block_colors[k_idx],
                                       linewidth=3))
    ax.set_title('MaxPool 输出 (2×2)\nmax(块) → 保留最强响应\n6, 8, 9, 9',
                 color=ACC_RED, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#444')

    # AvgPool 输出
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(Y_avg, cmap='RdBu_r', vmin=0, vmax=12)
    for r in range(2):
        for c in range(2):
            k_idx = r * 2 + c
            ax.text(c, r, f'{Y_avg[r, c]:.2f}',
                    ha='center', va='center', color='black',
                    fontsize=18, fontweight='bold')
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                       facecolor='none',
                                       edgecolor=block_colors[k_idx],
                                       linewidth=3))
    ax.set_title('AvgPool 输出 (2×2)\nmean(块) → 区域平均\n3.75, 5.25, 4.5, 5.75',
                 color=ACC_GREEN, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#444')

    save_fig(fig, '05_pooling', 'pool_numerical.png')


def fig_maxpool_translation_invariance():
    """4×4 输入里一个亮点平移 1 格，2×2 MaxPool 输出完全一样。"""
    X1 = np.zeros((4, 4), dtype=int); X1[0, 0] = 9
    X2 = np.zeros((4, 4), dtype=int); X2[0, 1] = 9

    def maxpool22(X):
        return np.array([[X[i:i+2, j:j+2].max() for j in (0, 2)]
                         for i in (0, 2)])

    Y1 = maxpool22(X1)
    Y2 = maxpool22(X2)

    fig = plt.figure(figsize=(13, 4))
    fig.suptitle('§2.3 MaxPool 的窗口内平移不变性：亮点在 2×2 窗口里挪 1 格，输出完全一样',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.4, 1, 0.4, 1], wspace=0.25)

    panels = [
        (gs[0, 0], '位移前 X1\n亮点在 (0, 0)', X1),
        (gs[0, 2], '位移后 X2\n亮点在 (0, 1)', X2),
    ]
    for cell, title, M in panels:
        ax = fig.add_subplot(cell)
        ax.imshow(M, cmap='Blues', vmin=0, vmax=10)
        for r in range(4):
            for c in range(4):
                v = M[r, c]
                ax.text(c, r, str(v),
                        ha='center', va='center',
                        color='black' if v == 0 else 'white',
                        fontsize=12, fontweight='bold')
        # 高亮第一个 2×2 块（含亮点的那一块）
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 2, 2,
                                   facecolor='none',
                                   edgecolor=ACC_ORANGE, linewidth=3))
        ax.set_title(title, color=TEXT_LT, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color('#444')

    # 中间箭头：位移前 → MaxPool / 位移后 → MaxPool
    for cell, label in [(gs[0, 1], '2×2 MaxPool'), (gs[0, 3], '2×2 MaxPool')]:
        ax = fig.add_subplot(cell); ax.axis('off')
        ax.text(0.5, 0.55, label, color=ACC_GREEN, fontsize=10,
                ha='center', va='center', fontweight='bold')
        ax.annotate('', xy=(1.0, 0.5), xytext=(0.0, 0.5),
                    arrowprops=dict(arrowstyle='->', color=ACC_GREEN, lw=2))

    # 右侧：两个 MaxPool 输出叠在一起 → 完全一样
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(Y1, cmap='Greens', vmin=0, vmax=10)
    for r in range(2):
        for c in range(2):
            v = Y1[r, c]
            v2 = Y2[r, c]
            ax.text(c, r, f'{v}',
                    ha='center', va='center',
                    color='black', fontsize=20, fontweight='bold')
    # 在边角写 "Y1 = Y2"
    ax.set_title('两个输出完全一样\nY1 = Y2 = [[9,0],[0,0]]\n→ MaxPool 对窗口内位移免疫',
                 color=ACC_GREEN, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ACC_GREEN); sp.set_linewidth(2)

    save_fig(fig, '05_pooling', 'maxpool_translation_invariance.png')


def fig_receptive_field_layered():
    """显示两层 3×3 conv 后 layer2 中心像素 → layer1 3×3 → 原图 5×5 的层级映射。"""
    fig = plt.figure(figsize=(14, 5.5))
    fig.suptitle('§5.2 感受野层级展开：layer2 一像素 → layer1 3×3 子区 → 原图 5×5 区域',
                 color=TEXT_LT, fontsize=13, y=0.98)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.4, 0.25, 1.0, 0.25, 0.7],
                          wspace=0.2)

    # 原图 7×7
    ax_in = fig.add_subplot(gs[0, 0])
    grid_in = np.zeros((7, 7))
    # 高亮中心 5×5（对应 layer2 中心像素的感受野）
    grid_in[1:6, 1:6] = 1
    # 进一步高亮中心 3×3（对应 layer1 中心像素的感受野）
    grid_in[2:5, 2:5] = 2
    # 中心 1×1
    grid_in[3, 3] = 3
    ax_in.imshow(grid_in, cmap='Blues', vmin=0, vmax=4)
    for r in range(7):
        for c in range(7):
            ax_in.text(c, r, '·', ha='center', va='center',
                       color='#888', fontsize=10)
    # 5×5 框
    ax_in.add_patch(plt.Rectangle((0.5, 0.5), 5, 5, facecolor='none',
                                  edgecolor=ACC_BLUE, linewidth=2.5))
    # 3×3 框
    ax_in.add_patch(plt.Rectangle((1.5, 1.5), 3, 3, facecolor='none',
                                  edgecolor=ACC_GREEN, linewidth=2.5))
    ax_in.set_title(
        '原图 (7×7)\n绿框 3×3 = layer1[1,1] 感受野  /  蓝框 5×5 = layer2[1,1] 感受野',
        color=TEXT_LT, fontsize=11)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for sp in ax_in.spines.values(): sp.set_color('#444')

    # 箭头 1
    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
    ax.text(0.5, 0.65, '3×3 conv\nstride=1', color=ACC_GREEN,
            fontsize=10, ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(1.0, 0.4), xytext=(0.0, 0.4),
                arrowprops=dict(arrowstyle='->', color=ACC_GREEN, lw=2))

    # Layer 1 (5×5)
    ax_l1 = fig.add_subplot(gs[0, 2])
    grid_l1 = np.zeros((5, 5))
    # 高亮中心 3×3（对应 layer2 中心像素能看到的 layer1 区域）
    grid_l1[1:4, 1:4] = 1
    grid_l1[2, 2] = 2
    ax_l1.imshow(grid_l1, cmap='Greens', vmin=0, vmax=3)
    for r in range(5):
        for c in range(5):
            ax_l1.text(c, r, '·', ha='center', va='center',
                       color='#888', fontsize=10)
    # 中心 3×3 框：layer2 直接看到的 layer1 区域
    ax_l1.add_patch(plt.Rectangle((0.5, 0.5), 3, 3, facecolor='none',
                                  edgecolor=ACC_BLUE, linewidth=2.5))
    # 中心点
    ax_l1.add_patch(plt.Circle((2, 2), 0.25, facecolor=ACC_RED,
                               edgecolor='white', linewidth=1))
    ax_l1.set_title('layer1 (5×5)\n蓝框 3×3 = layer2[1,1] 直接看到的子区',
                    color=ACC_GREEN, fontsize=11)
    ax_l1.set_xticks([]); ax_l1.set_yticks([])
    for sp in ax_l1.spines.values(): sp.set_color('#444')

    # 箭头 2
    ax = fig.add_subplot(gs[0, 3]); ax.axis('off')
    ax.text(0.5, 0.65, '3×3 conv\nstride=1', color=ACC_BLUE,
            fontsize=10, ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(1.0, 0.4), xytext=(0.0, 0.4),
                arrowprops=dict(arrowstyle='->', color=ACC_BLUE, lw=2))

    # Layer 2 (3×3)
    ax_l2 = fig.add_subplot(gs[0, 4])
    grid_l2 = np.zeros((3, 3))
    grid_l2[1, 1] = 1
    ax_l2.imshow(grid_l2, cmap='Reds', vmin=0, vmax=2)
    for r in range(3):
        for c in range(3):
            ax_l2.text(c, r, '·', ha='center', va='center',
                       color='#888', fontsize=10)
    ax_l2.add_patch(plt.Circle((1, 1), 0.25, facecolor=ACC_RED,
                               edgecolor='white', linewidth=1))
    ax_l2.set_title('layer2 (3×3)\n中心红点感受野 = 5×5',
                    color=ACC_RED, fontsize=11)
    ax_l2.set_xticks([]); ax_l2.set_yticks([])
    for sp in ax_l2.spines.values(): sp.set_color('#444')

    save_fig(fig, '05_pooling', 'receptive_field_layered.png')


def fig_filter_shape_extension():
    """单通道 (3,3) filter → 多通道 (3,3,3) filter 的形状扩展示意。"""
    W2d = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    # 三个独立的 3x3 子 filter
    W_R = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])    # Sobel-x
    W_G = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])    # Sobel-y
    W_B = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])       # identity 中心

    fig = plt.figure(figsize=(13, 4.5))
    fig.suptitle('单通道 filter (3,3) → 多通道 filter (C, 3, 3)：每个输入通道配一个独立的 2D 子 filter',
                 color=TEXT_LT, fontsize=12, y=0.99)
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 0.4, 0.6, 1, 1, 1, 0.5], wspace=0.25)

    # 左：单通道 2D filter
    ax = fig.add_subplot(gs[0, 0])
    _draw_grid_2d(ax, W2d, cmap='RdBu_r', vmin=-1.5, vmax=1.5, fontsize=18)
    ax.set_title('单通道 (T2)\nfilter shape = (3, 3)\n9 个权重',
                 color=TEXT_LT, fontsize=11)

    # 中：箭头 + 文字
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    ax.text(0.5, 0.6, '输入有\nC = 3 通道', color=ACC_GREEN,
            fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.3, '→', color=TEXT_LT,
            fontsize=28, ha='center', va='center')

    # 占位（视觉间隔）
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.text(0.5, 0.5,
            'C = 3, k = 3\n→\nfilter shape\n(3, 3, 3)\n27 个权重',
            color=ACC_ORANGE, fontsize=11, ha='center', va='center', fontweight='bold')

    # 右：3 个独立的子 filter（每个对应一个输入通道）
    for col, (W, name, color) in enumerate([
        (W_R, 'sub-filter for R\n(看 R 通道)', '#ef5350'),
        (W_G, 'sub-filter for G\n(看 G 通道)', '#66bb6a'),
        (W_B, 'sub-filter for B\n(看 B 通道)', '#4fc3f7'),
    ]):
        ax = fig.add_subplot(gs[0, 3 + col])
        _draw_grid_2d(ax, W, cmap='RdBu_r', vmin=-1.5, vmax=1.5, fontsize=16)
        ax.set_title(name, color=color, fontsize=11)
        # 加色边框
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(2)

    save_fig(fig, '04_multi_channel', 'filter_shape_extension.png')


def fig_multichannel_conv_numerical():
    """多通道卷积的具体数值算例：3×4×4 输入 + 3×3×3 filter → 2×2 输出。"""
    # R: 第 2 列是亮条；G: 第 2 行是亮条；B: 主对角线
    X_R = np.array([[0, 0, 1, 0],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0]])
    X_G = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0]])
    X_B = np.eye(4, dtype=int)

    # 三个不同的 sub filter
    W_R = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])    # 检测垂直边
    W_G = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])    # 检测水平边
    W_B = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])       # identity 中心

    def conv2d_simple(X, W):
        out = np.zeros((2, 2), dtype=int)
        for i in range(2):
            for j in range(2):
                out[i, j] = (X[i:i+3, j:j+3] * W).sum()
        return out

    mid_R = conv2d_simple(X_R, W_R)
    mid_G = conv2d_simple(X_G, W_G)
    mid_B = conv2d_simple(X_B, W_B)
    Y = mid_R + mid_G + mid_B

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('§2.1 多通道卷积的数值算例：每个通道独立卷 → 3 张中间结果对位相加 → 1 张输出',
                 color=TEXT_LT, fontsize=13, y=0.995)
    gs = fig.add_gridspec(4, 5,
                          width_ratios=[1, 0.25, 0.7, 0.25, 1],
                          height_ratios=[1, 1, 1, 1.3],
                          hspace=0.55, wspace=0.25)

    rows = [
        ('R 通道', '#ef5350', X_R, W_R, mid_R, '检测垂直边\n(Sobel-x 简化)'),
        ('G 通道', '#66bb6a', X_G, W_G, mid_G, '检测水平边\n(Sobel-y 简化)'),
        ('B 通道', '#4fc3f7', X_B, W_B, mid_B, 'identity 中心'),
    ]

    for r, (name, c, X_c, W_c, mid_c, fname) in enumerate(rows):
        # 列 0: 输入通道
        ax = fig.add_subplot(gs[r, 0])
        _draw_grid_2d(ax, X_c, cmap='Blues', vmin=0, vmax=1, fontsize=12)
        ax.set_title(f'输入 {name} (4×4)', color=c, fontsize=11)
        for sp in ax.spines.values(): sp.set_color(c); sp.set_linewidth(1.5)

        # 列 1: ⊛ 符号
        ax = fig.add_subplot(gs[r, 1]); ax.axis('off')
        ax.text(0.5, 0.5, '⊛', color=TEXT_LT, fontsize=22,
                ha='center', va='center')

        # 列 2: filter
        ax = fig.add_subplot(gs[r, 2])
        _draw_grid_2d(ax, W_c, cmap='RdBu_r', vmin=-1.5, vmax=1.5, fontsize=12)
        ax.set_title(f'sub-filter for {name[0]}\n{fname}', color=c, fontsize=10)
        for sp in ax.spines.values(): sp.set_color(c); sp.set_linewidth(1.5)

        # 列 3: = 符号
        ax = fig.add_subplot(gs[r, 3]); ax.axis('off')
        ax.text(0.5, 0.5, '=', color=TEXT_LT, fontsize=22,
                ha='center', va='center')

        # 列 4: 中间结果
        ax = fig.add_subplot(gs[r, 4])
        _draw_grid_2d(ax, mid_c, cmap='RdBu_r',
                      vmin=-3, vmax=3, fontsize=20)
        ax.set_title(f'中间结果 mid_{name[0]} (2×2)', color=c, fontsize=11)
        for sp in ax.spines.values(): sp.set_color(c); sp.set_linewidth(1.5)

    # 最后一行：求和 + 输出
    ax = fig.add_subplot(gs[3, :4])
    ax.axis('off')
    ax.text(0.5, 0.5,
            'mid_R   +   mid_G   +   mid_B  =  Y',
            color=TEXT_LT, fontsize=22, ha='center', va='center', fontweight='bold')
    ax.text(0.5, 0.15,
            '通道维被求和"折叠"掉 → 3 通道进, 1 通道出',
            color=ACC_GREEN, fontsize=12, ha='center', va='center', style='italic')

    ax = fig.add_subplot(gs[3, 4])
    _draw_grid_2d(ax, Y, cmap='RdBu_r', vmin=-7, vmax=7, fontsize=22)
    ax.set_title('最终输出 Y (2×2)\n= mid_R + mid_G + mid_B',
                 color=ACC_ORANGE, fontsize=11)
    for sp in ax.spines.values(): sp.set_color(ACC_ORANGE); sp.set_linewidth(2.5)

    save_fig(fig, '04_multi_channel', 'multichannel_conv_numerical.png')


def fig_K_filters_stack():
    """K=4 个 filter → 4 张 feature map → 沿通道堆叠成输出张量。"""
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('一层卷积 K=4 个 filter：每个 filter 各自做"多通道卷积"→ K 张 feature map → 沿通道堆叠',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 0.18, 1, 0.18, 1.4],
                          height_ratios=[1, 1], hspace=0.5, wspace=0.15)

    # 中：单一输入张量 (C_in, H, W)
    ax = fig.add_subplot(gs[:, 0])
    ax.axis('off')
    ax.text(0.5, 0.94, '输入 X', color=TEXT_LT, fontsize=12,
            ha='center', fontweight='bold')
    ax.text(0.5, 0.86, '(C_in=3, H=32, W=32)', color=TEXT_DIM, fontsize=10,
            ha='center')
    # 画堆叠的 3 个平面
    for k_layer in range(3):
        col = ['#ef5350', '#66bb6a', '#4fc3f7'][k_layer]
        x0, y0 = 0.15 + k_layer * 0.06, 0.40 + k_layer * 0.05
        ax.add_patch(plt.Rectangle((x0, y0), 0.55, 0.32,
                                    facecolor=col, alpha=0.55,
                                    edgecolor='white', linewidth=1.2,
                                    transform=ax.transAxes))
    ax.text(0.5, 0.30, '"一张图 = 3 个通道堆叠"',
            color=TEXT_DIM, fontsize=10, ha='center', style='italic')

    # 箭头
    ax = fig.add_subplot(gs[:, 1]); ax.axis('off')
    ax.text(0.5, 0.5, '→\n各自卷', color=ACC_GREEN, fontsize=14,
            ha='center', va='center')

    # 4 个 filter 块（左上 2×2 排列在 col=2）
    filter_colors = ['#ff8a65', '#ba68c8', '#26c6da', '#ffa726']
    filter_ax = fig.add_subplot(gs[:, 2])
    filter_ax.axis('off')
    filter_ax.text(0.5, 0.94, '4 个 filter（每个 (3, 3, 3)）',
                   color=TEXT_LT, fontsize=12, ha='center', fontweight='bold')
    filter_ax.text(0.5, 0.86, 'shape = (C_out=4, C_in=3, k=3, k=3)',
                   color=TEXT_DIM, fontsize=10, ha='center')
    for k_idx in range(4):
        col = filter_colors[k_idx]
        row, col_pos = k_idx // 2, k_idx % 2
        bx, by = 0.10 + col_pos * 0.45, 0.55 - row * 0.32
        # 三层堆叠示意 一个 filter
        for layer in range(3):
            x0 = bx + layer * 0.025
            y0 = by + layer * 0.02
            filter_ax.add_patch(plt.Rectangle((x0, y0), 0.32, 0.18,
                                              facecolor=col, alpha=0.6,
                                              edgecolor='white', linewidth=1,
                                              transform=filter_ax.transAxes))
        filter_ax.text(bx + 0.16, by - 0.06, f'filter {k_idx}',
                       color=col, ha='center', fontsize=10,
                       fontweight='bold', transform=filter_ax.transAxes)

    # 箭头 2
    ax = fig.add_subplot(gs[:, 3]); ax.axis('off')
    ax.text(0.5, 0.5, '→\n沿通道堆叠', color=ACC_GREEN, fontsize=12,
            ha='center', va='center')

    # 输出：4 张 feature map 堆叠
    out_ax = fig.add_subplot(gs[:, 4])
    out_ax.axis('off')
    out_ax.text(0.5, 0.94, '输出 Y',
                color=TEXT_LT, fontsize=12, ha='center', fontweight='bold')
    out_ax.text(0.5, 0.86, '(C_out=4, H\'=32, W\'=32)',
                color=TEXT_DIM, fontsize=10, ha='center')
    for k_idx in range(4):
        col = filter_colors[k_idx]
        x0, y0 = 0.20 + k_idx * 0.04, 0.30 + k_idx * 0.05
        out_ax.add_patch(plt.Rectangle((x0, y0), 0.55, 0.30,
                                        facecolor=col, alpha=0.6,
                                        edgecolor='white', linewidth=1.2,
                                        transform=out_ax.transAxes))
    out_ax.text(0.5, 0.18,
                '通道数 4 = filter 数\n→ 下一层的输入通道',
                color=ACC_GREEN, fontsize=10, ha='center')

    save_fig(fig, '04_multi_channel', 'K_filters_stack.png')


def fig_conv_tensor_shapes():
    """4D 张量形状对照：X (N,C_in,H,W) / W (C_out,C_in,k,k) / Y (N,C_out,H',W')。"""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axis('off')
    fig.suptitle('§5 一层卷积涉及的 4D 张量形状对照',
                 color=TEXT_LT, fontsize=13, y=0.99)

    # 三个张量框
    panels = [
        # (x_center, label, dims, color, sub_text)
        (0.18, 'X (输入 batch)', '(N, C_in, H, W)', ACC_BLUE,
         'N 张图 × 每张 C_in 通道 × H × W'),
        (0.5,  'W (filter 组)',  '(C_out, C_in, k, k)', ACC_ORANGE,
         'C_out 个 filter × 每个 C_in × k × k\n（不带 N: filter 与 batch 无关）'),
        (0.82, 'Y (输出 batch)', '(N, C_out, H\', W\')', ACC_GREEN,
         'N 张图 × 每张 C_out 通道 × H\' × W\''),
    ]
    for cx, name, shape, color, sub in panels:
        ax.add_patch(plt.Rectangle((cx-0.13, 0.35), 0.26, 0.45,
                                    facecolor=color, alpha=0.18,
                                    edgecolor=color, linewidth=2,
                                    transform=ax.transAxes))
        ax.text(cx, 0.74, name, color=color, fontsize=13,
                ha='center', fontweight='bold', transform=ax.transAxes)
        ax.text(cx, 0.65, shape, color=TEXT_LT, fontsize=14,
                ha='center', family='monospace', fontweight='bold',
                transform=ax.transAxes)
        ax.text(cx, 0.46, sub, color=TEXT_MD, fontsize=10,
                ha='center', va='center', transform=ax.transAxes)

    # 中间 + 右边箭头 (X ⊛ W = Y)
    ax.text(0.32, 0.575, '⊛', color=TEXT_LT, fontsize=28,
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.66, 0.575, '=', color=TEXT_LT, fontsize=28,
            ha='center', va='center', transform=ax.transAxes)

    # 底部公式
    ax.text(0.5, 0.16,
            r'$Y[n, k, i, j] \;=\; b[k] \;+\; '
            r'\sum_{c=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n_2=0}^{K-1}\;'
            r'X[n,\, c,\, i\!+\!m,\, j\!+\!n_2] \cdot W[k,\, c,\, m,\, n_2]$',
            color=TEXT_LT, fontsize=14, ha='center', va='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.04,
            'C_in 求和折叠掉; C_out 由 K 个不同 filter 各自产生一张 feature map',
            color=ACC_GREEN, fontsize=11, ha='center', va='center',
            style='italic', transform=ax.transAxes)

    save_fig(fig, '04_multi_channel', 'conv_tensor_shapes.png')


def fig_padding_visualization():
    """5×5 输入 → padding=1 后变 7×7 的概念图。"""
    H, W = 5, 5
    # 原图：用 1..25 让格子颜色有差异（教学性）
    X = np.arange(1, H*W+1).reshape(H, W) % 5 + 1
    Xp = np.pad(X, 1, mode='constant', constant_values=0)

    fig = plt.figure(figsize=(11, 4.6))
    fig.suptitle('Padding：在四周补 1 圈 0，让 5×5 变成 7×7',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.35, 1.4], wspace=0.2)

    ax_l = fig.add_subplot(gs[0, 0])
    _draw_grid_2d(ax_l, X, cmap='Blues', fontsize=14)
    ax_l.set_title('原始输入 (5×5)\n所有像素都"是数据"',
                   color=TEXT_LT, fontsize=11)

    ax_m = fig.add_subplot(gs[0, 1])
    ax_m.axis('off')
    ax_m.text(0.5, 0.6, 'padding = 1', color=ACC_GREEN,
              fontsize=14, ha='center', va='center', fontweight='bold')
    ax_m.text(0.5, 0.4, '→', color=TEXT_LT,
              fontsize=28, ha='center', va='center')

    ax_r = fig.add_subplot(gs[0, 2])
    # 用 mask 让 padding 区域显示 0 但视觉上区分
    display = Xp.astype(float)
    ax_r.imshow(display, cmap='Blues', vmin=0, vmax=display.max()*1.1)
    # 标注 padding 区域为绿色边框
    for i in range(Xp.shape[0]):
        for j in range(Xp.shape[1]):
            is_pad = (i == 0 or i == 6 or j == 0 or j == 6)
            v = Xp[i, j]
            ax_r.text(j, i, str(int(v)),
                      ha='center', va='center',
                      color=ACC_GREEN if is_pad else 'black',
                      fontsize=12, fontweight='bold')
            if is_pad:
                ax_r.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              facecolor='none',
                                              edgecolor=ACC_GREEN,
                                              linewidth=1.5, alpha=0.7))
    # 大框圈出原图区域
    ax_r.add_patch(plt.Rectangle((0.5, 0.5), 5, 5,
                                  facecolor='none',
                                  edgecolor=ACC_ORANGE,
                                  linewidth=2.5))
    ax_r.set_xticks([]); ax_r.set_yticks([])
    for sp in ax_r.spines.values(): sp.set_color('#444')
    ax_r.set_title('补 padding 后 (7×7)\n绿框=新增的 0 / 橙框=原数据',
                   color=TEXT_LT, fontsize=11)

    save_fig(fig, '03_padding_stride', 'padding_visualization.png')


def fig_stride_1d():
    """1D 上 stride=1 vs stride=2 的 filter 位置 + 输出长度对比。"""
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    fig = plt.figure(figsize=(13, 6.5))
    fig.suptitle('Stride：filter 不再每步移 1，而是每步移 s',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(2, 1, hspace=0.4)

    for row, (s, title) in enumerate([
        (1, 'stride = 1：每步右移 1 → 5 个位置 → 输出长度 5'),
        (2, 'stride = 2：每步右移 2 → 3 个位置 → 输出长度 3'),
    ]):
        ax = fig.add_subplot(gs[row, 0])
        ax.set_xlim(-1.0, 12)
        n_pos = (7 - 3) // s + 1
        ax.set_ylim(-1.5 - n_pos * 0.7, 1.4)
        ax.axis('off')
        ax.set_title(title, color=TEXT_LT, fontsize=12, loc='left')

        # 顶部 input
        _draw_cells_1d(ax, letters, 0.4, label='x:')
        # 每个 filter 起手位置
        palette = plt.cm.viridis(np.linspace(0.15, 0.85, n_pos))
        for k_idx, p in enumerate([i*s for i in range(n_pos)]):
            row_y = -0.7 - k_idx * 0.7
            for k in range(3):
                ax.add_patch(plt.Rectangle((p+k, row_y), 0.9, 0.55,
                                           facecolor=palette[k_idx], alpha=0.6,
                                           edgecolor=palette[k_idx], linewidth=1.5))
                ax.text(p+k+0.45, row_y+0.275, letters[p+k],
                        color='white', ha='center', va='center',
                        fontsize=10, fontweight='bold')
            ax.text(7.6, row_y+0.275,
                    f'pos {k_idx}:  y[{k_idx}] = '
                    f'{letters[p]}·w0 + {letters[p+1]}·w1 + {letters[p+2]}·w2',
                    color=palette[k_idx], fontsize=10, va='center')

    save_fig(fig, '03_padding_stride', 'stride_1d.png')


def fig_stride_starting_positions():
    """1D filter 起手位置：stride=1/2/3 + 输出尺寸公式验证。"""
    H, k = 7, 3
    cases = [
        (1, list(range(0, H - k + 1, 1))),
        (2, list(range(0, H - k + 1, 2))),
        (3, list(range(0, H - k + 1, 3))),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 6))
    fig.suptitle('§3.1.1 公式的几何含义：filter 起手位置数 = 输出尺寸',
                 color=TEXT_LT, fontsize=13, y=0.99)

    for ax, (s, starts) in zip(axes, cases):
        ax.set_xlim(-0.8, 11)
        ax.set_ylim(-0.7, 1.0)
        ax.axis('off')

        # 7 个输入位置 (用圆点)
        for i in range(H):
            color = ACC_GREEN if i in starts else '#555'
            ax.add_patch(plt.Circle((i, 0.3), 0.18,
                                    facecolor=color,
                                    edgecolor='white', linewidth=1.2))
            ax.text(i, -0.1, str(i),
                    color=TEXT_DIM if i not in starts else TEXT_LT,
                    ha='center', fontsize=10)

        # 起手位置数标注
        n = len(starts)
        ax.text(7.7, 0.3,
                f'stride = {s}：起点共 {n} 个 → 输出长度 = {n}',
                color=ACC_GREEN, fontsize=11, va='center', fontweight='bold')
        ax.text(7.7, -0.1,
                f'公式: floor((7 + 0 − 3) / {s}) + 1 = floor({4/s:.2f}) + 1 = {4//s + 1}',
                color=TEXT_DIM, fontsize=10, va='center')

    save_fig(fig, '03_padding_stride', 'stride_starting_positions.png')


def fig_dropped_pixels():
    """H=8, k=3, s=3, p=0：filter 站不到 row 6/7，被丢。"""
    H = 8
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.suptitle('§3.3 不能整除时的边界陷阱：H=8, k=3, p=0, s=3 → 输出 2 行，row 6/7 被丢',
                 color=TEXT_LT, fontsize=13, y=0.99)

    ax.set_xlim(-1.5, 12.5)
    ax.set_ylim(-0.6, H + 1.5)
    ax.invert_yaxis()
    ax.axis('off')

    # 8 行输入位置（用编号矩形）
    for i in range(H):
        is_dropped = i >= 6
        col = ACC_RED if is_dropped else PANEL_BG
        rect = plt.Rectangle((0.5, i-0.4), 0.8, 0.8,
                             facecolor=col, alpha=0.4 if is_dropped else 1,
                             edgecolor=ACC_RED if is_dropped else '#555',
                             linewidth=1.5)
        ax.add_patch(rect)
        ax.text(0.9, i, f'row {i}',
                color=ACC_RED if is_dropped else TEXT_LT,
                ha='center', va='center', fontsize=11,
                fontweight='bold' if is_dropped else 'normal')

    # 2 个 filter 站位
    cols = [ACC_BLUE, ACC_GREEN]
    for k_idx, start in enumerate([0, 3]):
        c = cols[k_idx]
        rect = plt.Rectangle((1.5, start-0.5), 1.0, 3,
                             facecolor=c, alpha=0.25,
                             edgecolor=c, linewidth=2)
        ax.add_patch(rect)
        ax.text(2.0, start+1, f'filter\nstart\n= {start}',
                color=c, ha='center', va='center',
                fontsize=10, fontweight='bold')
        ax.annotate('', xy=(3.5, start+1), xytext=(2.7, start+1),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        ax.text(4.5, start+1,
                f'覆盖 row {start}, {start+1}, {start+2}',
                color=c, ha='left', va='center', fontsize=10)

    # 解释下一步起手会出界
    ax.text(7.5, 6.5,
            '↓ 想再放下一个 filter，起点必须是 row 6\n'
            '   但 6 + 3 = 9 > 8，filter 伸出去 → 不能站\n'
            '   ⇒ row 6, 7 永远不会被任何 filter 覆盖',
            color=ACC_RED, fontsize=11, ha='left', va='center')

    # 公式验证（放在底部）
    ax.text(0.5, H + 0.9,
            '公式: floor((8 + 0 − 3) / 3) + 1 = floor(5/3) + 1 = 1 + 1 = 2 行输出',
            color=TEXT_LT, fontsize=11, fontweight='bold', va='center')

    save_fig(fig, '03_padding_stride', 'dropped_pixels.png')


def fig_padding_coverage():
    H, W, k = 5, 5, 3
    no_pad   = touch_count_map(H, W, k, padding=0)
    with_pad = touch_count_map(H, W, k, padding=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f'5×5 输入用 3×3 filter 卷积时, 每个像素被覆盖的次数\n'
        f'padding 把"角落 vs 中心"的差距从 9× 缩到 ~2.3×',
        color=TEXT_LT, fontsize=12, y=1.04,
    )

    for ax, mat, title in zip(
        axes,
        (no_pad, with_pad),
        ('padding = 0\n(角落只用 1 次, 中心用 9 次)',
         'padding = 1\n(角落用 4 次, 中心仍用 9 次)'),
    ):
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, str(int(v)),
                        ha='center', va='center',
                        color='white' if v < 6 else 'black',
                        fontsize=14, fontweight='bold')
        style_imshow_axes(ax, title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=TEXT_DIM)
        cbar.set_label('被覆盖次数', color=TEXT_MD)

    plt.tight_layout()
    save_fig(fig, '03_padding_stride', 'padding_coverage_heatmap.png')


# ─────────────────────────────────────────────
# 图 5：T5 §5.2 感受野逐层扩张
# ─────────────────────────────────────────────
def fig_receptive_field():
    """画 1/2/3 层 3×3 conv (s=1) 后, 中心输出像素能"看到"原图多大区域。"""
    size = 11
    center = size // 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        '感受野（receptive field）随层数扩张\n'
        '同一张 11×11 输入, 看深层一个像素究竟能看到多少',
        color=TEXT_LT, fontsize=12, y=1.05,
    )

    for layer, ax in zip([1, 2, 3], axes):
        rf = 1 + 2 * layer  # 3, 5, 7
        half = rf // 2
        mask = np.zeros((size, size), dtype=np.float32)
        mask[center-half:center+half+1, center-half:center+half+1] = 1.0

        ax.imshow(mask, cmap='Blues', vmin=-0.3, vmax=1.3,
                  interpolation='nearest')
        # 画网格 + 标注每个像素
        for i in range(size):
            for j in range(size):
                if mask[i, j] > 0:
                    if (i, j) == (center, center):
                        ax.text(j, i, '★', ha='center', va='center',
                                color=ACC_ORANGE, fontsize=16, fontweight='bold')
                    else:
                        ax.text(j, i, '●', ha='center', va='center',
                                color=ACC_BLUE, fontsize=10)
                else:
                    ax.text(j, i, '·', ha='center', va='center',
                            color='#555555', fontsize=8)

        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which='minor', color='#333', linewidth=0.5)
        ax.tick_params(which='minor', length=0)
        ax.set_xticks([]); ax.set_yticks([])

        ax.set_title(
            f'{layer} 层 3×3 conv (stride=1)\n'
            f'感受野 = {rf}×{rf}\n'
            f'★ 输出像素位置, ● 它能"看到"的输入',
            color=TEXT_LT, fontsize=10,
        )

    plt.tight_layout()
    save_fig(fig, '05_pooling', 'receptive_field_growth.png')


# ─────────────────────────────────────────────
# 图 6：T5 §3 MaxPool vs AvgPool 视觉对比
# ─────────────────────────────────────────────
def maxpool2d(X, k=2, s=2):
    H, W = X.shape
    Hout, Wout = H // s, W // s
    Y = np.zeros((Hout, Wout), dtype=X.dtype)
    for i in range(Hout):
        for j in range(Wout):
            Y[i, j] = X[i*s:i*s+k, j*s:j*s+k].max()
    return Y


def avgpool2d(X, k=2, s=2):
    H, W = X.shape
    Hout, Wout = H // s, W // s
    Y = np.zeros((Hout, Wout), dtype=X.dtype)
    for i in range(Hout):
        for j in range(Wout):
            Y[i, j] = X[i*s:i*s+k, j*s:j*s+k].mean()
    return Y


def fig_pool_compare(img_chw, label):
    gray = to_grayscale(img_chw)
    # 用 4×4 池化（缩 4 倍）效果对比更明显
    mp = maxpool2d(gray, k=4, s=4)
    ap = avgpool2d(gray, k=4, s=4)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    fig.suptitle(
        f'MaxPool vs AvgPool 视觉对比  ({label}, 32×32 → 8×8)\n'
        f'MaxPool 保留最强响应, AvgPool 平均后变模糊',
        color=TEXT_LT, fontsize=12, y=1.05,
    )

    axes[0].imshow(gray, cmap='gray', interpolation='nearest')
    style_imshow_axes(axes[0], '原图灰度 (32×32)')

    axes[1].imshow(mp, cmap='gray', interpolation='nearest')
    style_imshow_axes(axes[1], 'MaxPool 4×4 (8×8)\n锐利, 边缘强')

    axes[2].imshow(ap, cmap='gray', interpolation='nearest')
    style_imshow_axes(axes[2], 'AvgPool 4×4 (8×8)\n平滑, 像降分辨率')

    plt.tight_layout()
    save_fig(fig, '05_pooling', 'maxpool_vs_avgpool.png')


# ─────────────────────────────────────────────
# 图 7：T6 §3 ∇W = correlate(X, δ) — δ 在 X 上"滑动"
# ─────────────────────────────────────────────
def fig_grad_W_slide():
    """把 ∇W 的计算可视化为'把 δ 当 filter 在 X 上滑动'。"""
    # 4×4 X, 2×2 W → 3×3 Y → 3×3 δ → 2×2 ∇W
    X = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 0, 1, 2],
                  [3, 4, 5, 6]], dtype=np.float32)
    delta = np.array([[0.1, 0.2, 0.1],
                      [0.0, 0.3, 0.2],
                      [0.1, 0.0, 0.4]], dtype=np.float32)

    # 计算 ∇W = correlate(X, δ), shape (2, 2)
    grad_W = np.zeros((2, 2), dtype=np.float32)
    for m in range(2):
        for n in range(2):
            grad_W[m, n] = (X[m:m+3, n:n+3] * delta).sum()

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(
        '∇W = correlate(X, δ): 把"输出梯度 δ"当作 filter, 在输入 X 上滑动\n'
        '(X 是 4×4, W 是 2×2, 所以 δ 是 3×3, ∇W 是 2×2)',
        color=TEXT_LT, fontsize=13, y=0.99,
    )

    # 上半: 输入 X 与 δ
    ax_X = plt.subplot2grid((2, 5), (0, 0), colspan=2)
    ax_d = plt.subplot2grid((2, 5), (0, 2), colspan=1)
    # 下半: 4 个滑动位置 + ∇W 结果
    pos_axes = [plt.subplot2grid((2, 5), (1, i)) for i in range(4)]
    ax_gW = plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=1)

    # 画 X
    ax_X.imshow(X, cmap='Blues', vmin=-2, vmax=10)
    for i in range(4):
        for j in range(4):
            ax_X.text(j, i, f'{int(X[i,j])}', ha='center', va='center',
                      color='black', fontsize=14, fontweight='bold')
    style_imshow_axes(ax_X, '输入 X (4×4)')

    # 画 δ
    ax_d.imshow(delta, cmap='Oranges', vmin=0, vmax=0.5)
    for i in range(3):
        for j in range(3):
            ax_d.text(j, i, f'{delta[i,j]:.1f}', ha='center', va='center',
                      color='black', fontsize=12, fontweight='bold')
    style_imshow_axes(ax_d, '输出梯度 δ (3×3)\n— 当作 filter 用')

    # 画 ∇W
    ax_gW.imshow(grad_W, cmap='Greens', vmin=0, vmax=grad_W.max()*1.2)
    for i in range(2):
        for j in range(2):
            ax_gW.text(j, i, f'{grad_W[i,j]:.2f}', ha='center', va='center',
                       color='black', fontsize=18, fontweight='bold')
    style_imshow_axes(ax_gW, '最终 ∇W (2×2)')

    # 画 4 个滑动位置 (在 X 上覆盖 3×3 高亮框)
    for idx, (m, n) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax = pos_axes[idx]
        ax.imshow(X, cmap='Blues', vmin=-2, vmax=10, alpha=0.4)
        # 高亮被覆盖的 3×3 子块
        for i in range(4):
            for j in range(4):
                in_box = (m <= i < m+3) and (n <= j < n+3)
                if in_box:
                    val_x = X[i, j]
                    val_d = delta[i-m, j-n]
                    ax.text(j, i, f'{int(val_x)}\n×{val_d:.1f}',
                            ha='center', va='center',
                            color='#ff8a65', fontsize=8, fontweight='bold')
                else:
                    ax.text(j, i, f'{int(X[i,j])}', ha='center', va='center',
                            color='#666666', fontsize=10)
        rect = plt.Rectangle((n-0.5, m-0.5), 3, 3, linewidth=2.5,
                             edgecolor=ACC_ORANGE, facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f'(m,n)=({m},{n})\n∇W[{m},{n}] = sum = {grad_W[m,n]:.2f}',
                     color=TEXT_LT, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color('#444')

    plt.tight_layout()
    save_fig(fig, '06_conv_backprop', 'grad_W_slide.png')


# ─────────────────────────────────────────────
# 图 8：T6 §4 ∇X 的 W "翻转"现象
# ─────────────────────────────────────────────
def fig_grad_X_flip():
    """选 X[1,1] 这一个像素, 展示前向时 W 的哪些元素和它相乘,
       反向时这些元素以"翻转"顺序回乘 δ — 翻转的根因。"""
    # 用字母标记 W 的元素方便看
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        '∇X[a,b] 计算中 "W 翻转" 是怎么自然冒出来的\n'
        '以 X[1,1] (内部一点) 为例, X 4×4, W 2×2 → Y 3×3',
        color=TEXT_LT, fontsize=12, y=1.02,
    )

    # ── Panel 1: 前向 — X[1,1] 出现在 4 个 Y 位置, 各自和不同 W 元素相乘 ──
    ax = axes[0]
    ax.imshow(np.zeros((4, 4)), cmap='gray', alpha=0)
    for i in range(4):
        for j in range(4):
            if (i, j) == (1, 1):
                ax.text(j, i, '★', ha='center', va='center',
                        color=ACC_ORANGE, fontsize=24, fontweight='bold')
            else:
                ax.text(j, i, '·', ha='center', va='center',
                        color='#555', fontsize=14)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which='minor', color='#444', linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('① 输入 X (4×4)\n★ = X[1,1] (我们关心的像素)',
                 color=TEXT_LT, fontsize=11)
    for sp in ax.spines.values(): sp.set_color('#444')

    # ── Panel 2: 列出 X[1,1] 出现在哪些 Y 位置, 配什么 W 元素 ──
    ax = axes[1]
    ax.axis('off')
    text = (
        '② X[1,1] 在前向出现在 4 个 Y 位置\n'
        '   (因为 W 是 2×2, 任何 X 位置都被周围 4 个 Y 用到)\n\n'
        '   Y[0,0]:  ★ × W[1,1]   ←─ 注意 W 索引\n'
        '   Y[0,1]:  ★ × W[1,0]\n'
        '   Y[1,0]:  ★ × W[0,1]\n'
        '   Y[1,1]:  ★ × W[0,0]\n\n'
        '   ── W 的索引顺序: (1,1) → (1,0) → (0,1) → (0,0)\n'
        '   ── 这是 W "反着走" 一遍! ──\n\n'
        '   反向时 ∇X[1,1] 把这 4 个梯度按同样的反序累加:\n\n'
        '   ∇X[1,1] = δ[0,0]·W[1,1]\n'
        '           + δ[0,1]·W[1,0]\n'
        '           + δ[1,0]·W[0,1]\n'
        '           + δ[1,1]·W[0,0]'
    )
    ax.text(0.0, 0.95, text, transform=ax.transAxes,
            color=TEXT_MD, fontsize=10, verticalalignment='top',
            family=['Menlo', 'Monaco', 'Arial Unicode MS', 'monospace'])
    ax.set_title('③ 前向时 ★ 配的 W 索引顺序 = W 翻转',
                 color=TEXT_LT, fontsize=11)

    # ── Panel 3: flip(W) 让上面的运算变成一次干净的互相关 ──
    ax = axes[2]
    ax.axis('off')
    text2 = (
        '④ 用 flip(W) (180° 翻转) 重写, 上面的求和\n'
        '   就变成 δ 局部窗口和 flip(W) 的对位乘求和:\n\n'
        '   原 W:              flip(W):\n'
        '   ┌─────┬─────┐      ┌─────┬─────┐\n'
        '   │ a   │ b   │      │ d   │ c   │\n'
        '   ├─────┼─────┤  →   ├─────┼─────┤\n'
        '   │ c   │ d   │      │ b   │ a   │\n'
        '   └─────┴─────┘      └─────┴─────┘\n\n'
        '   ∇X[1,1] = δ[0:2,0:2] ⊙ flip(W)\n'
        '          = correlate(δ, flip(W)) [的对应位置]\n\n'
        '   ── 数学结论 ──\n'
        '   ∇X = correlate(  pad(δ, k-1) ,  flip(W)  )\n'
        '\n'
        '   "前向用 互相关、反向需要 翻转 filter"\n'
        '   就是从这个索引反序自然冒出来的'
    )
    ax.text(0.0, 0.95, text2, transform=ax.transAxes,
            color=TEXT_MD, fontsize=9.5, verticalalignment='top',
            family=['Menlo', 'Monaco', 'Arial Unicode MS', 'monospace'])
    ax.set_title('⑤ 翻转 W 把求和变回干净的互相关',
                 color=TEXT_LT, fontsize=11)

    plt.tight_layout()
    save_fig(fig, '06_conv_backprop', 'grad_X_flip.png')


# ─────────────────────────────────────────────
# 图 9：T6 §7 MaxPool 反向: 稀疏路由
# ─────────────────────────────────────────────
def fig_maxpool_backward():
    """前向: 4×4 → 2×2 取 max, 记录位置;
       反向: 2×2 δ → 4×4 ∇X, 只往 max 位置塞梯度。"""
    X = np.array([[1, 3, 2, 4],
                  [5, 6, 7, 8],
                  [9, 2, 1, 5],
                  [3, 4, 8, 9]], dtype=np.float32)
    # 2×2 池化, stride=2 → 输出 2×2
    Y = np.zeros((2, 2), dtype=np.float32)
    argmax_pos = np.zeros((2, 2, 2), dtype=np.int32)  # 存每个块 argmax 位置(i,j)
    for i in range(2):
        for j in range(2):
            block = X[i*2:i*2+2, j*2:j*2+2]
            Y[i, j] = block.max()
            local = np.unravel_index(block.argmax(), block.shape)
            argmax_pos[i, j] = (i*2 + local[0], j*2 + local[1])

    # 反向: 给一个 δ
    delta = np.array([[0.1, 0.2],
                      [0.3, 0.4]], dtype=np.float32)
    grad_X = np.zeros_like(X)
    for i in range(2):
        for j in range(2):
            ai, aj = argmax_pos[i, j]
            grad_X[ai, aj] = delta[i, j]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    fig.suptitle(
        'MaxPool 2×2 反向传播: 梯度只往"前向 argmax 位置"传, 其它 0\n'
        '(对比 AvgPool 的均匀分配)',
        color=TEXT_LT, fontsize=12, y=1.04,
    )

    # ── Panel 1: 前向 X + argmax 标记 ──
    ax = axes[0]
    ax.imshow(X, cmap='Blues', vmin=0, vmax=10)
    for i in range(4):
        for j in range(4):
            is_max = any((argmax_pos[bi, bj][0] == i and argmax_pos[bi, bj][1] == j)
                         for bi in range(2) for bj in range(2))
            color = ACC_ORANGE if is_max else 'black'
            weight = 'bold'
            text = f'{int(X[i,j])}'
            if is_max: text += '★'
            ax.text(j, i, text, ha='center', va='center',
                    color=color, fontsize=12, fontweight=weight)
    # 画 2×2 块的边界
    for k in [2]:
        ax.axhline(k - 0.5, color='#888', linewidth=2)
        ax.axvline(k - 0.5, color='#888', linewidth=2)
    style_imshow_axes(ax, '① 前向输入 X (4×4)\n★ = 各 2×2 块内的 argmax')

    # ── Panel 2: 前向输出 Y ──
    ax = axes[1]
    ax.imshow(Y, cmap='Blues', vmin=0, vmax=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{int(Y[i,j])}', ha='center', va='center',
                    color='black', fontsize=18, fontweight='bold')
    style_imshow_axes(ax, '② 前向输出 Y (2×2)\nY[i,j] = max(对应 2×2 块)')

    # ── Panel 3: 反向 δ ──
    ax = axes[2]
    ax.imshow(delta, cmap='Oranges', vmin=0, vmax=0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{delta[i,j]:.1f}', ha='center', va='center',
                    color='black', fontsize=18, fontweight='bold')
    style_imshow_axes(ax, '③ 收到的输出梯度 δ (2×2)\n(从下游传回来)')

    # ── Panel 4: 反向 ∇X ──
    ax = axes[3]
    ax.imshow(grad_X, cmap='Greens', vmin=0, vmax=0.5)
    for i in range(4):
        for j in range(4):
            v = grad_X[i, j]
            if v > 0:
                ax.text(j, i, f'{v:.1f}★', ha='center', va='center',
                        color=ACC_ORANGE, fontsize=12, fontweight='bold')
            else:
                ax.text(j, i, '0', ha='center', va='center',
                        color='#666', fontsize=11)
    for k in [2]:
        ax.axhline(k - 0.5, color='#888', linewidth=2)
        ax.axvline(k - 0.5, color='#888', linewidth=2)
    style_imshow_axes(ax, '④ ∇X (4×4)\nδ 只送到 ★ 位置, 其它全 0\n(稀疏路由)')

    plt.tight_layout()
    save_fig(fig, '06_conv_backprop', 'maxpool_backward.png')


# ─────────────────────────────────────────────
# 图 10：T6 §8 AvgPool 反向: 均匀分配
# ─────────────────────────────────────────────
def fig_avgpool_backward():
    """前向: 4×4 → 2×2 取平均;
       反向: 2×2 δ → 4×4 ∇X, 每个 δ 平均分到块内 4 个位置。"""
    X = np.array([[1, 3, 2, 4],
                  [5, 6, 7, 8],
                  [9, 2, 1, 5],
                  [3, 4, 8, 9]], dtype=np.float32)
    Y = np.zeros((2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            Y[i, j] = X[i*2:i*2+2, j*2:j*2+2].mean()

    delta = np.array([[0.4, 0.8],
                      [1.2, 1.6]], dtype=np.float32)  # 选大点的数好看
    grad_X = np.zeros_like(X)
    for i in range(2):
        for j in range(2):
            grad_X[i*2:i*2+2, j*2:j*2+2] = delta[i, j] / 4.0

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    fig.suptitle(
        'AvgPool 2×2 反向传播: 梯度均匀分到块内所有 4 个位置 (每个 = δ/k²)',
        color=TEXT_LT, fontsize=12, y=1.03,
    )

    # ── Panel 1: 前向 X ──
    ax = axes[0]
    ax.imshow(X, cmap='Blues', vmin=0, vmax=10)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{int(X[i,j])}', ha='center', va='center',
                    color='black', fontsize=12)
    for k in [2]:
        ax.axhline(k - 0.5, color='#888', linewidth=2)
        ax.axvline(k - 0.5, color='#888', linewidth=2)
    style_imshow_axes(ax, '① 前向输入 X (4×4)')

    # ── Panel 2: 前向输出 Y ──
    ax = axes[1]
    ax.imshow(Y, cmap='Blues', vmin=0, vmax=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{Y[i,j]:.1f}', ha='center', va='center',
                    color='black', fontsize=16, fontweight='bold')
    style_imshow_axes(ax, '② 前向输出 Y (2×2)\nY[i,j] = mean(对应 2×2 块)')

    # ── Panel 3: 反向 δ ──
    ax = axes[2]
    ax.imshow(delta, cmap='Oranges', vmin=0, vmax=2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{delta[i,j]:.1f}', ha='center', va='center',
                    color='black', fontsize=16, fontweight='bold')
    style_imshow_axes(ax, '③ 收到的输出梯度 δ (2×2)')

    # ── Panel 4: 反向 ∇X ──
    ax = axes[3]
    ax.imshow(grad_X, cmap='Greens', vmin=0, vmax=0.5)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{grad_X[i,j]:.2f}', ha='center', va='center',
                    color='black', fontsize=11, fontweight='bold')
    for k in [2]:
        ax.axhline(k - 0.5, color='#888', linewidth=2)
        ax.axvline(k - 0.5, color='#888', linewidth=2)
    style_imshow_axes(ax, '④ ∇X (4×4)\n块内 4 格都 = δ/4\n(密集均匀分配)')

    plt.tight_layout()
    save_fig(fig, '06_conv_backprop', 'avgpool_backward.png')


# ─────────────────────────────────────────────
# 图 11：T1 §1 MLP 第一层参数量爆炸（log 轴柱状图）
# ─────────────────────────────────────────────
def fig_param_explosion():
    cases = [
        ('MLP-128\nMNIST\n28×28×1',       784      * 128, ACC_GREEN),
        ('MLP-128\nCIFAR-10\n32×32×3',    3072     * 128, ACC_BLUE),
        ('MLP-128\nImageNet\n224×224×3',  150528   * 128, ACC_ORANGE),
        ('MLP-512\nImageNet',             150528   * 512, ACC_RED),
        ('CNN 一个\n3×3×3 filter',        3 * 3 * 3,      '#9c27b0'),
        ('CNN 32 个\n3×3×3 filter',       32 * 3 * 3 * 3, '#26c6da'),
    ]
    labels = [c[0] for c in cases]
    values = [c[1] for c in cases]
    colors = [c[2] for c in cases]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.suptitle('第一层参数量对比（log 轴）：MLP 随输入维度爆炸，CNN 不依赖输入大小',
                 color=TEXT_LT, fontsize=13, y=0.995)

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor='#222', linewidth=1.2)
    ax.set_yscale('log')
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('第一层权重个数（log）', color=TEXT_MD, fontsize=11)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    def fmt(v):
        if v >= 1e6: return f'{v/1e6:.1f}M'
        if v >= 1e3: return f'{v/1e3:.1f}K'
        return str(v)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v * 1.25, fmt(v),
                ha='center', va='bottom', color=TEXT_LT, fontsize=11, fontweight='bold')

    # 注释 ImageNet-512 vs ResNet-50 对比
    ax.axhline(2.5e7, color='#888', linestyle=':', linewidth=1)
    ax.text(len(values) - 0.5, 2.7e7, 'ResNet-50 总参数 ≈ 25M',
            ha='right', color=TEXT_DIM, fontsize=9, style='italic')

    ax.set_ylim(1, 2e8)
    plt.tight_layout()
    save_fig(fig, '01_why_conv', 'param_explosion.png')


# ─────────────────────────────────────────────
# 图 12：T1 §2 像素打乱实验（MLP 看不到空间结构的实证）
# ─────────────────────────────────────────────
def _train_tiny_mlp(X_train, y_train, X_test, y_test, epochs=5, batch=64, lr=0.1, seed=0):
    """784→64→10 ReLU MLP，纯 numpy。返回每个 epoch 的测试准确率。"""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((784, 64)).astype(np.float32) * np.sqrt(2.0 / 784)
    b1 = np.zeros(64,  dtype=np.float32)
    W2 = rng.standard_normal((64,  10)).astype(np.float32) * np.sqrt(2.0 / 64)
    b2 = np.zeros(10,  dtype=np.float32)

    def forward(X):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        z2 -= z2.max(axis=1, keepdims=True)
        p  = np.exp(z2); p /= p.sum(axis=1, keepdims=True)
        return a1, z1, p

    accs = []
    n = len(X_train)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            b_idx = idx[i:i+batch]
            X = X_train[b_idx]; y = y_train[b_idx]
            a1, z1, p = forward(X)
            # cross-entropy gradient
            dz2 = p.copy(); dz2[np.arange(len(y)), y] -= 1; dz2 /= len(y)
            dW2 = a1.T @ dz2; db2 = dz2.sum(0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = X.T @ dz1; db1 = dz1.sum(0)
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
        _, _, p_test = forward(X_test)
        acc = float((p_test.argmax(1) == y_test).mean())
        accs.append(acc)
    return accs


def fig_pixel_shuffle_invariance():
    print('  加载 MNIST 并训练两个 MLP（normal vs shuffled）...')
    X_tr, y_tr, X_te, y_te = load_mnist()
    # 取子集加快速度（足以让 MLP 收敛到 ~95%）
    rng = np.random.default_rng(0)
    sub_tr = rng.choice(len(X_tr), 10000, replace=False)
    sub_te = rng.choice(len(X_te), 2000,  replace=False)
    X_tr, y_tr = X_tr[sub_tr], y_tr[sub_tr]
    X_te, y_te = X_te[sub_te], y_te[sub_te]

    # 固定 permutation：训练集和测试集用同一个打乱顺序
    perm = rng.permutation(784)
    X_tr_s = X_tr[:, perm]
    X_te_s = X_te[:, perm]

    acc_normal   = _train_tiny_mlp(X_tr,   y_tr, X_te,   y_te, epochs=5, seed=1)
    acc_shuffled = _train_tiny_mlp(X_tr_s, y_tr, X_te_s, y_te, epochs=5, seed=1)
    print(f'  normal   acc per epoch: {[f"{a:.3f}" for a in acc_normal]}')
    print(f'  shuffled acc per epoch: {[f"{a:.3f}" for a in acc_shuffled]}')

    # 构图：上排展示 2 个数字的"原图 vs 打乱图"，下排画准确率曲线
    fig = plt.figure(figsize=(13, 6.5))
    fig.suptitle('像素打乱实验：MLP 完全感知不到打乱（→ 它本来就没用空间结构）',
                 color=TEXT_LT, fontsize=13, y=0.995)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.4], hspace=0.35, wspace=0.25)

    sample_ids = [0, 7]
    for col, sid in enumerate(sample_ids):
        digit = X_tr[sid].reshape(28, 28)
        digit_s = X_tr_s[sid].reshape(28, 28)
        ax_o = fig.add_subplot(gs[0, col*2])
        ax_o.imshow(digit, cmap='gray', interpolation='nearest')
        style_imshow_axes(ax_o, f'原图  label={int(y_tr[sid])}')
        ax_s = fig.add_subplot(gs[0, col*2 + 1])
        ax_s.imshow(digit_s, cmap='gray', interpolation='nearest')
        style_imshow_axes(ax_s, '同一张图按固定 permutation 打乱')

    ax = fig.add_subplot(gs[1, :])
    epochs = list(range(1, len(acc_normal) + 1))
    ax.plot(epochs, acc_normal,   'o-', color=ACC_GREEN, linewidth=2, markersize=8,
            label=f'正常 MNIST  (final acc = {acc_normal[-1]*100:.2f}%)')
    ax.plot(epochs, acc_shuffled, 's--', color=ACC_ORANGE, linewidth=2, markersize=8,
            label=f'打乱 MNIST  (final acc = {acc_shuffled[-1]*100:.2f}%)')
    ax.set_xlabel('epoch', color=TEXT_MD, fontsize=11)
    ax.set_ylabel('test accuracy', color=TEXT_MD, fontsize=11)
    ax.set_title('两条曲线几乎重合 → MLP 学的是"哪个像素亮"，不是"邻居关系"',
                 color=TEXT_LT, fontsize=11)
    ax.legend(facecolor=PANEL_BG, edgecolor='#444', labelcolor=TEXT_LT, fontsize=10)
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_ylim(0, 1)

    save_fig(fig, '01_why_conv', 'pixel_shuffle_invariance.png')


# ─────────────────────────────────────────────
# 图 13：T1 §3 平移等变性（MLP 特征剧烈变化 vs CNN 跟着平移）
# ─────────────────────────────────────────────
def fig_translation_equivariance(img_chw, label):
    """
    用一张 CIFAR 图，对它做循环平移 (np.roll)，分别比较：
      - MLP: 第一层激活向量与基准的相对 L2 距离（应迅速增大）
      - CNN: 把"先平移再卷积"的 feature map 反向平移回去后 与"直接卷积" 的差（应保持极小）
    """
    img_rgb = img_chw.astype(np.float32) / 255.0  # (3, 32, 32)
    rng = np.random.default_rng(0)

    # 随机 MLP 第一层 W ∈ (3072, 128)
    W_mlp = rng.standard_normal((3072, 128)).astype(np.float32) * np.sqrt(2.0 / 3072)
    # 随机 CNN：8 个 3×3×3 filter
    W_cnn = rng.standard_normal((8, 3, 3, 3)).astype(np.float32) * np.sqrt(2.0 / 27)

    def mlp_feat(img):
        return np.maximum(0, img.flatten() @ W_mlp)

    def correlate_same_circular(X, W):
        # 同尺寸输出 + 循环 padding，使 g(roll_s(x)) == roll_s(g(x))
        H, Wd = X.shape
        kh, kw = W.shape
        ph, pw = kh // 2, kw // 2
        Xp = np.pad(X, ((ph, ph), (pw, pw)), mode='wrap')
        out = np.zeros_like(X)
        for i in range(kh):
            for j in range(kw):
                out += Xp[i:i+H, j:j+Wd] * W[i, j]
        return out

    def cnn_feat(img):
        # img (3, 32, 32) → output (8, 32, 32)，同尺寸便于和 roll 对齐
        out = np.zeros((8, 32, 32), dtype=np.float32)
        for k in range(8):
            for c in range(3):
                out[k] += correlate_same_circular(img[c], W_cnn[k, c])
        return np.maximum(0, out)

    base_mlp = mlp_feat(img_rgb)
    base_cnn = cnn_feat(img_rgb)

    shifts = list(range(0, 13, 2))
    mlp_rel = []
    cnn_rel = []
    for s in shifts:
        shifted = np.roll(img_rgb, s, axis=2)         # 沿宽度循环平移 s 列
        mlp_s = mlp_feat(shifted)
        cnn_s = cnn_feat(shifted)
        # MLP：直接 L2 相对差
        mlp_rel.append(np.linalg.norm(mlp_s - base_mlp) / (np.linalg.norm(base_mlp) + 1e-9))
        # CNN：把 cnn_s 沿宽度反向 roll s 后再比，验证 f(T_s x) ≈ T_s f(x)
        cnn_s_aligned = np.roll(cnn_s, -s, axis=2)
        cnn_rel.append(np.linalg.norm(cnn_s_aligned - base_cnn) / (np.linalg.norm(base_cnn) + 1e-9))

    # 构图：上排 4 张平移示意 + 下排 1 张曲线
    fig = plt.figure(figsize=(13, 6.5))
    fig.suptitle(f'平移等变性 (translation equivariance)：CNN 跟着平移走，MLP 完全错位  '
                 f'(CIFAR-10 {label})',
                 color=TEXT_LT, fontsize=13, y=0.995)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.4], hspace=0.35, wspace=0.25)

    show_shifts = [0, 4, 8, 12]
    for i, s in enumerate(show_shifts):
        shifted = np.roll(img_rgb, s, axis=2)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(shifted.transpose(1, 2, 0), interpolation='nearest')
        style_imshow_axes(ax, f'shift = {s} 列')

    ax = fig.add_subplot(gs[1, :])
    ax.plot(shifts, mlp_rel, 'o-', color=ACC_RED, linewidth=2, markersize=8,
            label='MLP 第一层激活：‖f(Tₛx) − f(x)‖ / ‖f(x)‖')
    ax.plot(shifts, cnn_rel, 's-', color=ACC_GREEN, linewidth=2, markersize=8,
            label='CNN 对齐后：‖T₋ₛ g(Tₛx) − g(x)‖ / ‖g(x)‖')
    ax.set_xlabel('循环平移列数 s', color=TEXT_MD, fontsize=11)
    ax.set_ylabel('相对 L2 距离', color=TEXT_MD, fontsize=11)
    ax.set_title('MLP 的特征和原图毫无对应关系 (~1)；CNN 的特征整体跟着平移 (~0)',
                 color=TEXT_LT, fontsize=11)
    ax.legend(facecolor=PANEL_BG, edgecolor='#444', labelcolor=TEXT_LT, fontsize=10, loc='center right')
    ax.grid(alpha=0.25, linestyle='--')
    ax.set_ylim(-0.05, 1.4)

    save_fig(fig, '01_why_conv', 'translation_equivariance.png')


# ─────────────────────────────────────────────
# 02 章配图：1D 互相关全过程
# ─────────────────────────────────────────────
def _draw_cells_1d(ax, x_vals, y_pos, highlight=None, hcolor=ACC_ORANGE,
                   cell_size=0.9, label=None, value_color=TEXT_LT, fontsize=12):
    """在 ax 上画一行 1D 单元格。highlight=(start, length) 高亮区间。"""
    if label is not None:
        ax.text(-0.6, y_pos + cell_size/2, label, color=TEXT_LT,
                ha='right', va='center', fontsize=12, fontweight='bold')
    for i, v in enumerate(x_vals):
        is_h = highlight is not None and highlight[0] <= i < highlight[0] + highlight[1]
        rect = plt.Rectangle((i, y_pos), cell_size, cell_size,
                             facecolor=hcolor if is_h else PANEL_BG,
                             alpha=0.5 if is_h else 1.0,
                             edgecolor=hcolor if is_h else '#555',
                             linewidth=2 if is_h else 1)
        ax.add_patch(rect)
        ax.text(i + cell_size/2, y_pos + cell_size/2, str(v),
                color=value_color, ha='center', va='center',
                fontsize=fontsize, fontweight='bold' if is_h else 'normal')


def fig_1d_correlation():
    x = [3, 1, 4, 1, 5, 9, 2]
    w = [1, 0, -1]
    y_out = [sum(x[p+k]*w[k] for k in range(3)) for p in range(5)]

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.suptitle('1D 互相关：filter w = [1, 0, −1] 滑过 x = [3, 1, 4, 1, 5, 9, 2]',
                 color=TEXT_LT, fontsize=13, y=0.97)
    ax.set_xlim(-1.5, 11)
    ax.set_ylim(-7.5, 1.3)
    ax.axis('off')

    _draw_cells_1d(ax, x, 0.2, label='x:')

    for pos in range(5):
        row_y = -1.0 - pos * 1.15
        _draw_cells_1d(ax, x, row_y, highlight=(pos, 3),
                       label=f'pos {pos}:')
        result = y_out[pos]
        prods = ' + '.join([f'{x[pos+k]}·{w[k]}' for k in range(3)])
        col = ACC_GREEN if result > 0 else (ACC_RED if result < 0 else TEXT_DIM)
        ax.text(7.6, row_y + 0.45,
                f'= {prods}  =  {result:+d}',
                color=col, fontsize=11, va='center', family='monospace')

    out_y = -1.0 - 5 * 1.15 - 0.1
    _draw_cells_1d(ax, [f'{v:+d}' for v in y_out], out_y, label='y:',
                   value_color=ACC_GREEN if max(y_out) > 0 else TEXT_LT)
    ax.text(5.5, out_y - 0.4, '输出比输入短：7 − 3 + 1 = 5',
            color=TEXT_DIM, fontsize=10, ha='center', style='italic')

    save_fig(fig, '02_convolution', '1d_correlation.png')


# ─────────────────────────────────────────────
# 02 章配图：互相关 vs 真正的卷积
# ─────────────────────────────────────────────
def _draw_cells_1d_signed(ax, vals, y_pos, label=None, cell_size=0.9):
    """画带正负着色的单元格行：正→蓝，负→红，0→灰。"""
    if label is not None:
        ax.text(-0.6, y_pos + cell_size/2, label, color=TEXT_LT,
                ha='right', va='center', fontsize=12, fontweight='bold')
    for i, v in enumerate(vals):
        if v > 0:   c = ACC_BLUE
        elif v < 0: c = ACC_RED
        else:       c = '#555'
        rect = plt.Rectangle((i, y_pos), cell_size, cell_size,
                             facecolor=c, alpha=0.55,
                             edgecolor=c, linewidth=2)
        ax.add_patch(rect)
        ax.text(i + cell_size/2, y_pos + cell_size/2, f'{v:+d}',
                color=TEXT_LT, ha='center', va='center',
                fontsize=12, fontweight='bold')


def fig_correlation_vs_convolution():
    x = [3, 1, 4, 1, 5, 9, 2]
    w = [1, 0, -1]
    w_flip = w[::-1]
    y_corr = [sum(x[p+k]*w[k]      for k in range(3)) for p in range(5)]
    y_conv = [sum(x[p+k]*w_flip[k] for k in range(3)) for p in range(5)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle('互相关 vs 真正卷积：唯一区别是 filter 翻转 → 输出每一位符号相反',
                 color=TEXT_LT, fontsize=13, y=0.99)

    for ax, title, weights, y_vals, want_arrow in [
        (axes[0], '互相关 (深度学习实际算的)\nfilter 直接用',     w,      y_corr, False),
        (axes[1], '真正卷积 (数学定义)\nfilter 先左右翻转',         w_flip, y_conv, True),
    ]:
        ax.set_xlim(-1.5, 8.5)
        ax.set_ylim(-3.7, 2.0)
        ax.axis('off')
        ax.set_title(title, color=TEXT_LT, fontsize=12, pad=10)
        _draw_cells_1d(ax, x, 0.6, label='x:')
        _draw_cells_1d(ax, weights, -0.8,
                       label='w:' if not want_arrow else 'flip w:')
        _draw_cells_1d_signed(ax, y_vals, -2.5, label='out:')
        if want_arrow:
            ax.annotate('', xy=(2.2, -0.35), xytext=(0.3, -0.35),
                        arrowprops=dict(arrowstyle='->', color=ACC_ORANGE, lw=2))

    # 在两幅图中间画一对比对箭头：每个输出和对面输出符号相反
    fig.text(0.5, 0.18,
             '对应位置的输出符号互为相反数：  −1 ↔ +1   0 ↔ 0   −1 ↔ +1   −8 ↔ +8   +3 ↔ −3',
             color=ACC_GREEN, fontsize=11, ha='center', va='center',
             fontweight='bold')

    save_fig(fig, '02_convolution', 'corr_vs_conv.png')


# ─────────────────────────────────────────────
# 02 章配图：2D 互相关 raster scan 路径
# ─────────────────────────────────────────────
def _draw_grid_2d(ax, M, vmin=None, vmax=None, cmap='Blues', fontsize=11,
                  text_color='black'):
    """通用 2D 网格绘制."""
    M = np.asarray(M)
    if vmin is None: vmin = M.min()
    if vmax is None: vmax = M.max() * 1.1 if M.max() > 0 else 1
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = f'{int(v)}' if float(v).is_integer() else f'{v:.1f}'
            ax.text(j, i, txt, ha='center', va='center',
                    color=text_color, fontsize=fontsize, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#444')


def fig_2d_raster_scan():
    X = np.array([[1,2,3,0,1],[0,1,2,3,1],[3,0,1,2,2],[2,3,0,1,0],[1,2,3,0,1]])

    fig = plt.figure(figsize=(14, 6.5))
    fig.suptitle('2D 互相关：3×3 filter 在 5×5 输入上沿 raster scan 路径走 9 步 → 3×3 输出',
                 color=TEXT_LT, fontsize=13, y=0.98)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)

    ax_x = fig.add_subplot(gs[0, 0])
    _draw_grid_2d(ax_x, X)
    ax_x.set_title('输入 X (5×5)：filter 起手能站到的 9 个位置（彩色框）',
                   color=TEXT_LT, fontsize=11)

    palette = plt.cm.viridis(np.linspace(0.1, 0.9, 9))
    for idx, (i, j) in enumerate([(a, b) for a in range(3) for b in range(3)]):
        rect = plt.Rectangle((j-0.5, i-0.5), 3, 3, linewidth=2,
                             edgecolor=palette[idx], facecolor='none', alpha=0.7)
        ax_x.add_patch(rect)
        # 在 filter 左上角画一个小圆点标号
        ax_x.add_patch(plt.Circle((j-0.4, i-0.4), 0.16,
                                  facecolor=palette[idx], edgecolor='white', linewidth=1))
        ax_x.text(j-0.4, i-0.4, str(idx+1), color='white',
                  ha='center', va='center', fontsize=8, fontweight='bold')

    ax_y = fig.add_subplot(gs[0, 1])
    Y_idx = np.arange(9).reshape(3, 3) + 1
    ax_y.imshow(np.zeros((3,3)), cmap='Greys', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            ax_y.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                         facecolor=palette[idx], alpha=0.7,
                                         edgecolor='white', linewidth=1))
            ax_y.text(j, i, f'Y[{i},{j}]\n← #{idx+1}',
                      ha='center', va='center', color='white',
                      fontsize=11, fontweight='bold')
    ax_y.set_xticks([]); ax_y.set_yticks([])
    for sp in ax_y.spines.values(): sp.set_color('#444')
    ax_y.set_title('输出 Y (3×3)：每一格颜色对应左侧的 filter 位置',
                   color=TEXT_LT, fontsize=11)

    save_fig(fig, '02_convolution', 'raster_scan.png')


# ─────────────────────────────────────────────
# 02 章配图：前 3 步 filter 覆盖 + 完整 Y
# ─────────────────────────────────────────────
def fig_2d_step_overlay():
    X = np.array([[1,2,3,0,1],[0,1,2,3,1],[3,0,1,2,2],[2,3,0,1,0],[1,2,3,0,1]])
    W = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    Y = np.zeros((3, 3), dtype=np.int32)
    for i in range(3):
        for j in range(3):
            Y[i, j] = (X[i:i+3, j:j+3] * W).sum()

    fig = plt.figure(figsize=(17, 5.5))
    fig.suptitle('§3.2 前 3 步逐格分解 + filter W + 完整输出 Y',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.55, 0.85], wspace=0.3)

    # 前 3 步
    for col, (i, j) in enumerate([(0, 0), (0, 1), (0, 2)]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(X, cmap='Blues', alpha=0.35, vmin=-1, vmax=4)
        for r in range(5):
            for c in range(5):
                in_box = (i <= r < i+3) and (j <= c < j+3)
                if in_box:
                    val_x = X[r, c]
                    val_w = W[r-i, c-j]
                    ax.text(c, r, f'{val_x}·{val_w:+d}',
                            ha='center', va='center', color=ACC_ORANGE,
                            fontsize=11, fontweight='bold')
                else:
                    ax.text(c, r, str(X[r, c]),
                            ha='center', va='center', color='#666', fontsize=10)
        rect = plt.Rectangle((j-0.5, i-0.5), 3, 3, linewidth=2.5,
                             edgecolor=ACC_ORANGE, facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f'第 ({i},{j}) 步\nY[{i},{j}] = sum = {Y[i,j]:+d}',
                     color=TEXT_LT, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color('#444')

    # filter W
    ax_w = fig.add_subplot(gs[0, 3])
    _draw_grid_2d(ax_w, W, cmap='RdBu_r', vmin=-1.5, vmax=1.5, fontsize=20)
    ax_w.set_title('filter W\n左+1 / 中0 / 右−1',
                   color=TEXT_LT, fontsize=11)

    # 完整 Y
    ax_y = fig.add_subplot(gs[0, 4])
    _draw_grid_2d(ax_y, Y, cmap='RdBu_r', vmin=-3, vmax=3, fontsize=20)
    ax_y.set_title('完整 Y (3×3)\n红=正 蓝=负',
                   color=TEXT_LT, fontsize=11)

    save_fig(fig, '02_convolution', 'step_overlay.png')


# ─────────────────────────────────────────────
# 02 章配图：输入像素被 filter 覆盖次数（边界凋零）
# ─────────────────────────────────────────────
def fig_input_coverage_count():
    H, Win, k = 5, 5, 3
    cnt = touch_count_map(H, Win, k, padding=0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.suptitle('5×5 输入每个像素被 3×3 filter 覆盖的次数：'
                 '中心 9× / 边 6× / 角 1×',
                 color=TEXT_LT, fontsize=12, y=0.98)
    _draw_grid_2d(ax, cnt, cmap='YlOrRd', vmin=0, vmax=10, fontsize=18)
    ax.set_title('颜色越深 = 参与计算次数越多。'
                 '\n角落 (0,0) 像素只贡献给 Y[0,0] 一次 → 边界信息浪费',
                 color=TEXT_LT, fontsize=11)
    save_fig(fig, '02_convolution', 'coverage_count.png')


# ─────────────────────────────────────────────
# 02 章配图：合成"垂直亮条"上的 Sobel-x 输出
# ─────────────────────────────────────────────
def fig_vertical_bar_sobel():
    X = np.zeros((5, 7), dtype=np.int32)
    X[:, 3] = 9  # 中间一列亮
    Wx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Y = np.zeros((3, 5), dtype=np.int32)
    for i in range(3):
        for j in range(5):
            Y[i, j] = (X[i:i+3, j:j+3] * Wx).sum()

    fig = plt.figure(figsize=(15, 4.5))
    fig.suptitle('一根垂直亮条经过 Sobel-x：原图"实心条"变成两条"轮廓线"（左+ / 右−）',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.6, 1.0], wspace=0.3)

    ax_x = fig.add_subplot(gs[0, 0])
    _draw_grid_2d(ax_x, X, cmap='gray', vmin=0, vmax=10, text_color='#222')
    ax_x.set_title('输入 X (5×7)：第 3 列亮 (=9), 其余 0',
                   color=TEXT_LT, fontsize=11)

    ax_w = fig.add_subplot(gs[0, 1])
    _draw_grid_2d(ax_w, Wx, cmap='RdBu_r', vmin=-2.5, vmax=2.5, fontsize=16)
    ax_w.set_title('Sobel-x:\n右列 + / 左列 −', color=TEXT_LT, fontsize=11)

    ax_y = fig.add_subplot(gs[0, 2])
    _draw_grid_2d(ax_y, Y, cmap='RdBu_r', vmin=-40, vmax=40, fontsize=14)
    ax_y.set_title('输出 Y (3×5)：+36 标左边缘, −36 标右边缘',
                   color=TEXT_LT, fontsize=11)

    save_fig(fig, '02_convolution', 'vertical_bar_sobel.png')


# ─────────────────────────────────────────────
# 02 章配图：im2col 概念图（卷积 = 矩阵乘法）
# ─────────────────────────────────────────────
def fig_im2col():
    # 4×4 输入, 3×3 filter → 2×2 输出 → 4 个 patch
    X = np.array([[1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6]])
    W = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    patches = []
    for i in range(2):
        for j in range(2):
            patches.append(X[i:i+3, j:j+3].flatten())
    patch_mat = np.stack(patches)        # (4, 9)
    w_vec = W.flatten()                  # (9,)
    y_vec = patch_mat @ w_vec            # (4,)
    Y = y_vec.reshape(2, 2)

    fig = plt.figure(figsize=(15, 6.5))
    fig.suptitle('im2col：把 4 个 3×3 patch 拉平成 4 行 → 卷积变成一次矩阵乘法',
                 color=TEXT_LT, fontsize=13, y=0.99)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.15, 1.6, 0.5, 0.6], wspace=0.15)

    ax_x = fig.add_subplot(gs[0, 0])
    _draw_grid_2d(ax_x, X, cmap='Blues', fontsize=14)
    ax_x.set_title('输入 X (4×4)\n4 种颜色 = 4 个 patch',
                   color=TEXT_LT, fontsize=11)
    pcols = [ACC_ORANGE, ACC_GREEN, ACC_BLUE, '#9c27b0']
    for idx, (i, j) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        rect = plt.Rectangle((j-0.5+idx*0.05, i-0.5+idx*0.05), 3, 3, linewidth=2,
                             edgecolor=pcols[idx], facecolor='none', alpha=0.9)
        ax_x.add_patch(rect)

    fig.text(0.255, 0.5, '→\nim2col', color=TEXT_LT,
             fontsize=14, ha='center', va='center', fontweight='bold')

    ax_m = fig.add_subplot(gs[0, 2])
    _draw_grid_2d(ax_m, patch_mat, cmap='Blues', fontsize=11)
    # 行旁标颜色
    for idx in range(4):
        ax_m.add_patch(plt.Rectangle((-0.6, idx-0.5), 0.4, 1,
                                     facecolor=pcols[idx], edgecolor='white'))
    ax_m.set_title('patch 矩阵 (4 × 9)\n每行 = 一个 3×3 patch 拉平',
                   color=TEXT_LT, fontsize=11)
    ax_m.set_xlim(-0.8, 8.5)

    fig.text(0.685, 0.5, '×', color=TEXT_LT,
             fontsize=22, ha='center', va='center', fontweight='bold')

    ax_w = fig.add_subplot(gs[0, 3])
    _draw_grid_2d(ax_w, w_vec.reshape(-1, 1), cmap='RdBu_r',
                  vmin=-1.2, vmax=1.2, fontsize=12)
    ax_w.set_title('w 拉平\n(9 × 1)', color=TEXT_LT, fontsize=11)

    fig.text(0.835, 0.5, '=', color=TEXT_LT,
             fontsize=22, ha='center', va='center', fontweight='bold')

    ax_y = fig.add_subplot(gs[0, 4])
    _draw_grid_2d(ax_y, Y, cmap='RdBu_r', vmin=-12, vmax=12, fontsize=16)
    ax_y.set_title('Y reshape\n(2 × 2)', color=TEXT_LT, fontsize=11)

    save_fig(fig, '02_convolution', 'im2col.png')


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('Week 2 文档插图生成')
    print('=' * 60)

    print('\n[1/2] 准备 CIFAR-10 数据 ...')
    img, label = get_demo_image()
    print(f'  演示用图: CIFAR-10 第一张"{label}", shape = {img.shape}')

    print('\n[2/2] 生成 20 张图 ...')
    # T1 动机章
    fig_param_explosion()
    fig_pixel_shuffle_invariance()
    fig_translation_equivariance(img, label)
    # T2 卷积：直觉 + 算例
    fig_1d_correlation()
    fig_correlation_vs_convolution()
    fig_2d_raster_scan()
    fig_2d_step_overlay()
    fig_input_coverage_count()
    fig_vertical_bar_sobel()
    fig_im2col()
    # T2 真实图片
    fig_edge_detection(img, label)
    fig_classic_filters(img, label)
    # T4 多通道
    fig_filter_shape_extension()
    fig_multichannel_conv_numerical()
    fig_K_filters_stack()
    fig_conv_tensor_shapes()
    # T3 padding/stride
    fig_padding_visualization()
    fig_stride_1d()
    fig_stride_starting_positions()
    fig_dropped_pixels()
    fig_padding_coverage()
    # T4-T5
    fig_rgb_channels(img, label)
    fig_receptive_field()
    fig_pool_compare(img, label)
    # T5 池化数值算例 + 感受野分层
    fig_pool_numerical()
    fig_maxpool_translation_invariance()
    fig_receptive_field_layered()
    # T6 反向传播相关
    fig_grad_W_slide()
    fig_grad_X_flip()
    fig_maxpool_backward()
    fig_avgpool_backward()
    fig_backprop_full_example()
    fig_gradient_aggregation()
    fig_stride2_backward()

    print('\n全部完成。文件位于 assets/week2/figures/ 下')
