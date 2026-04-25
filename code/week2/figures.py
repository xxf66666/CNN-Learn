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
    targz = os.path.join(DATA_DIR, 'cifar-10-python.tar.gz')
    if not os.path.exists(targz):
        print(f'下载 CIFAR-10 (~170 MB) → {targz}')
        urllib.request.urlretrieve(CIFAR_URL, targz)
    extracted = os.path.join(DATA_DIR, 'cifar-10-batches-py')
    if not os.path.exists(extracted):
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
# 主程序
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('Week 2 文档插图生成')
    print('=' * 60)

    print('\n[1/2] 准备 CIFAR-10 数据 ...')
    img, label = get_demo_image()
    print(f'  演示用图: CIFAR-10 第一张"{label}", shape = {img.shape}')

    print('\n[2/2] 生成 10 张图 ...')
    fig_edge_detection(img, label)
    fig_classic_filters(img, label)
    fig_rgb_channels(img, label)
    fig_padding_coverage()
    fig_receptive_field()
    fig_pool_compare(img, label)
    # T6 反向传播相关
    fig_grad_W_slide()
    fig_grad_X_flip()
    fig_maxpool_backward()
    fig_avgpool_backward()

    print('\n全部完成。文件位于 assets/week2/figures/ 下')
