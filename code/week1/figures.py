"""
Week 1 文档插图生成脚本

补充 docs/week1/ 04-05 章节的 5 张可视化图：

  04 §6  非凸损失面             → 04_gradient/nonconvex_landscape.png
  05 §2  XOR 在隐藏层空间被拉直 → 05_mlp/xor_hidden_space.png
  05 §8  激活函数导数对比       → 05_mlp/activation_derivatives.png
  05 §8  梯度随层数指数衰减     → 05_mlp/gradient_decay_by_depth.png
  05 §11 MLP 无平移不变性 vs CNN → 05_mlp/mlp_vs_cnn_translation.png

运行:
  MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week1/figures.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─────────────────────────────────────────────
# 主题（与已有 Week 1 / Week 2 图保持一致）
# ─────────────────────────────────────────────
DARK_BG   = '#0f1117'
PANEL_BG  = '#1a1d27'
TEXT_LT   = 'white'
TEXT_MD   = '#cccccc'
TEXT_DIM  = '#aaaaaa'
GRID      = '#2a2e3a'
ACC_BLUE   = '#4fc3f7'
ACC_GREEN  = '#66bb6a'
ACC_ORANGE = '#ff8a65'
ACC_RED    = '#ef5350'
ACC_PURPLE = '#ba68c8'
ACC_YELLOW = '#ffd54f'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   PANEL_BG,
    'axes.edgecolor':   '#444444',
    'axes.labelcolor':  TEXT_MD,
    'xtick.color':      TEXT_DIM,
    'ytick.color':      TEXT_DIM,
    'text.color':       TEXT_LT,
    'axes.titlecolor':  TEXT_LT,
    'axes.titlesize':   13,
    'font.family':      ['Arial Unicode MS', 'PingFang SC', 'sans-serif'],
    'axes.unicode_minus': False,
    'savefig.facecolor': DARK_BG,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
FIG_BASE = os.path.join(ROOT, 'assets', 'week1', 'figures')


def save(fig, subdir, name):
    out_dir = os.path.join(FIG_BASE, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  saved → {os.path.relpath(path, ROOT)}')


# ─────────────────────────────────────────────
# 图 1: XOR 在隐藏层空间被拉直
# ─────────────────────────────────────────────
def fig_xor_hidden_space():
    """
    输入空间: (0,0)/(1,1)→class0,  (1,0)/(0,1)→class1
    经过 h1 = ReLU(x1+x2),  h2 = ReLU(x1+x2-1)  之后:
      (0,0)→(0,0), (1,0)→(1,0), (0,1)→(1,0), (1,1)→(2,1)
    在 (h1,h2) 空间上一条直线即可分开两类。
    """
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    colors = np.where(labels == 0, ACC_RED, ACC_BLUE)

    h1 = np.maximum(pts[:, 0] + pts[:, 1], 0)
    h2 = np.maximum(pts[:, 0] + pts[:, 1] - 1, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # 左：输入空间
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=300, zorder=3,
               edgecolors='white', linewidths=1.5)
    for (x, y), lab in zip(pts, labels):
        ax.annotate(f'({int(x)},{int(y)})', (x, y), xytext=(8, 8),
                    textcoords='offset points', fontsize=10, color=TEXT_MD)

    # 画几条尝试分离的直线
    xx = np.linspace(-0.4, 1.4, 50)
    for k, b, alpha in [(-1, 0.5, 0.35), (-1, 1.5, 0.35), (0, 0.5, 0.35)]:
        ax.plot(xx, k * xx + b, '--', color=TEXT_DIM, alpha=alpha, lw=1)
    ax.text(0.5, -0.45, '任何直线都无法分开 ✗', ha='center',
            color=ACC_RED, fontsize=11, fontweight='bold')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.6, 1.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('输入空间 $(x_1, x_2)$:  XOR 线性不可分', pad=12)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_aspect('equal')

    # 右：隐藏空间
    ax = axes[1]
    # (0,1) 和 (1,0) 都映射到 (1,0)，会重叠；稍微错开标注
    ax.scatter(h1, h2, c=colors, s=300, zorder=3,
               edgecolors='white', linewidths=1.5)
    annotations = [
        ((0, 0), '(0,0)→(0,0)', (8, 8)),
        ((1, 0), '(1,0)→(1,0)', (8, -18)),
        ((1, 0), '(0,1)→(1,0)', (8, 12)),
        ((2, 1), '(1,1)→(2,1)', (-90, -2)),
    ]
    for (x, y), txt, off in annotations:
        ax.annotate(txt, (x, y), xytext=off,
                    textcoords='offset points', fontsize=9, color=TEXT_MD)

    # 一条可分线: h1 - 2*h2 = 0.5
    xs = np.linspace(-0.3, 2.5, 50)
    ys = (xs - 0.5) / 2
    ax.plot(xs, ys, '-', color=ACC_GREEN, lw=2.2,
            label='$h_1 - 2h_2 = 0.5$ ✓')

    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(-0.6, 1.5)
    ax.set_xlabel('$h_1 = \\mathrm{ReLU}(x_1+x_2)$', fontsize=12)
    ax.set_ylabel('$h_2 = \\mathrm{ReLU}(x_1+x_2-1)$', fontsize=12)
    ax.set_title('隐藏空间 $(h_1, h_2)$:  一条直线即可分开 ✓', pad=12)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.legend(loc='upper left', facecolor=PANEL_BG, edgecolor='#444',
              labelcolor=TEXT_LT)

    # 图例标注红/蓝代表
    handles = [
        mpatches.Patch(color=ACC_RED, label='类别 0  (XOR=0)'),
        mpatches.Patch(color=ACC_BLUE, label='类别 1  (XOR=1)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               facecolor=PANEL_BG, edgecolor='#444',
               labelcolor=TEXT_LT, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('非线性把"不可分"变成"可分"', y=1.02, fontsize=14)
    plt.tight_layout()
    save(fig, '05_mlp', 'xor_hidden_space.png')


# ─────────────────────────────────────────────
# 图 2: 激活函数导数对比
# ─────────────────────────────────────────────
def fig_activation_derivatives():
    z = np.linspace(-5, 5, 400)
    sig  = 1.0 / (1.0 + np.exp(-z))
    sigp = sig * (1 - sig)
    tanp = 1 - np.tanh(z) ** 2
    relup = (z > 0).astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z, sigp,  color=ACC_RED,    lw=2.2, label="Sigmoid'  (max=0.25)")
    ax.plot(z, tanp,  color=ACC_ORANGE, lw=2.2, label="Tanh'      (max=1.0)")
    ax.plot(z, relup, color=ACC_GREEN,  lw=2.4, label="ReLU'       (z>0 时为 1)")

    # 关键标注
    ax.axhline(0.25, color=ACC_RED, ls=':', alpha=0.5, lw=1)
    ax.axhline(1.0,  color=ACC_GREEN, ls=':', alpha=0.5, lw=1)
    ax.text(-4.7, 0.27, '0.25', color=ACC_RED, fontsize=10)
    ax.text(-4.7, 1.03, '1.00', color=ACC_GREEN, fontsize=10)

    # 标注 Sigmoid 顶点
    ax.scatter([0], [0.25], color=ACC_RED, s=80, zorder=4,
               edgecolors='white', linewidths=1.2)
    ax.annotate('Sigmoid 峰值 0.25\n→ 每层最多保留 1/4 梯度',
                xy=(0, 0.25), xytext=(1.2, 0.55),
                color=ACC_RED, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=ACC_RED, alpha=0.7))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('$z$', fontsize=12)
    ax.set_ylabel("激活函数导数 $\\sigma'(z)$", fontsize=12)
    ax.set_title('Sigmoid 的最大导数只有 0.25，ReLU 正区恒为 1', pad=12)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.legend(loc='upper right', facecolor=PANEL_BG, edgecolor='#444',
              labelcolor=TEXT_LT, fontsize=10)

    plt.tight_layout()
    save(fig, '05_mlp', 'activation_derivatives.png')


# ─────────────────────────────────────────────
# 图 3: 梯度随层数指数衰减
# ─────────────────────────────────────────────
def fig_gradient_decay_by_depth():
    N = np.arange(1, 21)
    sigmoid = 0.25 ** N
    tanh_practical = 0.5 ** N      # tanh 在偏离原点时迅速饱和，实际衰减
    relu = np.ones_like(N, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(N, sigmoid, 'o-', color=ACC_RED,    lw=2.2, ms=6,
                label='Sigmoid:  $0.25^N$  (最坏情况上界)')
    ax.semilogy(N, tanh_practical, 's-', color=ACC_ORANGE, lw=2.2, ms=6,
                label='Tanh:  $\\sim 0.5^N$  (远离原点饱和后)')
    ax.semilogy(N, relu, '^-', color=ACC_GREEN, lw=2.4, ms=7,
                label='ReLU:  恒为 1  (无衰减)')

    # 关键警戒线
    ax.axhline(1e-6, color=ACC_RED, ls='--', alpha=0.4, lw=1)
    ax.text(0.5, 1.5e-6, '$10^{-6}$（梯度几乎消失）',
            color=ACC_RED, fontsize=9, alpha=0.85)

    # 标注 Sigmoid 在 N=10 时的点
    ax.scatter([10], [0.25 ** 10], color=ACC_RED, s=120, zorder=4,
               edgecolors='white', linewidths=1.5)
    ax.annotate(f'10 层 Sigmoid:\n$0.25^{{10}} \\approx 9.5\\times 10^{{-7}}$',
                xy=(10, 0.25 ** 10), xytext=(11.5, 1e-3),
                color=ACC_RED, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=ACC_RED, alpha=0.7))

    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(1e-12, 5)
    ax.set_xlabel('网络层数 $N$', fontsize=12)
    ax.set_ylabel('保留下来的梯度信号（log 轴）', fontsize=12)
    ax.set_title('为什么 ReLU 让深层网络变得可训', pad=12)
    ax.set_xticks(np.arange(1, 21, 2))
    ax.grid(True, color=GRID, linewidth=0.6, which='both', alpha=0.4)
    ax.legend(loc='lower left', facecolor=PANEL_BG, edgecolor='#444',
              labelcolor=TEXT_LT, fontsize=10)

    plt.tight_layout()
    save(fig, '05_mlp', 'gradient_decay_by_depth.png')


# ─────────────────────────────────────────────
# 图 4: MLP 无平移不变性 vs CNN 权重共享
# ─────────────────────────────────────────────
def fig_mlp_vs_cnn_translation():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # ── 共用：两张 8x8 灰度图，"耳朵"出现在不同位置 ──
    img_left = np.zeros((8, 8))
    img_left[1:3, 1:3] = 1.0          # 左上耳朵
    img_right = np.zeros((8, 8))
    img_right[5:7, 5:7] = 1.0         # 右下耳朵

    # ── 左面板: MLP ──
    ax = axes[0]
    ax.set_title('MLP: 每个像素位置都有独立权重', pad=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # 上下两张小图
    def imshow_inset(ax, img, x0, y0, size=2.2, highlight=None):
        H, W = img.shape
        for i in range(H):
            for j in range(W):
                v = img[i, j]
                color = '#e0e0e0' if v > 0.5 else '#2c3140'
                rect = mpatches.Rectangle(
                    (x0 + j * size / W, y0 + (H - 1 - i) * size / H),
                    size / W, size / H,
                    facecolor=color, edgecolor='#1a1d27', linewidth=0.4)
                ax.add_patch(rect)
        # 高亮区域
        if highlight is not None:
            i0, i1, j0, j1 = highlight
            rect = mpatches.Rectangle(
                (x0 + j0 * size / W, y0 + (H - i1) * size / H),
                (j1 - j0) * size / W, (i1 - i0) * size / H,
                fill=False, edgecolor=ACC_YELLOW, linewidth=2.5)
            ax.add_patch(rect)

    imshow_inset(ax, img_left,  0.5, 5.5, highlight=(1, 3, 1, 3))
    imshow_inset(ax, img_right, 0.5, 0.6, highlight=(5, 7, 5, 7))
    ax.text(1.6, 8.0, '图 A: 耳朵在左上', color=TEXT_MD, fontsize=10, ha='center')
    ax.text(1.6, 3.1, '图 B: 耳朵在右下', color=TEXT_MD, fontsize=10, ha='center')

    # MLP 第一层"权重网格"——不同位置不同颜色，强调互不相关
    rng = np.random.default_rng(7)
    weight_colors = rng.random((8, 8))
    for i in range(8):
        for j in range(8):
            c = plt.cm.viridis(weight_colors[i, j] * 0.9)
            rect = mpatches.Rectangle(
                (4 + j * 0.32, 3.6 + (7 - i) * 0.32),
                0.32, 0.32, facecolor=c, edgecolor='#1a1d27', linewidth=0.3)
            ax.add_patch(rect)
    ax.text(5.3, 7.6, '$W^{(1)}$ 的 64 个权重\n(每个像素位置独立)',
            color=TEXT_LT, fontsize=10, ha='center')

    # 箭头：A 的高亮区域 → 某些权重； B 的高亮区域 → 另外一些权重
    # （示意性，不画准确连线，只表达"不同位置走不同权重"）
    ax.annotate('', xy=(4.5, 5.2), xytext=(2.3, 6.5),
                arrowprops=dict(arrowstyle='->', color=ACC_ORANGE, lw=1.8))
    ax.annotate('', xy=(6.5, 4.0), xytext=(2.3, 1.6),
                arrowprops=dict(arrowstyle='->', color=ACC_PURPLE, lw=1.8))

    ax.text(5.3, 2.6,
            '左上耳朵 → 橙色权重\n右下耳朵 → 紫色权重\n这两组权重互不相关',
            color=TEXT_MD, fontsize=10, ha='center', va='top')

    # ── 右面板: CNN ──
    ax = axes[1]
    ax.set_title('CNN: 同一个滤波器在所有位置滑动', pad=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')

    imshow_inset(ax, img_left,  0.5, 5.5, highlight=(1, 3, 1, 3))
    imshow_inset(ax, img_right, 0.5, 0.6, highlight=(5, 7, 5, 7))
    ax.text(1.6, 8.0, '图 A: 耳朵在左上', color=TEXT_MD, fontsize=10, ha='center')
    ax.text(1.6, 3.1, '图 B: 耳朵在右下', color=TEXT_MD, fontsize=10, ha='center')

    # 一个 3x3 滤波器，画两次（图 A 上 + 图 B 上），同色同模式
    filter_pattern = np.array([
        [0.2, 0.9, 0.2],
        [0.9, 0.5, 0.9],
        [0.2, 0.9, 0.2],
    ])

    def draw_filter(ax, x0, y0, size=1.2):
        for i in range(3):
            for j in range(3):
                c = plt.cm.plasma(filter_pattern[i, j])
                rect = mpatches.Rectangle(
                    (x0 + j * size / 3, y0 + (2 - i) * size / 3),
                    size / 3, size / 3,
                    facecolor=c, edgecolor='#1a1d27', linewidth=0.4)
                ax.add_patch(rect)
        # 边框
        rect = mpatches.Rectangle((x0, y0), size, size,
                                  fill=False, edgecolor=ACC_YELLOW, lw=2)
        ax.add_patch(rect)

    draw_filter(ax, 5.5, 6.3)
    draw_filter(ax, 5.5, 1.2)
    ax.text(6.1, 7.7, '同一个 3×3 滤波器', color=TEXT_LT, fontsize=11, ha='center')
    ax.text(6.1, 2.6, '同一个 3×3 滤波器', color=TEXT_LT, fontsize=11, ha='center')

    # 箭头：A 的高亮区域 → 滤波器; B 的高亮区域 → 滤波器（同色，强调同一个）
    ax.annotate('', xy=(5.5, 6.8), xytext=(2.3, 6.5),
                arrowprops=dict(arrowstyle='->', color=ACC_GREEN, lw=1.8))
    ax.annotate('', xy=(5.5, 1.7), xytext=(2.3, 1.6),
                arrowprops=dict(arrowstyle='->', color=ACC_GREEN, lw=1.8))

    ax.text(7.5, 4.5,
            '权重共享\n→ 平移不变\n→ 参数大幅减少',
            color=ACC_GREEN, fontsize=11, ha='center', va='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#1a1d27', edgecolor=ACC_GREEN))

    plt.suptitle('为什么图像任务从 MLP 跳到 CNN', y=1.00, fontsize=14)
    plt.tight_layout()
    save(fig, '05_mlp', 'mlp_vs_cnn_translation.png')


# ─────────────────────────────────────────────
# 图 5: 非凸损失面（04 §6）
# ─────────────────────────────────────────────
def fig_nonconvex_landscape():
    """
    左：传统的"碗"   ½ (W1² + 2 W2²)
    右：非凸损失面，含多个局部极小和鞍点
        L = sin(1.4 W1) cos(1.4 W2) + 0.08 (W1² + W2²)
    """
    w1 = np.linspace(-3, 3, 120)
    w2 = np.linspace(-3, 3, 120)
    W1, W2 = np.meshgrid(w1, w2)

    L_convex = 0.5 * (W1 ** 2 + 2 * W2 ** 2)
    L_nonconv = np.sin(1.4 * W1) * np.cos(1.4 * W2) + 0.08 * (W1 ** 2 + W2 ** 2)

    fig = plt.figure(figsize=(13, 5.5))

    # 左：凸 / 碗
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_facecolor(PANEL_BG)
    ax.plot_surface(W1, W2, L_convex, cmap='viridis',
                    edgecolor='none', alpha=0.92,
                    rstride=2, cstride=2)
    ax.set_xlabel('$W_1$', color=TEXT_MD)
    ax.set_ylabel('$W_2$', color=TEXT_MD)
    ax.set_zlabel('$\\mathcal{L}$', color=TEXT_MD)
    ax.set_title('凸：1D / 2D 教学图（"碗"）', pad=12)
    ax.tick_params(colors=TEXT_DIM)
    ax.xaxis.pane.set_facecolor(PANEL_BG)
    ax.yaxis.pane.set_facecolor(PANEL_BG)
    ax.zaxis.pane.set_facecolor(PANEL_BG)
    ax.xaxis.pane.set_edgecolor('#333')
    ax.yaxis.pane.set_edgecolor('#333')
    ax.zaxis.pane.set_edgecolor('#333')
    ax.grid(True, color=GRID)

    # 右：非凸（神经网络真实情况的玩具版）
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_facecolor(PANEL_BG)
    ax.plot_surface(W1, W2, L_nonconv, cmap='magma',
                    edgecolor='none', alpha=0.92,
                    rstride=2, cstride=2)
    ax.set_xlabel('$W_1$', color=TEXT_MD)
    ax.set_ylabel('$W_2$', color=TEXT_MD)
    ax.set_zlabel('$\\mathcal{L}$', color=TEXT_MD)
    ax.set_title('非凸：神经网络真实损失面（玩具版）', pad=12)
    ax.tick_params(colors=TEXT_DIM)
    ax.xaxis.pane.set_facecolor(PANEL_BG)
    ax.yaxis.pane.set_facecolor(PANEL_BG)
    ax.zaxis.pane.set_facecolor(PANEL_BG)
    ax.xaxis.pane.set_edgecolor('#333')
    ax.yaxis.pane.set_edgecolor('#333')
    ax.zaxis.pane.set_edgecolor('#333')
    ax.grid(True, color=GRID)

    fig.text(0.5, 0.02,
             '左：只有一个最低点，梯度下降必收敛到全局最优。'
             '   右：多个局部极小 + 鞍点 + 平台，梯度下降只保证局部下降。',
             ha='center', color=TEXT_MD, fontsize=10)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, '04_gradient', 'nonconvex_landscape.png')


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('生成 Week 1 文档插图…')
    fig_xor_hidden_space()
    fig_activation_derivatives()
    fig_gradient_decay_by_depth()
    fig_mlp_vs_cnn_translation()
    fig_nonconvex_landscape()
    print('完成。')
