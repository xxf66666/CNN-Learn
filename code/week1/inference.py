"""
Week 1 T8 拓展：推理 + 预处理
配合 app.py 使用，把任意手绘图片喂给训好的 MLP。

关键点是 preprocess()，把"用户在画板上画的东西"对齐到 MNIST 训练分布：
反色 → 找笔画 bbox → 等比缩到 20×20 → 按重心放进 28×28 画布 → 归一化。
对应 docs/week1/09_handwriting_demo.md 的预处理章节。
"""

import os
import numpy as np
from PIL import Image

from mlp_numpy import forward  # 复用训练时的 forward

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), '../../assets/week1/outputs/mlp_weights.npz'
)


# ─────────────────────────────────────────────
# 1. 加载训练好的权重
# ─────────────────────────────────────────────

def load_model(path: str = WEIGHTS_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'找不到权重文件 {path}\n'
            f'请先跑一遍训练：\n'
            f'  MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week1/mlp_numpy.py'
        )
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ─────────────────────────────────────────────
# 2. 预处理（核心）
# ─────────────────────────────────────────────

def preprocess(img: np.ndarray) -> np.ndarray:
    """
    输入: 任意尺寸的 H×W 灰度数组（uint8 或 float），值约定为
         "笔画亮 / 背景暗" 或 "笔画暗 / 背景亮" 都可以，函数会自动判断。
    输出: (1, 784) 的 float32 数组，值在 [0, 1]，可直接喂 forward()。

    流程严格对齐 LeCun 1998 制作 MNIST 时的预处理：
      ① 转灰度 + 自动反色到"黑底白字"
      ② 阈值 + 找笔画 bounding box，裁掉空白
      ③ 等比缩放，让长边 = 20 像素
      ④ 算重心(center of mass)，把笔画"重心"放到 28×28 画布的正中
      ⑤ 归一化到 [0, 1] 并 flatten
    """
    arr = np.asarray(img)

    # ── ① 转单通道灰度 ─────────────────────────
    if arr.ndim == 3:
        # RGB / RGBA 都用亮度公式压成灰度
        if arr.shape[2] == 4:                  # 带 alpha 通道（画板常见）
            alpha = arr[:, :, 3].astype(np.float32) / 255.0
            rgb   = arr[:, :, :3].astype(np.float32)
            # 透明区域当成白色背景
            arr   = rgb * alpha[..., None] + 255.0 * (1 - alpha[..., None])
        arr = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])

    arr = arr.astype(np.float32)

    # ── ② 自动反色到"黑底白字"─────────────────
    # 判定：四角的平均亮度 > 整体平均，说明背景是亮的，需要反色
    h, w = arr.shape
    corners = np.mean([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    if corners > arr.mean():
        arr = 255.0 - arr

    # ── ③ 阈值 + 找笔画 bbox ──────────────────
    mask = arr > 30                            # 30/255 ≈ 12% 亮度
    if not mask.any():
        # 整张图全黑（用户没画东西）
        return np.zeros((1, 784), dtype=np.float32)

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = arr[y0:y1, x0:x1]

    # ── ④ 等比缩到长边 20 ────────────────────
    bh, bw = cropped.shape
    if bh > bw:
        new_h, new_w = 20, max(1, round(bw * 20 / bh))
    else:
        new_h, new_w = max(1, round(bh * 20 / bw)), 20

    pil = Image.fromarray(cropped.astype(np.uint8), mode='L')
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.asarray(pil, dtype=np.float32)

    # ── ⑤ 按重心放进 28×28 画布 ──────────────
    canvas = np.zeros((28, 28), dtype=np.float32)
    # 先粗放在中心，算出重心后再平移
    off_y = (28 - new_h) // 2
    off_x = (28 - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    cy, cx = _center_of_mass(canvas)
    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))
    canvas = _shift(canvas, shift_y, shift_x)

    # ── 归一化 + flatten ─────────────────────
    canvas /= 255.0
    return canvas.reshape(1, 784).astype(np.float32)


def _center_of_mass(arr: np.ndarray) -> tuple[float, float]:
    total = arr.sum()
    if total == 0:
        return 14.0, 14.0
    ys = np.arange(arr.shape[0])[:, None]
    xs = np.arange(arr.shape[1])[None, :]
    cy = float((arr * ys).sum() / total)
    cx = float((arr * xs).sum() / total)
    return cy, cx


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """整数像素平移，超出边界的部分丢掉，空出的位置补 0。"""
    h, w = arr.shape
    out = np.zeros_like(arr)
    y_src_start = max(0, -dy); y_src_end = min(h, h - dy)
    x_src_start = max(0, -dx); x_src_end = min(w, w - dx)
    y_dst_start = max(0, dy);  y_dst_end = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(0, dx);  x_dst_end = x_dst_start + (x_src_end - x_src_start)
    out[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
        arr[y_src_start:y_src_end, x_src_start:x_src_end]
    return out


# ─────────────────────────────────────────────
# 3. 预测接口
# ─────────────────────────────────────────────

def predict(img: np.ndarray, params: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """
    返回:
      pred:    int，预测的数字 0-9
      probs:   shape (10,)，每个类别的 softmax 概率
      x_view:  shape (28, 28)，预处理后喂给模型的图，供 UI 显示给用户看
    """
    x = preprocess(img)                        # (1, 784)
    logits, _ = forward(x, params)             # (1, 10)

    # 数值稳定 softmax（和训练里完全一致）
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = (e / e.sum(axis=1, keepdims=True))[0]

    pred   = int(probs.argmax())
    x_view = x.reshape(28, 28)
    return pred, probs, x_view


# ─────────────────────────────────────────────
# 4. CLI 自检
# ─────────────────────────────────────────────

if __name__ == '__main__':
    """跑一下：用测试集第一张验证 inference 链路通畅。"""
    from mlp_numpy import load_data

    print('加载权重...')
    params = load_model()

    print('加载测试集第 0 张...')
    _, _, X_test, y_test = load_data()
    img28 = (X_test[0].reshape(28, 28) * 255).astype(np.uint8)

    pred, probs, _ = predict(img28, params)
    print(f'真实标签: {y_test[0]}, 预测: {pred}')
    print(f'概率分布: {np.round(probs, 3)}')
    assert pred == y_test[0], '推理链路出问题了'
    print('✓ inference 链路 OK')
