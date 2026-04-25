"""
Week 2 T7: 手写 2D 卷积层 (前向 + 反向)

理论对应:
  - 前向公式      docs/week2/02_convolution.md §6
  - padding/stride docs/week2/03_padding_stride.md §3
  - 多通道/多 filter docs/week2/04_multi_channel.md §3
  - 反向 ∇W/∇b/∇X docs/week2/06_conv_backprop.md §3-§6

API 约定 (跟 Week 1 mlp_numpy.py 一致):
  Y, cache = conv2d_forward(X, W, b, padding, stride)
  grad_X, grad_W, grad_b = conv2d_backward(delta, cache)
  gradient_check(...) 是项目"事实测试套件" — 改了 forward/backward 必跑

用法:
  python code/week2/conv2d_numpy.py
"""

import numpy as np

# 抑制 macOS Accelerate BLAS 误报 (跟 Week 1 同样原因)
np.seterr(divide='ignore', over='ignore', invalid='ignore')


# ─────────────────────────────────────────────
# 1. 前向传播
# ─────────────────────────────────────────────

def conv2d_forward(X, W, b, padding=0, stride=1):
    """
    多通道、多 filter、可选 padding/stride 的 2D 卷积前向。

    参数:
      X       (N, C_in, H, W_in)        输入张量
      W       (C_out, C_in, k, k)       filter 组
      b       (C_out,)                  每个 filter 一个 bias
      padding int, 四周补 0 圈数
      stride  int, filter 滑动步长

    返回:
      Y       (N, C_out, H_out, W_out)  输出 feature maps
      cache   tuple, 反向时要用 (含 X_padded, W, padding, stride, k)
    """
    N, C_in, H, W_in = X.shape
    C_out, _, k, _   = W.shape

    # ① padding (T3 §1)
    if padding > 0:
        X_padded = np.pad(
            X, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant', constant_values=0,
        )
    else:
        X_padded = X

    H_padded, W_padded = X_padded.shape[2], X_padded.shape[3]

    # 输出尺寸 (T3 §3 公式)
    H_out = (H_padded - k) // stride + 1
    W_out = (W_padded - k) // stride + 1

    Y = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)

    # ② 4 重 for: batch / 输出通道 / 输出 H / 输出 W
    # (朴素实现, 性能差但意图清楚, im2col 优化作为可选)
    for n in range(N):
        for ko in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    # 抠出 (C_in, k, k) 的 patch
                    patch = X_padded[n, :,
                                     i*stride : i*stride + k,
                                     j*stride : j*stride + k]
                    # 与第 ko 个 filter (C_in, k, k) 对位乘求和
                    # 折叠掉 (C_in, k, k) 三个维度
                    Y[n, ko, i, j] = (patch * W[ko]).sum() + b[ko]

    # 缓存反向所需的中间量
    cache = (X_padded, W, padding, stride, k)
    return Y, cache


# ─────────────────────────────────────────────
# 2. 反向传播
# ─────────────────────────────────────────────

def conv2d_backward(delta, cache):
    """
    给定输出梯度 δ = ∂L/∂Y, 算 ∂L/∂X, ∂L/∂W, ∂L/∂b.

    实现策略:
      ∇W: 直接用 T6 §3 的 "δ 当 filter 在 X 上滑" 实现 (clean)
      ∇b: T6 §5 一行 sum
      ∇X: 用 "scatter" 实现 (T6 §4 的等价形式) — 把每个 δ·W
          散射回它在前向时影响过的输入位置. 比 "翻转+卷积"
          实现错误率低, gradient check 通过率高.

    参数:
      delta   (N, C_out, H_out, W_out)
      cache   conv2d_forward 返回的元组

    返回:
      grad_X  (N, C_in, H, W_in)        要传给上一层
      grad_W  (C_out, C_in, k, k)       用来更新 filter
      grad_b  (C_out,)                  用来更新 bias
    """
    X_padded, W, padding, stride, k = cache
    N, C_in, H_padded, W_padded = X_padded.shape
    C_out = W.shape[0]
    _, _, H_out, W_out = delta.shape

    # ── ∇b: 在 (N, H_out, W_out) 三个维度求和 (T6 §5) ──────
    grad_b = delta.sum(axis=(0, 2, 3))                 # (C_out,)

    # ── ∇W 和 ∇X: 一起在主循环里算 ──────────────────────
    grad_W        = np.zeros_like(W)                   # (C_out, C_in, k, k)
    grad_X_padded = np.zeros_like(X_padded)            # (N, C_in, H_pad, W_pad)

    for n in range(N):
        for ko in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    # 这个输出位置看的输入子块
                    h_slice = slice(i*stride, i*stride + k)
                    w_slice = slice(j*stride, j*stride + k)
                    patch = X_padded[n, :, h_slice, w_slice]   # (C_in, k, k)
                    d = delta[n, ko, i, j]                     # 标量

                    # ∇W 累加: T6 §3 公式
                    # ∇W[ko] += d * patch
                    grad_W[ko] += d * patch

                    # ∇X "scatter" 累加: 这个输出对前向输入子块的贡献
                    # ∂Y[n,ko,i,j]/∂X_padded[n,:,h_slice,w_slice] = W[ko]
                    grad_X_padded[n, :, h_slice, w_slice] += d * W[ko]

    # 把 padding 的部分裁掉, 还原成原 X 的形状
    if padding > 0:
        grad_X = grad_X_padded[:, :, padding:-padding, padding:-padding]
    else:
        grad_X = grad_X_padded

    return grad_X, grad_W, grad_b


# ─────────────────────────────────────────────
# 3. 梯度检验 (Week 1 §10 套路复用)
# ─────────────────────────────────────────────

def gradient_check(X, W, b, padding=0, stride=1, eps=1e-5, atol=1e-4):
    """
    用有限差分验证解析梯度.
      数值梯度 = (loss(p + eps) - loss(p - eps)) / (2*eps)
      解析梯度 = backward 算出来的
      相对误差 < atol 算通过.

    用一个 simple loss = Y.sum() (∂L/∂Y = 全 1) 来简化测试 — 这样
    backward 收到的 delta = ones_like(Y), 容易复现.

    重要: gradient check 内部用 float64. 用 float32 会因为
    finite differencing 的 catastrophic cancellation (∼7 位有效数字)
    导致数值梯度噪声过大, 误差堆到 1e-2 量级看似实现错了, 其实只是精度不够.
    """
    # 升精度做 grad check (训练时仍用 float32)
    X = X.astype(np.float64)
    W = W.astype(np.float64)
    b = b.astype(np.float64)

    # 前向 + 反向
    Y, cache = conv2d_forward(X, W, b, padding=padding, stride=stride)
    delta = np.ones_like(Y)                              # ∂L/∂Y, L = Y.sum()
    grad_X_ana, grad_W_ana, grad_b_ana = conv2d_backward(delta, cache)

    print(f'\n配置: padding={padding}, stride={stride}, '
          f'X={X.shape}, W={W.shape}, Y={Y.shape}')
    print(f'{"参数":>10} | {"解析梯度":>14} | {"数值梯度":>14} | {"相对误差":>12}')
    print('-' * 62)

    def num_grad(arr, idx):
        """对 arr[idx] 做一次 ±eps 的有限差分, 返回 ∂Loss/∂arr[idx]."""
        old = arr[idx]
        arr[idx] = old + eps
        loss_plus  = conv2d_forward(X, W, b, padding=padding, stride=stride)[0].sum()
        arr[idx] = old - eps
        loss_minus = conv2d_forward(X, W, b, padding=padding, stride=stride)[0].sum()
        arr[idx] = old
        return (loss_plus - loss_minus) / (2 * eps)

    def report(name, ana, num):
        rel = abs(ana - num) / (abs(ana) + abs(num) + 1e-12)
        status = '✓' if rel < atol else '✗'
        print(f'{name:>10} | {ana:>14.6f} | {num:>14.6f} | {rel:>10.2e} {status}')
        return rel < atol

    all_ok = True

    # 检 W 的前 3 个 (覆盖 C_out=0..2, C_in=0)
    for i in range(min(3, W.shape[0])):
        idx = (i, 0, 0, 0)
        all_ok &= report(f'W{idx}', grad_W_ana[idx], num_grad(W, idx))

    # 检 b 的前 2 个
    for i in range(min(2, b.shape[0])):
        all_ok &= report(f'b[{i}]', grad_b_ana[i], num_grad(b, (i,)))

    # 检 X 的前 3 个 (覆盖 N=0, 不同 C_in 和位置)
    test_X_idx = [(0, 0, 1, 1), (0, 1, 2, 2), (0, 0, 3, 3)]
    for idx in test_X_idx[:min(3, X.shape[0]*X.shape[1])]:
        all_ok &= report(f'X{idx}', grad_X_ana[idx], num_grad(X, idx))

    return all_ok


# ─────────────────────────────────────────────
# 4. CLI 自检 — 跑 3 个常见配置, 全部 ✓ 才算实现正确
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 62)
    print('Week 2 T7: conv2d_numpy 梯度检验')
    print('=' * 62)

    np.random.seed(42)
    # 小测试用例: batch=2, RGB, 8x8 输入; 4 个 3x3 filter
    X = np.random.randn(2, 3, 8, 8).astype(np.float32)
    W = (np.random.randn(4, 3, 3, 3) * 0.1).astype(np.float32)
    b = (np.random.randn(4) * 0.1).astype(np.float32)

    results = []
    results.append(('valid (p=0, s=1)', gradient_check(X, W, b, padding=0, stride=1)))
    results.append(('same  (p=1, s=1)', gradient_check(X, W, b, padding=1, stride=1)))
    results.append(('down  (p=1, s=2)', gradient_check(X, W, b, padding=1, stride=2)))

    print('\n' + '=' * 62)
    print('总结:')
    for name, ok in results:
        print(f'  {name}: {"✓ 全部通过" if ok else "✗ 有失败项"}')
    print('=' * 62)
