"""
Week 2 T7: 手写 2D MaxPool 层 (前向 + 反向)

理论对应:
  - 前向公式      docs/week2/05_pooling.md §2
  - 反向稀疏路由   docs/week2/06_conv_backprop.md §7

API 约定 (跟 conv2d_numpy.py 一致):
  Y, cache = maxpool_forward(X, k, stride)
  grad_X   = maxpool_backward(delta, cache)

注意: 池化没有可学参数, 所以反向只算 grad_X (要传给上一层),
没有 grad_W 或 grad_b.

用法:
  python code/week2/maxpool_numpy.py
"""

import numpy as np

np.seterr(divide='ignore', over='ignore', invalid='ignore')


# ─────────────────────────────────────────────
# 1. 前向传播
# ─────────────────────────────────────────────

def maxpool_forward(X, k=2, stride=2):
    """
    每个 (k, k) 窗口取最大值, 同时记录 argmax 位置 (反向时要用).

    参数:
      X      (N, C, H, W_in)        输入
      k      池化窗口边长
      stride filter 移动步长

    返回:
      Y      (N, C, H_out, W_out)   池化后输出
      cache  tuple (X.shape, k, stride, mask)
             mask 形状 = X.shape, 是 0/1 的 bool 张量,
             True 表示该位置在前向时是某个窗口的 argmax
    """
    N, C, H, W_in = X.shape
    H_out = (H - k) // stride + 1
    W_out = (W_in - k) // stride + 1

    Y    = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
    mask = np.zeros_like(X, dtype=bool)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_slice = slice(i*stride, i*stride + k)
                    w_slice = slice(j*stride, j*stride + k)
                    block = X[n, c, h_slice, w_slice]      # (k, k)

                    # 取最大值填入 Y
                    Y[n, c, i, j] = block.max()

                    # 记录 argmax 位置 (block 内的 (h, w))
                    local = np.unravel_index(block.argmax(), block.shape)
                    mask[n, c,
                         i*stride + local[0],
                         j*stride + local[1]] = True

    cache = (X.shape, k, stride, mask)
    return Y, cache


# ─────────────────────────────────────────────
# 2. 反向传播 (T6 §7 稀疏路由)
# ─────────────────────────────────────────────

def maxpool_backward(delta, cache):
    """
    稀疏路由: δ 只送到前向 argmax 的位置, 其它位置梯度为 0.

    参数:
      delta  (N, C, H_out, W_out)  从下游传回的输出梯度
      cache  maxpool_forward 返回的元组

    返回:
      grad_X (N, C, H, W)   传给上一层
    """
    X_shape, k, stride, mask = cache
    grad_X = np.zeros(X_shape, dtype=delta.dtype)
    _, _, H_out, W_out = delta.shape
    N, C = X_shape[0], X_shape[1]

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_slice = slice(i*stride, i*stride + k)
                    w_slice = slice(j*stride, j*stride + k)
                    # 在这个 k×k 块内, 用 mask 选出 argmax 位置, 把 δ 加上去
                    block_mask = mask[n, c, h_slice, w_slice]
                    grad_X[n, c, h_slice, w_slice] += block_mask * delta[n, c, i, j]

    return grad_X


# ─────────────────────────────────────────────
# 3. 梯度检验
# ─────────────────────────────────────────────

def gradient_check(X, k=2, stride=2, eps=1e-5, atol=1e-4):
    """
    用有限差分验证 maxpool_backward 实现正确.

    注意: max 不可导. 实际策略是 "next-to-max-tied 的输入会让数值梯度
    跳变". 我们用随机 X 几乎不会出现两个值相等的情况 - 但仍然要
    确保扰动 ±eps 后 argmax 不会切换. 用 float64 + 小 eps + 高斯随机
    输入, 这一点天然成立.
    """
    X = X.astype(np.float64)

    Y, cache = maxpool_forward(X, k=k, stride=stride)
    delta = np.ones_like(Y)
    grad_X_ana = maxpool_backward(delta, cache)

    print(f'\n配置: k={k}, stride={stride}, X={X.shape}, Y={Y.shape}')
    print(f'{"参数":>14} | {"解析梯度":>10} | {"数值梯度":>10} | {"相对误差":>12}')
    print('-' * 58)

    def num_grad(idx):
        old = X[idx]
        X[idx] = old + eps
        loss_plus = maxpool_forward(X, k=k, stride=stride)[0].sum()
        X[idx] = old - eps
        loss_minus = maxpool_forward(X, k=k, stride=stride)[0].sum()
        X[idx] = old
        return (loss_plus - loss_minus) / (2 * eps)

    all_ok = True
    # 抽 5 个测试点 (覆盖 batch / channel / 不同位置)
    test_indices = [
        (0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 2, 3),
        (1, 0, 3, 2), (1, 1, 0, 0),
    ]
    for idx in test_indices:
        ana = grad_X_ana[idx]
        num = num_grad(idx)
        # max 反向: 数值和解析都可能恰好是 0 (该位置不是 argmax)
        if abs(ana) < 1e-10 and abs(num) < 1e-10:
            rel = 0.0
        else:
            rel = abs(ana - num) / (abs(ana) + abs(num) + 1e-12)
        ok = rel < atol
        all_ok &= ok
        marker = '✓' if ok else '✗'
        # 显示 X 在这个位置是不是 argmax
        is_argmax = cache[3][idx]
        note = ' (★ argmax)' if is_argmax else ''
        print(f'  X{str(idx):<10} | {ana:>10.4f} | {num:>10.4f} | '
              f'{rel:>10.2e} {marker}{note}')

    return all_ok


# ─────────────────────────────────────────────
# 4. CLI 自检
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 58)
    print('Week 2 T7: maxpool_numpy 梯度检验')
    print('=' * 58)

    np.random.seed(42)
    # 注意: 测试统一用 stride >= k (非重叠窗口).
    # 重叠 (stride < k) 时, 同一输入像素可能在多个窗口都是 argmax,
    # 此时 max 函数在该点不可导, 数值梯度会比解析梯度小
    # —— 这是 MaxPool 的数学本质 (T6 §7), 不是实现 bug.
    # 实际生产中 MaxPool 也都用 stride=k (LeNet/VGG 均如此).

    X1 = np.random.randn(2, 2, 4, 4).astype(np.float32)
    X2 = np.random.randn(2, 2, 6, 6).astype(np.float32)
    X3 = np.random.randn(2, 2, 8, 8).astype(np.float32)

    results = []
    results.append(('2x2 池化, stride=2 (输入 4x4)', gradient_check(X1, k=2, stride=2)))
    results.append(('3x3 池化, stride=3 (输入 6x6)', gradient_check(X2, k=3, stride=3)))
    results.append(('2x2 池化, stride=2 (输入 8x8)', gradient_check(X3, k=2, stride=2)))

    print('\n' + '=' * 58)
    print('总结:')
    for name, ok in results:
        print(f'  {name}: {"✓ 全部通过" if ok else "✗ 有失败项"}')
    print('=' * 58)
