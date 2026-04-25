"""
Week 1 T8: 用纯 numpy 手写 MLP，在 MNIST 上训练
对应理论文档: docs/week1/01~06

网络结构: 784 → 128 → 64 → 10
激活函数: ReLU（隐藏层）+ Softmax（输出层）
损失函数: 交叉熵
优化器:   mini-batch SGD
"""

import numpy as np
import os, struct, gzip, urllib.request
import matplotlib.pyplot as plt

# macOS Accelerate (vecLib) BLAS 在 matmul 边角处理时会触发硬件 FPE 标志位
# 报"divide by zero / overflow / invalid"——是误报，结果数值正确（梯度检验可证）
# 详见 docs/week1/08_code_walkthrough.md §4.1
np.seterr(divide='ignore', over='ignore', invalid='ignore')

plt.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 1. 数据加载
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/mnist')

URLS = {
    'train_images': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
    'test_images':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels':  'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz',
}

def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(DATA_DIR, os.path.basename(url))
        if not os.path.exists(path):
            print(f'下载 {name}...')
            urllib.request.urlretrieve(url, path)
            print(f'  完成: {path}')

def load_images(filename):
    path = os.path.join(DATA_DIR, filename)
    with gzip.open(path, 'rb') as f:
        _, n, h, w = struct.unpack('>4I', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, h * w).astype(np.float32) / 255.0  # 归一化到 [0,1]

def load_labels(filename):
    path = os.path.join(DATA_DIR, filename)
    with gzip.open(path, 'rb') as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)

def load_data():
    download_mnist()
    X_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    X_test  = load_images('t10k-images-idx3-ubyte.gz')
    y_test  = load_labels('t10k-labels-idx1-ubyte.gz')
    print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')
    return X_train, y_train, X_test, y_test

# ─────────────────────────────────────────────
# 2. 激活函数（对应 docs/week1/05_mlp.md）
# ─────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    """ReLU 的导数：z>0 的位置为1，否则为0（对应 T7 的 𝟙[z>0]）"""
    return (z > 0).astype(np.float32)

def softmax(z):
    """数值稳定的 Softmax：先减最大值再算 exp（防止 overflow）"""
    z_shift = z - z.max(axis=1, keepdims=True)
    exp_z   = np.exp(z_shift)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# ─────────────────────────────────────────────
# 3. 参数初始化
# ─────────────────────────────────────────────

def init_params(layer_sizes, seed=42):
    """
    He 初始化：W ~ N(0, sqrt(2/n_in))
    适合 ReLU，防止初始梯度消失/爆炸
    """
    np.random.seed(seed)
    params = {}
    for i in range(len(layer_sizes) - 1):
        n_in  = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        params[f'W{i+1}'] = np.random.randn(n_in, n_out).astype(np.float32) * np.sqrt(2.0 / n_in)
        params[f'b{i+1}'] = np.zeros(n_out, dtype=np.float32)
    return params

# ─────────────────────────────────────────────
# 4. 前向传播（对应 T2, T5 的推导）
# ─────────────────────────────────────────────

def forward(X, params):
    """
    X:      (batch, 784)
    返回:   (logits, cache)
    cache 存储中间变量，反向传播时需要用
    """
    cache = {'X': X}

    # 隐藏层 1: z1 = X @ W1 + b1, h1 = ReLU(z1)
    z1 = X @ params['W1'] + params['b1']          # (batch, 128)
    h1 = relu(z1)
    cache['z1'], cache['h1'] = z1, h1

    # 隐藏层 2: z2 = h1 @ W2 + b2, h2 = ReLU(z2)
    z2 = h1 @ params['W2'] + params['b2']          # (batch, 64)
    h2 = relu(z2)
    cache['z2'], cache['h2'] = z2, h2

    # 输出层: z3 = h2 @ W3 + b3（logits，不加激活）
    z3 = h2 @ params['W3'] + params['b3']          # (batch, 10)
    cache['z3'] = z3

    return z3, cache

# ─────────────────────────────────────────────
# 5. 损失函数（对应 T3）
# ─────────────────────────────────────────────

def cross_entropy_loss(logits, y):
    """
    logits: (batch, 10)
    y:      (batch,) 整数标签
    返回:   标量损失值
    """
    batch = logits.shape[0]
    P     = softmax(logits)
    # 只取每个样本真实类别对应的概率
    correct_probs = P[np.arange(batch), y]          # (batch,)
    loss = -np.mean(np.log(correct_probs + 1e-12))  # 加 1e-12 防止 log(0)
    return loss, P

# ─────────────────────────────────────────────
# 6. 反向传播（对应 T7 完整推导）
# ─────────────────────────────────────────────

def backward(logits, y, P, cache, params):
    """
    手动实现反向传播，严格对应 docs/week1/06_backpropagation.md 第8节
    """
    batch = logits.shape[0]
    grads = {}

    # ① Softmax+CE 梯度：δ = P - P̂（对应 T7 公式）
    delta = P.copy()
    delta[np.arange(batch), y] -= 1          # 只有真实类的位置减1
    delta /= batch                            # 除以batch取平均

    # ② 输出层 W3, b3 的梯度（外积 δ · h2^T）
    grads['W3'] = cache['h2'].T @ delta      # (64, batch) @ (batch, 10) = (64, 10)
    grads['b3'] = delta.sum(axis=0)          # 对 batch 求和

    # ③ 梯度传回 h2（W3^T · δ）
    delta_h2 = delta @ params['W3'].T        # (batch, 64)

    # ④ 过 ReLU2：逐元素乘以 𝟙[z2>0]
    delta_z2 = delta_h2 * relu_grad(cache['z2'])   # ⊙ 操作

    # ⑤ 隐藏层 W2, b2 的梯度
    grads['W2'] = cache['h1'].T @ delta_z2   # (128, 64)
    grads['b2'] = delta_z2.sum(axis=0)

    # ⑥ 梯度传回 h1
    delta_h1 = delta_z2 @ params['W2'].T     # (batch, 128)

    # ⑦ 过 ReLU1
    delta_z1 = delta_h1 * relu_grad(cache['z1'])

    # ⑧ 第一层 W1, b1 的梯度
    grads['W1'] = cache['X'].T @ delta_z1    # (784, 128)
    grads['b1'] = delta_z1.sum(axis=0)

    return grads

# ─────────────────────────────────────────────
# 7. 参数更新（对应 T4 梯度下降）
# ─────────────────────────────────────────────

def update_params(params, grads, lr):
    """SGD: W = W - lr * ∂L/∂W"""
    for key in params:
        params[key] -= lr * grads[key]
    return params

# ─────────────────────────────────────────────
# 8. 训练循环
# ─────────────────────────────────────────────

def train(X_train, y_train, X_test, y_test,
          layer_sizes=(784, 128, 64, 10),
          lr=0.1, batch_size=256, epochs=20):

    params = init_params(layer_sizes)
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    n = X_train.shape[0]

    for epoch in range(epochs):
        # 每个 epoch 打乱数据
        idx = np.random.permutation(n)
        X_train, y_train = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]

            # 前向传播
            logits, cache = forward(X_batch, params)
            loss, P       = cross_entropy_loss(logits, y_batch)

            # 反向传播
            grads = backward(logits, y_batch, P, cache, params)

            # 参数更新
            params = update_params(params, grads, lr)

            epoch_loss += loss
            n_batches  += 1

        # 每个 epoch 评估
        train_acc = evaluate(X_train[:5000], y_train[:5000], params)
        test_acc  = evaluate(X_test, y_test, params)
        avg_loss  = epoch_loss / n_batches

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch+1:2d}/{epochs} | '
              f'Loss: {avg_loss:.4f} | '
              f'Train: {train_acc:.1%} | '
              f'Test: {test_acc:.1%}')

    return params, history

def evaluate(X, y, params):
    logits, _ = forward(X, params)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

# ─────────────────────────────────────────────
# 9. 可视化
# ─────────────────────────────────────────────

ASSET_DIR = os.path.join(os.path.dirname(__file__), '../../assets/week1/outputs')

def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f1117')

    epochs = range(1, len(history['train_loss']) + 1)

    for ax in axes:
        ax.set_facecolor('#1a1d27')
        for sp in ax.spines.values(): sp.set_color('#444')
        ax.tick_params(colors='#aaaaaa')

    axes[0].plot(epochs, history['train_loss'], color='#4fc3f7', lw=2.5, marker='o', ms=5)
    axes[0].set_title('训练损失曲线', color='white', fontsize=13)
    axes[0].set_xlabel('Epoch', color='#aaaaaa')
    axes[0].set_ylabel('Cross-Entropy Loss', color='#aaaaaa')
    axes[0].grid(alpha=0.15)

    axes[1].plot(epochs, history['train_acc'], color='#66bb6a', lw=2.5, marker='o', ms=5, label='训练集')
    axes[1].plot(epochs, history['test_acc'],  color='#ff8a65', lw=2.5, marker='s', ms=5, label='测试集')
    axes[1].set_title('准确率曲线', color='white', fontsize=13)
    axes[1].set_xlabel('Epoch', color='#aaaaaa')
    axes[1].set_ylabel('Accuracy', color='#aaaaaa')
    axes[1].legend(facecolor='#1a1d27', edgecolor='#555', labelcolor='white')
    axes[1].set_ylim(0.8, 1.0)
    axes[1].grid(alpha=0.15)

    plt.tight_layout()
    path = os.path.join(ASSET_DIR, 'training_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    print(f'曲线图已保存: {path}')
    plt.show()

def plot_predictions(X_test, y_test, params, n=20):
    """可视化预测结果：显示20张测试图+预测标签"""
    logits, _ = forward(X_test[:n], params)
    preds = logits.argmax(axis=1)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle('MNIST 测试集预测结果（绿=正确，红=错误）', color='white', fontsize=13)

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        correct = (preds[i] == y_test[i])
        color = '#66bb6a' if correct else '#ef5350'
        ax.set_title(f'预测:{preds[i]}\n真实:{y_test[i]}', color=color, fontsize=9)

    plt.tight_layout()
    path = os.path.join(ASSET_DIR, 'predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    print(f'预测图已保存: {path}')
    plt.show()

def plot_weight_visualization(params):
    """可视化 W1 的前64列（每列 reshape 成 28×28，就是第一层神经元学到的"模板"）"""
    W1 = params['W1']  # (784, 128)

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle('第一层权重可视化（每个格子=一个神经元学到的特征）', color='white', fontsize=13)

    for i, ax in enumerate(axes.flat):
        w = W1[:, i].reshape(28, 28)
        ax.imshow(w, cmap='RdBu_r', vmin=-w.std()*2, vmax=w.std()*2)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(ASSET_DIR, 'weights_layer1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    print(f'权重图已保存: {path}')
    plt.show()

# ─────────────────────────────────────────────
# 10. 梯度数值检验（验证反向传播正确性）
# ─────────────────────────────────────────────

def gradient_check(params, X, y, eps=1e-5):
    """
    用有限差分估算数值梯度，和反向传播梯度对比
    相对误差 < 1e-4 说明反向传播实现正确
    """
    logits, cache = forward(X, params)
    loss, P       = cross_entropy_loss(logits, y)
    grads         = backward(logits, y, P, cache, params)

    print('\n梯度检验（只检查 W3 的前5个元素）：')
    print(f'{"元素":>10} | {"解析梯度":>14} | {"数值梯度":>14} | {"相对误差":>12}')
    print('-' * 58)

    W3 = params['W3']
    for i in range(5):
        idx = (i, 0)
        old = W3[idx]

        W3[idx] = old + eps
        loss_plus, _ = cross_entropy_loss(forward(X, params)[0], y)

        W3[idx] = old - eps
        loss_minus, _ = cross_entropy_loss(forward(X, params)[0], y)

        W3[idx] = old

        num_grad = (loss_plus - loss_minus) / (2 * eps)
        ana_grad = grads['W3'][idx]
        rel_err  = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-12)

        status = '✓' if rel_err < 1e-4 else '✗'
        print(f'W3{str(idx):>6}   | {ana_grad:>14.8f} | {num_grad:>14.8f} | {rel_err:>10.2e} {status}')

# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 50)
    print('Week 1 T8: numpy MLP on MNIST')
    print('=' * 50)

    # 加载数据
    X_train, y_train, X_test, y_test = load_data()

    # 梯度检验（用小批量验证反向传播正确性）
    print('\n[ 步骤1 ] 梯度检验...')
    params_check = init_params((784, 128, 64, 10))
    gradient_check(params_check, X_train[:8], y_train[:8])

    # 训练
    print('\n[ 步骤2 ] 开始训练...')
    params, history = train(
        X_train, y_train, X_test, y_test,
        layer_sizes=(784, 128, 64, 10),
        lr=0.1,
        batch_size=256,
        epochs=20
    )

    # 保存权重（供 inference.py / app.py 使用）
    weights_path = os.path.join(ASSET_DIR, 'mlp_weights.npz')
    os.makedirs(ASSET_DIR, exist_ok=True)
    np.savez(weights_path, **params)
    print(f'\n权重已保存: {weights_path}')

    # 可视化
    print('\n[ 步骤3 ] 生成可视化图...')
    plot_training(history)
    plot_predictions(X_test, y_test, params)
    plot_weight_visualization(params)

    print(f'\n最终测试准确率: {history["test_acc"][-1]:.2%}')
