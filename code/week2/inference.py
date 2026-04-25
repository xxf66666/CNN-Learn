"""
Week 2 LeNet 推理 + 预处理 (供 app.py 使用)

核心难点 (跟 Week 1 inference.py 完全不同):
  Week 1 的 MNIST demo 主要解决的是 "用户手绘 vs MNIST 训练分布" 的差异,
  Week 2 LeNet 训练在 CIFAR-10 (32×32 RGB 中心物体简单背景), 真实用户上传
  的高清自然照片跟训练分布差距极大. 所以这里的预处理只能做到:
    1. 把任意图压成 32×32
    2. 用训练时的均值/标准差归一化
  分布差异本身是无法通过预处理消除的 - 这是模型容量和训练数据决定的.
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from lenet_pytorch import LeNet, OUTPUT_DIR, CIFAR_CLASSES, ANIMAL_CLASSES, pick_device


WEIGHTS_PATH = os.path.join(OUTPUT_DIR, 'lenet_weights.pth')

# 跟训练时完全一致的预处理 (lenet_pytorch.py::load_cifar10)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD  = (0.5, 0.5, 0.5)


# ─────────────────────────────────────────────
# 1. 加载训练好的 LeNet
# ─────────────────────────────────────────────

def load_model(path: str = WEIGHTS_PATH, device=None):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'找不到权重文件 {path}\n'
            f'请先跑训练: python code/week2/lenet_pytorch.py'
        )
    if device is None:
        device = pick_device()
    model = LeNet()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model = model.to(device).eval()
    return model, device


# ─────────────────────────────────────────────
# 2. 预处理: PIL Image → 模型输入 tensor
# ─────────────────────────────────────────────

# 标准 transform (跟训练时一致)
_TRANSFORM = T.Compose([
    T.Resize((32, 32)),                                # 任意尺寸 → 32×32
    T.ToTensor(),                                       # PIL → (3, 32, 32) in [0, 1]
    T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),         # 平移到约 [-1, 1]
])


def preprocess(pil_img: Image.Image):
    """
    输入: 任意尺寸的 PIL Image (RGB / RGBA / L 都行)
    输出:
      tensor: (1, 3, 32, 32) float32, 已归一化, 直接喂模型
      view_32:  (32, 32, 3) uint8, 模型实际"看到"的 32×32 图, 供 UI 显示
    """
    # 统一到 RGB (RGBA / L 都转过去)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # 标准 pipeline
    tensor = _TRANSFORM(pil_img).unsqueeze(0)         # 加 batch 维, (1, 3, 32, 32)

    # 单独算一遍"未归一化"的 32×32 给 UI 看 (Normalize 后的负值不能直接显示)
    resized = pil_img.resize((32, 32), Image.BILINEAR)
    view_32 = np.array(resized, dtype=np.uint8)        # (32, 32, 3)

    return tensor, view_32


# ─────────────────────────────────────────────
# 3. 预测接口
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(pil_img: Image.Image, model, device):
    """
    返回:
      pred_idx:   int, 预测类别索引 (0-9)
      pred_name:  str, 类别名 (e.g., 'horse')
      probs:      np.ndarray (10,), softmax 概率
      view_32:    np.ndarray (32, 32, 3), 模型实际看到的图
    """
    tensor, view_32 = preprocess(pil_img)
    tensor = tensor.to(device)
    logits = model(tensor)                             # (1, 10)
    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_name = CIFAR_CLASSES[pred_idx]
    return pred_idx, pred_name, probs, view_32


# ─────────────────────────────────────────────
# 4. CLI 自检
# ─────────────────────────────────────────────

if __name__ == '__main__':
    """跑一下: 用导出的 horse 第 0 张验证 inference 链路通畅."""
    print('加载 LeNet 权重...')
    model, device = load_model()
    print(f'  device: {device}')

    sample_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'assets', 'week2', 'samples', 'horse', '00.png',
    )
    print(f'\n加载测试样本: {sample_path}')
    img = Image.open(sample_path)
    print(f'  shape: {img.size} mode: {img.mode}')

    pred_idx, pred_name, probs, view_32 = predict(img, model, device)
    print(f'\n预测: [{pred_idx}] {pred_name}')
    print(f'置信度: {probs[pred_idx]:.2%}')
    print(f'\n概率分布:')
    for i, name in enumerate(CIFAR_CLASSES):
        marker = ' ★' if i == pred_idx else ''
        is_animal = ' (animal)' if i in ANIMAL_CLASSES else ''
        print(f'  [{i}] {name:12s}: {probs[i]:.2%}{marker}{is_animal}')
