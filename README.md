# CNN-Learn

CNN-Learn 是一个从基础神经网络到卷积神经网络的学习项目。当前阶段重点是 Week 1：用纯 NumPy 手写一个 MLP，在 MNIST 手写数字数据集上跑通完整训练流程。

## 项目结构

```text
CNN-Learn/
├── code/          # 代码实现
│   └── week1/
│       └── mlp_numpy.py
├── data/          # 数据集
│   └── mnist/
├── docs/          # 学习笔记和推导文档
└── assets/        # 训练曲线、预测结果、权重可视化等输出图片
```

## 环境准备

本项目建议使用 Conda。假设你已经创建好了名为 `cnn` 的环境：

```bash
conda activate cnn
python -m pip install -r requirements.txt
```

如果还没有创建环境，可以使用：

```bash
conda create -n cnn python=3.12
conda activate cnn
python -m pip install -r requirements.txt
```

如果已经激活了 `cnn`，但仍然出现 `ModuleNotFoundError: No module named 'numpy'`，通常说明 `pip` 指向了系统 Python，而不是 Conda 环境。用下面命令检查：

```bash
which python
python -m pip -V
```

正确情况下，路径里应该包含类似：

```text
miniforge3/envs/cnn/
```

不要直接依赖 `pip install ...`，优先使用：

```bash
python -m pip install -r requirements.txt
```

## 运行 Week 1 MLP

从项目根目录执行：

```bash
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week1/mlp_numpy.py
```

说明：

- `MPLCONFIGDIR=/tmp/mplconfig` 用于避免 Matplotlib 默认缓存目录不可写的问题。
- `MPLBACKEND=Agg` 使用非交互绘图后端，适合在终端中直接运行并保存图片。

脚本会完成：

- 加载 MNIST 数据。
- 执行反向传播梯度检验。
- 训练一个 `784 -> 128 -> 64 -> 10` 的 MLP。
- 保存训练曲线、预测结果和第一层权重可视化图片。

## MNIST 数据

MNIST 是手写数字识别数据集，包含 `0-9` 共 10 类数字图片。每张图片大小为 `28x28`，在代码中会被拉平成 `784` 维向量。

当前项目的数据目录是：

```text
data/mnist/
```

如果数据文件不存在，`code/week1/mlp_numpy.py` 会尝试从网络下载。

## 预期输出

运行完成后会生成：

```text
assets/week1/outputs/training_curve.png
assets/week1/outputs/predictions.png
assets/week1/outputs/weights_layer1.png
```

在本地测试中，默认参数训练 20 个 epoch 后，测试集准确率约为 `97%`。

## 学习路线

完整计划见：

```text
docs/00_learning_plan.md
```

Week 1 的理论推导和任务拆分见：

```text
docs/week1/
```
