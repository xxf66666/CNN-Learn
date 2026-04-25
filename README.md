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

训练完成会顺带保存权重到 `assets/week1/outputs/mlp_weights.npz`，供下面的手绘 demo 加载使用。

## 手绘数字识别 demo（可选）

用 Gradio 起一个浏览器画板，鼠标手写一个数字让训好的 MLP 实时识别。

**前置条件**：先跑一次训练（上一节）生成 `mlp_weights.npz`。

启动 UI：

```bash
python code/week1/app.py
```

启动成功后控制台会打印：

```text
* Running on local URL:  http://127.0.0.1:7860
```

浏览器打开这个地址，**用鼠标在左边白色画板上画一个 0–9 的数字**，模型实时给出预测和 10 类概率分布。中间那张 28×28 灰度图就是模型实际"看到"的输入（经反色 + 重心居中等预处理后的结果）。

> **注意**：笔刷是黑色（默认），白底黑笔才能看见自己画的笔画。

可选：跑 inference 自检（确认权重加载和 forward 链路通畅，不启动 UI）：

```bash
python code/week1/inference.py
# 期待输出：✓ inference 链路 OK
```

关闭 UI：

```bash
pkill -f "python code/week1/app.py"
```

如果遇到 `Couldn't start the app ... 502` 错误，是系统 HTTP 代理（Clash / V2Ray 等）拦截 localhost 健康检查导致的，`app.py` 已经在本进程里清掉了相关代理变量，不影响系统全局设置。详细原理、预处理细节、为什么自己手写的数字准确率比测试集低，见 `docs/week1/09_handwriting_demo.md`。

## 学习路线

完整计划见：

```text
docs/00_learning_plan.md
```

Week 1 的理论推导和任务拆分见：

```text
docs/week1/
```
