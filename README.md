# CNN-Learn

CNN-Learn 是一个从基础神经网络到卷积神经网络的学习项目，按"问题驱动 → 数学推导 → 手写代码 → 工业级实现 → 可玩 demo"的节奏推进。当前进度：

- **Week 1**：纯 NumPy 手写 MLP，MNIST 上 97.5% + Gradio 手绘画板 demo
- **Week 2**：纯 NumPy 手写卷积/池化（grad check 通过）+ PyTorch 复现 LeNet 在 CIFAR-10 上 62.4% + MLP-vs-CNN 对比实验 + LeNet 双模式 demo
- Week 3 / Week 4：见 `docs/00_learning_plan.md`

## 项目结构

```text
CNN-Learn/
├── code/
│   ├── week1/         MLP numpy + Gradio 手绘 demo
│   └── week2/         conv2d/maxpool numpy + LeNet PyTorch + Gradio demo + 教学插图
├── data/
│   ├── mnist/         源 .gz 文件（入仓库供复现）
│   └── cifar10/       源 .tar.gz 文件
├── docs/              学习笔记、理论推导、思考记录、汇报总结
└── assets/
    ├── week1/         训练曲线 + 预测可视化 + 权重热图
    └── week2/
        ├── figures/   10 张教学插图（CIFAR horse 演示）
        ├── outputs/   LeNet 训练产出（曲线 / 对比图 / 权重）
        └── samples/   100 张 CIFAR-10 测试集 PNG（每类 10 张, demo 用）
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

---

## Week 2：CNN 核心 + LeNet on CIFAR-10

Week 2 把 Week 1 的"手写一切"路径走完整套 CNN：先用纯 NumPy 实现卷积层和池化层（含反向传播 + gradient check 全过），然后切到 PyTorch 复现 LeNet-5 在 CIFAR-10 上训练（62.4% 测试准确率），最后用对比实验把"为什么需要 CNN"从口号变成数据。

### 1. 生成 10 张教学插图（首次会下载 CIFAR-10 ~163 MB 源 tar.gz）

```bash
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week2/figures.py
```

输出到 `assets/week2/figures/`，包含真实图片的边缘检测、RGB 通道分解、padding 覆盖热图、感受野扩张、卷积反向传播步骤等。`docs/week2/02-06_*.md` 各章节直接引用。

### 2. 验证 numpy 卷积/池化数学正确性

```bash
python code/week2/conv2d_numpy.py     # 期望: 24/24 ✓ (相对误差 1e-11 ~ 1e-13)
python code/week2/maxpool_numpy.py    # 期望: 15/15 ✓
```

跟 Week 1 §10 同样的 gradient_check 套路。注意内部强制 `float64`——Week 2 踩到的工程坑，见 `docs/week2/07_thinking_log.md`。

### 3. 训练 LeNet（约 8 分钟 on Apple MPS）

```bash
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week2/lenet_pytorch.py
```

会保存：

- `assets/week2/outputs/lenet_weights.pth` — 训好的 LeNet 权重 (~250 KB)
- `assets/week2/outputs/lenet_training_curve.png` — loss + 准确率曲线
- `assets/week2/outputs/lenet_per_class_acc.png` — 10 类柱图（动物用橙色突出）
- `assets/week2/outputs/lenet_history.json` — epoch-by-epoch 指标

预期末尾：`最终测试准确率: 61.xx%`、`其中动物类准确率: 56.xx%`。

### 4. 跑 MLP-vs-LeNet 对比实验（约 5 分钟）

```bash
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week2/compare_mlp_vs_lenet.py
```

同样训练设置下训一个 MLP（1.7M 参数）和一个 LeNet（62K 参数），分别在原测试集 + 平移 ±4 像素的测试集上评估。预期结果：

```
         |  原测试集 |  平移 ±4 px |  下降幅度
   MLP   |   55.0%  |   42.4%    |  -12.6%
  LeNet  |   62.4%  |   54.0%    |   -8.4%
```

LeNet 用 MLP 1/27 的参数赢了 7.4 个百分点，平移鲁棒性是 MLP 的 1.5 倍——把 `docs/week2/01_why_conv.md` 的"为什么需要 CNN"从口号变成数字。对比图存到 `assets/week2/outputs/mlp_vs_lenet_comparison.png`。

### 5. LeNet 双模式 demo（可选）

跟 Week 1 手绘 demo 配套，把训好的 LeNet 接到一个 Gradio 双 tab UI：

- **Tab 1 测试集浏览**：从 100 张 CIFAR-10 测试样本随机抽，看模型在分布内的表现（~62%）
- **Tab 2 上传识别**：用户上传任意图，顶部橙色提示框说明 32×32 / 10 类的训练分布限制，中间显示"模型实际看到的 32×32"

```bash
# 首次必做：训练（步骤 3）+ 导出 100 张 PNG
python code/week2/export_cifar_samples.py

# 自检 inference 链路
python code/week2/inference.py
# 期望: horse 第 0 张 → 预测 horse, 置信度 ~97%

# 启动 UI
python code/week2/app.py
# 浏览器打开 http://127.0.0.1:7861 (跟 Week 1 demo 7860 错开)
```

完整设计、跟 Week 1 demo 的对照、警告样式说明见 `docs/week2/12_lenet_demo.md`。

### 完整学习路径

每周的理论推导、代码走读、思考记录、汇报总结都按统一格式：

```text
docs/week2/
  00_tasks.md             任务总规划
  01_why_conv.md          T1 为什么需要卷积
  02_convolution.md       T2 卷积运算 (含真实图边缘检测)
  03_padding_stride.md    T3 padding/stride 与输出尺寸
  04_multi_channel.md     T4 多通道 + 多 filter
  05_pooling.md           T5 池化 + 感受野
  06_conv_backprop.md     T6 卷积反向传播 (Week 2 数学密度最高一节)
  07_thinking_log.md      学习思考 + 工程踩坑记录
  08_code_walkthrough.md  T7 numpy 实现走读
  09_pytorch_intro.md     T8 PyTorch 思维切换
  10_lenet_pytorch.md     T9 LeNet 实现 + 对比实验解读
  11_week2_summary.md     汇报性总结 (适合做 PPT / 知乎博客底稿)
  12_lenet_demo.md        拓展 demo 完整说明
```

---

## 学习路线

完整计划见：

```text
docs/00_learning_plan.md
```

Week 1 的理论推导和任务拆分见：

```text
docs/week1/
```
