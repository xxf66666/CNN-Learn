# Week 2 任务拆分

> 目标：理解卷积要解决的问题，手写卷积/池化的前向+反向，并用 PyTorch 复现 LeNet-5 在 CIFAR-10 上做动物分类
> 数据集：**CIFAR-10**（32×32 RGB，10 类，其中 6 类是动物：bird / cat / deer / dog / frog / horse）
> 终极交付：MLP（Week 1 风格）vs LeNet（CNN）在同一数据集上的对比实验

---

## 任务列表

| 编号 | 任务 | 关键问题 | 状态 |
|------|------|----------|------|
| T1 | 为什么需要卷积 | MLP 在图像上死在哪？参数爆炸 + 看不到空间结构 + 没有平移不变性 | ⬜ |
| T2 | 卷积运算本身 | 互相关 vs 卷积；单通道单 filter 怎么算 | ⬜ |
| T3 | padding & stride | 输出尺寸公式 H'=(H+2p−k)/s + 1 怎么来的 | ⬜ |
| T4 | 多通道 + 多滤波器 | 输入 C 通道、K 个 filter 怎么组织 | ⬜ |
| T5 | 池化 + 感受野 | MaxPool / AvgPool 各自意义；感受野是如何叠加的 | ⬜ |
| T6 | 卷积反向传播 | dY 经过卷积层时梯度怎么往回传（卷积+翻转） | ⬜ |
| T7 | 手写实现 + 梯度检验 | conv2d_numpy + maxpool_numpy + grad check 复用 Week 1 套路 | ⬜ |
| T8 | PyTorch 思维切换 | autograd 是怎么把 Week 1 手写的 backward 自动化的 | ⬜ |
| T9 | LeNet-5 复现 + MLP 对比 | 在 CIFAR-10 上跑 LeNet，与同样训练的 MLP 比准确率和参数量 | ⬜ |
| 拓展 | LeNet 双模式 demo | Gradio UI: ① 测试集浏览 (分布内) + ② 上传识别 (分布外限制说明) | ⬜ |

---

## 学习方式说明

延续 Week 1 的"问题驱动 → 公式 → 代码"节奏：

- 每节先问"上一节留下什么没解决的问题"，再说本节怎么回答
- 数学先口语建立直觉，再写公式，最后用代码验证
- 写代码时严格保留 Week 1 的事实测试套路：`gradient_check` 必须通过

---

## 文件规划

```
docs/week2/
  00_tasks.md              ← 本文件
  01_why_conv.md           ← T1
  02_convolution.md        ← T2
  03_padding_stride.md     ← T3
  04_multi_channel.md      ← T4
  05_pooling.md            ← T5
  06_conv_backprop.md      ← T6
  07_thinking_log.md       ← 思考过程记录
  08_code_walkthrough.md   ← T7（代码走读）
  09_pytorch_intro.md      ← T8（PyTorch 思维切换）
  10_lenet_pytorch.md      ← T9（LeNet 实现 + MLP 对比实验）
  11_week2_summary.md      ← 汇报性总结（同 Week 1 §10 格式）
  12_lenet_demo.md         ← 拓展 demo 文档

code/week2/
  figures.py               ← 一键生成 10 张教学插图
  conv2d_numpy.py          ← 手写卷积前向 + 反向 + 梯度检验
  maxpool_numpy.py         ← 手写池化前向 + 反向
  lenet_pytorch.py         ← PyTorch LeNet-5 跑 CIFAR-10
  compare_mlp_vs_lenet.py  ← 拓展实验：同数据集下 MLP vs LeNet
  export_cifar_samples.py  ← 拓展 demo: 导出 100 张测试集 PNG
  inference.py             ← 拓展 demo: 加载 + 预处理 + 预测
  app.py                   ← 拓展 demo: Gradio 双 tab UI

assets/week2/
  outputs/                 ← 训练曲线、混淆矩阵、对比图
  figures/<chapter>/       ← 各章插图（卷积/池化/感受野动图等）
```

---

## 与 Week 1 的关系

Week 1 的产出在本周会被反复对照：

| Week 1 产物 | Week 2 怎么用 |
|---|---|
| `mlp_numpy.py::forward/backward` | T6 卷积反向传播会和它做镜像对比 |
| `gradient_check()` 的写法 | T7 直接复用同一套数值梯度对比逻辑 |
| `inference.py` 的预处理思路 | T9 中给 LeNet 写预处理时复用 |
| 训练好的 MLP 权重 | T9 对比实验里直接拉来比准确率 |
| MNIST 97.53% 的基线 | T9 对比 CIFAR-10 上 MLP 的成绩——预期会暴跌 |

**Week 2 不是另起炉灶，是在 Week 1 的脚手架上加卷积。**
