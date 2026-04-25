# CNN 卷积神经网络学习计划

> 适合对象：有 C++ / Python 基础，首次系统学习 CNN
> 目标：理解 CNN 原理 → 手写核心模块 → 完成图像分类任务 → 输出理论+代码展示材料

---

## 目录结构

```
CNN-Learn/
├── docs/          # 学习笔记、计划、理论文档
├── code/          # 所有代码（按阶段子目录）
├── data/          # 数据集（本地存放或软链接）
└── assets/        # 图片、特征图可视化输出等
```

---

## 学习路线总览（4 周）

| 周次 | 主题 | 产出 |
|------|------|------|
| Week 1 | 数学基础 + 感知机 → MLP 回顾 | 笔记 + 手写 MLP |
| Week 2 | CNN 核心结构精讲 | 笔记 + 手写卷积/池化 |
| Week 3 | 经典网络复现 + 训练技巧 | LeNet / VGG 复现代码 |
| Week 4 | 完整项目 + 可视化 + 材料整理 | 图像分类项目 + 展示材料 |

---

## Week 1：数学基础与神经网络回顾

### 核心概念
- 矩阵乘法、链式法则、梯度下降
- 激活函数：Sigmoid / ReLU / Softmax
- 损失函数：CrossEntropy / MSE
- 反向传播（Backpropagation）推导

### 学习资源

**视频**
- [3Blue1Brown - 神经网络系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)（直觉建立，强烈推荐，共 4 集）
- [李沐 - 动手学深度学习 B站](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)（第 1-4 章）

**文档 / 书籍**
- 《动手学深度学习》(d2l.ai) 第 1-4 章，有中文版
- CS231n Lecture Notes: [Linear Classification](https://cs231n.github.io/linear-classify/)

### 实践任务
```
code/week1/
  ├── mlp_numpy.py       # 纯 numpy 手写两层 MLP，跑 MNIST
  └── backprop_demo.py   # 手推反向传播，打印梯度对比
```

---

## Week 2：CNN 核心结构精讲

### 核心概念
- 卷积操作（互相关 vs 卷积）、感受野
- Padding / Stride
- 池化层（MaxPool / AvgPool）
- 多通道卷积（输入 C 通道 → 输出 K 个 Feature Map）
- BatchNorm、Dropout

### 必读论文

| 论文 | 说明 | 链接 |
|------|------|------|
| LeCun et al. 1998 - LeNet | CNN 奠基之作 | [论文PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) |
| Krizhevsky et al. 2012 - AlexNet | 深度学习复兴之作，ImageNet冠军 | [arXiv](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
| Simonyan & Zisserman 2014 - VGGNet | 深而窄的设计思想 | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |
| He et al. 2015 - ResNet | 残差连接，解决梯度消失 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |

### 学习资源

**视频**
- [CS231n 2017 Lecture 5 - Convolutional Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo)（斯坦福，最权威）
- [李沐讲 AlexNet / VGG / ResNet](https://space.bilibili.com/1567748478)
- [Andrew Ng - CNN 课程（Coursera 第4课）](https://www.coursera.org/learn/convolutional-neural-networks)

**图解**
- [A guide to convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic)（动图理解卷积/转置卷积）
- [CNN Explainer 在线可视化](https://poloclub.github.io/cnn-explainer/)

### 实践任务
```
code/week2/
  ├── conv2d_numpy.py    # 纯 numpy 手写卷积前向传播
  ├── maxpool_numpy.py   # 手写最大池化
  └── lenet_pytorch.py  # PyTorch 复现 LeNet-5，跑 MNIST
```

---

## Week 3：经典网络复现 + 训练技巧

### 核心概念
- 数据增强（RandomCrop / Flip / Normalize）
- 学习率调度（StepLR / CosineAnnealing）
- 权重初始化（Xavier / He）
- 过拟合诊断与正则化

### 推荐数据集

| 数据集 | 规模 | 适用阶段 | 获取方式 |
|--------|------|----------|----------|
| MNIST | 7万张，10类手写数字，28×28灰度 | Week 1-2 入门 | `torchvision.datasets.MNIST` |
| CIFAR-10 | 6万张，10类彩色，32×32 | Week 2-3 主力 | `torchvision.datasets.CIFAR10` |
| CIFAR-100 | 6万张，100类 | Week 3 进阶 | `torchvision.datasets.CIFAR100` |
| Tiny-ImageNet | 10万张，200类，64×64 | Week 4 挑战 | [下载地址](http://cs231n.stanford.edu/tiny-imagenet-200.zip) |
| Flower102 | 8189张，102类花卉 | 迁移学习演示 | `torchvision.datasets.Flowers102` |

> **课程作业推荐：CIFAR-10**，规模适中，10类，有大量benchmark可对比

### 开源项目参考

| 项目 | 说明 | 链接 |
|------|------|------|
| pytorch/vision | 官方 torchvision，含所有经典网络实现 | [GitHub](https://github.com/pytorch/vision) |
| kuangliu/pytorch-cifar | CIFAR-10 各网络跑分，代码极简 | [GitHub](https://github.com/kuangliu/pytorch-cifar) |
| d2l-ai/d2l-zh | 动手学深度学习配套代码 | [GitHub](https://github.com/d2l-ai/d2l-zh) |
| weiaicunzai/pytorch-cifar100 | CIFAR-100 完整训练框架 | [GitHub](https://github.com/weiaicunzai/pytorch-cifar100) |

### 实践任务
```
code/week3/
  ├── vgg_pytorch.py         # 复现 VGG-11，CIFAR-10
  ├── resnet_pytorch.py      # 复现 ResNet-18，CIFAR-10
  ├── train_utils.py         # 训练器封装（含 lr_scheduler）
  └── data_augmentation.py  # 数据增强对比实验
```

---

## Week 4：完整项目 + 可视化 + 材料整理

### 完整项目结构
```
code/project/
  ├── dataset.py      # 数据加载与增强
  ├── model.py        # 网络定义
  ├── train.py        # 训练主循环，保存 loss/acc 曲线数据
  ├── evaluate.py     # 测试集评估，混淆矩阵
  └── visualize.py    # 特征图可视化、Grad-CAM
```

### 可视化（课程展示必备）

**特征图可视化**
```python
# 注册 hook 提取中间层输出
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.conv1.register_forward_hook(get_activation('conv1'))
```

**Grad-CAM 热力图**
- 库：`pip install grad-cam`
- 项目：[jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

**训练曲线**
- 使用 `matplotlib` 绘制 loss / accuracy 曲线
- 或接入 `tensorboard`：`tensorboard --logdir=runs`

### 输出材料 checklist
- [ ] `docs/` 理论笔记（卷积/池化/BN/结构演进）
- [ ] 训练 loss + accuracy 曲线图 → `assets/`
- [ ] 各层 Feature Map 可视化图 → `assets/`
- [ ] Grad-CAM 热力图 → `assets/`
- [ ] 混淆矩阵 → `assets/`
- [ ] 完整可运行代码 → `code/project/`

---

## 环境配置

```bash
# 推荐用 conda 隔离环境
conda create -n cnn python=3.10
conda activate cnn

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy jupyter grad-cam tensorboard scikit-learn
```

> 没有 GPU：使用 CPU 跑 MNIST/CIFAR-10 的小网络完全可行；
> 或免费使用 [Google Colab](https://colab.research.google.com/) / [Kaggle Notebook](https://www.kaggle.com/code)（自带 T4 GPU）

---

## 推荐学习顺序

```
3Blue1Brown 视频(直觉) 
    → d2l 第1-5章(理论+代码) 
    → CS231n Lecture 5-7(深度)
    → 手写 numpy 卷积(理解本质)
    → PyTorch 复现 LeNet → VGG → ResNet
    → 完整 CIFAR-10 项目 + 可视化
```

---

## 参考资源汇总

| 类型 | 名称 | 说明 |
|------|------|------|
| 书 | 动手学深度学习 (d2l.ai) | 最佳中文入门书，代码+理论 |
| 书 | Deep Learning (Goodfellow) | 理论深度，第9章CNN |
| 课 | CS231n Stanford | 最权威的视觉深度学习课 |
| 课 | Andrew Ng CNN (Coursera) | 适合入门，讲解细致 |
| 课 | 李沐 B站 | 中文，论文精读+代码 |
| 论文 | LeNet/AlexNet/VGG/ResNet | 按顺序读，理解设计演进 |
| 工具 | PyTorch 官方文档 | 实践必备 |
| 工具 | Papers With Code | 查各数据集 SOTA |
