# Week 3 任务拆分

> 目标：把 Week 2 LeNet 的 **62.4%** 推到现代水平 **90%+**，用 VGG-11 + ResNet-18 + 5 件训练技巧（数据增强 / BatchNorm / lr schedule / weight decay / dropout）。
>
> 数据集：**继续 CIFAR-10**（跟 Week 2 横向可比）
>
> 终极交付：**LeNet vs VGG vs ResNet** 三模型对比 + 训练技巧 **ablation study**（每次只关一个技巧看 acc 掉多少）

---

## 任务列表

| 编号 | 任务 | 关键问题 | 状态 |
|------|------|----------|------|
| T1 | 为什么 LeNet 不够 | Week 2 留下的 5 件事 + 把 90% 拆成"加什么涨多少"路线图 | ⬜ |
| T2 | 数据增强 | RandomCrop / Flip / Normalize / ColorJitter；"扩大训练分布"的精确含义 | ⬜ |
| T3 | BatchNorm | 标准化激活分布让深网能训；训练 vs 推理两套行为；顺带正则 | ⬜ |
| T4 | lr schedule | StepLR / CosineAnnealing / Warmup；训练后期为什么必须降 lr | ⬜ |
| T5 | 正则化 + 过拟合诊断 | weight decay / dropout / early stopping；train/val/test 三划分 | ⬜ |
| T6 | VGG 设计哲学 | 深而窄、全部 3×3 conv 堆叠；为什么两个 3×3 比一个 5×5 好 | ⬜ |
| T7 | ResNet 残差连接（核心） | 梯度消失/爆炸的工程解；跳连让 50+ 层可训；identity vs projection | ⬜ |
| T8 | VGG-11 + ResNet-18 复现 | 在 CIFAR-10 上跑两个网络（从 nn.Module 一字字翻论文） | ⬜ |
| T9 | Ablation + 三模型对比 | LeNet 62% → VGG ?% → ResNet ?%；每个技巧贡献几个百分点 | ⬜ |
| 拓展 | 三模型对比 demo | 同一张图喂 LeNet / VGG / ResNet，看预测一致性 + 置信度差异 | ⬜ |

---

## 学习方式说明

延续 Week 1/2 的 **"问题驱动 → 公式 → 代码"** 节奏：

- 每节先问"上一节留下了什么问题"，再说本节要回答什么
- 数学先口语建立直觉，再写公式，最后用代码验证
- **Week 3 引入工程实践维度**：每讲一个技巧（augment / BN / lr schedule …），都用一组 ablation 数字证明"它真的有用"

---

## 训练强度策略：Fast 模式优先

CIFAR-10 上冲 90%+ 的训练时间是 Week 2 的 5–10 倍。本周采用 **Fast 模式**：

| | 设置 | 单模型耗时 (Apple MPS) | 预期 acc |
|---|---|---|---|
| **Fast (本周采用)** | 30 epoch | ~30 min | VGG ~85% / ResNet ~88% |
| Full (可选) | 100+ epoch | ~2 h | VGG ~92% / ResNet ~94% |

Ablation study 5 组 × 30 epoch ≈ 2.5 h。**所有数字先用 Fast 模式跑、写到 docs**；想冲更高准确率的人后续可以自己跑 Full（脚本里留 `EPOCHS` 变量）。

---

## 文件规划

```
docs/week3/
  00_tasks.md                     ← 本文件
  01_why_lenet_not_enough.md      ← T1（动机）
  02_data_augmentation.md         ← T2
  03_batchnorm.md                 ← T3
  04_lr_schedule.md               ← T4
  05_regularization.md            ← T5
  06_vgg_design.md                ← T6
  07_residual_connection.md       ← T7（Week 3 数学/直觉密度最高）
  08_thinking_log.md              ← 思考过程记录
  09_code_walkthrough.md          ← T8 代码走读
  10_ablation_experiment.md       ← T9 实验解读
  11_week3_summary.md             ← 汇报性总结
  12_three_model_demo.md          ← 拓展 demo

code/week3/
  figures.py                      ← 教学插图（数据增强对比 / 残差块 / 训练曲线对比）
  data_augmentation.py            ← T2 demo: 同一张图增强后多版本可视化
  train_utils.py                  ← 公用训练循环（含 augment / schedule / checkpoint）
  vgg_pytorch.py                  ← T8: VGG-11 实现 + 训练
  resnet_pytorch.py               ← T8: ResNet-18 实现 + 训练
  ablation_experiment.py          ← T9: 5 组训练（baseline + 逐个加 trick）
  compare_three_models.py         ← T9 拓展: LeNet vs VGG vs ResNet
  inference.py + app.py           ← 拓展 demo（端口 7862，跟 Week 1/2 demo 错开）

assets/week3/
  figures/                        ← 教学插图
  outputs/                        ← 训练产出（曲线 / 对比图 / 权重）
```

---

## 与 Week 1/2 的关系

Week 3 严格在 Week 1/2 的脚手架上扩展：

| Week 1/2 产物 | Week 3 怎么用 |
|---|---|
| `code/week2/lenet_pytorch.py::LeNet` | 三模型对比里直接 import 作为基线 |
| `code/week2/lenet_pytorch.py::load_cifar10` | 训练数据 pipeline 可复用，只是加 augment transform |
| `code/week2/lenet_pytorch.py::train` | 改造成 `train_utils.py`，加 lr schedule / weight decay 钩子 |
| `assets/week2/outputs/lenet_weights.pth` | 三模型对比的 LeNet 权重直接复用 |
| Week 2 的 62.4% 基线 | T9 ablation 里作为"未加任何技巧"的对照 |

**Week 3 不是另起炉灶，是把 Week 2 的训练管线产业化：加 augment、加 BN、加 schedule、加 dropout、加深网络。**

---

## 与 Week 4 的衔接

Week 3 完成后，Week 4 的"完整项目 + 可视化"已经无障碍可推：

- Week 4 用 ResNet-18 作为骨干（Week 3 已经训好）
- 加 Grad-CAM 特征可视化
- 加混淆矩阵 + per-class precision/recall
- 完整工程化（dataset.py / model.py / train.py / evaluate.py / visualize.py 五件套）
- 做对外汇报材料（PPT / 技术博客 / 视频 demo）

所以 Week 3 的"产业化训练管线 + 90%+ 模型"是 Week 4 的直接基础。
