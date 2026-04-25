# Week 1 拓展：手绘数字识别 demo

> 把训练好的 MLP 接到一个 web 画板上，用鼠标画数字让它实时识别。
> 对应文件：`code/week1/inference.py`、`code/week1/app.py`。

---

## 1. 这一节的真正目的

代码加 UI 不是重点，重点是会撞上一道在 ML 圈最常见、也最让人沮丧的坎：

> **训练时 97.53% 的模型，到了真实输入上准确率会跌到 50% 上下。**

不是模型坏了，是 **训练分布 ≠ 推理分布**。这一节的全部精华就在于看清这个 gap，然后用一段预处理把它补上。

| MNIST 训练集长什么样 | 鼠标画板的默认输入长什么样 |
|---|---|
| **黑底白字**（背景=0，笔画≈255） | 白底黑字（要反色） |
| 笔画粗细约 2~3 px（28×28 下） | 鼠标笔画太细或太粗 |
| 数字按**重心**居中，留 4 px 边距 | 画在画布任意位置 |
| 抗锯齿灰度过渡 | 缩放后可能锯齿 |
| 280×280 缩到 28×28 直接喂 → 灾难 | — |

**直接 resize 不做对齐，准确率会从 97% 掉到 30~50%**。这就是 ML 工程里"模型只是冰山一角，预处理才是水面下的 90%"那句话的来源。

---

## 2. 整条数据流

```mermaid
flowchart LR
    Sketch["浏览器画板<br/>280×280 RGBA"] --> Pre["preprocess()<br/>反色+bbox+缩放+重心居中"]
    Pre --> X["x &nbsp;[1, 784]<br/>归一化到 [0,1]"]
    X --> Fwd["forward()<br/>(复用训练时的实现)"]
    Fwd --> Soft["softmax → P [10]"]
    Soft --> UI["UI<br/>预测数字 + 概率柱状图<br/>+ 28×28 模型视角预览"]

    classDef io   fill:#1a1d27,stroke:#4fc3f7,color:#fff
    classDef proc fill:#0f1117,stroke:#66bb6a,color:#fff
    class Sketch,UI io
    class Pre,Fwd,Soft proc
    class X io
```

模型这一段（`forward → softmax`）一行没改，**全部新代码都在数据预处理上**——这本身就说明了问题在哪。

---

## 3. 预处理五步详解

`inference.py::preprocess()` 严格对齐 LeCun 1998 年制作 MNIST 时的预处理流程：

### ① 转灰度 + 自动反色到"黑底白字"

```python
if arr.ndim == 3:  # RGB 或 RGBA
    if arr.shape[2] == 4:  # 处理画板的 alpha 通道
        ...
    arr = 0.299*R + 0.587*G + 0.114*B  # 标准亮度公式

corners = mean(arr 四角)
if corners > arr.mean():
    arr = 255 - arr        # 背景亮 → 反色
```

**为什么要自动判断**：用户画板可能是白底黑笔，也可能是黑底白笔。用四角的平均亮度推断"哪边是背景"，比硬编码鲁棒得多。

### ② 阈值 + 找笔画的 bounding box

```python
mask = arr > 30       # ≈12% 亮度，过滤抗锯齿边缘的噪点
ys, xs = np.where(mask)
y0, y1 = ys.min(), ys.max() + 1
x0, x1 = xs.min(), xs.max() + 1
cropped = arr[y0:y1, x0:x1]
```

**作用**：把画布周围一大圈空白裁掉。如果用户在 280×280 画布的左上角画了一个小小的 30 像素的"3"，不裁直接缩到 28×28，那个 3 会变成 3 像素大的一坨——模型完全不认。

### ③ 等比缩到长边 = 20 像素

```python
if bh > bw:
    new_h, new_w = 20, round(bw * 20 / bh)
else:
    new_h, new_w = round(bh * 20 / bw), 20
```

**为什么是 20，不是 28**：MNIST 制作时数字本身就被规范化到 20×20，再放进 28×28 画布留 4 像素边距。这是图像分类里的硬约定——你看 §2.6 训练完那张 `predictions.png` 里每个数字四周都有空白，原因就在这。

### ④ 按"重心"放进 28×28 画布

```python
canvas = zeros(28, 28)
canvas[粗略居中位置] = resized
cy, cx = center_of_mass(canvas)        # 算笔画的灰度加权重心
canvas = shift(canvas, 14-cy, 14-cx)   # 把重心平移到 (14, 14)
```

**为什么是重心，不是 bounding box 中心**：考虑数字 "1" —— 它的 bounding box 中心和笔画重心明显错位（"1" 多数像素集中在右半部分）。LeCun 的论文明确说 MNIST 是按 **center of mass** 居中的，不按 bbox 中心。这一步做不对，"1" 的识别率会显著下降。

### ⑤ 归一化 + flatten

```python
canvas /= 255.0
return canvas.reshape(1, 784).astype(np.float32)
```

跟训练时 `load_images()` 里那行 `astype(np.float32) / 255.0` 完全对齐——数值范围、dtype、shape 三件事都要和训练时一字不差。

---

## 4. 为什么 UI 上要显示"模型实际看到的 28×28"

`app.py` 把预处理后的张量也渲染出来给用户看：

```python
preview = (x_view * 255).astype(np.uint8)  # 28×28 灰度
gr.Image(label='模型实际看到的 28×28 输入', ...)
```

**这是这个 demo 的灵魂功能**。当你画一个数字模型却预测错了，你需要立刻能判断：

- **预处理对了，模型不行** → 28×28 预览看着像一个正常的 MNIST 数字，但模型给的是错的概率分布。这是模型的锅（容量、训练不足、过拟合……）。
- **预处理不对，模型再好也没用** → 28×28 预览看上去就是一坨歪在角落里的细线，那不怪模型——MNIST 训练集里没见过这种东西。

没有这个预览，用户只会得出"模型很差"的结论；有了它，调试方向立刻分两条。

---

## 5. 启动方式

```bash
# 1. 先跑一次训练把权重存下来（首次必做）
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week1/mlp_numpy.py

# 2. 跑 inference 自检（确认权重加载 + forward 链路通）
python code/week1/inference.py
# 输出应有：✓ inference 链路 OK

# 3. 启动 Gradio
python code/week1/app.py
# 浏览器打开终端打印的 http://127.0.0.1:7860
```

权重文件路径：`assets/week1/outputs/mlp_weights.npz`，约 440 KB。

---

## 6. 已知的环境坑

### 6.1 系统代理导致 502

如果你装了 Clash / V2Ray / Surge 一类的全局代理（`http_proxy=http://127.0.0.1:7897` 这种），**Gradio 6 启动时的内部健康检查会被代理拦截，报：**

```
Exception: Couldn't start the app because
'http://127.0.0.1:7860/gradio_api/startup-events' failed (code 502).
```

`app.py` 顶部已经在本进程内 `os.environ.pop` 掉了 `http_proxy / https_proxy / all_proxy`，**不影响你系统全局代理设置**，只是让这个 Python 进程本身的请求绕过代理。如果你换别的网络栈（比如 SSL 翻墙工具），同样的思路加进去就行。

### 6.2 `RuntimeWarning: ... encountered in matmul`

详见 `08_code_walkthrough.md §4.1`——是 macOS Accelerate BLAS 的误报。`mlp_numpy.py` 顶部已加 `np.seterr(divide='ignore', over='ignore', invalid='ignore')` 屏蔽，控制台从此干净。

### 6.3 第一次画完没反应

Gradio 5/6 的 `Sketchpad` 在用户**画完一笔松开鼠标**才会触发 `change` 事件。如果你按住鼠标一直画，预测不会实时刷新——这是设计如此。点页面里的 **识别** 按钮可以强制触发一次。

---

## 7. 这个 demo 暴露的局限

跑起来玩一会儿就会发现：

1. **数字 "1" 经常被识别成 "7"**：你画的 1 多数有一个明显的起笔小钩，MNIST 训练集里大量样本是直竖一条，模型对"带钩的 1"就脆。
2. **画得潦草、连笔会崩**：MNIST 的笔迹相对工整，模型没见过艺术体。
3. **画粗了/画细了都有影响**：MNIST 笔画粗细是有典型分布的，远离这个分布准确率掉得很快。

这三个问题的本质都是**训练分布太窄**，对应的解药分别是：

- **数据增强**（旋转、平移、缩放、笔画粗细抖动）
- **更大的训练集**（EMNIST、QuickDraw）
- **更强的归纳偏置**（CNN 的平移不变性 → Week 2）

所以这个 demo 也是 Week 2 引入卷积的一个很自然的引子：当你看到 MLP 在自己手写的 1 上反复失败时，"为什么需要 CNN"这个问题就从抽象的教学口号变成切身的痛点了。
