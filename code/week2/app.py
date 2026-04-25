"""
Week 2 LeNet on CIFAR-10 — 双模式 Gradio Demo

Tab 1 (测试集浏览): 模型在分布内的真实表现 (~62% 准确率)
Tab 2 (上传识别):    模型在分布外的退化情况 + 醒目限制说明

启动:
  MPSCONFIGDIR=/tmp/mplconfig python code/week2/app.py
浏览器访问: http://127.0.0.1:7861
"""

import os
# 系统若设了 http_proxy (Clash/V2Ray), Gradio 6 启动会被代理拦截
# 直接清掉本进程内的代理变量
for _k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
           'all_proxy', 'ALL_PROXY'):
    os.environ.pop(_k, None)
os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'

import random
from pathlib import Path

import numpy as np
from PIL import Image
import gradio as gr

from inference import load_model, predict
from lenet_pytorch import CIFAR_CLASSES, ANIMAL_CLASSES


HERE        = Path(__file__).parent
ROOT        = HERE.parent.parent
SAMPLES_DIR = ROOT / 'assets' / 'week2' / 'samples'

# 全局加载一次模型 (启动时)
print('加载 LeNet 模型...')
MODEL, DEVICE = load_model()
print(f'  device: {DEVICE}')

# 收集所有样本图路径, 按类别组织
SAMPLES_BY_CLASS = {
    cls: sorted((SAMPLES_DIR / cls).glob('*.png'))
    for cls in CIFAR_CLASSES
}
ALL_SAMPLES = [(cls, p) for cls, ps in SAMPLES_BY_CLASS.items() for p in ps]


# ─────────────────────────────────────────────
# 共用: 推理 + 输出格式化
# ─────────────────────────────────────────────

def infer_and_format(pil_img, true_label=None):
    """
    返回 (preview_32_pil, probs_dict, verdict_md, big_view_pil).
    true_label: 若有真实标签, verdict 里加一个 ✓/✗.
    """
    pred_idx, pred_name, probs, view_32 = predict(pil_img, MODEL, DEVICE)

    # 概率字典 (Gradio Label 用)
    probs_dict = {f'{i}.{name}': float(probs[i])
                  for i, name in enumerate(CIFAR_CLASSES)}

    # 32×32 的"模型视角"图 (放大到 256×256 显示, 用 NEAREST 保留像素感)
    preview_pil = Image.fromarray(view_32, mode='RGB').resize(
        (256, 256), Image.NEAREST,
    )

    # 原图也放大 (上传场景下原图可能是大图, 这里统一展示成相同尺寸)
    big_pil = pil_img.resize((256, 256), Image.LANCZOS) \
        if min(pil_img.size) < 256 else pil_img

    # 文字判断
    is_animal = pred_idx in ANIMAL_CLASSES
    confidence = probs[pred_idx]
    animal_tag = ' 🐾 (动物)' if is_animal else ''

    if true_label is None:
        # 上传模式: 没有 ground truth
        verdict = (f'### 预测: **{pred_name}** {animal_tag}\n'
                   f'置信度: **{confidence:.1%}**')
    else:
        # 测试集模式: 有 ground truth, 显示对错
        correct = (pred_name == true_label)
        emoji = '✅' if correct else '❌'
        verdict = (f'### {emoji} 真实: **{true_label}** &nbsp;|&nbsp; '
                   f'预测: **{pred_name}** {animal_tag}\n'
                   f'置信度: **{confidence:.1%}**')

    return preview_pil, probs_dict, verdict, big_pil


# ─────────────────────────────────────────────
# Tab 1: 测试集浏览
# ─────────────────────────────────────────────

def random_sample(class_filter):
    """随机抽一张, 可选限定类别."""
    if class_filter == '随机 (全部 10 类)':
        cls, path = random.choice(ALL_SAMPLES)
    else:
        path = random.choice(SAMPLES_BY_CLASS[class_filter])
        cls = class_filter
    img = Image.open(path)
    preview, probs, verdict, big = infer_and_format(img, true_label=cls)
    return big, preview, probs, verdict


# ─────────────────────────────────────────────
# Tab 2: 上传识别
# ─────────────────────────────────────────────

def upload_predict(upload_img):
    if upload_img is None:
        return (None, None, {c: 0.0 for c in CIFAR_CLASSES},
                '### 请先上传一张图')
    # gr.Image(type='pil') 直接给 PIL Image
    preview, probs, verdict, big = infer_and_format(upload_img, true_label=None)
    return big, preview, probs, verdict


# ─────────────────────────────────────────────
# UI 构造
# ─────────────────────────────────────────────

WARNING_HTML = """
<div style="
    background: #1a1d27;
    border-left: 4px solid #ff8a65;
    border-radius: 4px;
    padding: 14px 18px;
    margin: 8px 0 16px 0;
    color: #f5f5f5 !important;
    font-size: 14px;
    line-height: 1.75;
">
<div style="font-size: 15px; font-weight: 600; color: #ff8a65 !important; margin-bottom: 8px;">
⚠️ 关于本模型的限制
</div>
<div style="color: #f5f5f5 !important;">
LeNet 训练在 <b style="color:#ffcc80 !important;">CIFAR-10 (32×32 RGB)</b>, 对真实高清照片识别效果有限. 几个要注意的点:
</div>
<ul style="margin: 8px 0; padding-left: 22px;">
<li style="color: #f5f5f5 !important; margin-bottom: 4px;">只覆盖 <b style="color:#fff !important;">10 个类别</b>: airplane / automobile / bird / cat / deer / dog / frog / horse / ship / truck. 其它东西会被强行归到这 10 类</li>
<li style="color: #f5f5f5 !important; margin-bottom: 4px;">训练图都是 <b style="color:#fff !important;">32×32 像素</b>, 物体居中, 背景简单. 高清照片压到 32×32 会丢大量信息</li>
<li style="color: #f5f5f5 !important;">测试集准确率 <b style="color:#ffcc80 !important;">62.4%</b> (Tab 1 可验证), 用户照片上一般会更低</li>
</ul>
<div style="color: #b0b0b0 !important; font-size: 13px; margin-top: 8px; font-style: italic;">
这是 Week 2 教学的真实展示 — 模型表现受训练分布严格限制. Week 3 用 ResNet + 数据增强会改善.
</div>
</div>
"""

INFO_HTML = """
<div style="
    background: #1a1d27;
    border-left: 4px solid #4fc3f7;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #f5f5f5 !important;
    font-size: 14px;
    line-height: 1.7;
">
<div style="color: #4fc3f7 !important; font-weight: 600; margin-bottom: 4px;">CIFAR-10 测试集浏览</div>
<span style="color: #f5f5f5 !important;">这里是测试集里的图 (每类 10 张, 共 100 张). 模型在这个分布上的表现接近论文水平 <b style="color:#81d4fa !important;">~62%</b>, 反复点 "随机抽一张" 可以看模型在不同类别上的强弱.</span>
</div>
"""


with gr.Blocks(title='LeNet CIFAR-10 Demo') as demo:
    gr.Markdown('# Week 2 · LeNet 在 CIFAR-10 上的图像分类 Demo')
    gr.Markdown(
        '训练好的 LeNet-5 模型 · 测试集准确率 **62.4%** · '
        '其中 6 类是动物 (bird / cat / deer / dog / frog / horse) · '
        '权重位于 `assets/week2/outputs/lenet_weights.pth`'
    )

    with gr.Tabs():
        # ─────────────────────────────────────
        # Tab 1: 测试集浏览
        # ─────────────────────────────────────
        with gr.Tab('① 测试集浏览 (分布内, 模型擅长)'):
            gr.HTML(INFO_HTML)
            with gr.Row():
                class_filter = gr.Dropdown(
                    choices=['随机 (全部 10 类)'] + CIFAR_CLASSES,
                    value='随机 (全部 10 类)',
                    label='选择类别',
                    scale=3,
                )
                btn_random = gr.Button('🎲 随机抽一张', variant='primary', scale=1)

            with gr.Row():
                with gr.Column(scale=1):
                    img_orig_t1 = gr.Image(
                        label='原图 (32×32, 显示放大到 256)',
                        type='pil', height=280, interactive=False,
                    )
                with gr.Column(scale=1):
                    img_view_t1 = gr.Image(
                        label='模型实际看到的 (32×32)',
                        type='pil', height=280, interactive=False,
                    )
                with gr.Column(scale=1):
                    verdict_t1 = gr.Markdown('（点上方按钮抽样）')
                    probs_t1 = gr.Label(num_top_classes=10, label='10 类概率')

            btn_random.click(
                random_sample, inputs=class_filter,
                outputs=[img_orig_t1, img_view_t1, probs_t1, verdict_t1],
            )

        # ─────────────────────────────────────
        # Tab 2: 上传识别
        # ─────────────────────────────────────
        with gr.Tab('② 上传识别 (分布外, 看模型怎么"翻车")'):
            gr.HTML(WARNING_HTML)
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.Image(
                        label='上传一张图片 (任意尺寸, 自动 resize 到 32×32)',
                        type='pil', height=320,
                    )
                    btn_predict = gr.Button('🔍 识别', variant='primary')

                with gr.Column(scale=1):
                    img_view_t2 = gr.Image(
                        label='⚡ 模型实际看到的 32×32 (灵魂)',
                        type='pil', height=320, interactive=False,
                    )

                with gr.Column(scale=1):
                    verdict_t2 = gr.Markdown('（上传图后点识别）')
                    probs_t2 = gr.Label(num_top_classes=10, label='10 类概率')

            # 隐藏 (上传 tab 不显示原图大图, 中间那栏是 32×32 视角)
            big_dummy = gr.Image(visible=False)

            btn_predict.click(
                upload_predict, inputs=upload,
                outputs=[big_dummy, img_view_t2, probs_t2, verdict_t2],
            )
            # 上传后自动触发一次预测
            upload.change(
                upload_predict, inputs=upload,
                outputs=[big_dummy, img_view_t2, probs_t2, verdict_t2],
            )


if __name__ == '__main__':
    demo.launch(theme=gr.themes.Soft(), share=False, server_port=7861)
