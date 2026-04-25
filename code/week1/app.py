"""
Week 1 T8 拓展：手绘数字识别 demo

启动:
  MPLCONFIGDIR=/tmp/mplconfig python code/week1/app.py
浏览器访问命令行打印的 http://127.0.0.1:7860
"""

import os
# 系统若设了 http_proxy（Clash/V2Ray 等），Gradio 6 的启动健康检查会被代理拦截返回 502
# 直接清掉本进程内的代理变量，仅影响这个 app，不动你的全局代理设置
for _k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
    os.environ.pop(_k, None)
os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'

import numpy as np
import gradio as gr

from inference import load_model, predict

PARAMS = load_model()
LABELS = [str(i) for i in range(10)]


def infer(sketch):
    """Gradio 回调：sketch 是 Sketchpad 输出。"""
    empty_msg = '请在左边用鼠标画一个数字'
    if sketch is None:
        return {l: 0.0 for l in LABELS}, None, empty_msg

    # Gradio 5/6 的 Sketchpad 返回 dict {'background', 'layers', 'composite'}
    # composite 才是用户实际看到的合成图
    img = sketch['composite'] if isinstance(sketch, dict) else sketch
    if img is None:
        return {l: 0.0 for l in LABELS}, None, empty_msg

    arr = np.asarray(img)
    # 提前判一下是不是空画布。RGBA 看 alpha，灰度/RGB 看亮度方差
    if arr.ndim == 3 and arr.shape[2] == 4:
        has_ink = arr[:, :, 3].max() > 10
    else:
        has_ink = arr.std() > 1.0
    if not has_ink:
        return {l: 0.0 for l in LABELS}, None, empty_msg

    pred, probs, x_view = predict(arr, PARAMS)
    label_dict = {LABELS[i]: float(probs[i]) for i in range(10)}
    preview    = (x_view * 255).astype(np.uint8)
    msg        = f'预测: **{pred}** &nbsp;&nbsp; 置信度: {probs[pred]:.1%}'
    return label_dict, preview, msg


with gr.Blocks(title='MNIST 手绘识别') as demo:
    gr.Markdown('# 手绘数字识别 — Week 1 MLP')
    gr.Markdown(
        '在左边画一个 0–9 的数字，模型会实时识别。\n\n'
        '> 训练集是 MNIST（黑底白字、按重心居中），所以代码里做了一套'
        '"反色 → bbox → 等比缩 20px → 重心居中"的预处理来对齐分布。'
        '中间那张 28×28 灰度图就是模型实际"看到"的输入。'
    )

    with gr.Row():
        with gr.Column(scale=1):
            sketch = gr.Sketchpad(
                label='画板（用鼠标画一个数字 0–9）',
                type='numpy',
                image_mode='RGBA',
                canvas_size=(280, 280),
                # Sketchpad 默认是白底，所以笔刷必须用黑色，否则用户看不见
                # preprocess() 里有自动反色逻辑，输入"白底黑字"也能正确处理
                brush=gr.Brush(default_size=18, colors=['#000000'], color_mode='fixed'),
                layers=False,
            )
            with gr.Row():
                btn_clear = gr.Button('清空', variant='secondary')
                btn_run   = gr.Button('识别', variant='primary')

        with gr.Column(scale=1):
            preview = gr.Image(
                label='模型实际看到的 28×28 输入',
                image_mode='L',
                height=280,
                width=280,
                interactive=False,
            )
            verdict = gr.Markdown('（等待识别）')
            probs   = gr.Label(num_top_classes=10, label='10 类概率')

    # 鼠标画完一笔就实时识别，按下"识别"也会触发
    sketch.change(infer, sketch, [probs, preview, verdict])
    btn_run.click(infer, sketch, [probs, preview, verdict])

    # Gradio 6 的 Sketchpad 直接传 None 不清画板，需要返回完整的 EditorValue 结构
    # 同时用 JS 触发 Sketchpad 内置的 "Erase" 按钮，双保险
    def _clear():
        return (
            {'background': None, 'layers': [], 'composite': None},
            None,
            {l: 0.0 for l in LABELS},
            '（已清空，请重新画一个数字）',
        )
    btn_clear.click(
        _clear, None, [sketch, preview, probs, verdict],
        js="""() => {
            // 找 Sketchpad 工具栏里那个 'Erase' / 'Clear' / 'Undo all' 图标按钮，模拟点击一下
            const btns = document.querySelectorAll(
                'button[aria-label*="rase" i], button[title*="rase" i], button[aria-label*="lear" i]'
            );
            btns.forEach(b => b.click());
        }"""
    )


if __name__ == '__main__':
    # share=True 会生成 72h 公网链接，本地玩不需要
    demo.launch(theme=gr.themes.Soft(), share=False)
