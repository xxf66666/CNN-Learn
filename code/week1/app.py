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


# ─────────────────────────────────────────────
# 推理结果 → 大字预测块 (主题自适应: 不指定背景色和文字色,
# 让 inherit/currentColor 跟随浏览器或 Gradio 主题, 浅色/深色都协调)
# ─────────────────────────────────────────────

def _verdict_html(pred=None, confidence=None, status='waiting'):
    if status == 'waiting':
        return """
<div style="text-align:center; padding:24px 12px;">
  <div style="font-size:13px; opacity:0.6; letter-spacing:1px; text-transform:uppercase;">
    等待识别
  </div>
  <div style="font-size:48px; font-weight:300; opacity:0.25; margin:8px 0;
              font-variant-numeric:tabular-nums;">—</div>
  <div style="font-size:13px; opacity:0.55;">
    在左边画一个 <b>0–9</b> 的数字, 松开鼠标自动识别
  </div>
</div>
"""
    if status == 'empty':
        return """
<div style="text-align:center; padding:24px 12px;">
  <div style="font-size:13px; color:#f57c00; letter-spacing:1px; text-transform:uppercase;
              font-weight:600;">画布为空</div>
  <div style="font-size:48px; font-weight:300; opacity:0.25; margin:8px 0;
              font-variant-numeric:tabular-nums;">?</div>
  <div style="font-size:13px; opacity:0.55;">
    请在左边画一个数字
  </div>
</div>
"""
    # status == 'done' — 根据置信度取色
    if confidence >= 0.9:
        accent = '#2e7d32'   # 高: 深绿
        accent_light = '#66bb6a'
        tag = '识别结果'
    elif confidence >= 0.5:
        accent = '#e65100'   # 中: 深橙
        accent_light = '#ff9800'
        tag = '识别结果 · 置信度一般'
    else:
        accent = '#c62828'   # 低: 深红
        accent_light = '#ef5350'
        tag = '识别结果 · 置信度较低'

    return f"""
<div style="text-align:center; padding:20px 12px;">
  <div style="font-size:13px; color:{accent}; letter-spacing:1px; text-transform:uppercase;
              font-weight:600;">{tag}</div>
  <div style="font-size:84px; font-weight:700; line-height:1.05;
              color:{accent_light}; margin:4px 0;
              font-variant-numeric:tabular-nums;">{pred}</div>
  <div style="font-size:14px; opacity:0.7; margin-top:4px;">
    置信度 <b style="font-variant-numeric:tabular-nums;">{confidence:.1%}</b>
  </div>
</div>
"""


# ─────────────────────────────────────────────
# 推理回调
# ─────────────────────────────────────────────

def infer(sketch):
    if sketch is None:
        return {l: 0.0 for l in LABELS}, None, _verdict_html(status='empty')

    img = sketch['composite'] if isinstance(sketch, dict) else sketch
    if img is None:
        return {l: 0.0 for l in LABELS}, None, _verdict_html(status='empty')

    arr = np.asarray(img)
    # 提前判一下是不是空画布. RGBA 看 alpha, 灰度/RGB 看亮度方差
    if arr.ndim == 3 and arr.shape[2] == 4:
        has_ink = arr[:, :, 3].max() > 10
    else:
        has_ink = arr.std() > 1.0
    if not has_ink:
        return {l: 0.0 for l in LABELS}, None, _verdict_html(status='empty')

    pred, probs, x_view = predict(arr, PARAMS)
    label_dict = {LABELS[i]: float(probs[i]) for i in range(10)}
    preview    = (x_view * 255).astype(np.uint8)
    return label_dict, preview, _verdict_html(pred=int(pred),
                                              confidence=float(probs[pred]),
                                              status='done')


# ─────────────────────────────────────────────
# UI 元素 (主题自适应)
# ─────────────────────────────────────────────

HEADER_MD = """
<div style="text-align:center; padding:4px 0;">
  <div style="font-size:28px; font-weight:700; letter-spacing:-0.3px;">
    手绘数字识别 · Week 1 MLP
  </div>
  <div style="margin-top:6px; font-size:13px; opacity:0.65;">
    纯 NumPy 手写两层 MLP &nbsp;·&nbsp;
    MNIST 测试准确率 <b style="color:#2e7d32;">97.5%</b> &nbsp;·&nbsp;
    每张图被预处理成 <b>28×28</b> 喂给模型
  </div>
</div>
"""

FOOTER_MD = """
<div style="text-align:center; padding-top:8px; font-size:11px; opacity:0.45;
            line-height:1.7;">
  完整说明: <code>docs/week1/09_handwriting_demo.md</code> &nbsp;·&nbsp;
  训练流程: <code>code/week1/mlp_numpy.py</code> &nbsp;·&nbsp;
  权重: <code>assets/week1/outputs/mlp_weights.npz</code>
</div>
"""


# ─────────────────────────────────────────────
# Gradio Blocks 布局
# ─────────────────────────────────────────────
#
# 三块构思:
#   ① 画板        - 左, 占 3/5 宽, 最大化绘图空间
#   ② 预测 + 视角 - 右, 占 2/5 宽, 大字预测在上, 小预览在下
#   ③ 概率分布   - 底部全宽横向铺开, 不挤压侧栏
#
# 整体高度让 ① 和 ② 等高 (equal_height=True), ③ 单独成行
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Gradio 6 Sketchpad 默认不激活画笔工具, 用户每次进/清空后
# 都得先点画笔图标. 用 JS 模拟点击让它"开箱即用".
# 黑色 + size 已通过 brush=gr.Brush(...) 锁定, 这里只解决"工具激活"问题.
# ─────────────────────────────────────────────

JS_ACTIVATE_BRUSH = """
() => {
    const findBtn = (keywords, exclude=[]) => {
        const all = document.querySelectorAll('button');
        for (const b of all) {
            const label = ((b.getAttribute('aria-label') || '') + ' ' +
                           (b.title || '') + ' ' +
                           (b.textContent || '')).toLowerCase();
            if (exclude.some(kw => label.includes(kw))) continue;
            if (keywords.some(kw => label.includes(kw))) return b;
        }
        return null;
    };
    // 找画笔工具按钮 (Draw / Pencil / Brush), 排除 erase / clear
    setTimeout(() => {
        const drawBtn = findBtn(
            ['draw', 'pencil', 'brush', '画笔', '绘'],
            ['erase', 'clear', 'rase']
        );
        if (drawBtn) drawBtn.click();
    }, 300);
}
"""

JS_CLEAR_AND_REACTIVATE = """
() => {
    const findBtn = (keywords, exclude=[]) => {
        const all = document.querySelectorAll('button');
        for (const b of all) {
            const label = ((b.getAttribute('aria-label') || '') + ' ' +
                           (b.title || '') + ' ' +
                           (b.textContent || '')).toLowerCase();
            if (exclude.some(kw => label.includes(kw))) continue;
            if (keywords.some(kw => label.includes(kw))) return b;
        }
        return null;
    };
    // ① 触发 Sketchpad 内置的 "Erase all" 清空 (用 aria-label 包含 'rase' 的按钮)
    document.querySelectorAll(
        'button[aria-label*="rase" i], button[title*="rase" i]'
    ).forEach(b => b.click());
    // ② 清空后立刻重新激活画笔, 用户不用再手动选
    setTimeout(() => {
        const drawBtn = findBtn(
            ['draw', 'pencil', 'brush', '画笔', '绘'],
            ['erase', 'clear', 'rase']
        );
        if (drawBtn) drawBtn.click();
    }, 200);
}
"""


with gr.Blocks(title='MNIST 手绘识别') as demo:

    gr.HTML(HEADER_MD)

    with gr.Row(equal_height=True):

        # ① 画板 (左, 大)
        with gr.Column(scale=3):
            sketch = gr.Sketchpad(
                label='画板 · 用鼠标画一个 0–9',
                type='numpy',
                image_mode='RGBA',
                canvas_size=(360, 360),
                # Sketchpad 默认白底, 笔刷必须用黑色否则用户看不见;
                # preprocess() 里有自动反色逻辑, "白底黑字" 也能正确处理
                brush=gr.Brush(default_size=22, colors=['#000000'], color_mode='fixed'),
                layers=False,
            )
            with gr.Row():
                btn_clear = gr.Button('清空画布', variant='secondary', size='lg')
                btn_run   = gr.Button('手动识别', variant='primary',  size='lg')

        # ② 预测 + 模型视角 (右, 紧凑)
        with gr.Column(scale=2):
            verdict = gr.HTML(_verdict_html(status='waiting'))
            preview = gr.Image(
                label='模型实际看到的 28×28 (经反色 + 重心居中预处理)',
                image_mode='L',
                height=200,
                width=200,
                interactive=False,
            )

    # ③ 10 类概率分布 (底部全宽)
    probs = gr.Label(num_top_classes=10, label='10 类概率分布')

    gr.HTML(FOOTER_MD)

    # ─────────────────────────────────────────
    # 事件绑定
    # ─────────────────────────────────────────

    sketch.change(infer, sketch, [probs, preview, verdict])
    btn_run.click(infer, sketch, [probs, preview, verdict])

    # Gradio 6 的 Sketchpad 直接传 None 不清画板, 需要返回完整 EditorValue
    # 同时用 JS 触发 Sketchpad 内置 "Erase" 按钮做双保险
    def _clear():
        return (
            {'background': None, 'layers': [], 'composite': None},
            None,
            {l: 0.0 for l in LABELS},
            _verdict_html(status='waiting'),
        )
    btn_clear.click(
        _clear, None, [sketch, preview, probs, verdict],
        js=JS_CLEAR_AND_REACTIVATE,
    )

    # 页面首次加载时, 自动激活画笔工具 — 用户进来直接画, 不用先点工具
    demo.load(None, None, None, js=JS_ACTIVATE_BRUSH)


if __name__ == '__main__':
    demo.launch(theme=gr.themes.Soft(primary_hue='blue'),
                share=False, server_port=7860)
