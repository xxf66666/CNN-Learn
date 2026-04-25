# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Nature

This is a **pedagogical learning project** following a 4-week roadmap from MLP fundamentals to a full CNN image-classification project. The current state is Week 1 only. The plan in `docs/00_learning_plan.md` is the source of truth for what each week should produce — keep new code aligned with that plan rather than introducing structure of your own invention.

**Hard constraint for Week 1**: implementations are pure NumPy by design. Do not introduce PyTorch, scikit-learn, or any autograd library into `code/week1/`. PyTorch enters in Week 2 (`lenet_pytorch.py`) and onward. The whole point of `mlp_numpy.py` is that forward + backward + grad-check are written by hand.

## Run Commands

The project assumes a Conda env named `cnn`:

```bash
conda activate cnn
python -m pip install -r requirements.txt
```

Run the Week 1 script **from the project root** (paths inside the script are relative to `__file__`, but `assets/` and `data/` are anchored at the repo root):

```bash
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python code/week1/mlp_numpy.py
```

The two env vars are non-negotiable on this machine:
- `MPLCONFIGDIR=/tmp/mplconfig` — matplotlib's default cache dir is not writable here, plotting will crash without this.
- `MPLBACKEND=Agg` — non-interactive backend; the script saves PNGs and calls `plt.show()`, which would block or fail on a headless run.

Always use `python -m pip` rather than bare `pip` — README documents this is a recurring issue where `pip` resolves to system Python instead of the conda env. If `ModuleNotFoundError: No module named 'numpy'` appears with `cnn` active, that's the cause.

There is **no test runner, linter, or build system**. Correctness for `mlp_numpy.py` is verified by the in-script `gradient_check()` (Section 10), which compares analytic vs. finite-difference gradients on `W3`; relative error `< 1e-4` is the pass criterion. Treat that check as the test suite — if you change `forward`/`backward`/`cross_entropy_loss`, gradient_check must still print `✓` for all rows.

## Code Architecture

`code/week1/mlp_numpy.py` is a single-file, top-to-bottom implementation organized as 10 numbered sections that mirror the docs:

| Section | Code | Doc |
|---|---|---|
| 1 | data loading (IDX format, gzip) | — |
| 2 | activations: ReLU, softmax (numerically stable, subtract max) | `docs/week1/05_mlp.md` |
| 3 | He init | `05_mlp.md` |
| 4 | forward, builds `cache` for backward | `01,02_*.md` (T2, T5) |
| 5 | cross-entropy loss | `03_loss_function.md` (T3) |
| 6 | backward, hand-derived | `06_backpropagation.md` §8 (T7) |
| 7 | SGD update | `04_gradient_descent.md` (T4) |
| 8 | training loop | — |
| 9 | three plots: training curve, predictions, W1 visualization | — |
| 10 | gradient check | — |

Code comments contain references like `T2`, `T7`, `对应 docs/week1/06_backpropagation.md 第8节`. **These references are load-bearing** — when modifying the math (changing the network shape, swapping activation, adding a layer), update the corresponding doc and its `T*` task ID, not just the code. The docs are the derivation; the code is the realization.

Network shape `784 → 128 → 64 → 10` and parameter naming `W1/b1/W2/b2/W3/b3` are referenced throughout `forward`, `backward`, `gradient_check`, and `plot_weight_visualization` (which reshapes `W1[:, i]` to 28×28). Changing the shape requires touching all of these, not just `init_params`.

Training-script outputs go to `assets/week1/outputs/` (`training_curve.png`, `predictions.png`, `weights_layer1.png`); doc illustrations live under `assets/week1/figures/<chapter>/`. Plots use a fixed dark theme (`#0f1117` figure, `#1a1d27` axes); when adding new plots in future weeks, follow the same palette so the assets look coherent across the project.

## Future Weeks (per `docs/00_learning_plan.md`)

When adding code for later weeks, place it in `code/weekN/`, docs in `docs/weekN/`, outputs in `assets/weekN/`. Week 2 adds `conv2d_numpy.py`, `maxpool_numpy.py` (still pure NumPy) and `lenet_pytorch.py` (first PyTorch file). Week 3 is full PyTorch (VGG, ResNet on CIFAR-10). Week 4 is the capstone project under `code/project/` with the multi-file split listed in the plan (`dataset.py`, `model.py`, `train.py`, `evaluate.py`, `visualize.py`).
