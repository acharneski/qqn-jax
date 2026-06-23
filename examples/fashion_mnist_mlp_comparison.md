# Fashion-MNIST / MNIST MLP Optimizer Comparison

A full-batch benchmark that pits **QQN** (Quadratic-path Quasi-Newton) against
three established baselines — **SGD**, **Adam**, and **Optax L-BFGS** — on a
configurable, *non-convex* multi-layer perceptron trained on (Fashion-)MNIST.

> Run it with:
>
> ```bash
> python examples/fashion_mnist_mlp_comparison.py
> ```

---

## 1. Overview

Unlike the linear softmax classifier in `mnist_comparison.py`, this script
trains a **two-or-more-layer fully-connected network with nonlinear hidden
activations**. The hidden nonlinearity makes the loss surface genuinely
non-convex — introducing saddle points, flat regions, and non-unique minima.
This is a much sterner test for the curvature-aware methods (QQN and L-BFGS).

The objective is framed as a **full-batch, deterministic** cross-entropy loss
(with optional L2 regularization). Keeping it full-batch makes the comparison
apples-to-apples for second-order methods: every optimizer sees the exact same
objective, the same initial parameters, and the same termination criteria.

### Why full-batch and non-convex?

A larger full-batch objective has a **richer, more anisotropic Hessian** —
precisely the regime where QQN's gradient + curvature-oracle blending along the
quadratic path is most competitive against L-BFGS. The deeper / wider the
network, the more ill-conditioned and compositional the curvature becomes,
widening the gap where second-order information pays off.

### Headline findings (from the design notes)

- QQN wins decisively on **iterations-to-target** (up to ~1.9x vs L-BFGS), and
  the speedup **widens monotonically as the target tightens**.
- The historical per-iteration cost penalty was *resolved* by moving to a
  richer, more anisotropic Hessian (larger batch + deeper network), which
  shrank the deep-oracle ms/it gap below the iteration advantage — converting
  the iteration win into a **wall-clock** win.
- The decisive enabler is the **deep L-BFGS oracle** itself (with a plain
  Armijo line search), *not* a cheaper line search.

### Hard-won negative lessons (documented as negative controls)

1. **Warm-started backtracking backfires** on this smooth surface — it *raises*
   the iteration count for little-to-no ms/it saving, a net wall-clock loss.
   The cheap-probe variants are retained only as documented negative controls
   with *tamed* (gentle) warm-starts.
2. **Spline (cubic-Hermite) variants diverge** to the chance solution on the
   `tanh,gelu,tanh` surface — the cubic model's stationary-point probes are
   untrustworthy near init. Spline variants are quarantined behind a
   gentle-Armijo guard and clearly marked as negative controls.

---

## 2. Model architecture

The network maps a flattened image vector through one or more hidden layers to
class logits:

```
x -> [W1, b1] -> act -> ... -> [Wk, bk] -> act -> [Wout, bout] -> logits
```

- Hidden layers apply the configured **activation function**.
- The **output layer is always linear** (produces logits).
- All parameters are stored in a **single flat vector** (laid out as
  `W_1, b_1, W_2, b_2, ..., W_L, b_L`) so they slot directly into both QQN and
  Optax, which operate on flat arrays.

Weight initialization is activation-aware:

- **He init** (`std = sqrt(2 / fan_in)`) for `relu` layers.
- **Glorot/Xavier-style init** (`std = sqrt(1 / fan_in)`) otherwise.

The default architecture is **width 256 × depth 3** hidden layers with mixed
`tanh,gelu` activations on a 10-class problem.

---

## 3. Configuration guide

All configuration is via **environment variables**. Defaults are chosen to
reproduce the headline experiment described in the script.

### 3.1 Dataset selection

| Variable  | Values                        | Default          | Description                          |
| --------- | ----------------------------- | ---------------- | ------------------------------------ |
| `DATASET` | `mnist`, `fashion_mnist`      | `fashion_mnist`  | Which corpus to train on.            |

```bash
DATASET=fashion_mnist python examples/fashion_mnist_mlp_comparison.py
```

> An unknown value falls back to `mnist` with a warning.

### 3.2 Dataset size

| Variable  | Type | Default | Description                                       |
| --------- | ---- | ------- | ------------------------------------------------- |
| `N_TRAIN` | int  | `25000` | Full-batch **training** subset size.              |
| `N_TEST`  | int  | `5000`  | Full-batch **test** subset size (for accuracy).   |

A larger full-batch objective has a richer, more anisotropic Hessian — the
regime where QQN's gradient + oracle blending is most competitive.

> **VRAM note:** the deep L-BFGS history materializes
> `f32[history, n_train, width]` JVP tensors, which can OOM a ~6.5 GiB GPU at
> very large batch sizes. The default 25k is chosen to keep evaluation
> dominant while staying VRAM-safe with the width-256 × depth-3 network.
> Lower `N_TRAIN` if you have less VRAM.

The subset is drawn as a **reproducible, class-balanced random sample** (seed
`0`) rather than the first-N examples, giving a better-conditioned and more
representative Hessian.

### 3.3 Network topology

Precedence (highest first):

1. `HIDDEN_SIZES` — explicit comma-separated widths.
2. `DEPTH` × `HIDDEN` — uniform-width network.
3. Default: width 256 × depth 3.

| Variable       | Type            | Default | Description                                                       |
| -------------- | --------------- | ------- | ----------------------------------------------------------------- |
| `HIDDEN_SIZES` | comma-int list  | (unset) | Explicit per-layer widths, e.g. `256,128,64`. Takes precedence.   |
| `HIDDEN`       | int             | `256`   | Width of each hidden layer (uniform-width mode).                  |
| `DEPTH`        | int             | `3`     | Number of hidden layers (uniform-width mode).                     |

Examples:

```bash
# 3-layer MLP with two hidden layers of width 128 and 64
HIDDEN_SIZES=128,64 python examples/fashion_mnist_mlp_comparison.py

# Uniform 4 hidden layers of width 128
DEPTH=4 HIDDEN=128 python examples/fashion_mnist_mlp_comparison.py

# Deep, tapering network
HIDDEN_SIZES=256,128,64 python examples/fashion_mnist_mlp_comparison.py
```

> Invalid `HIDDEN_SIZES` (non-positive or non-numeric) falls back to
> `DEPTH`/`HIDDEN`; invalid `HIDDEN`/`DEPTH` falls back to `[64]`.

### 3.4 Activation function(s)

| Variable     | Values (single or comma-list) | Default     | Description                       |
| ------------ | ----------------------------- | ----------- | --------------------------------- |
| `ACTIVATION` | see table below               | `tanh,gelu` | Hidden-layer activation(s).       |

Supported activations:

| Name        | Definition                              | Notes                                     |
| ----------- | --------------------------------------- | ----------------------------------------- |
| `relu`      | `max(0, x)`                             | Triggers He init.                         |
| `sigmoid`   | `1 / (1 + e^-x)`                        | Default fallback for unknown names.       |
| `sine`      | `sin(x)`                                | Periodic.                                 |
| `gaussian`  | `exp(-x^2)`                             | Localized, RBF-like bump.                 |
| `triangle`  | periodic triangle wave in `[-1, 1]`     | Piecewise-linear, periodic.               |
| `sawtooth`  | periodic ramp in `[-1, 1)`              | Periodic.                                 |
| `logabs`    | `sign(x) * ln(|x| + 1)`                 | Heavy-tailed, odd.                        |
| `tanh`      | `tanh(x)`                               | Bounded squashing.                        |
| `gelu`      | Gaussian Error Linear Unit              | Smooth ReLU-like.                         |
| `swish`     | `x * sigmoid(x)`                        | Smooth, non-monotonic (SiLU).             |
| `softplus`  | `ln(1 + e^x)`                           | Smooth ReLU approximation.                |
| `abs`       | `|x|`                                   | V-shaped, even.                           |
| `identity`  | `x`                                     | Linear (useful in mixes).                 |

**Single activation** (applied to every hidden layer):

```bash
ACTIVATION=relu python examples/fashion_mnist_mlp_comparison.py
```

**Mixed activations** — a comma-separated list assigns activations per hidden
layer. The list is **cycled** if shorter than the number of hidden layers (and
truncated if longer):

```bash
# layer 1: relu, layer 2: sine, layer 3: gaussian
ACTIVATION=relu,sine,gaussian python examples/fashion_mnist_mlp_comparison.py

# 4 hidden layers, activations cycle: tanh, gaussian, tanh, gaussian
ACTIVATION=tanh,gaussian DEPTH=4 python examples/fashion_mnist_mlp_comparison.py
```

> Unknown activation names fall back to `sigmoid` with a warning. The output
> layer is always linear regardless of `ACTIVATION`.

### 3.5 GPU / XLA tuning (set automatically)

The script sets these *before* importing JAX to avoid speculative
multi-GiB workspace allocations that can OOM small GPUs:

| Variable           | Default value         | Purpose                                              |
| ------------------ | --------------------- | ---------------------------------------------------- |
| `XLA_FLAGS`        | `--xla_gpu_autotune_level=0` | Disables the cuBLAS-Lt autotuner's parallel probing. |
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async`   | Falls back to host memory instead of hard OOM.       |

Both use `setdefault`, so you can override them in your environment if needed.

---

## 4. Data loading

The script attempts to load a real corpus, in order:

1. **TensorFlow / Keras** (`tensorflow.keras.datasets`).
2. **torchvision** (`torchvision.datasets`).
3. **Synthetic fallback** — Gaussian-blob "MNIST-like" data so the experiment
   always runs.

Install one of the backends to use real data:

```bash
# Option A — TensorFlow / Keras (ships both MNIST + Fashion-MNIST)
pip install tensorflow

# Option B — torchvision (ships both MNIST + Fashion-MNIST)
pip install torch torchvision
```

Images are flattened to shape `(N, 784)` and scaled to `float32` in `[0, 1]`.

---

## 5. Termination criteria (shared by every optimizer)

All optimizers stop under the **same** conditions, so the comparison is fair:

| Key           | Value     | Meaning                                                |
| ------------- | --------- | ------------------------------------------------------ |
| `f_target`    | `2.0e-2`  | Headline target loss. First crossing is the win point. |
| `gtol`        | `1.0e-8`  | Gradient-norm convergence tolerance.                   |
| `time_budget` | `150.0` s | Wall-clock cap (so deep stacks aren't truncated early).|
| `milestones`  | `1e0, 5e-1, 2e-1, 1e-1` | Loss levels for the convergence-rate profile. |

The headline `f_target` is deliberately pushed tighter (`2e-2`) into the regime
where QQN's curvature blend dominates hardest, while staying reachable within
the time budget.
### Overriding termination & training hyper-parameters
Every value above (and the baseline learning rates / regularization) is
overridable via environment variables, with the documented value as default:
| Variable         | Type             | Default                       | Description                                         |
| ---------------- | ---------------- | ----------------------------- | --------------------------------------------------- |
| `F_TARGET`       | float            | `2.0e-2`                      | Headline target loss (first-crossing win point).    |
| `GTOL`           | float            | `1.0e-8`                      | Gradient-norm convergence tolerance.                |
| `TIME_BUDGET`    | float (seconds)  | `150.0`                       | Wall-clock cap.                                     |
| `MILESTONES`     | comma-float list | `1.0,0.5,0.2,0.1`             | Loss levels for the convergence-rate profile.       |
| `TARGET_PROFILE` | comma-float list | `0.2,0.1,0.06,0.04,0.02`      | Targets for the target-sensitivity speedup curve.   |
| `MAXITER`        | int              | `1000000`                     | Hard iteration cap (shared by all optimizers).      |
| `L2`             | float            | `1.0e-4`                      | L2 ridge penalty on the flat parameter vector.      |
| `SGD_LR`         | float            | `0.05`                        | SGD baseline learning rate.                         |
| `ADAM_LR`        | float            | `0.01`                        | Adam baseline learning rate.                        |
| `SEED`           | int              | `42`                          | PRNG seed for the shared initial parameters.        |
```bash
# Tighter target with a shorter budget and a higher Adam LR
F_TARGET=1e-2 TIME_BUDGET=300 ADAM_LR=0.02 \
    python examples/fashion_mnist_mlp_comparison.py
```

### Target-sensitivity profile

To address selection-bias concerns, the speedup is reported as a *curve* across
multiple targets rather than a single point:

```
target_profile = (2.0e-1, 1.0e-1, 6.0e-2, 4.0e-2, 2.0e-2)
```

---

## Quick reference — common invocations

```bash
# Default headline experiment (Fashion-MNIST, 256x3, tanh,gelu)
python examples/fashion_mnist_mlp_comparison.py

# MNIST instead of Fashion-MNIST
DATASET=mnist python examples/fashion_mnist_mlp_comparison.py

# Deeper, narrower ReLU network
DEPTH=5 HIDDEN=128 ACTIVATION=relu python examples/fashion_mnist_mlp_comparison.py

# Explicit tapering topology with mixed activations
HIDDEN_SIZES=256,128,64 ACTIVATION=tanh,gelu,gaussian \
    python examples/fashion_mnist_mlp_comparison.py

# Smaller problem for a low-VRAM GPU
N_TRAIN=8000 N_TEST=2000 HIDDEN=128 DEPTH=2 \
    python examples/fashion_mnist_mlp_comparison.py
```