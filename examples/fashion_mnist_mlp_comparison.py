"""2-layer ReLU MLP benchmark: QQN vs SGD vs Adam vs L-BFGS.

Trains a small two-layer fully-connected ReLU network (one hidden layer)
on a subset of MNIST or Fashion-MNIST and compares the convergence
behaviour of QQN against three common baselines: SGD, Adam, and Optax's
L-BFGS.

Unlike the linear softmax classifier in ``mnist_comparison.py``, this
model is *non-convex* (a hidden ReLU layer), which is a much sterner test
for the second-order methods (QQN and L-BFGS): the loss surface now has
saddle points, flat regions, and non-unique minima. Framing it as a
*full-batch* deterministic objective keeps the comparison apples-to-apples
for the curvature-aware methods.

Dataset selection:
  Set the environment variable ``DATASET`` to either ``mnist`` (default)
  or ``fashion_mnist`` to choose which corpus to train on, e.g.:

      DATASET=fashion_mnist python examples/fashion_mnist_mlp_comparison.py
Network architecture selection:
   The hidden-layer topology is configurable via environment variables:
     HIDDEN_SIZES   Comma-separated list of hidden-layer widths, e.g.
                    ``HIDDEN_SIZES=128,64`` builds a 3-layer MLP with two
                    hidden layers of width 128 and 64. Takes precedence over
                    ``HIDDEN`` / ``DEPTH`` when set.
     HIDDEN         Width of each hidden layer (default 64). Used together
                    with ``DEPTH`` to build a uniform-width network.
     DEPTH          Number of hidden layers (default 1). Used together with
                    ``HIDDEN``.
   Examples:
       HIDDEN_SIZES=256,128,64 python examples/fashion_mnist_mlp_comparison.py
       DEPTH=3 HIDDEN=128 python examples/fashion_mnist_mlp_comparison.py
Activation function selection:
    The hidden-layer activation is configurable via the ``ACTIVATION``
    environment variable. Supported values: ``relu``, ``sigmoid`` (default),
    and ``sine``. The output layer is always linear (logits). Example:
        ACTIVATION=relu python examples/fashion_mnist_mlp_comparison.py


Data loading / install instructions:
  The script tries to load the chosen dataset via ``torchvision`` or
  ``tensorflow`` if available. To install one of these:

      # Option A — TensorFlow / Keras (ships both MNIST + Fashion-MNIST):
      pip install tensorflow

      # Option B — torchvision (ships both MNIST + Fashion-MNIST):
      pip install torch torchvision

  If neither is installed, the script falls back to a synthetic
  Gaussian-blob "MNIST-like" dataset so the experiment always runs, and
  prints a reminder of the install commands above.

Run with:  python examples/fashion_mnist_mlp_comparison.py
"""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

from qqn_jax import QQN
from qqn_jax.oracles import (
    LBFGSOracle,
    MomentumOracle,
    SecantOracle,
    AndersonOracle,
    Fallback,
)
from qqn_jax.regions import (
    BoxRegion,
    TrustRegion,
)


# --------------------------------------------------------------------------
# Install hint (shown when no real dataset backend is available)
# --------------------------------------------------------------------------

_INSTALL_HINT = (
    "[data] No dataset backend found. Install ONE of the following to use a\n"
    "       real (Fashion-)MNIST corpus instead of the synthetic fallback:\n"
    "           pip install tensorflow            # Keras datasets (MNIST + Fashion)\n"
    "           pip install torch torchvision     # torchvision datasets\n"
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def _load_dataset_numpy(dataset, n_train, n_test, n_classes):
    """Try to load a real (Fashion-)MNIST subset; fall back to synthetic.

    Args:
        dataset: ``"mnist"`` or ``"fashion_mnist"``.

    Returns:
        (X_train, y_train, X_test, y_test) as numpy arrays with images
        flattened to shape (N, 784) and float32 in [0, 1].
    """
    # --- Attempt 1: tensorflow / keras ---
    try:
        if dataset == "fashion_mnist":
            from tensorflow.keras.datasets import fashion_mnist as ds  # type: ignore
        else:
            from tensorflow.keras.datasets import mnist as ds  # type: ignore

        (xtr, ytr), (xte, yte) = ds.load_data()
        xtr = xtr.reshape(xtr.shape[0], -1).astype(np.float32) / 255.0
        xte = xte.reshape(xte.shape[0], -1).astype(np.float32) / 255.0
        print(f"[data] Loaded {dataset} via tensorflow.keras.")
        return _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes)
    except Exception:
        pass

    # --- Attempt 2: torchvision ---
    try:
        from torchvision import datasets  # type: ignore

        if dataset == "fashion_mnist":
            cls = datasets.FashionMNIST
            root = "./_fashion_mnist_data"
        else:
            cls = datasets.MNIST
            root = "./_mnist_data"

        train = cls(root=root, train=True, download=True)
        test = cls(root=root, train=False, download=True)
        xtr = train.data.numpy().reshape(len(train), -1).astype(np.float32) / 255.0
        ytr = train.targets.numpy()
        xte = test.data.numpy().reshape(len(test), -1).astype(np.float32) / 255.0
        yte = test.targets.numpy()
        print(f"[data] Loaded {dataset} via torchvision.")
        return _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes)
    except Exception:
        pass

    # --- Fallback: synthetic "MNIST-like" Gaussian blobs ---
    print(_INSTALL_HINT)
    print(f"[data] Real {dataset} unavailable; using synthetic Gaussian blobs.")
    return _synthetic(n_train, n_test, n_classes, dim=784)


def _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes):
    """Keep only the first ``n_classes`` classes and subsample."""
    train_mask = ytr < n_classes
    test_mask = yte < n_classes
    xtr, ytr = xtr[train_mask][:n_train], ytr[train_mask][:n_train]
    xte, yte = xte[test_mask][:n_test], yte[test_mask][:n_test]
    return xtr, ytr.astype(np.int32), xte, yte.astype(np.int32)


def _synthetic(n_train, n_test, n_classes, dim):
    """Generate a linearly-separable-ish synthetic classification set."""
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=3.0, size=(n_classes, dim)).astype(np.float32)

    def make(n):
        y = rng.integers(0, n_classes, size=n).astype(np.int32)
        x = centers[y] + rng.normal(scale=1.0, size=(n, dim)).astype(np.float32)
        return x.astype(np.float32), y

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte


# --------------------------------------------------------------------------
# Model: configurable multi-layer ReLU MLP
#   x -> [W1, b1] -> ReLU -> ... -> [Wk, bk] -> ReLU -> [Wout, bout] -> logits
#
# The parameter vector is a single flat array laying out, in order:
#   for each layer (in order): W_i (fan_in x fan_out), b_i (fan_out)
# so it slots directly into QQN / Optax which both operate on flat vectors.
#
# The full list of layer dimensions is:
#   [dim, hidden_1, hidden_2, ..., hidden_k, n_classes]
# --------------------------------------------------------------------------


def _layer_dims(dim, hidden_sizes, n_classes):
    """Return the full list of layer dimensions [dim, *hidden, n_classes]."""
    return [dim, *list(hidden_sizes), n_classes]


# --------------------------------------------------------------------------
# Activation function selection
# --------------------------------------------------------------------------
_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "sine": jnp.sin,
}


def _parse_activation():
    """Resolve the hidden-layer activation from the ``ACTIVATION`` env var.
    Supported values: ``relu``, ``sigmoid`` (default), ``sine``.
    Returns:
        A tuple ``(name, fn)`` of the activation name and callable.
    """
    raw = os.environ.get("ACTIVATION", "sigmoid").strip().lower()
    if raw not in _ACTIVATIONS:
        print(
            f"[config] Unknown ACTIVATION={raw!r}; falling back to 'sigmoid'. "
            f"Valid values: {', '.join(sorted(_ACTIVATIONS))}."
        )
        raw = "sigmoid"
    return raw, _ACTIVATIONS[raw]


def _param_layout(dim, hidden_sizes, n_classes):
    """Return cumulative offsets delimiting each W/b block in the flat vector.

    The blocks are laid out as W_1, b_1, W_2, b_2, ..., W_L, b_L where
    ``L = len(hidden_sizes) + 1`` is the number of weight layers.
    """
    dims = _layer_dims(dim, hidden_sizes, n_classes)
    sizes = [0]
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        sizes.append(fan_in * fan_out)  # W block
        sizes.append(fan_out)  # b block
    return np.cumsum(sizes)


def init_params(dim, hidden_sizes, n_classes, key, activation="sigmoid"):
    """Flat parameter vector for a multi-layer MLP.

    Uses He-style init for ReLU and Xavier/Glorot-style init otherwise
    (sigmoid/sine), which keeps activations well-scaled at init.
    """
    dims = _layer_dims(dim, hidden_sizes, n_classes)
    keys = jax.random.split(key, len(dims) - 1)
    blocks = []
    for k, fan_in, fan_out in zip(keys, dims[:-1], dims[1:]):
        if activation == "relu":
            # He initialization for ReLU: std = sqrt(2 / fan_in).
            scale = jnp.sqrt(2.0 / fan_in)
        else:
            # Glorot/Xavier-style init for sigmoid/sine.
            scale = jnp.sqrt(1.0 / fan_in)
        w = jax.random.normal(k, (fan_in * fan_out,)) * scale
        b = jnp.zeros((fan_out,))
        blocks.append(w)
        blocks.append(b)
    return jnp.concatenate(blocks)


def _unpack(params, dim, hidden_sizes, n_classes):
    """Split the flat vector into a list of (W, b) tuples, one per layer."""
    dims = _layer_dims(dim, hidden_sizes, n_classes)
    o = _param_layout(dim, hidden_sizes, n_classes)
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        w_start, w_end = o[2 * i], o[2 * i + 1]
        b_start, b_end = o[2 * i + 1], o[2 * i + 2]
        w = params[w_start:w_end].reshape(fan_in, fan_out)
        b = params[b_start:b_end]
        layers.append((w, b))
    return layers


def _forward(params, X, dim, hidden_sizes, n_classes, activation=jax.nn.sigmoid):
    layers = _unpack(params, dim, hidden_sizes, n_classes)
    h = X
    # Apply the activation after every layer except the final (output) layer.
    for i, (w, b) in enumerate(layers):
        h = h @ w + b
        if i < len(layers) - 1:
            h = activation(h)
    return h


def make_loss(X, y, dim, hidden_sizes, n_classes, l2=1e-4, activation=jax.nn.sigmoid):
    """Build a full-batch cross-entropy loss ``f(params) -> scalar``."""
    Y = jax.nn.one_hot(y, n_classes)

    def loss(params):
        logits = _forward(params, X, dim, hidden_sizes, n_classes, activation)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.mean(jnp.sum(Y * log_probs, axis=-1))
        reg = 0.5 * l2 * jnp.sum(params**2)
        return ce + reg

    return loss


def accuracy(params, X, y, dim, hidden_sizes, n_classes, activation=jax.nn.sigmoid):
    logits = _forward(params, X, dim, hidden_sizes, n_classes, activation)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == y).astype(jnp.float32))


def _parse_hidden_sizes():
    """Resolve the hidden-layer topology from environment variables.
    Precedence:
      1. ``HIDDEN_SIZES`` (comma-separated explicit widths).
      2. ``DEPTH`` x ``HIDDEN`` (uniform-width network).
      3. Default: a single hidden layer of width 64.
    Returns:
        A list of positive ints (may be empty for a pure linear model).
    """
    raw = os.environ.get("HIDDEN_SIZES")
    if raw is not None and raw.strip() != "":
        try:
            sizes = [int(tok) for tok in raw.split(",") if tok.strip() != ""]
            if any(s <= 0 for s in sizes):
                raise ValueError("hidden sizes must be positive")
            return sizes
        except ValueError as exc:
            print(
                f"[config] Invalid HIDDEN_SIZES={raw!r} ({exc}); "
                "falling back to DEPTH/HIDDEN."
            )
    try:
        hidden = int(os.environ.get("HIDDEN", "64"))
        depth = int(os.environ.get("DEPTH", "1"))
        if hidden <= 0 or depth < 0:
            raise ValueError
    except ValueError:
        print("[config] Invalid HIDDEN/DEPTH; using default [64].")
        return [64]
    return [hidden] * depth


# --------------------------------------------------------------------------
# Optimizer runners (shared termination logic)
# --------------------------------------------------------------------------


def _converged(value, gnorm, f_target, gtol):
    """Shared convergence test: target loss reached OR gradient ~ 0."""
    if f_target is not None and value <= f_target:
        return True
    if gtol is not None and gnorm <= gtol:
        return True
    return False


def _update_milestones(milestones, hit, value, it, now):
    """Record the first iteration/time each loss milestone is crossed."""
    if not milestones:
        return
    for m in milestones:
        if hit.get(m) is None and value <= m:
            hit[m] = (it, now)


def _grad_norm(loss_fn, params):
    g = jax.grad(loss_fn)(params)
    return float(jnp.linalg.norm(g))


def _run_qqn(loss_fn, params0, maxiter, stop=None, **qqn_kwargs):
    """Run a configurable QQN variant and return a standard result tuple."""
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    solver = QQN(loss_fn, maxiter=maxiter, **qqn_kwargs)
    state = solver.init_state(params0)
    params = params0
    history = [float(state.value)]
    times = [0.0]
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    _update_milestones(milestones, milestone_hits, history[-1], 0, 0.0)
    t0 = time.perf_counter()
    update = jax.jit(solver.update)
    for it in range(maxiter):
        params, state = update(params, state)
        history.append(float(state.value))
        now = time.perf_counter() - t0
        times.append(now)
        gnorm = _grad_norm(loss_fn, params)
        _update_milestones(milestones, milestone_hits, history[-1], it + 1, now)
        if iters_to_target is None and _converged(history[-1], gnorm, f_target, gtol):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
        if bool(state.done):
            break
    wall = time.perf_counter() - t0
    return (
        params,
        history,
        wall,
        times,
        iters_to_target,
        time_to_target,
        milestone_hits,
    )


def run_optax(loss_fn, params0, optimizer, maxiter, stop=None):
    """Run a generic Optax optimizer; returns a standard result tuple."""
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    opt_state = optimizer.init(params0)

    @jax.jit
    def step(params, opt_state):
        value, grad = value_and_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, jnp.linalg.norm(grad)

    params = params0
    history = [float(loss_fn(params))]
    times = [0.0]
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    _update_milestones(milestones, milestone_hits, history[-1], 0, 0.0)
    t0 = time.perf_counter()
    for it in range(maxiter):
        params, opt_state, value, gnorm = step(params, opt_state)
        history.append(float(value))
        now = time.perf_counter() - t0
        times.append(now)
        _update_milestones(milestones, milestone_hits, history[-1], it + 1, now)
        if iters_to_target is None and _converged(
            history[-1], float(gnorm), f_target, gtol
        ):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
    wall = time.perf_counter() - t0
    return (
        params,
        history,
        wall,
        times,
        iters_to_target,
        time_to_target,
        milestone_hits,
    )


def run_optax_lbfgs(loss_fn, params0, maxiter, stop=None):
    """Run Optax's L-BFGS (with zoom line search) on the full-batch loss."""
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    optimizer = optax.lbfgs()
    opt_state = optimizer.init(params0)

    @jax.jit
    def step(params, opt_state):
        value, grad = value_and_grad(params)
        updates, opt_state = optimizer.update(
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=loss_fn,
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, value, jnp.linalg.norm(grad)

    params = params0
    history = [float(loss_fn(params))]
    times = [0.0]
    iters_to_target = None
    time_to_target = None
    milestone_hits = {m: None for m in milestones}
    _update_milestones(milestones, milestone_hits, history[-1], 0, 0.0)
    t0 = time.perf_counter()
    for it in range(maxiter):
        params, opt_state, value, gnorm = step(params, opt_state)
        history.append(float(value))
        now = time.perf_counter() - t0
        times.append(now)
        _update_milestones(milestones, milestone_hits, history[-1], it + 1, now)
        if iters_to_target is None and _converged(
            history[-1], float(gnorm), f_target, gtol
        ):
            iters_to_target = it + 1
            time_to_target = now
            break
        if time_budget is not None and now >= time_budget:
            break
    wall = time.perf_counter() - t0
    return (
        params,
        history,
        wall,
        times,
        iters_to_target,
        time_to_target,
        milestone_hits,
    )


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------


def main():
    # --- Dataset selection (env var) ---
    dataset = os.environ.get("DATASET", "fashion_mnist").lower()
    if dataset not in ("mnist", "fashion_mnist"):
        print(
            f"[config] Unknown DATASET={dataset!r}; falling back to 'mnist'. "
            "Valid values: 'mnist', 'fashion_mnist'."
        )
        dataset = "mnist"

    # Problem configuration
    n_classes = 10
    n_train = 5000
    n_test = 1000
    # Hidden-layer topology is configurable via env vars (see module docstring).
    hidden_sizes = _parse_hidden_sizes()
    # Hidden-layer activation is configurable via the ACTIVATION env var.
    activation_name, activation_fn = _parse_activation()
    maxiter = 1000

    # --- Shared, fair termination bounds applied to EVERY optimizer ---
    # The non-convex MLP loss does not descend as far as the linear model
    # within the budget, so the loss target / milestones are looser than in
    # mnist_comparison.py to keep the ``->target`` columns informative.
    stop = {
        "f_target": 1.0e-1,
        "gtol": 1.0e-8,
        "time_budget": 10.0,
        "milestones": (1.0e0, 7.0e-1, 5.0e-1, 4.0e-1),
    }

    n_hidden_layers = len(hidden_sizes)
    arch_str = "->".join(str(s) for s in (["x"] + hidden_sizes + [n_classes]))
    print(
        f"=== {n_hidden_layers + 1}-layer ReLU MLP comparison: "
        "QQN vs SGD vs Adam vs L-BFGS ==="
    )
    print(
        f"    dataset={dataset}  hidden_sizes={hidden_sizes}  "
        f"arch={arch_str}  activation={activation_name}  (non-convex objective)"
    )
    print(
        f"  classes={n_classes}  n_train={n_train}  n_test={n_test}  "
        f"maxiter={maxiter}\n"
    )
    print(
        f"  shared stop: f_target={stop['f_target']:.1e}  "
        f"gtol={stop['gtol']:.1e}  time_budget={stop['time_budget']:.1f}s\n"
    )

    xtr, ytr, xte, yte = _load_dataset_numpy(dataset, n_train, n_test, n_classes)
    dim = xtr.shape[1]

    X_train = jnp.asarray(xtr)
    y_train = jnp.asarray(ytr)
    X_test = jnp.asarray(xte)
    y_test = jnp.asarray(yte)

    loss_fn = make_loss(
        X_train, y_train, dim, hidden_sizes, n_classes, activation=activation_fn
    )

    # Shared initial parameters so every optimizer starts identically.
    params0 = init_params(
        dim, hidden_sizes, n_classes, jax.random.PRNGKey(42), activation=activation_name
    )
    n_params = int(params0.shape[0])
    print(f"  model parameters: {n_params}\n")

    runners = {
        # --- Baseline QQN (L-BFGS oracle, Armijo line search) ---
        "QQN": lambda: _run_qqn(loss_fn, params0, maxiter, stop=stop),
        "QQN-S": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            stop=stop,
            spline=True,
        ),
        # --- QQN with backtracking line search (cheap, robust) ---
        "QQN-BT": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            stop=stop,
        ),
        "QQN-BT-S": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            stop=stop,
            spline=True,
        ),
        # --- QQN with a deeper L-BFGS history (richer curvature memory) ---
        "QQN-L20": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=20),
            stop=stop,
        ),
        # --- Deep L-BFGS memory (size 50) ---
        "QQN-L50": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            stop=stop,
        ),
        # --- Momentum oracle (first-order accelerator) ---
        "QQN-Mom": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.9),
            stop=stop,
        ),
        "QQN-Mom-S": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.9),
            stop=stop,
            spline=True,
        ),
        "QQN-Mom-S-BT": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=MomentumOracle(beta=0.9),
            stop=stop,
            spline=True,
        ),
        # --- Matrix-free secant curvature oracle ---
        "QQN-Sec": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            oracle=SecantOracle(),
            stop=stop,
        ),
        # --- Anderson acceleration oracle ---
        "QQN-And": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            oracle=AndersonOracle(window=5),
            stop=stop,
        ),
        # --- Deep L-BFGS with an Anderson fallback (robust safety net) ---
        "QQN-L50And": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            oracle=Fallback([LBFGSOracle(history_size=50), AndersonOracle(window=5)]),
            stop=stop,
        ),
        # --- QQN with an adaptive trust-region sphere (non-convex safeguard) ---
        "QQN-TR": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        # --- Best-of-breed: deep memory + backtracking + fixed trust-region ---
        "QQN-Fast": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={
                "init_step": 2.0,
                "shrink": 0.7,
                "max_iter": 40,
            },
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.5, adaptive=False),
            stop=stop,
        ),
        # --- QQN constrained to a box region (bounded weights) ---
        "QQN-Box": lambda: _run_qqn(
            loss_fn,
            params0,
            maxiter,
            region=BoxRegion(lo=-2.0, hi=2.0),
            stop=stop,
        ),
        "SGD": lambda: run_optax(
            loss_fn, params0, optax.sgd(learning_rate=0.5), maxiter, stop=stop
        ),
        "Adam": lambda: run_optax(
            loss_fn, params0, optax.adam(learning_rate=0.01), maxiter, stop=stop
        ),
        "L-BFGS": lambda: run_optax_lbfgs(loss_fn, params0, maxiter, stop=stop),
    }

    results = {}
    for name, runner in runners.items():
        (
            params,
            history,
            wall,
            times,
            iters_to_target,
            time_to_target,
            milestone_hits,
        ) = runner()
        train_acc = float(
            accuracy(
                params, X_train, y_train, dim, hidden_sizes, n_classes, activation_fn
            )
        )
        test_acc = float(
            accuracy(
                params, X_test, y_test, dim, hidden_sizes, n_classes, activation_fn
            )
        )
        reached = iters_to_target is not None
        n_iters = max(len(history) - 1, 1)
        ms_per_iter = (wall / n_iters) * 1e3
        log_hist = np.log10(np.maximum(np.asarray(history), 1e-12))
        if len(log_hist) > 1:
            x_axis = np.linspace(0.0, 1.0, len(log_hist))
            traj_auc = float(np.trapezoid(log_hist, x_axis))
        else:
            traj_auc = float(log_hist[-1])
        results[name] = {
            "final_loss": history[-1],
            "best_loss": min(history),
            "iters": len(history) - 1,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "wall": wall,
            "history": history,
            "times": times,
            "reached": reached,
            "iters_to_target": iters_to_target,
            "time_to_target": time_to_target,
            "milestone_hits": milestone_hits,
            "ms_per_iter": ms_per_iter,
            "traj_auc": traj_auc,
        }

    # --- Summary table ---
    ordered = sorted(results.items(), key=lambda kv: kv[1]["final_loss"])
    lbfgs_ref = results.get("L-BFGS", {}).get("iters_to_target")
    print(
        f"{'optimizer':<12}{'final_loss':>14}{'iters':>8}"
        f"{'train_acc':>12}{'test_acc':>11}{'time(s)':>10}"
        f"{'ms/it':>8}{'->target':>10}{'t->tgt':>9}{'vs LBFGS':>10}{'AUC':>8}"
    )
    print("-" * 120)
    for name, r in ordered:
        it_tgt = "—" if r["iters_to_target"] is None else f"{r['iters_to_target']}"
        t_tgt = "—" if r["time_to_target"] is None else f"{r['time_to_target']:.3f}"
        if lbfgs_ref is not None and r["iters_to_target"] is not None:
            spd = f"{lbfgs_ref / r['iters_to_target']:.2f}x"
        else:
            spd = "—"
        print(
            f"{name:<12}{r['final_loss']:>14.6e}{r['iters']:>8}"
            f"{r['train_acc']:>12.4f}{r['test_acc']:>11.4f}"
            f"{r['wall']:>10.3f}"
            f"{r['ms_per_iter']:>8.2f}{it_tgt:>10}{t_tgt:>9}{spd:>10}"
            f"{r['traj_auc']:>8.2f}"
        )

    # --- Pareto frontier (loss vs. wall-time) ---
    print("\nPareto frontier (loss vs. time — non-dominated variants):")
    pareto = []
    for name, r in ordered:
        dominated = any(
            (o["final_loss"] <= r["final_loss"] and o["wall"] < r["wall"])
            or (o["final_loss"] < r["final_loss"] and o["wall"] <= r["wall"])
            for on, o in results.items()
            if on != name
        )
        if not dominated:
            pareto.append((name, r))
    for name, r in sorted(pareto, key=lambda kv: kv[1]["wall"]):
        print(f"  {name:<12} loss={r['final_loss']:.4e}  time={r['wall']:.3f}s")

    # --- Iteration-efficiency leaderboard (converging variants only) ---
    print("\nIteration-efficiency leaderboard (target reached, fewest iters):")
    converged = [
        (name, r) for name, r in results.items() if r["iters_to_target"] is not None
    ]
    converged.sort(key=lambda kv: (kv[1]["iters_to_target"], kv[1]["wall"]))
    for name, r in converged[:12]:
        spd = (
            f"{lbfgs_ref / r['iters_to_target']:.2f}x" if lbfgs_ref is not None else "—"
        )
        print(
            f"  {name:<14} iters={r['iters_to_target']:>4}  "
            f"time={r['time_to_target']:.3f}s  vs_LBFGS={spd:>6}  "
            f"final={r['final_loss']:.4e}"
        )

    # --- Convergence-rate profile (loss milestones) ---
    milestones = stop.get("milestones", ())
    if milestones:
        print("\nConvergence-rate profile (iteration first reaching each loss):")
        header = (
            "  "
            + f"{'optimizer':<12}"
            + "".join(f"{f'<={m:.1e}':>12}" for m in milestones)
        )
        print(header)
        tightest = milestones[-1]

        def _sort_key(kv):
            hit = kv[1]["milestone_hits"].get(tightest)
            return hit[0] if hit is not None else 10**9

        for name, r in sorted(results.items(), key=_sort_key):
            cells = []
            for m in milestones:
                hit = r["milestone_hits"].get(m)
                cells.append("—" if hit is None else f"{hit[0]}")
            print("  " + f"{name:<12}" + "".join(f"{c:>12}" for c in cells))

    # --- Stall report (non-converging variants) ---
    stalled = [(name, r) for name, r in results.items() if r["iters_to_target"] is None]
    if stalled:
        print("\nStall report (never reached the shared target):")
        stalled.sort(key=lambda kv: kv[1]["final_loss"])
        for name, r in stalled:
            if r["wall"] >= stop.get("time_budget", float("inf")) - 0.5:
                cause = "time-budget exhausted"
            elif r["final_loss"] > 0.7:
                cause = "stalled (plateau)"
            else:
                cause = "slow (no target in maxiter)"
            print(
                f"  {name:<14} final={r['final_loss']:.4e}  "
                f"iters={r['iters']:>3}  time={r['wall']:.3f}s  [{cause}]"
            )

    # --- Loss trajectory (compact ASCII view at log10 scale) ---
    print("\nLoss trajectory (log10, sampled):")
    sample_points = 10
    for name, r in results.items():
        hist = r["history"]
        idxs = np.linspace(0, len(hist) - 1, sample_points).astype(int)
        vals = [f"{np.log10(max(hist[i], 1e-12)):6.2f}" for i in idxs]
        print(f"  {name:<10} " + " ".join(vals))

    # Optional: save a matplotlib plot if available.
    try:
        import matplotlib.pyplot as plt  # type: ignore

        baselines = {"SGD", "Adam", "L-BFGS"}
        # Ensure the output directory exists and build a timestamped basename.
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # A shared title describing all the important configuration knobs so
        # the saved figures are self-documenting (architecture, dataset,
        # activation, class count, and training-set size).
        config_title = (
            f"{n_hidden_layers + 1}-layer MLP {arch_str} on {dataset}\n"
            f"activation={activation_name}  classes={n_classes}  "
            f"n_train={n_train}  maxiter={maxiter}  (QQN variants vs baselines)"
        )

        def _draw(x_key, x_label, file_suffix):
            """Render one convergence plot keyed on iteration or wall-time."""
            plt.figure(figsize=(7, 5))
            for name, r in results.items():
                # ``times`` and ``history`` share the same length, so either
                # can index the loss values along the chosen x-axis.
                xs = r["times"] if x_key == "times" else range(len(r["history"]))
                if name in baselines:
                    plt.semilogy(
                        xs, r["history"], label=name, linestyle="--", linewidth=2
                    )
                else:
                    plt.semilogy(xs, r["history"], label=name, alpha=0.85)
            plt.xlabel(x_label)
            plt.ylabel("full-batch loss")
            plt.title(config_title)
            plt.legend(ncol=2, fontsize=8)
            plt.grid(True, which="both", alpha=0.3)
            out = os.path.join(
                results_dir,
                f"{dataset}_mlp_comparison_{file_suffix}_{timestamp}.png",
            )
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"[plot] Saved {file_suffix} convergence plot to {out}")

        print()
        # Loss vs. iteration (classic view).
        _draw("iteration", "iteration", "vs_iter")
        # Loss vs. wall-clock time (captures per-iteration cost differences).
        _draw("times", "wall-clock time (s)", "vs_time")
    except Exception:
        print("\n[plot] matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
