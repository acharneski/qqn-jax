"""Benchmark: sparse MNIST classification with QQN + OrthantRegion.

This example trains a small multi-layer perceptron (MLP) on MNIST using
QQN. The ``OrthantRegion`` (OWL-QN style) is applied to encourage weight
sparsity by constraining each update to remain within the orthant defined
by the current weights' signs, zeroing coordinates that would cross zero.

We compare three configurations and report final loss, test accuracy and
weight sparsity (fraction of near-zero parameters):

  1. ``region=None``     -- baseline (dense).
  2. ``OrthantRegion``   -- sparsity via orthant projection.
  3. ``Sequential([Orthant, TrustRegion])`` -- sparsity + step control.

The benchmark is intentionally lightweight (small MLP, subset of MNIST)
so it runs quickly on CPU while still exercising the region machinery
through ``jit``/``vmap``-compatible code paths.
Configurability:
   Every major knob is overridable via an environment variable so the
   benchmark can be re-tuned without editing the source. The available
   variables (with defaults in parentheses) are:
     DATASET        ``mnist`` (default) or ``fashion_mnist`` -- which corpus
                    to train on (both ship via tensorflow.keras / torchvision).
     N_TRAIN (10000)  Training-subset size.
     N_TEST  (5000)   Test-subset size.
     HIDDEN_SIZES   Comma-separated hidden-layer widths, e.g. ``128,64``.
                    Takes precedence over ``HIDDEN`` / ``DEPTH``.
     HIDDEN  (64)   Width of each hidden layer (with ``DEPTH``).
     DEPTH   (2)    Number of hidden layers (with ``HIDDEN``).
     ACTIVATION     Hidden activation: ``relu``, ``tanh`` (default), ``sigmoid``,
                    ``sine``, ``gaussian``, ``gelu``, ``swish``, ``softplus``,
                    ``abs``, ``identity``. A comma-separated list mixes
                    activations per hidden layer (cycled if too short).
     MAXITER (50000)        Raw-training iteration budget.
     POLISH_MAXITER         Polishing iteration budget (default MAXITER // 10).
     LINE_SEARCH    Line search type (default ``strong_wolfe``).
     HISTORY_SIZE (10)      L-BFGS history length.
     L2 (1e-4)      Squared-L2 weight-decay coefficient.
     L1_SCALE (1e-5)        L1 sparsity-penalty scale.
     QUANT_SCALE (1e-4)     Quantization-delta penalty scale.
     QBITS (4)      Quantization grid bit-depth.
     SEED (0)       RNG seed.
Examples::
     ACTIVATION=relu DEPTH=3 HIDDEN=128 python -m examples.mnist_sparse_benchmark
     DATASET=fashion_mnist N_TRAIN=20000 python -m examples.mnist_sparse_benchmark
     ACTIVATION=tanh,gelu HIDDEN_SIZES=128,64 python -m examples.mnist_sparse_benchmark


Note:
    The current L-BFGS oracle (``qqn_jax.lbfgs``) operates on a *flat*
    parameter vector, so this example flattens the MLP parameters into a
    single 1-D array via ``jax.flatten_util.ravel_pytree`` and unflattens
    them inside the loss function.

Run:
    python -m examples.mnist_sparse_benchmark

Requires:
    - jax, jaxlib
    - numpy
    - tensorflow-datasets OR a local MNIST .npz (see ``load_mnist``).
"""

import os

import time
from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from qqn_jax.solver import QQN
from qqn_jax.regions import (
    OrthantRegion,
    QuantizationRegion,
)
from qqn_jax.regularizers import (
    l1_penalty,
    quantization_delta_penalty,
)


# --- Environment-variable parsing helpers -----------------------------
def _env_int(name, default):
    """Parse an int environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[config] Invalid {name}={raw!r}; using default {default}.")
        return default


def _env_float(name, default):
    """Parse a float environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[config] Invalid {name}={raw!r}; using default {default}.")
        return default


def _env_str(name, default):
    """Parse a string environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


# --- Activation function selection ------------------------------------
_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
    "sine": jnp.sin,
    # Gaussian "bump" activation, exp(-x^2): localized, smooth, RBF-like.
    "gaussian": lambda x: jnp.exp(-(x**2)),
    "gelu": jax.nn.gelu,
    "swish": jax.nn.swish,
    "softplus": jax.nn.softplus,
    "abs": jnp.abs,
    "identity": lambda x: x,
}


def _resolve_activation_name(name):
    """Resolve a single activation name to ``(name, fn)``; fall back to tanh."""
    name = name.strip().lower()
    if name not in _ACTIVATIONS:
        print(
            f"[config] Unknown ACTIVATION={name!r}; falling back to 'tanh'. "
            f"Valid values: {', '.join(sorted(_ACTIVATIONS))}."
        )
        name = "tanh"
    return name, _ACTIVATIONS[name]


def parse_activation(n_hidden_layers=None):
    """Resolve the hidden-layer activation(s) from the ``ACTIVATION`` env var.
    Accepts a single name (applied uniformly) or a comma-separated list that
    is cycled/truncated to one activation per hidden layer.
    Returns:
        ``(names, fns)`` where each is a list of length ``n_hidden_layers``
        (or a single-element list when the count is unknown).
    """
    raw = os.environ.get("ACTIVATION", "tanh").strip().lower()
    tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
    if not tokens:
        tokens = ["tanh"]
    resolved = [_resolve_activation_name(t) for t in tokens]
    names = [n for n, _ in resolved]
    fns = [f for _, f in resolved]
    if n_hidden_layers is not None and n_hidden_layers > 0:
        names = [names[i % len(names)] for i in range(n_hidden_layers)]
        fns = [fns[i % len(fns)] for i in range(n_hidden_layers)]
    return names, fns


def parse_hidden_sizes():
    """Resolve the hidden-layer topology from environment variables.
    Precedence: ``HIDDEN_SIZES`` -> ``DEPTH`` x ``HIDDEN`` -> default ``[64, 64]``.
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
    hidden = _env_int("HIDDEN", 64)
    depth = _env_int("DEPTH", 2)
    if hidden <= 0 or depth < 0:
        print("[config] Invalid HIDDEN/DEPTH; using default [64, 64].")
        return [64, 64]
    return [hidden] * depth


# --- Data loading -----------------------------------------------------


def load_mnist(
    n_train: int = 2000,
    n_test: int = 1000,
    seed: int = 0,
    dataset: str = "mnist",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a subset of MNIST.

     Tries ``tensorflow.keras``, ``tensorflow_datasets``, and
     ``torchvision`` in turn; if none are available, falls back to a
     small synthetic dataset so the example still runs end-to-end.
     Args:
         dataset: ``"mnist"`` (default) or ``"fashion_mnist"``.


    Returns:
        (x_train, y_train, x_test, y_test) with images flattened to
        ``(n, 784)`` float32 in ``[0, 1]`` and integer labels.
    """

    x_train = y_train = x_test = y_test = None
    is_fashion = dataset == "fashion_mnist"

    # --- Attempt 1: tensorflow.keras ---
    if x_train is None:
        try:
            if is_fashion:
                from tensorflow.keras.datasets import (  # type: ignore
                    fashion_mnist as mnist,
                )
            else:
                from tensorflow.keras.datasets import mnist  # type: ignore

            (xtr, ytr), (xte, yte) = mnist.load_data()
            x_train = xtr.reshape(xtr.shape[0], -1).astype(np.float32) / 255.0
            y_train = ytr.astype(np.int32)
            x_test = xte.reshape(xte.shape[0], -1).astype(np.float32) / 255.0
            y_test = yte.astype(np.int32)
            print(f"[load_mnist] Loaded {dataset} via tensorflow.keras.")
        except Exception:
            pass

    # --- Attempt 2: tensorflow_datasets ---
    if x_train is None:
        try:
            import tensorflow_datasets as tfds  # type: ignore

            name = "fashion_mnist" if is_fashion else "mnist"
            ds = tfds.load(name, split=["train", "test"], batch_size=-1)
            train, test = tfds.as_numpy(ds[0]), tfds.as_numpy(ds[1])
            x_train = train["image"].reshape(-1, 784).astype(np.float32) / 255.0
            y_train = train["label"].astype(np.int32)
            x_test = test["image"].reshape(-1, 784).astype(np.float32) / 255.0
            y_test = test["label"].astype(np.int32)
            print(f"[load_mnist] Loaded {dataset} via tensorflow_datasets.")
        except Exception:
            pass

    # --- Attempt 3: torchvision ---
    if x_train is None:
        try:
            from torchvision import datasets  # type: ignore

            if is_fashion:
                cls = datasets.FashionMNIST
                root = "./_fashion_mnist_data"
            else:
                cls = datasets.MNIST
                root = "./_mnist_data"
            train = cls(root=root, train=True, download=True)
            test = cls(root=root, train=False, download=True)
            x_train = (
                train.data.numpy().reshape(len(train), -1).astype(np.float32) / 255.0
            )
            y_train = train.targets.numpy().astype(np.int32)
            x_test = test.data.numpy().reshape(len(test), -1).astype(np.float32) / 255.0
            y_test = test.targets.numpy().astype(np.int32)
            print(f"[load_mnist] Loaded {dataset} via torchvision.")
        except Exception:
            pass

    # --- Fallback: synthetic data ---
    if x_train is None:
        print(f"[load_mnist] Real {dataset} unavailable; using synthetic data.")
        rng = np.random.default_rng(seed)
        x_train = rng.random((n_train, 784)).astype(np.float32)
        y_train = rng.integers(0, 10, size=n_train).astype(np.int32)
        x_test = rng.random((n_test, 784)).astype(np.float32)
        y_test = rng.integers(0, 10, size=n_test).astype(np.int32)
        return x_train, y_train, x_test, y_test
    assert x_train is not None
    assert y_train is not None
    assert x_test is not None
    assert y_test is not None

    # Subsample to the requested sizes.
    rng = np.random.default_rng(seed)
    tr_idx = rng.permutation(len(x_train))[:n_train]
    te_idx = rng.permutation(len(x_test))[:n_test]
    return (
        x_train[tr_idx],
        y_train[tr_idx],
        x_test[te_idx],
        y_test[te_idx],
    )


# --- Model ------------------------------------------------------------


def init_params(
    key, sizes: List[int], activation: Any = "tanh"
) -> List[Dict[str, jnp.ndarray]]:
    """Initialize MLP parameters with scaled Gaussian weights.

    Uses He-style init for ReLU hidden layers and Xavier/Glorot-style init
    otherwise, which keeps activations well-scaled at initialization.
    ``activation`` may be a single name string (applied uniformly) or a
    per-hidden-layer list of names for mixed-activation networks.
    """
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    n_layers = len(sizes) - 1
    n_hidden = n_layers - 1
    if isinstance(activation, (list, tuple)):
        hidden_names = [
            activation[i % len(activation)] for i in range(max(n_hidden, 0))
        ]
    else:
        hidden_names = [activation] * max(n_hidden, 0)
    # Output layer is always linear (logits) -> Glorot-style init.
    layer_names = hidden_names + ["identity"]
    for li, (k, (n_in, n_out)) in enumerate(zip(keys, zip(sizes[:-1], sizes[1:]))):
        wk, bk = jax.random.split(k)
        act_name = layer_names[li] if li < len(layer_names) else "identity"
        if act_name == "relu":
            # He initialization for ReLU: std = sqrt(2 / fan_in).
            scale = jnp.sqrt(2.0 / n_in)
        else:
            # Glorot/Xavier-style init.
            scale = 1.0 / jnp.sqrt(n_in)
        params.append(
            {
                "w": scale * jax.random.normal(wk, (n_in, n_out)),
                "b": jnp.zeros((n_out,)),
            }
        )
    return params


def mlp_forward(params, x, activation=jnp.tanh):
    """Forward pass: configurable activation hidden layers, linear logits.

    ``activation`` may be a single callable (applied to every hidden layer)
    or a list/tuple of callables (one per hidden layer, cycled if short).
    """
    n_hidden = len(params) - 1
    if isinstance(activation, (list, tuple)):
        acts = [activation[i % len(activation)] for i in range(n_hidden)]
    else:
        acts = [activation] * n_hidden
    h = x
    for i, layer in enumerate(params[:-1]):
        h = acts[i](h @ layer["w"] + layer["b"])
    last = params[-1]
    return h @ last["w"] + last["b"]


def cross_entropy_loss(
    params, x, y, l2: float = 1e-4, regularizer=None, activation=jnp.tanh
):
    """Softmax cross-entropy with optional extra regularization.

    Args:
        params: MLP parameter pytree.
        x, y: inputs and integer labels.
        l2: base squared-L2 weight-decay coefficient (always applied).
        regularizer: optional ``params -> scalar`` callable whose value is
            *added* to the loss. Use this to inject L1 sparsity or the
            quantization-delta (precision) penalty alongside training.
         activation: hidden-layer activation callable(s).
    """
    logits = mlp_forward(params, x, activation)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.mean(jnp.take_along_axis(log_probs, y[:, None], axis=-1))
    reg = l2 * sum(jnp.sum(layer["w"] ** 2) for layer in params)
    if regularizer is not None:
        reg = reg + regularizer(params)
    return nll + reg


def test_loss(params, x, y, activation=jnp.tanh):
    """Plain softmax cross-entropy (no regularization) on a dataset.

    Reported in the same units as the training loss so all metrics are
    directly comparable.
    """
    logits = mlp_forward(params, x, activation)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, y[:, None], axis=-1))


# --- Sparsity metric --------------------------------------------------


def sparsity(params, threshold: float = 1e-6) -> float:
    """Fraction of weight entries with magnitude below ``threshold``."""
    total = 0
    zeros = 0
    for layer in params:
        w = layer["w"]
        total += w.size
        zeros += int(jnp.sum((jnp.abs(w) < threshold).astype(jnp.int32)))
    return zeros / max(total, 1)


def round_params_to_grid(
    params,
    bits: int = 4,
    lo: float = -1.0,
    hi: float = 1.0,
):
    """Round each layer's weights onto a ``bits``-level grid over ``[lo, hi]``.

    Returns a new parameter pytree with quantized weights (biases kept as-is).
    Used to measure the *post-rounding* loss: a precision-optimized network
    should show (almost) no loss increase after this rounding.
    """
    delta = (hi - lo) / ((2**bits) - 1)
    quantized = []
    for layer in params:
        w = jnp.clip(layer["w"], lo, hi)
        k = jnp.round((w - lo) / delta)
        k = jnp.clip(k, 0.0, jnp.floor((hi - lo) / delta))
        grid = lo + k * delta
        quantized.append({"w": grid, "b": layer["b"]})
    return quantized


# --- Benchmark driver -------------------------------------------------
def plot_convergence(results: List[Dict[str, Any]], fname: str = "convergence.png"):
    """Plot loss-vs-evaluation convergence curves for all configs.
    Saves a PNG (and tries to show it interactively). Gracefully degrades
    if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # safe default for headless environments
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plot_convergence] matplotlib unavailable ({exc!r}); skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        history = r.get("loss_history", [])
        if not history:
            continue
        ax.plot(range(len(history)), history, label=r["name"], linewidth=1.5)
    ax.set_xlabel("loss evaluation")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("QQN convergence: sparse MNIST")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"\nSaved convergence plot to {fname!r}")
    try:
        plt.show()
    except Exception:
        pass


def run_config(
    name: str,
    region,
    x_train,
    y_train,
    x_test,
    y_test,
    sizes: List[int],
    maxiter: int = 100,
    seed: int = 0,
    line_search: str = "strong_wolfe",
    regularizer=None,
    quant_bits: int = 4,
    init_flat=None,
    unravel_fn=None,
    activation=jnp.tanh,
    activation_names: Union[str, List[str]] = "tanh",
    l2: float = 1e-4,
    history_size: int = 10,
    quant_lo: float = -1.0,
    quant_hi: float = 1.0,
) -> Dict[str, Any]:
    """Train one configuration and collect metrics.

    The MLP parameter pytree is flattened to a single 1-D vector so it is
    compatible with the flat-array L-BFGS oracle. The loss closure
    unflattens the vector before evaluating the network.
     ``regularizer`` (optional) is a ``params -> scalar`` penalty added to the
     loss — used to inject L1 sparsity or the quantization-delta (precision)
     penalty that turns ordinary training into precision-optimized training.
     ``init_flat`` (optional) warm-starts the optimizer from a previously
     trained flat parameter vector. This is used to run the quantization
     *polishing* phase on top of a dense- or sparse-trained model. When
     provided, ``unravel_fn`` (the matching ``ravel_pytree`` unflatten) must
     also be supplied so the structure matches the warm-start vector.
      ``activation`` is the hidden-layer activation callable(s); ``l2`` the
      base weight-decay coefficient; ``history_size`` the L-BFGS memory length.
    """
    key = jax.random.PRNGKey(seed)

    if init_flat is not None and unravel_fn is not None:
        # Warm-start (polishing phase): reuse a pre-trained parameter vector.
        flat_params0 = init_flat
        unravel = unravel_fn
    else:
        # Cold-start (raw training phase): fresh random initialization.
        params0_tree = init_params(key, sizes, activation=activation_names)
        # Flatten the pytree to a flat vector; keep the unflatten fn to
        # reconstruct the structured params inside the loss / metrics.
        flat_params0, unravel = ravel_pytree(params0_tree)

    # Closure over the (static) training data, operating on flat params.
    def loss_fn(flat_params):
        params = unravel(flat_params)
        return cross_entropy_loss(
            params,
            x_train,
            y_train,
            l2=l2,
            regularizer=regularizer,
            activation=activation,
        )

    # Record the loss at each evaluation via a host callback so we can
    # plot a convergence curve afterwards. This is jit-compatible.
    loss_history: List[float] = []

    def _record(val):
        loss_history.append(float(val))

    def loss_fn_recorded(flat_params):
        val = loss_fn(flat_params)
        jax.debug.callback(_record, val)
        return val

    solver = QQN(
        loss_fn_recorded,
        maxiter=maxiter,
        tol=1e-6,
        history_size=history_size,
        line_search=line_search,
        region=region,
    )

    run = jax.jit(solver.run)

    t0 = time.perf_counter()
    final_flat, final_state = run(flat_params0)
    # Block until computation is complete for accurate timing.
    jax.block_until_ready(final_flat)
    elapsed = time.perf_counter() - t0

    final_params = unravel(final_flat)
    final_loss = float(final_state.value)
    test_loss_val = float(test_loss(final_params, x_test, y_test, activation))
    spars = sparsity(final_params)
    # Post-rounding loss: evaluate the (regularizer-free) test loss after
    # quantizing weights onto the grid. Reported in loss units so it is
    # directly comparable to ``loss`` and ``test_loss``.
    quant_params = round_params_to_grid(
        final_params, bits=quant_bits, lo=quant_lo, hi=quant_hi
    )
    quant_loss_val = float(test_loss(quant_params, x_test, y_test, activation))

    return {
        "name": name,
        "iters": int(final_state.iter),
        "loss": final_loss,
        "test_loss": test_loss_val,
        "sparsity": spars,
        "quant_loss": quant_loss_val,
        "time_s": elapsed,
        "loss_history": loss_history,
        # Trained parameters (flat) + unflatten fn, so this result can serve
        # as the warm-start for a subsequent quantization polishing phase.
        "final_flat": final_flat,
        "unravel": unravel,
    }


def main():
    # --- Configuration (environment-variable driven) ---
    dataset = _env_str("DATASET", "mnist").lower()
    if dataset not in ("mnist", "fashion_mnist"):
        print(
            f"[config] Unknown DATASET={dataset!r}; falling back to 'mnist'. "
            "Valid values: 'mnist', 'fashion_mnist'."
        )
        dataset = "mnist"
    n_train = _env_int("N_TRAIN", 10000)
    n_test = _env_int("N_TEST", 5000)
    seed = _env_int("SEED", 0)
    line_search = _env_str("LINE_SEARCH", "strong_wolfe")
    history_size = _env_int("HISTORY_SIZE", 10)
    l2 = _env_float("L2", 1e-4)
    l1_scale = _env_float("L1_SCALE", 1e-5)
    quant_scale = _env_float("QUANT_SCALE", 1e-4)

    hidden_sizes = parse_hidden_sizes()
    activation_names, activation_fns = parse_activation(len(hidden_sizes))
    activation_str = ",".join(activation_names)

    print(f"Loading {dataset} subset...")
    x_train, y_train, x_test, y_test = load_mnist(
        n_train=n_train, n_test=n_test, seed=seed, dataset=dataset
    )
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    dim = x_train.shape[1]
    n_classes = int(max(int(y_train.max()), int(y_test.max())) + 1)
    sizes = [dim, *hidden_sizes, n_classes]
    maxiter = _env_int("MAXITER", 50000)
    # Quantization grid used by both the precision regularizer and the
    # QuantizationRegion below (4-bit symmetric grid over [-1, 1]).
    QBITS = _env_int("QBITS", 4)
    QLO, QHI = -1.0, 1.0
    arch_str = "->".join(str(s) for s in sizes)
    print(
        f"  arch={arch_str}  activation={activation_str}  "
        f"line_search={line_search}  history_size={history_size}"
    )
    print(
        f"  maxiter={maxiter}  l2={l2:.1e}  l1_scale={l1_scale:.1e}  "
        f"quant_scale={quant_scale:.1e}  qbits={QBITS}  seed={seed}\n"
    )

    # Penalty callables (params -> scalar) added to the training loss.
    def l1_reg(params):
        return l1_penalty(params, scale=l1_scale, weights_only=True)

    def quant_reg(params):
        # L1 norm of the rounding delta — draws weights onto the grid so the
        # trained network is precision-optimized (quantization-aware).
        return quantization_delta_penalty(
            params, scale=quant_scale, bits=QBITS, lo=QLO, hi=QHI, weights_only=True
        )

    # ------------------------------------------------------------------
    # Phase 1: raw (cold-start) training of base models.
    # These are full training runs from random init. Each produces a
    # trained parameter vector that the polishing phase can warm-start from.
    # ------------------------------------------------------------------
    base_configs = [
        # (name, region, line_search, regularizer)
        ("baseline (dense)", None, "strong_wolfe", None),
        ("orthant (sparse)", OrthantRegion(), "strong_wolfe", None),
        # --- L1-regularized sparsity (objective-space sparsity) ---
        ("l1-orthant-penalty (sparse)", OrthantRegion(), "strong_wolfe", l1_reg),
        ("l1-penalty (sparse)", None, "strong_wolfe", l1_reg),
    ]

    # ------------------------------------------------------------------
    # Phase 2: quantization polishing variants. Each of these is applied
    # *on top of* every base-trained model (cross-product), as a short
    # polishing phase rather than a from-scratch training run.
    # ------------------------------------------------------------------
    polish_configs = [
        # (suffix, region, line_search, regularizer)
        # --- Precision-optimized: quantization-delta penalty only ---
        ("quant-penalty (prec)", None, "strong_wolfe", quant_reg),
        # --- Precision-optimized: QuantizationRegion confines the step to
        #     each weight's rounding cell (geometric quantization). ---
        (
            "quant-region (prec)",
            QuantizationRegion(bits=QBITS, lo=QLO, hi=QHI),
            "strong_wolfe",
            None,
        ),
        (
            "quant-region-penalty (prec)",
            QuantizationRegion(bits=QBITS, lo=QLO, hi=QHI),
            "strong_wolfe",
            quant_reg,
        ),
    ]

    # Polishing uses a much smaller iteration budget than raw training: it
    # only needs to nudge an already-trained model onto the quant grid.
    polish_maxiter = _env_int("POLISH_MAXITER", max(1, maxiter // 10))

    results = []
    base_results = []
    for name, region, line_search, regularizer in base_configs:
        print(f"\n=== [base] Running: {name} ===")
        res = run_config(
            name,
            region,
            x_train,
            y_train,
            x_test,
            y_test,
            sizes,
            maxiter=maxiter,
            line_search=line_search,
            regularizer=regularizer,
            quant_bits=QBITS,
            seed=seed,
            activation=activation_fns,
            activation_names=activation_names,
            l2=l2,
            history_size=history_size,
            quant_lo=QLO,
            quant_hi=QHI,
        )
        results.append(res)
        base_results.append(res)
        print(
            f"  iters={res['iters']:3d}  loss={res['loss']:.4f}  "
            f"test_loss={res['test_loss']:.4f}  "
            f"sparsity={res['sparsity']:.3f}  "
            f"quant_loss={res['quant_loss']:.4f}  time={res['time_s']:.2f}s"
        )

    # Cross-product: every quant polishing variant on every base model.
    for base in base_results:
        for suffix, region, line_search, regularizer in polish_configs:
            name = f"{base['name']} -> {suffix}"
            print(f"\n=== [polish] Running: {name} ===")
            res = run_config(
                name,
                region,
                x_train,
                y_train,
                x_test,
                y_test,
                sizes,
                maxiter=polish_maxiter,
                line_search=line_search,
                regularizer=regularizer,
                quant_bits=QBITS,
                init_flat=base["final_flat"],
                unravel_fn=base["unravel"],
                seed=seed,
                activation=activation_fns,
                activation_names=activation_names,
                l2=l2,
                history_size=history_size,
                quant_lo=QLO,
                quant_hi=QHI,
            )
            results.append(res)
            print(
                f"  iters={res['iters']:3d}  loss={res['loss']:.4f}  "
                f"test_loss={res['test_loss']:.4f}  "
                f"sparsity={res['sparsity']:.3f}  "
                f"quant_loss={res['quant_loss']:.4f}  time={res['time_s']:.2f}s"
            )
    # Warm-start vectors are only needed during polishing; release them so
    # large parameter arrays are not held for the rest of the run.
    for base in base_results:
        base.pop("final_flat", None)
        base.pop("unravel", None)

    # Summary table.
    print("\n" + "=" * 90)
    header = (
        f"{'config':<48}{'iters':>7}{'loss':>10}"
        f"{'test_loss':>11}{'sparsity':>10}{'quant_loss':>11}{'time(s)':>10}"
    )
    print(header)
    print("-" * 114)
    for r in results:
        print(
            f"{r['name']:<48}{r['iters']:>7}{r['loss']:>10.4f}"
            f"{r['test_loss']:>11.4f}{r['sparsity']:>10.3f}"
            f"{r.get('quant_loss', 0.0):>11.4f}{r['time_s']:>10.2f}"
        )
    print("=" * 114)
    # --- Pareto frontier (test accuracy vs. sparsity) ---
    # Surface the non-dominated configs trading off accuracy against
    # compression (higher sparsity + higher accuracy is better).
    print("\nPareto frontier (test_loss vs. sparsity — non-dominated configs):")
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, o in enumerate(results):
            if i == j:
                continue
            # ``o`` dominates ``r`` iff it is no worse on both objectives
            # (lower test_loss, higher sparsity) and strictly better on one.
            no_worse = (
                o["test_loss"] <= r["test_loss"] and o["sparsity"] >= r["sparsity"]
            )
            strictly_better = (
                o["test_loss"] < r["test_loss"] or o["sparsity"] > r["sparsity"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    for r in sorted(pareto, key=lambda d: d["sparsity"], reverse=True):
        print(
            f"  {r['name']:<48} test_loss={r['test_loss']:.4f}  "
            f"sparsity={r['sparsity']:.3f}"
        )
    # --- Best precision-optimized config (lowest quant_err) ---
    prec = [r for r in results if "prec" in r["name"]]
    if prec:
        best = min(prec, key=lambda d: d["quant_loss"])
        print(
            f"\nBest precision config (lowest quant_loss): {best['name']}\n"
            f"  quant_loss={best['quant_loss']:.4f}  test_loss={best['test_loss']:.4f}  "
            f"sparsity={best['sparsity']:.3f}"
        )

    # Plot convergence curves across configurations.
    plot_convergence(results)


if __name__ == "__main__":
    main()
