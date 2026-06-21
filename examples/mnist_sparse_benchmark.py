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

import time
from functools import partial
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from qqn_jax.solver import QQN
from qqn_jax.regions import OrthantRegion, TrustRegion, Sequential


# --- Data loading -----------------------------------------------------


def load_mnist(
    n_train: int = 2000,
    n_test: int = 1000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a subset of MNIST.

     Tries ``tensorflow.keras``, ``tensorflow_datasets``, and
     ``torchvision`` in turn; if none are available, falls back to a
     small synthetic dataset so the example still runs end-to-end.

    Returns:
        (x_train, y_train, x_test, y_test) with images flattened to
        ``(n, 784)`` float32 in ``[0, 1]`` and integer labels.
    """

    x_train = y_train = x_test = y_test = None

    # --- Attempt 1: tensorflow.keras ---
    if x_train is None:
        try:
            from tensorflow.keras.datasets import mnist  # type: ignore

            (xtr, ytr), (xte, yte) = mnist.load_data()
            x_train = xtr.reshape(xtr.shape[0], -1).astype(np.float32) / 255.0
            y_train = ytr.astype(np.int32)
            x_test = xte.reshape(xte.shape[0], -1).astype(np.float32) / 255.0
            y_test = yte.astype(np.int32)
        except Exception:
            pass

    # --- Attempt 2: tensorflow_datasets ---
    if x_train is None:
        try:
            import tensorflow_datasets as tfds  # type: ignore

            ds = tfds.load("mnist", split=["train", "test"], batch_size=-1)
            train, test = tfds.as_numpy(ds[0]), tfds.as_numpy(ds[1])
            x_train = train["image"].reshape(-1, 784).astype(np.float32) / 255.0
            y_train = train["label"].astype(np.int32)
            x_test = test["image"].reshape(-1, 784).astype(np.float32) / 255.0
            y_test = test["label"].astype(np.int32)
        except Exception:
            pass

    # --- Attempt 3: torchvision ---
    if x_train is None:
        try:
            from torchvision import datasets  # type: ignore

            train = datasets.MNIST(root="./_mnist_data", train=True, download=True)
            test = datasets.MNIST(root="./_mnist_data", train=False, download=True)
            x_train = (
                train.data.numpy().reshape(len(train), -1).astype(np.float32) / 255.0
            )
            y_train = train.targets.numpy().astype(np.int32)
            x_test = test.data.numpy().reshape(len(test), -1).astype(np.float32) / 255.0
            y_test = test.targets.numpy().astype(np.int32)
        except Exception:
            pass

    # --- Fallback: synthetic data ---
    if x_train is None:
        print("[load_mnist] Real MNIST unavailable; using synthetic data.")
        rng = np.random.default_rng(seed)
        x_train = rng.random((n_train, 784)).astype(np.float32)
        y_train = rng.integers(0, 10, size=n_train).astype(np.int32)
        x_test = rng.random((n_test, 784)).astype(np.float32)
        y_test = rng.integers(0, 10, size=n_test).astype(np.int32)
        return x_train, y_train, x_test, y_test

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


def init_params(key, sizes: List[int]) -> List[Dict[str, jnp.ndarray]]:
    """Initialize MLP parameters with scaled Gaussian weights."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        wk, bk = jax.random.split(k)
        scale = 1.0 / jnp.sqrt(n_in)
        params.append(
            {
                "w": scale * jax.random.normal(wk, (n_in, n_out)),
                "b": jnp.zeros((n_out,)),
            }
        )
    return params


def mlp_forward(params, x):
    """Forward pass: ReLU hidden layers, linear logits output."""
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    last = params[-1]
    return h @ last["w"] + last["b"]


def cross_entropy_loss(params, x, y, l2: float = 1e-4):
    """Softmax cross-entropy with a small L2 regularizer."""
    logits = mlp_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.mean(jnp.take_along_axis(log_probs, y[:, None], axis=-1))
    reg = l2 * sum(jnp.sum(layer["w"] ** 2) for layer in params)
    return nll + reg


def accuracy(params, x, y):
    logits = mlp_forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == y).astype(jnp.float32))


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
) -> Dict[str, Any]:
    """Train one configuration and collect metrics.

    The MLP parameter pytree is flattened to a single 1-D vector so it is
    compatible with the flat-array L-BFGS oracle. The loss closure
    unflattens the vector before evaluating the network.
    """
    key = jax.random.PRNGKey(seed)
    params0_tree = init_params(key, sizes)

    # Flatten the pytree to a flat vector; keep the unflatten fn to
    # reconstruct the structured params inside the loss / metrics.
    flat_params0, unravel = ravel_pytree(params0_tree)

    # Closure over the (static) training data, operating on flat params.
    def loss_fn(flat_params):
        params = unravel(flat_params)
        return cross_entropy_loss(params, x_train, y_train)

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
        history_size=10,
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
    test_acc = float(accuracy(final_params, x_test, y_test))
    spars = sparsity(final_params)

    return {
        "name": name,
        "iters": int(final_state.iter),
        "loss": final_loss,
        "test_acc": test_acc,
        "sparsity": spars,
        "time_s": elapsed,
        "loss_history": loss_history,
    }


def main():
    print("Loading MNIST subset...")
    x_train, y_train, x_test, y_test = load_mnist(n_train=10000, n_test=5000)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    sizes = [784, 64, 64, 10]
    maxiter = 5000

    configs = [
        ("baseline (dense)", None, "strong_wolfe"),
        ("baseline (spline)", None, "spline"),
        ("orthant (sparse)", OrthantRegion(l1=1e-1), "strong_wolfe"),
        ("orthant (spline)", OrthantRegion(l1=1e-1), "spline"),
        (
            "orthant + trust",
            Sequential([OrthantRegion(l1=1e-1), TrustRegion(radius=1.0)]),
            "strong_wolfe",
        ),
        (
            "orthant + trust (spline)",
            Sequential([OrthantRegion(l1=1e-1), TrustRegion(radius=1.0)]),
            "spline",
        ),
    ]

    results = []
    for name, region, line_search in configs:
        print(f"\n=== Running: {name} ===")
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
        )
        results.append(res)
        print(
            f"  iters={res['iters']:3d}  loss={res['loss']:.4f}  "
            f"test_acc={res['test_acc']:.4f}  "
            f"sparsity={res['sparsity']:.3f}  time={res['time_s']:.2f}s"
        )

    # Summary table.
    print("\n" + "=" * 78)
    header = (
        f"{'config':<20}{'iters':>7}{'loss':>10}"
        f"{'test_acc':>10}{'sparsity':>10}{'time(s)':>10}"
    )
    print(header)
    print("-" * 78)
    for r in results:
        print(
            f"{r['name']:<20}{r['iters']:>7}{r['loss']:>10.4f}"
            f"{r['test_acc']:>10.4f}{r['sparsity']:>10.3f}{r['time_s']:>10.2f}"
        )
    print("=" * 78)
    # Plot convergence curves across configurations.
    plot_convergence(results)


if __name__ == "__main__":
    main()
