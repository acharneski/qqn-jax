"""MNIST validation experiment: QQN vs SGD vs Adam vs L-BFGS.

Trains a small softmax (logistic regression) classifier on a subset of
MNIST and compares the convergence behaviour of QQN against three
common baselines: SGD, Adam, and Optax's L-BFGS.

The optimization is framed as a *full-batch* deterministic problem so
that the comparison is apples-to-apples for the second-order methods
(QQN and L-BFGS), which assume a smooth, deterministic objective.

Data loading:
    The script tries to load MNIST via ``torchvision`` or ``tensorflow``
    if available. If neither is installed, it falls back to a synthetic
    Gaussian-blob "MNIST-like" dataset so the experiment always runs.

Run with:  python examples/mnist_comparison.py
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qqn_jax import QQN
from qqn_jax.oracles import (
    LBFGSOracle,
    MomentumOracle,
    ShampooOracle,
    Fallback,
)
from qqn_jax.regions import (
    BoxRegion,
    TrustRegion,
    OrthantRegion,
    Sequential,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def _load_mnist_numpy(n_train: int, n_test: int, n_classes: int):
    """Try to load a real MNIST subset; fall back to synthetic data.

    Returns:
        (X_train, y_train, X_test, y_test) as numpy arrays with images
        flattened to shape (N, 784) and float32 in [0, 1].
    """
    # --- Attempt 1: tensorflow_datasets / keras ---
    try:
        from tensorflow.keras.datasets import mnist  # type: ignore

        (xtr, ytr), (xte, yte) = mnist.load_data()
        xtr = xtr.reshape(xtr.shape[0], -1).astype(np.float32) / 255.0
        xte = xte.reshape(xte.shape[0], -1).astype(np.float32) / 255.0
        return _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes)
    except Exception:
        pass

    # --- Attempt 2: torchvision ---
    try:
        from torchvision import datasets  # type: ignore

        train = datasets.MNIST(root="./_mnist_data", train=True, download=True)
        test = datasets.MNIST(root="./_mnist_data", train=False, download=True)
        xtr = train.data.numpy().reshape(len(train), -1).astype(np.float32) / 255.0
        ytr = train.targets.numpy()
        xte = test.data.numpy().reshape(len(test), -1).astype(np.float32) / 255.0
        yte = test.targets.numpy()
        return _subset(xtr, ytr, xte, yte, n_train, n_test, n_classes)
    except Exception:
        pass

    # --- Fallback: synthetic "MNIST-like" Gaussian blobs ---
    print("[data] Real MNIST unavailable; using synthetic Gaussian blobs.")
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
# Model: softmax / multinomial logistic regression
# --------------------------------------------------------------------------


def init_params(dim: int, n_classes: int, key) -> jnp.ndarray:
    """Flat parameter vector: W (dim x n_classes) followed by b (n_classes)."""
    w = 0.01 * jax.random.normal(key, (dim * n_classes,))
    b = jnp.zeros((n_classes,))
    return jnp.concatenate([w, b])


def _unpack(params, dim, n_classes):
    w = params[: dim * n_classes].reshape(dim, n_classes)
    b = params[dim * n_classes :]
    return w, b


def make_loss(X, y, dim, n_classes, l2: float = 1e-4):
    """Build a full-batch cross-entropy loss ``f(params) -> scalar``."""
    Y = jax.nn.one_hot(y, n_classes)

    def loss(params):
        w, b = _unpack(params, dim, n_classes)
        logits = X @ w + b
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.mean(jnp.sum(Y * log_probs, axis=-1))
        reg = 0.5 * l2 * jnp.sum(params**2)
        return ce + reg

    return loss


def accuracy(params, X, y, dim, n_classes):
    w, b = _unpack(params, dim, n_classes)
    logits = X @ w + b
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == y).astype(jnp.float32))


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------


def run_qqn(loss_fn, params0, maxiter):
    """Run QQN and return (final_params, history_of_losses, wall_time)."""
    return _run_qqn_configured(loss_fn, params0, maxiter)


def _run_qqn_configured(
    loss_fn,
    params0,
    maxiter,
    line_search="armijo",
    line_search_options=None,
    oracle="lbfgs",
    region=None,
    spline: bool = False,
):
    """Run a configurable QQN variant.

    Exposes QQN's swappable components — the *oracle* (curvature source),
    the *line search* (step-size selection), and the *region* (projective
    constraint) — so we can benchmark several QQN flavours side-by-side.
    """
    solver = QQN(
        loss_fn,
        maxiter=maxiter,
        line_search=line_search,
        line_search_options=line_search_options,
        oracle=oracle,
        region=region,
        spline=spline,
    )

    # Run one update at a time to record the loss trajectory.
    state = solver.init_state(params0)
    params = params0
    history = [float(state.value)]
    t0 = time.perf_counter()
    update = jax.jit(solver.update)
    for _ in range(maxiter):
        params, state = update(params, state)
        history.append(float(state.value))
        if bool(state.done):
            break
    wall = time.perf_counter() - t0
    return params, history, wall


def run_optax(loss_fn, params0, optimizer, maxiter):
    """Run a generic Optax optimizer; returns (params, history, wall)."""
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    opt_state = optimizer.init(params0)

    @jax.jit
    def step(params, opt_state):
        value, grad = value_and_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    params = params0
    history = [float(loss_fn(params))]
    t0 = time.perf_counter()
    for _ in range(maxiter):
        params, opt_state, value = step(params, opt_state)
        history.append(float(value))
    wall = time.perf_counter() - t0
    return params, history, wall


def run_optax_lbfgs(loss_fn, params0, maxiter):
    """Run Optax's L-BFGS (with zoom line search) on the full-batch loss."""
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
        return params, opt_state, value

    params = params0
    history = [float(loss_fn(params))]
    t0 = time.perf_counter()
    for _ in range(maxiter):
        params, opt_state, value = step(params, opt_state)
        history.append(float(value))
    wall = time.perf_counter() - t0
    return params, history, wall


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------


def main():
    # Problem configuration
    n_classes = 10
    n_train = 5000
    n_test = 1000
    maxiter = 100

    print("=== MNIST optimizer comparison: QQN vs SGD vs Adam vs L-BFGS ===")
    print("    (QQN variants: line search / oracle / region)")
    print(
        f"  classes={n_classes}  n_train={n_train}  n_test={n_test}  "
        f"maxiter={maxiter}\n"
    )

    xtr, ytr, xte, yte = _load_mnist_numpy(n_train, n_test, n_classes)
    dim = xtr.shape[1]

    X_train = jnp.asarray(xtr)
    y_train = jnp.asarray(ytr)
    X_test = jnp.asarray(xte)
    y_test = jnp.asarray(yte)

    loss_fn = make_loss(X_train, y_train, dim, n_classes)

    # Shared initial parameters so every optimizer starts identically.
    params0 = init_params(dim, n_classes, jax.random.PRNGKey(42))

    runners = {
        # --- Baseline QQN (L-BFGS oracle, Armijo line search) ---
        "QQN": lambda: run_qqn(loss_fn, params0, maxiter),
        # --- QQN with a strong-Wolfe line search (tighter curvature) ---
        "QQN-SW": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="strong_wolfe",
        ),
        # --- QQN with backtracking line search (cheap, robust) ---
        "QQN-BT": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
        ),
        # --- QQN with Hager-Zhang line search (efficient Wolfe variant) ---
        "QQN-HZ": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="hager_zhang",
        ),
        # --- QQN with a cubic Hermite spline line search ---
        "QQN-Spln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            spline=True,
        ),
        # --- QQN with a momentum oracle instead of L-BFGS ---
        "QQN-Mom": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.9),
        ),
        # --- A/B (oracle): momentum with heavier damping for stability ---
        "QQN-Mom99": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.99),
        ),
        # --- A/B (oracle): lighter momentum damping (completes beta sweep) ---
        "QQN-Mom50": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.5),
        ),
        # --- A/B (oracle): lighter L-BFGS history (size 5) — cheap memory ---
        "QQN-L5": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=5),
        ),
        # --- QQN with a deeper L-BFGS history (richer curvature memory) ---
        "QQN-L20": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=20),
        ),
        # --- A/B (oracle): even deeper L-BFGS memory (size 50) ---
        "QQN-L50": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
        ),
        # --- A/B (oracle): very deep L-BFGS memory (size 100) — extends the
        #     monotone history sweep (L5<L10<L20<L50) to probe diminishing
        #     returns at the extreme end of curvature memory. ---
        "QQN-L100": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=100),
        ),
        # --- QQN with a Shampoo (structure-aware) oracle ---
        "QQN-Shmp": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=ShampooOracle(update_freq=1),
        ),
        # --- A/B (oracle): Shampoo with amortized updates (freq=5) — cheaper ---
        "QQN-Shmp5": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=ShampooOracle(update_freq=5),
        ),
        # --- A/B (oracle): Shampoo with heavy amortization (freq=20) to test
        #     whether the 174s cost is dominated by preconditioner updates;
        #     extends the update_freq sweep (1 -> 5 -> 20). ---
        "QQN-Shmp20": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=ShampooOracle(update_freq=20),
        ),
        # --- QQN with a Fallback oracle: L-BFGS, else momentum ---
        "QQN-Fall": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=Fallback([LBFGSOracle(history_size=10), MomentumOracle()]),
        ),
        # --- QQN constrained to a box region (bounded weights) ---
        "QQN-Box": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=BoxRegion(lo=-2.0, hi=2.0),
        ),
        # --- QQN with an adaptive trust-region sphere ---
        "QQN-TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=1.0, adaptive=True),
        ),
        # --- A/B (region): tighter adaptive trust-region (radius=0.5) ---
        "QQN-TR05": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=0.5, adaptive=True),
        ),
        # --- A/B (region): wider adaptive trust-region (radius=2.0) ---
        "QQN-TR20": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=2.0, adaptive=True),
        ),
        # --- A/B (region): very tight adaptive trust-region (radius=0.25),
        #     extends the radius sweep (0.25 -> 0.5 -> 1.0 -> 2.0) to probe
        #     whether over-constraining the step harms convergence. ---
        "QQN-TR025": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=0.25, adaptive=True),
        ),
        # --- A/B (region): fixed (non-adaptive) trust-region control ---
        "QQN-TRfix": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=1.0, adaptive=False),
        ),
        # --- QQN with an orthant region (OWL-QN-style sparsity) ---
        "QQN-Orth": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=OrthantRegion(),
        ),
        # --- Combined: strong-Wolfe search + adaptive trust-region ---
        "QQN-SW+TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="strong_wolfe",
            region=TrustRegion(radius=1.0, adaptive=True),
        ),
        # --- Best-of-breed: deep L-BFGS (size 20) + Hager-Zhang line search.
        #     Combines the fastest-converging oracle (L20: 53 iters) with the
        #     efficient Wolfe line search to probe for a new pareto winner. ---
        "QQN-L20HZ": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="hager_zhang",
            oracle=LBFGSOracle(history_size=20),
        ),
        # --- Best-of-breed: deep L-BFGS (size 20) + backtracking line search.
        #     Pairs the strongest oracle with the cheapest robust search
        #     (BT was the fastest wall-time at 0.500s) for speed. ---
        "QQN-L20BT": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=LBFGSOracle(history_size=20),
        ),
        # --- Best-of-breed: the strongest oracle (L50: lowest loss 1.025e-01)
        #     paired with cheap robust backtracking. Probes whether the deepest
        #     practical history reaches a new pareto point on loss vs. time. ---
        "QQN-L50BT": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=LBFGSOracle(history_size=50),
        ),
        # --- Best-of-breed: L50 oracle + Hager-Zhang Wolfe line search.
        #     Combines the lowest-loss oracle with the efficient Wolfe variant
        #     to push final loss below the current 1.025e-01 frontier. ---
        "QQN-L50HZ": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="hager_zhang",
            oracle=LBFGSOracle(history_size=50),
        ),
        # --- Best-of-breed: L50 oracle + adaptive trust-region. Tests whether
        #     curvature-rich steps benefit from the trust-region safeguard. ---
        "QQN-L50TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=True),
        ),
        # --- Combined: deep L-BFGS oracle + box constraint ---
        "QQN-L20Box": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=20),
            region=BoxRegion(lo=-2.0, hi=2.0),
        ),
        # --- Combined: Fallback oracle + Sequential (box ∩ trust) region ---
        "QQN-Stack": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=Fallback([LBFGSOracle(history_size=10), MomentumOracle(beta=0.9)]),
            region=Sequential([BoxRegion(lo=-2.0, hi=2.0), TrustRegion(radius=2.0)]),
        ),
        "SGD": lambda: run_optax(
            loss_fn, params0, optax.sgd(learning_rate=0.5), maxiter
        ),
        "Adam": lambda: run_optax(
            loss_fn, params0, optax.adam(learning_rate=0.05), maxiter
        ),
        "L-BFGS": lambda: run_optax_lbfgs(loss_fn, params0, maxiter),
    }

    results = {}
    for name, runner in runners.items():
        params, history, wall = runner()
        train_acc = float(accuracy(params, X_train, y_train, dim, n_classes))
        test_acc = float(accuracy(params, X_test, y_test, dim, n_classes))
        # Fraction of (near-)zero weights — illuminating for the orthant region.
        sparsity = float(jnp.mean((jnp.abs(params) < 1e-6).astype(jnp.float32)))
        results[name] = {
            "final_loss": history[-1],
            "iters": len(history) - 1,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "wall": wall,
            "sparsity": sparsity,
            "history": history,
        }

    # --- Summary table ---
    print(
        f"{'optimizer':<10}{'final_loss':>14}{'iters':>8}"
        f"{'train_acc':>12}{'test_acc':>11}{'sparsity':>10}{'time(s)':>10}"
    )
    print("-" * 75)
    for name, r in results.items():
        print(
            f"{name:<10}{r['final_loss']:>14.6e}{r['iters']:>8}"
            f"{r['train_acc']:>12.4f}{r['test_acc']:>11.4f}"
            f"{r['sparsity']:>10.4f}{r['wall']:>10.3f}"
        )

    # --- Loss trajectory (compact ASCII view at log10 scale) ---
    print("\nLoss trajectory (log10, sampled):")
    sample_points = 10
    for name, r in results.items():
        hist = r["history"]
        idxs = np.linspace(0, len(hist) - 1, sample_points).astype(int)
        vals = [f"{np.log10(max(hist[i], 1e-12)):6.2f}" for i in idxs]
        print(f"  {name:<8} " + " ".join(vals))

    # --- A/B comparison report -------------------------------------------
    # Each pair isolates a single variable (oracle depth, region radius,
    # line search, etc.) against a named baseline so the effect is causal.
    ab_pairs = [
        (
            "oracle: L-BFGS history",
            "QQN-L5",
            "QQN",
            "QQN-L20",
            "QQN-L50",
            "QQN-L100",
        ),
        ("oracle: momentum beta", "QQN-Mom50", "QQN-Mom", "QQN-Mom99"),
        (
            "oracle: Shampoo update_freq",
            "QQN-Shmp",
            "QQN-Shmp5",
            "QQN-Shmp20",
        ),
        (
            "region: trust radius",
            "QQN-TR025",
            "QQN-TR05",
            "QQN-TR",
            "QQN-TR20",
        ),
        ("region: trust adaptivity", "QQN-TRfix", "QQN-TR"),
        (
            "search: L20 line search",
            "QQN-L20",
            "QQN-L20BT",
            "QQN-L20HZ",
        ),
        (
            "search: L50 line search",
            "QQN-L50",
            "QQN-L50BT",
            "QQN-L50HZ",
        ),
        (
            "best-of-breed: L50 region",
            "QQN-L50",
            "QQN-L50TR",
        ),
    ]

    print("\nA/B controlled comparisons (vs first column = baseline):")

    for title, *variants in ab_pairs:
        present = [v for v in variants if v in results]
        if len(present) < 2:
            continue
        base = results[present[0]]
        print(f"  [{title}]")
        for v in present:
            r = results[v]
            d_iters = r["iters"] - base["iters"]
            d_wall = r["wall"] - base["wall"]
            marker = " (baseline)" if v == present[0] else ""
            print(
                f"    {v:<11} iters={r['iters']:>3} (Δ{d_iters:+d})"
                f"  loss={r['final_loss']:.3e}"
                f"  time={r['wall']:.3f}s (Δ{d_wall:+.3f}){marker}"
            )

    # Optional: save a matplotlib plot if available.
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(7, 5))
        baselines = {"SGD", "Adam", "L-BFGS"}
        for name, r in results.items():
            if name in baselines:
                plt.semilogy(r["history"], label=name, linestyle="--", linewidth=2)
            else:
                plt.semilogy(r["history"], label=name, alpha=0.85)
        plt.xlabel("iteration")
        plt.ylabel("full-batch loss")
        plt.title("MNIST optimizer comparison (QQN variants vs baselines)")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        out = "mnist_comparison.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\n[plot] Saved convergence plot to {out}")
    except Exception:
        print("\n[plot] matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
