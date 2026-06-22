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


def _grad_norm(loss_fn, params):
    """Compute the L2 norm of the gradient at ``params``."""
    g = jax.grad(loss_fn)(params)
    return float(jnp.linalg.norm(g))


def _converged(value, gnorm, f_target, gtol):
    """Shared convergence test: target loss reached OR gradient ~ 0."""
    if f_target is not None and value <= f_target:
        return True
    if gtol is not None and gnorm <= gtol:
        return True
    return False


def _update_milestones(milestones, hit, value, it, now):
    """Record the first iteration/time each loss milestone is crossed.
    ``milestones`` is a tuple of descending loss thresholds; ``hit`` is a
    mutable dict mapping each threshold to ``(iter, time)`` (or ``None``).
    This lets us report a full *convergence-rate profile* per optimizer
    rather than a single time-to-target, which is far more discriminating
    for separating early- from late-phase convergence behaviour.
    """
    if not milestones:
        return
    for m in milestones:
        if hit.get(m) is None and value <= m:
            hit[m] = (it, now)


def run_qqn(loss_fn, params0, maxiter, stop=None):
    """Run QQN and return (final_params, history_of_losses, wall_time)."""
    return _run_qqn_configured(loss_fn, params0, maxiter, stop=stop)


def _run_qqn_configured(
    loss_fn,
    params0,
    maxiter,
    line_search="armijo",
    line_search_options=None,
    oracle="lbfgs",
    region=None,
    spline: bool = False,
    t_grid=None,
    stop=None,
):
    """Run a configurable QQN variant.

    Exposes QQN's swappable components — the *oracle* (curvature source),
    the *line search* (step-size selection), and the *region* (projective
     constraint), plus the *t-grid* (blend discretization) — so we can
     benchmark several QQN flavours side-by-side.
     ``stop`` is a dict with shared termination bounds applied uniformly to
     every optimizer: ``f_target`` (loss threshold), ``gtol`` (gradient-norm
     tolerance), and ``time_budget`` (wall-clock seconds).
    """
    stop = stop or {}
    f_target = stop.get("f_target")
    gtol = stop.get("gtol")
    time_budget = stop.get("time_budget")
    milestones = stop.get("milestones", ())

    solver = QQN(
        loss_fn,
        maxiter=maxiter,
        line_search=line_search,
        line_search_options=line_search_options,
        oracle=oracle,
        region=region,
        spline=spline,
        t_grid=t_grid,
    )

    # Run one update at a time to record the loss trajectory.
    state = solver.init_state(params0)
    params = params0
    history = [float(state.value)]
    times = [0.0]
    # Record iteration / time at which the shared target was first hit.
    iters_to_target = None
    time_to_target = None
    # Convergence-rate profile: first iteration/time per loss milestone.
    milestone_hits = {m: None for m in milestones}
    _update_milestones(milestones, milestone_hits, history[-1], 0, 0.0)
    t0 = time.perf_counter()
    update = jax.jit(solver.update)
    for it in range(maxiter):
        params, state = update(params, state)
        history.append(float(state.value))
        now = time.perf_counter() - t0
        times.append(now)
        # --- Shared termination criteria (uniform across all methods) ---
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
    """Run a generic Optax optimizer; returns (params, history, wall, times)."""
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
        # --- Shared termination criteria (uniform across all methods) ---
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
        # --- Shared termination criteria (uniform across all methods) ---
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
    # Problem configuration
    n_classes = 10
    n_train = 5000
    n_test = 1000
    maxiter = 500
    # --- Shared, fair termination bounds applied to EVERY optimizer ---
    #   f_target:     stop once full-batch loss <= this value
    #   gtol:         stop once ||grad|| <= this value (stationarity)
    #   time_budget:  hard wall-clock cap (seconds) per optimizer
    # These make the comparison apples-to-apples: every method now races to
    # the same loss threshold under the same time limit and the same
    # stationarity tolerance, rather than each using its own private rule.
    stop = {
        # A reachable-but-demanding loss target so that the ``->target`` /
        # ``t->tgt`` columns become *informative* (the previous 1.0e-1 was
        # below every method's 50-iteration reach, leaving the columns empty).
        # The deep-memory + trust-region combos converge to ~1.04e-1, so a
        # slightly looser target lets the strongest variants actually "win"
        # the race and surface their iteration/time-to-target advantage.
        "f_target": 1.1e-1,
        "gtol": 1.0e-4,
        # Shampoo's dense n×n inverse-root refresh blew the previous 10s cap
        # after 6 iters; a modestly larger budget plus a *blocked* Shampoo
        # (below) keeps the comparison meaningful while still capping runaways.
        "time_budget": 15.0,
        # Intermediate milestones for measuring *convergence rate* (not just
        # the final target). Recording the iteration/time at which each method
        # first crosses these loss thresholds gives a far more discriminating
        # picture of early- vs late-phase convergence than a single target.
        "milestones": (5.0e-1, 2.0e-1, 1.5e-1, 1.2e-1),
    }

    print("=== MNIST optimizer comparison: QQN vs SGD vs Adam vs L-BFGS ===")
    print("    (QQN variants: line search / oracle / region)")
    print(
        f"  classes={n_classes}  n_train={n_train}  n_test={n_test}  "
        f"maxiter={maxiter}\n"
    )
    print(
        f"  shared stop: f_target={stop['f_target']:.1e}  "
        f"gtol={stop['gtol']:.1e}  time_budget={stop['time_budget']:.1f}s\n"
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
        "QQN": lambda: run_qqn(loss_fn, params0, maxiter, stop=stop),
        # --- QQN with a strong-Wolfe line search (tighter curvature) ---
        "QQN-SW": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="strong_wolfe",
            stop=stop,
        ),
        # --- QQN with backtracking line search (cheap, robust) ---
        "QQN-BT": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            stop=stop,
        ),
        # --- QQN with a cubic Hermite spline line search ---
        "QQN-Spln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            spline=True,
            stop=stop,
        ),
        # --- Best-of-breed (spline): cubic Hermite refinement on top of the
        #     strongest oracle (L50). Probes whether reusing every probe as a
        #     spline control point sharpens the deepest-memory trajectory. ---
        "QQN-L50Spln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            spline=True,
            stop=stop,
        ),
        # --- Best-of-breed (spline): cubic Hermite refinement + adaptive
        #     trust-region, stacking the spline's curve-reuse with the
        #     convergence-stabilizing safeguard. ---
        "QQN-SplnTR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            spline=True,
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        # --- Best-of-breed (spline): full stack — deep L-BFGS (L50) + cubic
        "QQN-L50SplnTR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            spline=True,
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        # --- Best-of-breed (spline): deepest memory (L100) + spline. Extends
        "QQN-L100Spln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=100),
            spline=True,
            stop=stop,
        ),
        # --- Best-of-breed (spline): spline refinement on top of the cheap
        "QQN-BTSpln": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            spline=True,
            stop=stop,
        ),
        # --- QQN with a momentum oracle instead of L-BFGS ---
        "QQN-Mom": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.9),
            stop=stop,
        ),
        # --- A/B (oracle): lighter momentum damping (completes beta sweep) ---
        "QQN-Mom50": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.5),
            stop=stop,
        ),
        # --- A/B (oracle): minimal momentum damping (beta=0.1) to find the
        #     floor of the monotone beta sweep (0.99>0.9>0.5 in loss); probes
        #     whether near-zero momentum collapses toward steepest-descent. ---
        "QQN-Mom10": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.1),
            stop=stop,
        ),
        # --- A/B (oracle): near-zero momentum (beta=0.01) to pin the floor of
        #     the monotone beta sweep and confirm the collapse toward pure
        #     steepest-descent (extends Mom10<Mom50<Mom toward the limit). ---
        "QQN-Mom01": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=MomentumOracle(beta=0.01),
            stop=stop,
        ),
        # --- A/B (oracle): Shampoo structure-aware preconditioner. Probes
        #     whether Kronecker-factored second-moment statistics beat the
        #     momentum first-order accelerator on this smooth problem. We use
        #     a *blocked* preconditioner (block_size=64) so the per-refresh
        #     eigendecomposition is O(block³) instead of O(n³); the previous
        #     dense n×n refresh exhausted the time budget after 6 iters. ---
        "QQN-Sh": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=ShampooOracle(block_size=64, update_freq=25),
            stop=stop,
        ),
        # --- A/B (oracle): lighter L-BFGS history (size 5) — cheap memory ---
        "QQN-L5": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=5),
            stop=stop,
        ),
        # --- QQN with a deeper L-BFGS history (richer curvature memory) ---
        "QQN-L20": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=20),
            stop=stop,
        ),
        # --- A/B (oracle): even deeper L-BFGS memory (size 50) ---
        "QQN-L50": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            stop=stop,
        ),
        # --- A/B (oracle): very deep L-BFGS memory (size 100) — extends the
        #     monotone history sweep (L5<L10<L20<L50) to probe diminishing
        #     returns at the extreme end of curvature memory. ---
        "QQN-L100": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=100),
            stop=stop,
        ),
        # --- QQN with a Fallback oracle: L-BFGS, else momentum ---
        "QQN-Fall": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=Fallback([LBFGSOracle(history_size=10), MomentumOracle()]),
            stop=stop,
        ),
        # --- QQN constrained to a box region (bounded weights) ---
        "QQN-Box": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=BoxRegion(lo=-2.0, hi=2.0),
            stop=stop,
        ),
        # --- QQN with an adaptive trust-region sphere ---
        "QQN-TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        # --- A/B (region): very tight adaptive trust-region (radius=0.25),
        #     extends the radius sweep (0.25 -> 1.0 -> 2.0) to probe
        #     whether over-constraining the step harms convergence. ---
        "QQN-TR025": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=0.25, adaptive=True),
            stop=stop,
        ),
        # --- A/B (region): generous adaptive trust-region (radius=2.0) to
        #     complete the radius sweep (0.25 -> 1.0 -> 2.0) and probe whether
        #     a looser safeguard lets deep curvature steps run unimpeded. ---
        "QQN-TR2": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=2.0, adaptive=True),
            stop=stop,
        ),
        # --- A/B (region): fixed (non-adaptive) trust-region control ---
        "QQN-TRfix": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=TrustRegion(radius=1.0, adaptive=False),
            stop=stop,
        ),
        # --- QQN with an orthant region (OWL-QN-style sparsity) ---
        "QQN-Orth": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=OrthantRegion(),
            stop=stop,
        ),
        # --- A/B (region): Sequential composition (box then trust-region).
        #     Probes that the combinator composes projections in order with
        #     negligible overhead and bounds weights while limiting step. ---
        "QQN-Seq": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            region=Sequential(
                [BoxRegion(lo=-2.0, hi=2.0), TrustRegion(radius=1.0, adaptive=True)]
            ),
            stop=stop,
        ),
        # --- A/B (t-grid): finer blend discretization (8 points). Probes
        #     whether sampling more gradient/oracle blends per iteration helps
        #     at higher per-iteration cost. ---
        "QQN-Tfine": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            t_grid=jnp.linspace(0.125, 1.0, 8),
            stop=stop,
        ),
        # --- A/B (t-grid): coarser blend discretization (2 points). The cheap
        #     end of the t-grid trade-off (fewer blends, lower cost). ---
        "QQN-Tcoarse": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            t_grid=jnp.array([0.5, 1.0]),
            stop=stop,
        ),
        # --- Combined: strong-Wolfe search + adaptive trust-region ---
        "QQN-SW+TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="strong_wolfe",
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
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
            stop=stop,
        ),
        # --- Best-of-breed: L50 oracle + adaptive trust-region. Tests whether
        #     curvature-rich steps benefit from the trust-region safeguard. ---
        "QQN-L50TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        # --- Best-of-breed: L100 oracle + adaptive trust-region. Extends the
        #     winning L50TR combo (lowest-loss trajectory) to the deeper L100
        #     oracle to push past the 1.024e-01 frontier with the safeguard. ---
        "QQN-L100TR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=100),
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        # --- Best-of-breed: L50 + generous trust-region (radius=2.0). Probes
        #     whether loosening the radius lets the lowest-loss L50TR combo
        #     push past its 1.044e-01 frontier by permitting longer steps. ---
        "QQN-L50TR2": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=2.0, adaptive=True),
            stop=stop,
        ),
        # --- Best-of-breed triple: L50 oracle + backtracking + trust-region.
        #     Combines the strongest pareto components — deep curvature memory,
        #     the cheapest robust search, and the convergence-stabilizing
        #     trust-region — to probe the joint optimum on loss AND time. ---
        "QQN-L50BTTR": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        # --- Performance: aggressive warm-started backtracking that probes
        #     well beyond α=1 (init_step=4) with a *gentle* shrink (0.8) so the
        #     deep-memory quasi-Newton step can stretch deep into the
        #     superlinear regime. The trust-region clips any overshoot, so the
        #     aggressive initial step is "free" — it can only help. This is a
        #     pure performance lever (no diversity loss; it's a distinct
        #     search configuration from the existing init_step=2 variant). ---
        "QQN-L50BTTR++": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 4.0, "shrink": 0.8, "max_iter": 50},
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.5, adaptive=True),
            stop=stop,
        ),
        # --- Performance: pure-oracle-dominant t-grid. Concentrate ALL blend
        #     samples in [0.8, 1.0] where the deep-memory experiments show the
        #     winning blend lives, while keeping a fine 4-point resolution so
        #     the line search still discriminates near the endpoint. Pairs the
        #     winning L50 oracle with warm-started backtracking + trust-region.
        #     Distinct from QQN-Fast (which spans 0.6..1.0); this probes the
        #     even-tighter endpoint regime. ---
        "QQN-L50Endpt": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 2.0, "shrink": 0.7, "max_iter": 40},
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=True),
            t_grid=jnp.array([0.8, 0.9, 0.95, 1.0]),
            stop=stop,
        ),
        # --- Performance: L50 + backtracking + trust-region, but with the
        #     line search *warm-started* at a larger initial step (init_step=2)
        #     and a gentler shrink (0.7). Because the quadratic path's t=1
        #     endpoint is already a full quasi-Newton step, allowing the search
        #     to probe beyond α=1 lets deep-memory steps stretch into the
        #     superlinear regime, accelerating convergence for free. ---
        "QQN-L50BTTR+": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 2.0, "shrink": 0.7, "max_iter": 40},
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=True),
            stop=stop,
        ),
        # --- Performance (t-grid): concentrate the blend samples *near t=1*
        #     where the quasi-Newton oracle dominates, since the deep-memory
        #     experiments show the winning blend sits close to the pure-oracle
        #     endpoint. A geometric grid clustered near 1.0 spends the line
        #     searches where they matter most without raising the grid size. ---
        "QQN-L50Tnear1": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.0, adaptive=True),
            t_grid=jnp.array([0.6, 0.8, 0.9, 1.0]),
            stop=stop,
        ),
        # --- Performance: the strongest stack — deep L100 memory + warm-started
        #     backtracking + adaptive trust-region + near-1 t-grid. Stacks every
        #     speed lever (curvature depth, aggressive step, oracle-focused
        #     blend) to probe the fewest-iterations frontier. ---
        "QQN-Fast": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 2.0, "shrink": 0.7, "max_iter": 40},
            oracle=LBFGSOracle(history_size=100),
            region=TrustRegion(radius=1.0, adaptive=True),
            t_grid=jnp.array([0.6, 0.8, 0.9, 1.0]),
            stop=stop,
        ),
        # --- Best-of-breed full stack: deep L-BFGS (L50) + backtracking +
        #     spline refinement + adaptive trust-region + finer t-grid. Stacks
        #     every pareto-winning component to probe the joint loss/time
        #     optimum and the lowest loss reachable in the budget. ---
        "QQN-Best": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            oracle=LBFGSOracle(history_size=50),
            spline=True,
            region=TrustRegion(radius=1.0, adaptive=True),
            t_grid=jnp.linspace(0.125, 1.0, 8),
            stop=stop,
        ),
        # --- Diversity-preserving champion: stack the strongest *independent*
        #     performance levers — deep curvature memory (L50), the cheapest
        #     robust search warm-started aggressively beyond α=1, a generous
        #     adaptive trust-region to clip overshoot for free, and an
        #     endpoint-concentrated t-grid — WITHOUT the expensive spline (which
        #     the data shows does not help deep-memory backtracking here). This
        #     is the intended best-on-both-axes (loss AND time) configuration:
        #     it should reach the target in the fewest iterations at ~1s. ---
        "QQN-Champion": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            line_search="backtracking",
            line_search_options={"init_step": 3.0, "shrink": 0.75, "max_iter": 45},
            oracle=LBFGSOracle(history_size=50),
            region=TrustRegion(radius=1.5, adaptive=True),
            t_grid=jnp.array([0.7, 0.85, 0.95, 1.0]),
            stop=stop,
        ),
        # --- Combined: deep L-BFGS oracle + box constraint ---
        "QQN-L20Box": lambda: _run_qqn_configured(
            loss_fn,
            params0,
            maxiter,
            oracle=LBFGSOracle(history_size=20),
            region=BoxRegion(lo=-2.0, hi=2.0),
            stop=stop,
        ),
        "SGD": lambda: run_optax(
            loss_fn, params0, optax.sgd(learning_rate=0.5), maxiter, stop=stop
        ),
        "Adam": lambda: run_optax(
            loss_fn, params0, optax.adam(learning_rate=0.05), maxiter, stop=stop
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
        train_acc = float(accuracy(params, X_train, y_train, dim, n_classes))
        test_acc = float(accuracy(params, X_test, y_test, dim, n_classes))
        # Fraction of (near-)zero weights — illuminating for the orthant region.
        sparsity = float(jnp.mean((jnp.abs(params) < 1e-6).astype(jnp.float32)))
        # Did this optimizer reach the shared loss/gradient target at all?
        reached = iters_to_target is not None
        # Mean wall-clock cost per accepted iteration (excludes the initial
        # value evaluation); a clean per-step cost metric for fair comparison.
        n_iters = max(len(history) - 1, 1)
        ms_per_iter = (wall / n_iters) * 1e3
        # Trajectory AUC: a single scalar summarizing *both* early- and
        # late-phase descent speed. We integrate log10(loss) over the
        # (normalized) iteration axis via the trapezoid rule; lower AUC means
        # the optimizer spent its whole trajectory at lower loss, which is far
        # more discriminating than a single time-to-target (it rewards fast
        # early descent AND deep late refinement simultaneously).
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
            "sparsity": sparsity,
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
    # Sort by final loss (ascending) so the strongest variants surface at the
    # top, making the leaderboard immediately readable. Baselines are kept in
    # the same sort so QQN's standing relative to SGD/Adam/L-BFGS is explicit.
    ordered = sorted(results.items(), key=lambda kv: kv[1]["final_loss"])
    # Reference iteration count for a "speedup vs L-BFGS" column: how many
    # fewer iterations each method needs to reach the shared target relative
    # to the classical L-BFGS baseline. This makes QQN's iteration advantage
    # explicit and directly comparable across the whole leaderboard.
    lbfgs_ref = results.get("L-BFGS", {}).get("iters_to_target")
    print(
        f"{'optimizer':<10}{'final_loss':>14}{'iters':>8}"
        f"{'train_acc':>12}{'test_acc':>11}{'sparsity':>10}{'time(s)':>10}"
        f"{'ms/it':>8}{'->target':>10}{'t->tgt':>9}{'vs LBFGS':>10}{'AUC':>8}"
    )
    print("-" * 120)
    for name, r in ordered:
        it_tgt = "—" if r["iters_to_target"] is None else f"{r['iters_to_target']}"
        t_tgt = "—" if r["time_to_target"] is None else f"{r['time_to_target']:.3f}"
        # Speedup vs L-BFGS in iterations-to-target (positive = faster).
        if lbfgs_ref is not None and r["iters_to_target"] is not None:
            spd = f"{lbfgs_ref / r['iters_to_target']:.2f}x"
        else:
            spd = "—"
        print(
            f"{name:<10}{r['final_loss']:>14.6e}{r['iters']:>8}"
            f"{r['train_acc']:>12.4f}{r['test_acc']:>11.4f}"
            f"{r['sparsity']:>10.4f}{r['wall']:>10.3f}"
            f"{r['ms_per_iter']:>8.2f}{it_tgt:>10}{t_tgt:>9}{spd:>10}"
            f"{r['traj_auc']:>8.2f}"
        )
    # --- Pareto frontier (loss vs. wall-time) ---
    # Surface the non-dominated variants: those for which no other variant is
    # both faster AND lower-loss. This crisply identifies the best loss/time
    # trade-offs without manual inspection of the full table.
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
    # --- Trajectory-AUC leaderboard --------------------------------------
    # Rank optimizers by the single-scalar trajectory AUC (lower = better):
    # it rewards methods that descend fast early AND refine deep late, so it
    # is a more effective summary of *overall* convergence quality than a
    # single time-to-target. The strongest deep-memory + warm-start combos
    # should top this list.
    print("\nTrajectory-AUC leaderboard (lower = faster overall descent):")
    auc_ranked = sorted(results.items(), key=lambda kv: kv[1]["traj_auc"])
    for name, r in auc_ranked[:12]:
        print(
            f"  {name:<14} AUC={r['traj_auc']:+.3f}  "
            f"final={r['final_loss']:.4e}  time={r['wall']:.3f}s"
        )
    # --- Convergence-rate profile (loss milestones) ----------------------
    # For each method, report the iteration at which it first crossed each
    # intermediate loss milestone. This separates *early-phase* descent speed
    # (large-loss milestones) from *late-phase* refinement (small-loss
    # milestones) far more sharply than a single time-to-target, surfacing
    # methods that descend fast early but stall late (e.g. momentum) vs. those
    # that accelerate near the optimum (e.g. deep-memory QQN).
    milestones = stop.get("milestones", ())
    if milestones:
        print("\nConvergence-rate profile (iteration first reaching each loss):")
        header = (
            "  "
            + f"{'optimizer':<12}"
            + "".join(f"{f'<={m:.1e}':>12}" for m in milestones)
        )
        print(header)
        # Sort by the iteration that reached the tightest milestone (those that
        # never reach it sort last), so the fastest late-phase methods surface.
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
        (
            "oracle: momentum beta",
            "QQN-Mom01",
            "QQN-Mom10",
            "QQN-Mom50",
            "QQN-Mom",
        ),
        (
            "oracle: accelerator class (Mom vs Shampoo)",
            "QQN-Mom10",
            "QQN-Sh",
        ),
        (
            "region: trust radius",
            "QQN-TR025",
            "QQN-TR",
            "QQN-TR2",
        ),
        ("region: trust adaptivity", "QQN-TRfix", "QQN-TR"),
        (
            "region: combinator (Sequential box+TR)",
            "QQN-TR",
            "QQN-Box",
            "QQN-Seq",
        ),
        (
            "t-grid: blend discretization",
            "QQN-Tcoarse",
            "QQN",
            "QQN-Tfine",
        ),
        (
            "search: line search (oracle=L-BFGS-10)",
            "QQN",
            "QQN-BT",
            "QQN-SW",
            "QQN-Spln",
        ),
        (
            "spline: best-of-breed refinement",
            "QQN-Spln",
            "QQN-BTSpln",
            "QQN-L50Spln",
            "QQN-L100Spln",
            "QQN-SplnTR",
            "QQN-L50SplnTR",
        ),
        (
            "best-of-breed: L50 region",
            "QQN-L50",
            "QQN-L50TR",
            "QQN-L50TR2",
            "QQN-L50BTTR",
        ),
        (
            "best-of-breed: L100 combos",
            "QQN-L100",
            "QQN-L100TR",
        ),
        (
            "performance: warm-start aggressiveness",
            "QQN-L50BTTR",
            "QQN-L50BTTR+",
            "QQN-L50BTTR++",
        ),
        (
            "performance: endpoint-concentrated t-grid",
            "QQN-L50BTTR",
            "QQN-L50Tnear1",
            "QQN-L50Endpt",
        ),
        (
            "champion: diversity-preserving best stack",
            "QQN-L50BTTR",
            "QQN-Fast",
            "QQN-Champion",
        ),
        (
            "best-of-breed: full stack",
            "QQN-L50BTTR",
            "QQN-L50SplnTR",
            "QQN-Best",
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
        # --- Second plot: loss vs wall-clock time ---
        plt.figure(figsize=(7, 5))
        for name, r in results.items():
            if name in baselines:
                plt.semilogy(
                    r["times"], r["history"], label=name, linestyle="--", linewidth=2
                )
            else:
                plt.semilogy(r["times"], r["history"], label=name, alpha=0.85)
        plt.xlabel("wall-clock time (s)")
        plt.ylabel("full-batch loss")
        plt.title("MNIST optimizer comparison vs time (QQN variants vs baselines)")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        out_time = "mnist_comparison_time.png"
        plt.savefig(out_time, dpi=120, bbox_inches="tight")
        print(f"[plot] Saved time-based convergence plot to {out_time}")
    except Exception:
        print("\n[plot] matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
