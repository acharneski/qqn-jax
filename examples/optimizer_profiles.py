"""Optimizer profile registry for the MLP comparison benchmark.

This module externalizes the (large) set of QQN / baseline optimizer
configurations from ``fashion_mnist_mlp_comparison.py`` so the headline
suite can be re-tuned without touching the experiment driver.

Two things are exported:

  * ``build_runners(...)`` — constructs the ``{name: runner_lambda}`` map
    and the companion ``{name: qqn_kwargs}`` map (used purely for the
    evaluation-cost display estimate). Only the profiles whose names appear
    in ``ENABLED`` are returned, so enabling/disabling a variant is a
    one-line edit to the ``ENABLED`` list below.

  * ``ENABLED`` — the ordered list of profile names that are actually run.
    Comment a name out (or delete it) to disable that variant; the order of
    this list also determines the order in which the variants execute.

To add a new variant: register it in ``_PROFILES`` (a factory taking the
shared context object and returning ``(runner_lambda, qqn_kwargs)``) and add
its name to ``ENABLED``.
"""

import optax

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
# Index of ENABLED optimizer profiles.
#
# Only the names listed here are built and run. Reorder to change execution
# order; comment a line out to disable a variant. Every name MUST have a
# matching entry in ``_PROFILES`` below.
# --------------------------------------------------------------------------
ENABLED = [
    # --- Baseline QQN + negative-control spline variants ---
    "QQN",
    "QQN-S",
    "QQN-BT",
    "QQN-BT-S",
    # --- L-BFGS memory-depth sweep ---
    "QQN-L20",
    "QQN-L50",
    "QQN-L80",
    "QQN-Cheap",
    "QQN-L120",
    "QQN-L160",
    "QQN-L80And",
    "QQN-L80-BT",
    # --- Alternative oracles ---
    "QQN-Mom",
    "QQN-Mom-S",
    "QQN-Sec",
    "QQN-And",
    "QQN-L50And",
    # --- Regions / best-of-breed stacks ---
    "QQN-TR",
    "QQN-Fast",
    "QQN-Max",
    "QQN-Champ",
    "QQN-Box",
    "QQN-Lean",
    # --- Baselines ---
    "SGD",
    "Adam",
    "L-BFGS",
]


# --------------------------------------------------------------------------
# Profile factories.
#
# Each factory receives a ``ctx`` namespace carrying the shared experiment
# objects/parameters and returns ``(runner_lambda, qqn_kwargs)`` where:
#   * ``runner_lambda`` is a zero-arg callable returning the standard result
#     tuple (it closes over the shared loss/params/stop).
#   * ``qqn_kwargs`` is the small dict used ONLY for the evaluation-cost
#     display estimate (see ``_estimate_evals_per_iter``).
# --------------------------------------------------------------------------


def _profiles():
    """Return the ``{name: factory}`` registry.

    ``ctx`` attributes used by the factories:
      loss_fn, params0, maxiter, stop, sgd_lr, adam_lr,
      run_qqn, run_optax, run_optax_lbfgs
    """

    def QQN(ctx):
        return (
            lambda: ctx.run_qqn(ctx.loss_fn, ctx.params0, ctx.maxiter, stop=ctx.stop),
            {},
        )

    def QQN_S(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn, ctx.params0, ctx.maxiter, stop=ctx.stop, spline=True
            ),
            {"spline": True},
        )

    def QQN_BT(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                line_search="backtracking",
                stop=ctx.stop,
            ),
            {"line_search": "backtracking"},
        )

    def QQN_BT_S(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                line_search="backtracking",
                stop=ctx.stop,
                spline=True,
            ),
            {"line_search": "backtracking", "spline": True},
        )

    def QQN_L20(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=LBFGSOracle(history_size=20),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_L50(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=LBFGSOracle(history_size=50),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_L80(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=LBFGSOracle(history_size=80),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_Cheap(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                line_search="backtracking",
                line_search_options={
                    "init_step": 1.0,
                    "shrink": 0.7,
                    "c1": 1e-4,
                    "max_iter": 20,
                },
                oracle=LBFGSOracle(history_size=80),
                stop=ctx.stop,
            ),
            {
                "line_search": "backtracking",
                "line_search_options": {"max_iter": 20},
            },
        )

    def QQN_L120(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=LBFGSOracle(history_size=120),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_L160(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=LBFGSOracle(history_size=160),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_L80And(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=Fallback(
                    [LBFGSOracle(history_size=80), AndersonOracle(window=5)]
                ),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_L80_BT(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                line_search="backtracking",
                line_search_options={
                    "init_step": 1.0,
                    "shrink": 0.7,
                    "c1": 1e-3,
                    "max_iter": 40,
                },
                oracle=LBFGSOracle(history_size=80),
                stop=ctx.stop,
            ),
            {
                "line_search": "backtracking",
                "line_search_options": {"max_iter": 40},
            },
        )

    def QQN_Mom(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=MomentumOracle(beta=0.9),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_Mom_S(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=MomentumOracle(beta=0.9),
                stop=ctx.stop,
                spline=True,
            ),
            {"spline": True},
        )

    def QQN_Sec(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=SecantOracle(),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_And(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=AndersonOracle(window=5),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_L50And(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=Fallback(
                    [LBFGSOracle(history_size=50), AndersonOracle(window=5)]
                ),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_TR(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                region=TrustRegion(radius=1.0, adaptive=True),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_Fast(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=LBFGSOracle(history_size=120),
                region=TrustRegion(radius=2.0, adaptive=False),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_Max(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=Fallback(
                    [LBFGSOracle(history_size=80), AndersonOracle(window=5)]
                ),
                region=TrustRegion(radius=2.0, adaptive=False),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_Champ(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=LBFGSOracle(history_size=120),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_Box(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                region=BoxRegion(lo=-2.0, hi=2.0),
                stop=ctx.stop,
            ),
            {},
        )

    def QQN_Lean(ctx):
        return (
            lambda: ctx.run_qqn(
                ctx.loss_fn,
                ctx.params0,
                ctx.maxiter,
                oracle=LBFGSOracle(history_size=80),
                stop=ctx.stop,
            ),
            {},
        )

    def SGD(ctx):
        return (
            lambda: ctx.run_optax(
                ctx.loss_fn,
                ctx.params0,
                optax.sgd(learning_rate=ctx.sgd_lr),
                ctx.maxiter,
                stop=ctx.stop,
            ),
            {},
        )

    def Adam(ctx):
        return (
            lambda: ctx.run_optax(
                ctx.loss_fn,
                ctx.params0,
                optax.adam(learning_rate=ctx.adam_lr),
                ctx.maxiter,
                stop=ctx.stop,
            ),
            {},
        )

    def LBFGS(ctx):
        return (
            lambda: ctx.run_optax_lbfgs(
                ctx.loss_fn, ctx.params0, ctx.maxiter, stop=ctx.stop
            ),
            {},
        )

    return {
        "QQN": QQN,
        "QQN-S": QQN_S,
        "QQN-BT": QQN_BT,
        "QQN-BT-S": QQN_BT_S,
        "QQN-L20": QQN_L20,
        "QQN-L50": QQN_L50,
        "QQN-L80": QQN_L80,
        "QQN-Cheap": QQN_Cheap,
        "QQN-L120": QQN_L120,
        "QQN-L160": QQN_L160,
        "QQN-L80And": QQN_L80And,
        "QQN-L80-BT": QQN_L80_BT,
        "QQN-Mom": QQN_Mom,
        "QQN-Mom-S": QQN_Mom_S,
        "QQN-Sec": QQN_Sec,
        "QQN-And": QQN_And,
        "QQN-L50And": QQN_L50And,
        "QQN-TR": QQN_TR,
        "QQN-Fast": QQN_Fast,
        "QQN-Max": QQN_Max,
        "QQN-Champ": QQN_Champ,
        "QQN-Box": QQN_Box,
        "QQN-Lean": QQN_Lean,
        "SGD": SGD,
        "Adam": Adam,
        "L-BFGS": LBFGS,
    }


def build_runners(ctx):
    """Build ``(runners, qqn_kwarg_map)`` for every ENABLED profile.

    Args:
        ctx: a namespace carrying the shared experiment objects/parameters
            (loss_fn, params0, maxiter, stop, sgd_lr, adam_lr) and the three
            runner helpers (run_qqn, run_optax, run_optax_lbfgs).

    Returns:
        ``(runners, qqn_kwarg_map)`` ordered dicts keyed by profile name,
        containing only the profiles listed in ``ENABLED``.
    """
    registry = _profiles()
    runners = {}
    qqn_kwarg_map = {}
    for name in ENABLED:
        if name not in registry:
            raise KeyError(f"ENABLED profile {name!r} has no factory in _PROFILES.")
        runner, kwargs = registry[name](ctx)
        runners[name] = runner
        qqn_kwarg_map[name] = kwargs
    return runners, qqn_kwarg_map
