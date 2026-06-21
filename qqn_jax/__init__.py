"""QQN (Quasi-Quadratic-Newton) optimizer for JAX.

QQN combines steepest descent and L-BFGS through a quadratic interpolation
path:

    d(t) = t(1-t)(-∇f) + t²(-H∇f)

and uses a line search over this path to select the optimal blend of
gradient and quasi-Newton directions.
"""

from qqn_jax.solver import QQN, QQNState
from qqn_jax.line_search import strong_wolfe_search, backtracking_search
from qqn_jax.spline_search import spline_search
from qqn_jax.oracles import (
    Oracle,
    OracleInfo,
    LBFGSOracle,
    MomentumOracle,
    ShampooOracle,
    Fallback,
)
from qqn_jax.regions import (
    Region,
    RegionInfo,
    IdentityRegion,
    BoxRegion,
    OrthantRegion,
    TrustRegion,
    Sequential,
)

__version__ = "0.1.0"

__all__ = [
    "QQN",
    "QQNState",
    "strong_wolfe_search",
    "backtracking_search",
    "spline_search",
    "Oracle",
    "OracleInfo",
    "LBFGSOracle",
    "MomentumOracle",
    "ShampooOracle",
    "Fallback",
    "Region",
    "RegionInfo",
    "IdentityRegion",
    "BoxRegion",
    "OrthantRegion",
    "TrustRegion",
    "Sequential",
    "__version__",
]
