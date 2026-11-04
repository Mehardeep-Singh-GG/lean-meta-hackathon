"""
campaigns/rigorous_benchmarks/ablations/__init__.py
===================================================
Ablation matrix harness and factor definitions for Project LEAN.
"""

from .ablation_matrix import (
    AblationFactorConfig,
    AblationPoint,
    STEP_STACKING_LADDER,
    DROP_ONE_ABLATIONS,
    AblationMatrix,
    get_ablation_run,
    list_all_ablation_runs,
)

__all__ = [
    "AblationFactorConfig",
    "AblationPoint",
    "STEP_STACKING_LADDER",
    "DROP_ONE_ABLATIONS",
    "AblationMatrix",
    "get_ablation_run",
    "list_all_ablation_runs",
]
