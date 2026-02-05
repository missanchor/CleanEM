from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class RuleStats:
    column: str
    rule_name: str
    rule_type: str
    alpha: float
    beta: float
    support_error: float
    support_clean: float


class EMLabelModel:
    """
    Deprecated placeholder kept for backward compatibility.
    """

    def __init__(
        self,
        error_prior: float = 0.05,
        max_iter: int = 50,
        tol: float = 1e-4,
        a1: float = 1.0,
        b1: float = 1.0,
        a0: float = 1.0,
        b0: float = 1.0,
    ) -> None:
        self.error_prior = error_prior
        self.max_iter = max_iter
        self.tol = tol
        self.a1 = a1
        self.b1 = b1
        self.a0 = a0
        self.b0 = b0
        self.rule_stats: Dict[Tuple[str, str, str], RuleStats] = {}
        self.cell_posteriors: Dict[str, np.ndarray] = {}

    def fit_from_vr_results(
        self,
        df: pd.DataFrame,
        base_rules: Dict[str, List[Dict[str, Any]]],
        clean_rules: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        self.rule_stats = {}
        self.cell_posteriors = {}

    def get_cell_posteriors(self) -> Dict[str, np.ndarray]:
        return self.cell_posteriors

    def get_rule_statistics(self) -> List[RuleStats]:
        return list(self.rule_stats.values())
