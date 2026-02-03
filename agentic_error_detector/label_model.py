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
        columns = set(base_rules.keys()) | set(clean_rules.keys())
        for column in columns:
            base_col = base_rules.get(column, [])
            clean_col = clean_rules.get(column, [])
            total_rows = self._infer_total_rows(base_col, clean_col)
            if total_rows <= 0:
                continue

            rule_entries: List[Tuple[str, str, np.ndarray]] = []

            for result in base_col:
                agent_name = str(result.get("agent", ""))
                violations = result.get("violations", [])
                z_vec = self._build_z_vector_from_violations(violations, total_rows)
                rule_entries.append(("dirty", agent_name, z_vec))

            for result in clean_col:
                rule_name = str(result.get("agent", ""))
                violations = result.get("violations", [])
                z_vec = self._build_z_vector_from_violations(violations, total_rows)
                rule_entries.append(("clean", rule_name, z_vec))

            if not rule_entries:
                continue

            Z = np.stack([entry[2] for entry in rule_entries], axis=0)
            alpha, beta, posterior = self._run_em(Z)
            self.cell_posteriors[column] = posterior

            for idx, (rule_type, rule_name, z_vec) in enumerate(rule_entries):
                idx1 = z_vec == 1
                idx0 = ~idx1
                p = posterior
                one_p = 1.0 - p

                n11 = float(np.sum(p[idx1]))
                n10 = float(np.sum(p[idx0]))
                n01 = float(np.sum(one_p[idx1]))
                n00 = float(np.sum(one_p[idx0]))

                support_error = n11 + n10
                support_clean = n01 + n00

                key = (column, rule_name, rule_type)
                self.rule_stats[key] = RuleStats(
                    column=column,
                    rule_name=rule_name,
                    rule_type=rule_type,
                    alpha=float(alpha[idx]),
                    beta=float(beta[idx]),
                    support_error=float(support_error),
                    support_clean=float(support_clean),
                )

    def get_cell_posteriors(self) -> Dict[str, np.ndarray]:
        return self.cell_posteriors

    def get_rule_statistics(self) -> List[RuleStats]:
        return list(self.rule_stats.values())

    def _run_em(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = Z.astype(np.int64)
        num_rules, num_cells = Z.shape

        alpha = np.full(num_rules, 0.8, dtype=float)
        beta = np.full(num_rules, 0.8, dtype=float)
        posterior = np.full(num_cells, self.error_prior, dtype=float)

        pi = self.error_prior

        eps = 1e-8

        for _ in range(self.max_iter):
            log_p1 = np.full(num_cells, np.log(pi + eps), dtype=float)
            log_p0 = np.full(num_cells, np.log(1.0 - pi + eps), dtype=float)

            for r in range(num_rules):
                z = Z[r]
                idx1 = z == 1
                idx0 = ~idx1

                if np.any(idx1):
                    log_p1[idx1] += np.log(alpha[r] + eps)
                    log_p0[idx1] += np.log(1.0 - beta[r] + eps)
                if np.any(idx0):
                    log_p1[idx0] += np.log(1.0 - alpha[r] + eps)
                    log_p0[idx0] += np.log(beta[r] + eps)

            max_log = np.maximum(log_p1, log_p0)
            log_p1 -= max_log
            log_p0 -= max_log

            new_posterior = 1.0 / (1.0 + np.exp(log_p0 - log_p1))
            new_posterior = np.clip(new_posterior, eps, 1.0 - eps)

            for r in range(num_rules):
                z = Z[r]
                idx1 = z == 1
                idx0 = ~idx1

                n11 = float(np.sum(new_posterior[idx1]))
                n10 = float(np.sum(new_posterior[idx0]))
                n01 = float(np.sum((1.0 - new_posterior)[idx1]))
                n00 = float(np.sum((1.0 - new_posterior)[idx0]))

                alpha[r] = (n11 + self.a1) / (n11 + n10 + self.a1 + self.b1 + eps)
                beta[r] = (n00 + self.a0) / (n00 + n01 + self.a0 + self.b0 + eps)

            delta = float(np.max(np.abs(new_posterior - posterior)))
            posterior = new_posterior
            pi = float(np.mean(posterior))

            if delta < self.tol:
                break

        return alpha, beta, posterior

    @staticmethod
    def _infer_total_rows(
        base_col: List[Dict[str, Any]],
        clean_col: List[Dict[str, Any]],
    ) -> int:
        for result in base_col:
            total = result.get("total_rows")
            if isinstance(total, int) and total > 0:
                return total
        for result in clean_col:
            total = result.get("total_rows")
            if isinstance(total, int) and total > 0:
                return total
        return 0

    @staticmethod
    def _build_z_vector_from_violations(
        violations: List[Dict[str, Any]],
        total_rows: int,
    ) -> np.ndarray:
        z = np.zeros(total_rows, dtype=np.int64)
        for item in violations:
            idx = item.get("row_index")
            if isinstance(idx, (int, np.integer)) and 0 <= idx < total_rows:
                z[int(idx)] = 1
        return z

