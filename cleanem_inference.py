import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from cleanem_models import CellRecord, EvidenceObservation, ValueRecord


@dataclass
class EvidenceMatrix:
    """Dense feature view over per-cell weak evidence."""

    cell_keys: List[str]
    feature_names: List[str]
    values: np.ndarray
    key_to_index: Dict[str, int]


@dataclass
class CalibrationTrace:
    """Diagnostics for active-learning calibration."""

    selected_cell_keys: List[str] = field(default_factory=list)
    selected_indices: List[int] = field(default_factory=list)
    selected_labels: Dict[str, int] = field(default_factory=dict)
    queried_cells: List[Dict[str, object]] = field(default_factory=list)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    fitted: bool = False
    reason: str = ""
    feature_count: int = 0
    label_count: int = 0


def _clip_probability(value: float, low: float = 1e-4, high: float = 1 - 1e-4) -> float:
    return float(min(high, max(low, value)))


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return float(math.log(p / (1.0 - p)))


def _sigmoid_array(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits.astype(float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _prediction_confidence(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)
    return np.clip(1.0 - entropy, 0.0, 1.0)


def _signed_strength(observation: EvidenceObservation) -> float:
    if observation.polarity == "dirty":
        return float(observation.strength)
    if observation.polarity == "clean":
        return -float(observation.strength)
    return 0.0


def _feature_name(scope: str, observation: EvidenceObservation) -> str:
    return f"{scope}:{observation.family}:{observation.source_id}"


def build_cell_evidence_matrix(
    cell_registry: Dict[str, CellRecord],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
) -> EvidenceMatrix:
    """Convert value-level and cell-level observations into signed evidence features.

    A positive feature value means the source emitted dirty evidence; a negative
    value means the same source emitted clean evidence. The model learns whether
    that signed source is reliable for the current dataset.
    """

    cell_keys = list(cell_registry.keys())
    feature_index: Dict[str, int] = {}
    sparse_rows: List[Dict[int, float]] = []

    for cell_key in cell_keys:
        record = cell_registry[cell_key]
        row_features: Dict[int, float] = {}
        observations = []
        observations.extend(value_observations.get(record.value_key, []))
        observations.extend(cell_observations.get(cell_key, []))

        for observation in observations:
            signed = _signed_strength(observation)
            if signed == 0.0:
                continue
            name = _feature_name(observation.target_scope, observation)
            idx = feature_index.setdefault(name, len(feature_index))
            row_features[idx] = row_features.get(idx, 0.0) + signed
        sparse_rows.append(row_features)

    values = np.zeros((len(cell_keys), len(feature_index)), dtype=float)
    for row_idx, row_features in enumerate(sparse_rows):
        for feature_idx, feature_value in row_features.items():
            values[row_idx, feature_idx] = feature_value

    feature_names = [None] * len(feature_index)
    for name, idx in feature_index.items():
        feature_names[idx] = name

    return EvidenceMatrix(
        cell_keys=cell_keys,
        feature_names=[str(name) for name in feature_names],
        values=values,
        key_to_index={key: idx for idx, key in enumerate(cell_keys)},
    )


def select_active_queries(
    evidence_matrix: EvidenceMatrix,
    budget: int,
    ridge: float = 1.0,
) -> List[str]:
    """Select cells by greedy Bayesian D-optimal design over evidence vectors.

    The objective favors labels that reduce uncertainty about evidence-source
    reliability for many cells, not merely cells whose current posterior is near
    0.5.
    """

    budget = max(0, int(budget))
    if budget == 0 or evidence_matrix.values.size == 0:
        return []

    x_all = evidence_matrix.values
    nonzero_mask = np.linalg.norm(x_all, axis=1) > 1e-9
    if not np.any(nonzero_mask):
        return []

    pattern_representatives: Dict[Tuple[Tuple[int, float], ...], int] = {}
    for row_idx in np.where(nonzero_mask)[0].tolist():
        nz = np.nonzero(np.abs(x_all[row_idx]) > 1e-9)[0]
        signature = tuple((int(idx), round(float(x_all[row_idx, idx]), 3)) for idx in nz)
        pattern_representatives.setdefault(signature, int(row_idx))

    candidate_indices = list(pattern_representatives.values())
    if not candidate_indices:
        return []

    dim = x_all.shape[1]
    inverse_information = np.eye(dim, dtype=float) / max(float(ridge), 1e-6)
    selected: List[int] = []
    remaining = set(candidate_indices)

    for _ in range(min(budget, len(candidate_indices))):
        best_idx = None
        best_score = -float("inf")
        for candidate_idx in remaining:
            x = x_all[candidate_idx]
            score = float(x @ inverse_information @ x.T)
            if score > best_score:
                best_score = score
                best_idx = candidate_idx

        if best_idx is None or best_score <= 1e-12:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        x = x_all[best_idx]
        x_col = x.reshape(-1, 1)
        denom = 1.0 + float(x @ inverse_information @ x.T)
        if denom > 1e-12:
            inverse_information = inverse_information - (
                inverse_information @ x_col @ x_col.T @ inverse_information
            ) / denom

    return [evidence_matrix.cell_keys[idx] for idx in selected]


def select_adaptive_stratified_queries(
    evidence_matrix: EvidenceMatrix,
    cell_columns: Dict[str, str],
    budget: int,
    already_selected: Optional[List[str]] = None,
    current_probabilities: Optional[Dict[str, float]] = None,
    column_risk: Optional[Dict[str, float]] = None,
    ridge: float = 1.0,
) -> List[str]:
    """Select one adaptive batch while preserving per-column coverage.

    The first pass gives the least-covered columns one representative each.
    Remaining slots use a global score combining D-optimal leverage, posterior
    uncertainty, conflicting evidence, dirty-signal strength, and semantic risk.
    Repeated evidence signatures within a column are collapsed so a batch does
    not spend labels on interchangeable cells.
    """

    budget = max(0, int(budget))
    if budget == 0 or evidence_matrix.values.size == 0:
        return []

    already_selected = list(already_selected or [])
    selected_set: Set[str] = set(already_selected)
    current_probabilities = current_probabilities or {}
    column_risk = column_risk or {}
    x_all = evidence_matrix.values

    # Keep one deterministic representative for each evidence pattern per
    # column. Identical patterns provide the same calibration information.
    representatives: Dict[Tuple[str, Tuple[Tuple[int, float], ...]], int] = {}
    for row_idx, cell_key in enumerate(evidence_matrix.cell_keys):
        if cell_key in selected_set:
            continue
        x = x_all[row_idx]
        nz = np.nonzero(np.abs(x) > 1e-9)[0]
        if len(nz) == 0:
            continue
        column = cell_columns.get(cell_key)
        if column is None:
            continue
        signature = tuple(
            (int(feature_idx), round(float(x[feature_idx]), 3))
            for feature_idx in nz
        )
        representatives.setdefault((column, signature), int(row_idx))

    remaining: Set[int] = set(representatives.values())
    if not remaining:
        return []

    dim = x_all.shape[1]
    information = np.eye(dim, dtype=float) * max(float(ridge), 1e-6)
    selected_indices = [
        evidence_matrix.key_to_index[key]
        for key in already_selected
        if key in evidence_matrix.key_to_index
    ]
    if selected_indices:
        selected_values = x_all[selected_indices]
        information += selected_values.T @ selected_values
    inverse_information = np.linalg.pinv(information)

    selected_counts: Dict[str, int] = {}
    for key in already_selected:
        column = cell_columns.get(key)
        if column is not None:
            selected_counts[column] = selected_counts.get(column, 0) + 1

    def candidate_score(candidate_idx: int) -> float:
        cell_key = evidence_matrix.cell_keys[candidate_idx]
        column = cell_columns[cell_key]
        x = x_all[candidate_idx]
        leverage = max(0.0, float(x @ inverse_information @ x.T))
        leverage_score = leverage / (1.0 + leverage)
        probability = float(current_probabilities.get(cell_key, 0.5))
        uncertainty = 1.0 - min(1.0, 2.0 * abs(probability - 0.5))
        positive = float(np.maximum(x, 0.0).sum())
        negative = float(np.maximum(-x, 0.0).sum())
        total = positive + negative
        disagreement = 0.0 if total <= 1e-12 else 2.0 * min(positive, negative) / total
        dirty_signal = 0.0 if total <= 1e-12 else positive / total
        semantic_risk = float(np.clip(column_risk.get(column, 0.5), 0.0, 1.0))
        return (
            0.35 * leverage_score
            + 0.25 * uncertainty
            + 0.15 * disagreement
            + 0.15 * dirty_signal
            + 0.10 * semantic_risk
        )

    def choose_best(candidate_pool: Set[int]) -> Optional[int]:
        if not candidate_pool:
            return None
        return max(
            candidate_pool,
            key=lambda idx: (
                candidate_score(idx),
                -idx,
            ),
        )

    def accept(candidate_idx: int) -> None:
        nonlocal inverse_information
        batch_indices.append(candidate_idx)
        remaining.remove(candidate_idx)
        cell_key = evidence_matrix.cell_keys[candidate_idx]
        column = cell_columns[cell_key]
        selected_counts[column] = selected_counts.get(column, 0) + 1
        x = x_all[candidate_idx]
        x_col = x.reshape(-1, 1)
        denom = 1.0 + float(x @ inverse_information @ x.T)
        if denom > 1e-12:
            inverse_information = inverse_information - (
                inverse_information @ x_col @ x_col.T @ inverse_information
            ) / denom

    batch_indices: List[int] = []
    available_columns = sorted(
        {cell_columns[evidence_matrix.cell_keys[idx]] for idx in remaining},
        key=lambda column: (
            selected_counts.get(column, 0),
            -float(column_risk.get(column, 0.5)),
            column,
        ),
    )

    # Hard coverage pass: at most one new label per column in this batch,
    # prioritizing columns that have received fewer labels in earlier rounds.
    for column in available_columns:
        if len(batch_indices) >= budget:
            break
        column_candidates = {
            idx
            for idx in remaining
            if cell_columns[evidence_matrix.cell_keys[idx]] == column
        }
        best_idx = choose_best(column_candidates)
        if best_idx is not None:
            accept(best_idx)

    # Spend any remaining budget on globally informative, non-duplicate cells.
    while remaining and len(batch_indices) < budget:
        best_idx = choose_best(remaining)
        if best_idx is None:
            break
        accept(best_idx)

    return [evidence_matrix.cell_keys[idx] for idx in batch_indices]


def _fit_prior_centered_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_prior: float,
    l2: float = 0.25,
    intercept_l2: float = 0.25,
    max_iters: int = 2000,
) -> Tuple[float, np.ndarray, str]:
    n_features = x_train.shape[1]
    base_logit = _logit(base_prior)
    theta = np.zeros(n_features + 1, dtype=float)
    theta[0] = base_logit
    prior_theta = np.zeros_like(theta)
    prior_theta[0] = base_logit

    x_aug = np.column_stack([np.ones(x_train.shape[0], dtype=float), x_train])
    penalties = np.full(n_features + 1, float(l2), dtype=float)
    penalties[0] = float(intercept_l2)

    spectral_norm = float(np.linalg.norm(x_aug, ord=2)) if x_aug.size else 1.0
    lipschitz = 0.25 * spectral_norm * spectral_norm + float(np.max(penalties))
    step_size = 1.0 / max(lipschitz, 1e-6)
    reason = "projected_gradient_converged"

    for _ in range(max_iters):
        logits = np.clip(x_aug @ theta, -30.0, 30.0)
        probs = _sigmoid_array(logits)
        gradient = x_aug.T @ (probs - y_train) + penalties * (theta - prior_theta)
        old_theta = theta.copy()
        theta -= step_size * gradient
        theta[1:] = np.maximum(theta[1:], 0.0)
        if float(np.linalg.norm(theta - old_theta)) < 1e-6:
            break
    else:
        reason = "max_iters"

    return float(theta[0]), theta[1:], reason


def estimate_calibrated_posteriors(
    value_registry: Dict[str, ValueRecord],
    cell_registry: Dict[str, CellRecord],
    evidence_matrix: EvidenceMatrix,
    selected_cell_keys: List[str],
    selected_labels: Dict[str, int],
    base_prior: float,
    queried_cells: List[Dict[str, object]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], CalibrationTrace]:
    """Fit a small calibrated label model and infer posteriors for all cells."""

    queried_cells = queried_cells or []
    clipped_prior = _clip_probability(min(0.30, max(0.01, base_prior)))
    base_logit = _logit(clipped_prior)
    x_all = evidence_matrix.values
    n_cells = len(evidence_matrix.cell_keys)

    trace = CalibrationTrace(
        selected_cell_keys=list(selected_cell_keys),
        selected_indices=[
            evidence_matrix.key_to_index[key]
            for key in selected_cell_keys
            if key in evidence_matrix.key_to_index
        ],
        selected_labels=dict(selected_labels),
        queried_cells=list(queried_cells),
        intercept=base_logit,
        fitted=False,
        reason="no_active_labels",
        feature_count=len(evidence_matrix.feature_names),
        label_count=len(selected_labels),
    )

    train_indices = [
        evidence_matrix.key_to_index[key]
        for key in selected_cell_keys
        if key in selected_labels and key in evidence_matrix.key_to_index
    ]
    if train_indices and x_all.shape[1] > 0:
        y_train = np.array([float(selected_labels[evidence_matrix.cell_keys[idx]]) for idx in train_indices])
        x_train = x_all[train_indices]
        intercept, weights, reason = _fit_prior_centered_logistic(x_train, y_train, clipped_prior)
        logits = intercept + x_all @ weights
        probabilities = _sigmoid_array(logits)
        trace.intercept = float(intercept)
        trace.feature_weights = {
            name: float(weight)
            for name, weight in zip(evidence_matrix.feature_names, weights)
            if abs(float(weight)) > 1e-8
        }
        trace.fitted = True
        trace.reason = reason
        if len(set(int(label) for label in selected_labels.values())) == 1:
            trace.reason = f"{reason};single_class_labels"
    else:
        probabilities = np.full(n_cells, clipped_prior, dtype=float)
        if x_all.shape[1] == 0:
            trace.reason = "no_evidence_features"

    confidences = _prediction_confidence(probabilities)
    cell_posteriors: Dict[str, Dict[str, float]] = {}
    for idx, cell_key in enumerate(evidence_matrix.cell_keys):
        cell_posteriors[cell_key] = {
            "posterior": float(probabilities[idx]),
            "confidence": float(confidences[idx]),
        }

    value_priors: Dict[str, Dict[str, float]] = {}
    for value_key, record in value_registry.items():
        posterior_values = []
        confidence_values = []
        for row_idx in record.row_indices:
            cell_key = f"{row_idx}::{record.column}"
            bundle = cell_posteriors.get(cell_key)
            if bundle:
                posterior_values.append(bundle["posterior"])
                confidence_values.append(bundle["confidence"])
        if posterior_values:
            value_priors[value_key] = {
                "prior": float(np.mean(posterior_values)),
                "confidence": float(np.mean(confidence_values)),
            }
        else:
            value_priors[value_key] = {"prior": clipped_prior, "confidence": 0.0}

    return value_priors, cell_posteriors, trace
