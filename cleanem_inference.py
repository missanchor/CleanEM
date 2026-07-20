import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from cleanem_models import CellRecord, EvidenceContribution, EvidenceObservation, ValueRecord


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


def _sigmoid_scalar(logit: float) -> float:
    return float(_sigmoid_array(np.asarray([logit], dtype=float))[0])


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
    fixed_calibration: Optional[CalibrationTrace] = None,
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
    if fixed_calibration is not None and fixed_calibration.fitted and x_all.shape[1] > 0:
        weights = np.array(
            [float(fixed_calibration.feature_weights.get(name, 0.0)) for name in evidence_matrix.feature_names],
            dtype=float,
        )
        logits = float(fixed_calibration.intercept) + x_all @ weights
        probabilities = _sigmoid_array(logits)
        trace.intercept = float(fixed_calibration.intercept)
        trace.feature_weights = {
            name: float(weight)
            for name, weight in zip(evidence_matrix.feature_names, weights)
            if abs(float(weight)) > 1e-8
        }
        trace.fitted = True
        trace.reason = f"fixed_after_gating:{fixed_calibration.reason}"
    elif train_indices and x_all.shape[1] > 0:
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
        logits = np.full(n_cells, base_logit, dtype=float)
        if x_all.shape[1] == 0:
            trace.reason = "no_evidence_features"

    confidences = _prediction_confidence(probabilities)
    cell_posteriors: Dict[str, Dict[str, float]] = {}
    for idx, cell_key in enumerate(evidence_matrix.cell_keys):
        cell_posteriors[cell_key] = {
            "posterior": float(probabilities[idx]),
            "confidence": float(confidences[idx]),
            # Keep the exact evaluated model logit for faithful counterfactual
            # explanations. Reconstructing it from a clipped probability loses
            # information for saturated predictions.
            "logit": float(np.clip(logits[idx], -30.0, 30.0))
            if n_cells and x_all.shape[1] > 0
            else float(base_logit),
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


def compute_leave_one_evidence_out_contributions(
    cell_registry: Dict[str, CellRecord],
    value_observations: Dict[str, List[EvidenceObservation]],
    cell_observations: Dict[str, List[EvidenceObservation]],
    cell_posteriors: Dict[str, Dict[str, float]],
    calibration_trace: CalibrationTrace,
) -> Dict[str, List[EvidenceContribution]]:
    """Measure each observation's exact posterior contribution with LOEO.

    The fitted calibration parameters remain fixed. For observation ``e`` with
    signed feature value ``x_e`` and learned weight ``w_e``, its contribution is
    ``p(full) - p(full logit - w_e*x_e)``. Positive values support the dirty
    class, while negative values support the clean class. When several
    observations share a feature, each is removed independently rather than
    dropping the entire aggregated feature.
    """

    contributions: Dict[str, List[EvidenceContribution]] = {}
    weights = calibration_trace.feature_weights or {}

    for cell_key, record in cell_registry.items():
        posterior_bundle = cell_posteriors.get(cell_key, {})
        posterior = float(posterior_bundle.get("posterior", 0.5))
        full_logit = float(posterior_bundle.get("logit", _logit(posterior)))
        observations: List[EvidenceObservation] = []
        observations.extend(value_observations.get(record.value_key, []))
        observations.extend(cell_observations.get(cell_key, []))

        cell_contributions: List[EvidenceContribution] = []
        for observation in observations:
            feature_name = _feature_name(observation.target_scope, observation)
            signed_value = _signed_strength(observation)
            feature_weight = float(weights.get(feature_name, 0.0))
            weighted_logit = float(feature_weight * signed_value)
            posterior_without = _sigmoid_scalar(full_logit - weighted_logit)
            cell_contributions.append(EvidenceContribution(
                target_scope=observation.target_scope,
                source_id=observation.source_id,
                family=observation.family,
                polarity=observation.polarity,
                reason_code=observation.reason_code,
                strength=float(observation.strength),
                hard=bool(observation.hard),
                feature_name=feature_name,
                feature_weight=feature_weight,
                signed_feature_value=float(signed_value),
                weighted_logit=weighted_logit,
                posterior_without=posterior_without,
                posterior_contribution=float(posterior - posterior_without),
                metadata=dict(observation.metadata or {}),
            ))

        cell_contributions.sort(
            key=lambda item: (
                abs(item.posterior_contribution),
                item.hard,
                item.strength,
            ),
            reverse=True,
        )
        contributions[cell_key] = cell_contributions

    return contributions
