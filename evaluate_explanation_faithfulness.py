#!/usr/bin/env python3
"""Offline counterfactual faithfulness evaluation for CleanEM traces."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path


EPS = 1e-12


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(probability):
    p = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return math.log(p / (1.0 - p))


def decision_probability(logit_value, suspicious):
    posterior = sigmoid(logit_value)
    return posterior if suspicious else 1.0 - posterior


def evidence_weight(item):
    value = item.get("weighted_logit")
    return float(value) if value is not None else 0.0


def support_evidence(record):
    suspicious = record.get("decision") == "suspicious"
    evidence = list(record.get("evidence") or [])
    if suspicious:
        support = [e for e in evidence if float(e.get("posterior_contribution") or 0.0) > EPS]
        support.sort(key=lambda e: float(e.get("posterior_contribution") or 0.0), reverse=True)
    else:
        support = [e for e in evidence if float(e.get("posterior_contribution") or 0.0) < -EPS]
        support.sort(key=lambda e: float(e.get("posterior_contribution") or 0.0))
    return support


def stable_rng(seed, cell_key):
    digest = hashlib.sha256(f"{seed}:{cell_key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def mean(values):
    return sum(values) / len(values) if values else None


def evaluate_record(record, random_trials, seed, max_rationale):
    evidence = list(record.get("evidence") or [])
    support = support_evidence(record)
    if not support:
        return None

    rationale = support[:max_rationale]
    suspicious = record.get("decision") == "suspicious"
    threshold = float(record.get("threshold", 0.5))
    posterior = float(record.get("posterior", 0.5))
    raw_full_logit = record.get("full_logit")
    full_logit = float(raw_full_logit) if raw_full_logit is not None else logit(posterior)
    intercept = full_logit - sum(evidence_weight(e) for e in evidence)
    rationale_weight = sum(evidence_weight(e) for e in rationale)

    q_full = decision_probability(full_logit, suspicious)
    removed_logit = full_logit - rationale_weight
    only_logit = intercept + rationale_weight
    q_removed = decision_probability(removed_logit, suspicious)
    q_only = decision_probability(only_logit, suspicious)

    full_prediction = sigmoid(full_logit) >= threshold
    removed_prediction = sigmoid(removed_logit) >= threshold
    only_prediction = sigmoid(only_logit) >= threshold

    # Match rationale length but sample from every non-zero evidence item,
    # following random-rationale baselines rather than sampling only from the
    # already filtered decision-support set (which degenerates when <=4 items
    # support the decision).
    random_pool = [
        e for e in evidence
        if abs(float(e.get("posterior_contribution") or 0.0)) > EPS
    ]
    rng = stable_rng(seed, str(record.get("cell_key", "")))
    random_comprehensiveness = []
    random_sufficiency_gap = []
    random_flip = []
    k = len(rationale)
    for _ in range(random_trials):
        sampled = rng.sample(random_pool, k) if len(random_pool) > k else list(random_pool)
        sampled_weight = sum(evidence_weight(e) for e in sampled)
        random_removed_logit = full_logit - sampled_weight
        random_only_logit = intercept + sampled_weight
        random_comprehensiveness.append(max(
            0.0,
            q_full - decision_probability(random_removed_logit, suspicious),
        ))
        random_sufficiency_gap.append(max(
            0.0,
            q_full - decision_probability(random_only_logit, suspicious),
        ))
        random_flip.append(float((sigmoid(random_removed_logit) >= threshold) != full_prediction))

    total_support_weight = sum(abs(evidence_weight(e)) for e in support)
    displayed_support_weight = sum(abs(evidence_weight(e)) for e in rationale)
    return {
        "decision": record.get("decision", ""),
        "comprehensiveness": max(0.0, q_full - q_removed),
        "sufficiency_gap": max(0.0, q_full - q_only),
        "decision_flip": float(removed_prediction != full_prediction),
        "sufficiency_agreement": float(only_prediction == full_prediction),
        "contribution_coverage": (
            displayed_support_weight / total_support_weight
            if total_support_weight > EPS else 0.0
        ),
        "random_comprehensiveness": mean(random_comprehensiveness),
        "random_sufficiency_gap": mean(random_sufficiency_gap),
        "random_decision_flip": mean(random_flip),
        "support_count": len(support),
        "displayed_count": len(rationale),
    }


def summarize(rows, decision):
    selected = rows if decision == "all" else [r for r in rows if r["decision"] == decision]
    result = {"group": decision, "cells": len(selected)}
    metric_names = [
        "comprehensiveness",
        "sufficiency_gap",
        "decision_flip",
        "sufficiency_agreement",
        "contribution_coverage",
        "random_comprehensiveness",
        "random_sufficiency_gap",
        "random_decision_flip",
    ]
    for name in metric_names:
        result[name] = mean([float(row[name]) for row in selected])
    return result


def parse_trace_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use DATASET=PATH for --trace")
    dataset, path = value.split("=", 1)
    return dataset.strip(), Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True, type=parse_trace_spec)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--random_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max_rationale", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset, trace_path in args.trace:
        evaluated = []
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                result = evaluate_record(
                    json.loads(line),
                    args.random_trials,
                    args.seed,
                    max(1, args.max_rationale),
                )
                if result is not None:
                    evaluated.append(result)
        for group in ("suspicious", "not_suspicious", "all"):
            summary = summarize(evaluated, group)
            summary["dataset"] = dataset
            summaries.append(summary)

    json_path = args.output_dir / "faithfulness_summary.json"
    csv_path = args.output_dir / "faithfulness_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False, allow_nan=False)
    fieldnames = ["dataset", "group", "cells"] + [
        key for key in summaries[0] if key not in {"dataset", "group", "cells"}
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
