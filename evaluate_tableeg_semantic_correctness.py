#!/usr/bin/env python3
"""Evaluate CleanEM detection, explanation semantics, repair, and constraints."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


ERROR_TYPES = ("missing_value", "pattern_violation", "rule_violation")
MISSING_SOURCES = {"missing", "missing_token", "missing_value"}


def normalize(value):
    if value is None:
        return "<NA>"
    text = str(value).strip()
    if text.lower() in {"", "nan", "na", "n/a", "null", "none"}:
        return "<NA>"
    try:
        number = float(text)
        if not math.isfinite(number):
            return "<NA>"
        return str(int(number)) if number.is_integer() else f"{number:.12g}"
    except ValueError:
        return re.sub(r"\s+", " ", text).lower()


def exact_equal(left, right):
    if left is None or right is None:
        return left is None and right is None
    return str(left).strip() == str(right).strip()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def support_evidence(trace, max_items=4):
    evidence = list(trace.get("evidence") or [])
    if trace.get("decision") == "suspicious":
        support = [e for e in evidence if float(e.get("posterior_contribution") or 0.0) > 1e-12]
        support.sort(key=lambda e: float(e.get("posterior_contribution") or 0.0), reverse=True)
    else:
        support = [e for e in evidence if float(e.get("posterior_contribution") or 0.0) < -1e-12]
        support.sort(key=lambda e: float(e.get("posterior_contribution") or 0.0))
    return support[:max_items]


def evidence_error_type(evidence):
    source = str(evidence.get("source_id") or "").lower()
    family = str(evidence.get("family") or "").lower()
    reason = str(evidence.get("reason_code") or "").lower()
    if source in MISSING_SOURCES or family == "missing" or "missing" in reason:
        return "missing_value"
    relation_text = " ".join((source, family, reason))
    if any(token in relation_text for token in ("relationship", "contextual", "consistency")):
        return "rule_violation"
    return "pattern_violation"


def trace_error_type(trace):
    support = support_evidence(trace, 1)
    return evidence_error_type(support[0]) if support else None


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def parse_tuple_pair_rows(value):
    return [int(item) for item in re.findall(r"\d+", str(value or ""))]


def gold_determinant_columns(constraint):
    columns = []
    pattern = re.compile(
        r"The\s+([A-Za-z0-9_]+)\s+of\s+Row\s+\d+\s+is\s+equal\s+to\s+"
        r"the\s+([A-Za-z0-9_]+)\s+of\s+Row\s+\d+",
        re.IGNORECASE,
    )
    for left, right in pattern.findall(str(constraint or "")):
        if left.lower() == right.lower():
            columns.append(left)
    return columns


def rule_explanation_alignment(
    annotation,
    trace,
    row_id_to_index,
):
    constraint = str(annotation.get("constraint") or "").strip().lower()
    if not constraint:
        return None
    gold_target_row_id = str(annotation.get("row_id"))
    gold_reference_rows = {
        row_id_to_index[str(row_id)]
        for row_id in parse_tuple_pair_rows(annotation.get("tuple_pairs"))
        if str(row_id) != gold_target_row_id
        and str(row_id) in row_id_to_index
    }
    gold_determinants = {
        value.lower()
        for value in gold_determinant_columns(annotation.get("constraint"))
    }
    gold_dependent = str(annotation.get("column") or "").strip().lower()

    type_aligned = False
    column_aligned = False
    detail_available = False
    reference_available = False
    expected_value_matches = None
    determinant_matches = False
    dependent_matches = False
    column_pair_matches = False
    reference_hit_at_1 = False
    reference_hit_at_5 = False
    grounded_matches = False
    for evidence in support_evidence(trace, 4):
        if evidence_error_type(evidence) != "rule_violation":
            continue
        type_aligned = True
        metadata = evidence.get("metadata") or {}
        columns = []
        for key in (
            "context_column",
            "left_column",
            "right_column",
            "determinant_column",
            "dependent_column",
        ):
            if metadata.get(key):
                columns.append(str(metadata[key]))
        related = metadata.get("related_columns") or []
        if isinstance(related, str):
            related = [related]
        columns.extend(str(value) for value in related)
        if any(column.lower() in constraint for column in columns if column):
            column_aligned = True
        determinant = str(metadata.get("determinant_column") or "").strip()
        dependent = str(metadata.get("dependent_column") or "").strip()
        rule_text = str(
            metadata.get("rule_text")
            or metadata.get("description")
            or ""
        ).strip()
        detail_available = detail_available or bool(
            determinant and dependent and rule_text
        )
        determinant_match = (
            determinant.lower() in gold_determinants
            if determinant and gold_determinants
            else False
        )
        dependent_match = bool(
            dependent and dependent.lower() == gold_dependent
        )
        determinant_matches = determinant_matches or determinant_match
        dependent_matches = dependent_matches or dependent_match
        column_pair_match = determinant_match and dependent_match
        column_pair_matches = column_pair_matches or column_pair_match

        references = metadata.get("reference_rows") or []
        if not isinstance(references, list):
            references = [references]
        normalized_references = []
        for value in references:
            try:
                normalized_references.append(int(value))
            except (TypeError, ValueError):
                continue
        reference_available = reference_available or bool(normalized_references)
        hit_at_1 = bool(
            normalized_references
            and normalized_references[0] in gold_reference_rows
        )
        hit_at_5 = bool(
            set(normalized_references[:5]) & gold_reference_rows
        )
        reference_hit_at_1 = reference_hit_at_1 or hit_at_1
        reference_hit_at_5 = reference_hit_at_5 or hit_at_5

        candidate_match = None
        if metadata.get("expected_value") is not None:
            candidate_match = (
                normalize(metadata.get("expected_value"))
                == normalize(annotation.get("right_value"))
            )
            expected_value_matches = bool(
                candidate_match
                if expected_value_matches is None
                else expected_value_matches or candidate_match
            )
        grounded_matches = grounded_matches or bool(
            column_pair_match and hit_at_5 and candidate_match
        )
    return {
        "type_aligned": type_aligned,
        "column_aligned": column_aligned,
        "detail_available": detail_available,
        "reference_available": reference_available,
        "expected_value_matches": expected_value_matches,
        "determinant_matches": determinant_matches,
        "dependent_matches": dependent_matches,
        "column_pair_matches": column_pair_matches,
        "reference_hit_at_1": reference_hit_at_1,
        "reference_hit_at_5": reference_hit_at_5,
        "grounded_matches": grounded_matches,
        "gold_reference_rows": sorted(gold_reference_rows),
        "gold_determinants": sorted(gold_determinants),
        "gold_dependent": gold_dependent,
    }


def evaluate_dataset(name, trace_path, dataset_dir, excluded_columns=None):
    dataset_dir = Path(dataset_dir)
    excluded_columns = set(excluded_columns or [])
    dirty_rows = read_csv(dataset_dir / "dirty.csv")
    clean_rows = read_csv(dataset_dir / "clean.csv")
    row_map = read_csv(dataset_dir / "row_map.csv")
    annotations = read_csv(dataset_dir / "annotation_aligned.csv")
    row_id_to_index = {str(row["row_id"]): int(row["row_index"]) for row in row_map}
    if len(dirty_rows) != len(clean_rows):
        raise ValueError(f"{name}: dirty/clean row count mismatch")

    traces = {}
    with Path(trace_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trace = json.loads(line)
                traces[(int(trace["row_index"]), str(trace["column"]))] = trace

    ground_truth = set()
    for row_index, (dirty, clean) in enumerate(zip(dirty_rows, clean_rows)):
        for column in dirty:
            if (
                column not in excluded_columns
                and column in clean
                and normalize(dirty[column]) != normalize(clean[column])
            ):
                ground_truth.add((row_index, column))
    predicted = {
        key for key, trace in traces.items()
        if (
            trace.get("decision") == "suspicious"
            and key[1] not in excluded_columns
        )
    }
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    annotation_records = []
    type_confusion = Counter()
    repairs = []
    constraint_type_scores = []
    constraint_column_scores = []
    rule_detail_scores = []
    rule_reference_scores = []
    rule_expected_value_scores = []
    rule_determinant_scores = []
    rule_dependent_scores = []
    rule_column_pair_scores = []
    rule_reference_hit_at_1_scores = []
    rule_reference_hit_at_5_scores = []
    grounded_rule_scores = []
    gold_rule_violations = 0
    detected_rule_violations = 0
    for annotation in annotations:
        if str(annotation.get("valid_key", "")).lower() != "true":
            continue
        if str(annotation.get("table_difference", "")).lower() != "true":
            continue
        row_id = str(annotation["row_id"])
        column = str(annotation["column"])
        key = (row_id_to_index[row_id], column)
        trace = traces.get(key)
        detected = bool(trace and trace.get("decision") == "suspicious")
        gold_type = str(annotation.get("error_type") or "")
        predicted_type = trace_error_type(trace) if detected else None
        correct_type = detected and predicted_type == gold_type
        type_confusion[(gold_type, predicted_type or "not_detected")] += 1

        repair_candidate = trace.get("repair_candidate") if detected else None
        repair_recommended = bool(
            trace and trace.get("repair_recommended") and repair_candidate is not None
        )
        if repair_recommended:
            right_value = annotation.get("right_value")
            repairs.append({
                "exact": exact_equal(repair_candidate, right_value),
                "normalized": normalize(repair_candidate) == normalize(right_value),
            })

        alignment = None
        if gold_type == "rule_violation":
            gold_rule_violations += 1
        if gold_type == "rule_violation" and detected:
            detected_rule_violations += 1
            alignment = rule_explanation_alignment(
                annotation,
                trace,
                row_id_to_index,
            )
            if alignment is not None:
                constraint_type_scores.append(float(alignment["type_aligned"]))
                constraint_column_scores.append(float(alignment["column_aligned"]))
                rule_detail_scores.append(float(alignment["detail_available"]))
                rule_reference_scores.append(float(alignment["reference_available"]))
                rule_determinant_scores.append(
                    float(alignment["determinant_matches"])
                )
                rule_dependent_scores.append(
                    float(alignment["dependent_matches"])
                )
                rule_column_pair_scores.append(
                    float(alignment["column_pair_matches"])
                )
                rule_reference_hit_at_1_scores.append(
                    float(alignment["reference_hit_at_1"])
                )
                rule_reference_hit_at_5_scores.append(
                    float(alignment["reference_hit_at_5"])
                )
                grounded_rule_scores.append(
                    float(alignment["grounded_matches"])
                )
                if alignment["expected_value_matches"] is not None:
                    rule_expected_value_scores.append(
                        float(alignment["expected_value_matches"])
                    )

        annotation_records.append({
            "detected": detected,
            "gold_type": gold_type,
            "predicted_type": predicted_type,
            "correct_type": correct_type,
            "repair_recommended": repair_recommended,
            "rule_alignment": alignment,
        })

    valid_annotations = len(annotation_records)
    detected_annotations = sum(row["detected"] for row in annotation_records)
    correctly_typed = sum(row["correct_type"] for row in annotation_records)
    conditional_type_correct = sum(
        row["correct_type"] for row in annotation_records if row["detected"]
    )

    per_type = {}
    type_f1s = []
    for error_type in ERROR_TYPES:
        class_tp = type_confusion[(error_type, error_type)]
        class_fn = sum(
            count for (gold, pred), count in type_confusion.items()
            if gold == error_type and pred != error_type
        )
        class_fp = sum(
            count for (gold, pred), count in type_confusion.items()
            if pred == error_type and gold != error_type
        )
        class_precision = safe_div(class_tp, class_tp + class_fp)
        class_recall = safe_div(class_tp, class_tp + class_fn)
        class_f1 = f1(class_precision, class_recall)
        support = sum(
            count for (gold, _), count in type_confusion.items() if gold == error_type
        )
        if support:
            type_f1s.append(class_f1)
        per_type[error_type] = {
            "support": support,
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
        }

    metrics = {
        "dataset": name,
        "excluded_columns": sorted(excluded_columns),
        "table_errors": len(ground_truth),
        "detected_cells": len(predicted),
        "cell_detection_precision": precision,
        "cell_detection_recall": recall,
        "cell_detection_f1": f1(precision, recall),
        "valid_annotated_errors": valid_annotations,
        "annotated_detection_recall": safe_div(detected_annotations, valid_annotations),
        "error_type_macro_f1": safe_div(sum(type_f1s), len(type_f1s)),
        "conditional_error_type_accuracy": safe_div(
            conditional_type_correct, detected_annotations
        ),
        "end_to_end_explanation_accuracy": safe_div(
            correctly_typed, valid_annotations
        ),
        "repair_coverage": safe_div(len(repairs), detected_annotations),
        "exact_repair_accuracy": safe_div(
            sum(item["exact"] for item in repairs), len(repairs)
        ),
        "normalized_repair_accuracy": safe_div(
            sum(item["normalized"] for item in repairs), len(repairs)
        ),
        "constraint_type_alignment": safe_div(
            sum(constraint_type_scores), len(constraint_type_scores)
        ),
        "constraint_column_alignment": safe_div(
            sum(constraint_column_scores), len(constraint_column_scores)
        ),
        "rule_explanation_detail_coverage": safe_div(
            sum(rule_detail_scores), len(rule_detail_scores)
        ),
        "rule_reference_row_coverage": safe_div(
            sum(rule_reference_scores), len(rule_reference_scores)
        ),
        "rule_expected_value_accuracy": safe_div(
            sum(rule_expected_value_scores), len(rule_expected_value_scores)
        ),
        "rule_determinant_column_accuracy": safe_div(
            sum(rule_determinant_scores), len(rule_determinant_scores)
        ),
        "rule_dependent_column_accuracy": safe_div(
            sum(rule_dependent_scores), len(rule_dependent_scores)
        ),
        "rule_column_pair_accuracy": safe_div(
            sum(rule_column_pair_scores), len(rule_column_pair_scores)
        ),
        "rule_reference_row_hit_at_1": safe_div(
            sum(rule_reference_hit_at_1_scores),
            len(rule_reference_hit_at_1_scores),
        ),
        "rule_reference_row_hit_at_5": safe_div(
            sum(rule_reference_hit_at_5_scores),
            len(rule_reference_hit_at_5_scores),
        ),
        "conditional_grounded_rule_accuracy": safe_div(
            sum(grounded_rule_scores), len(grounded_rule_scores)
        ),
        "end_to_end_grounded_rule_accuracy": safe_div(
            sum(grounded_rule_scores), gold_rule_violations
        ),
        "gold_rule_violations": gold_rule_violations,
        "detected_rule_violations": detected_rule_violations,
        "constraint_evaluated": len(constraint_type_scores),
        "rule_expected_value_evaluated": len(rule_expected_value_scores),
        "repair_evaluated": len(repairs),
        "per_type": per_type,
        "type_confusion": {
            f"{gold}->{pred}": count
            for (gold, pred), count in sorted(type_confusion.items())
        },
    }
    return metrics


def parse_spec(value):
    parts = value.split("=", 1)
    if len(parts) != 2 or ":" not in parts[1]:
        raise argparse.ArgumentTypeError("Use NAME=TRACE:DATASET_DIR")
    trace, dataset_dir = parts[1].split(":", 1)
    return parts[0], Path(trace), Path(dataset_dir)


def parse_exclusion(value):
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use NAME=COL1,COL2")
    return parts[0], {
        column.strip() for column in parts[1].split(",") if column.strip()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, type=parse_spec)
    parser.add_argument("--exclude_columns", action="append", default=[], type=parse_exclusion)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclusions = dict(args.exclude_columns)
    results = [
        evaluate_dataset(*spec, excluded_columns=exclusions.get(spec[0], set()))
        for spec in args.dataset
    ]
    with (args.output_dir / "experiment_b_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)
    scalar_keys = [key for key, value in results[0].items() if not isinstance(value, dict)]
    with (args.output_dir / "experiment_b_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_keys} for row in results)
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
