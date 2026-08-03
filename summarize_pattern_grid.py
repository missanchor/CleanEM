import ast
import csv
import json
import re
from pathlib import Path
from statistics import mean


root = Path("results/agent_pattern_weight_grid/canonicalization_on")
scales = ["0p5", "1p0", "1p5", "2p0"]
datasets = ["beers", "hospital", "flights"]
rows = []

for dataset in datasets:
    for scale_tag in scales:
        directory = root / dataset / f"scale_{scale_tag}"
        complete = []
        for log in sorted(directory.glob("*_cleanem.log"), reverse=True):
            text = log.read_text(encoding="utf-8", errors="replace")
            if "[4/4] clean_em pipeline complete." in text:
                complete.append((log, text))
        if not complete:
            raise RuntimeError(f"no complete log in {directory}")
        log, text = complete[0]
        match = re.search(r"  - overall: (\{[^\n]+\})", text)
        if not match:
            raise RuntimeError(f"overall metrics missing in {log}")
        metrics = ast.literal_eval(match.group(1))
        candidate_match = re.search(
            r"Canonicalization/Repair Agent: candidate_cells=(\d+), "
            r"candidates=(\d+)",
            text,
        )
        scale_match = re.search(
            r"agent_pattern_rule_scaling: scale=([^,]+), counts=(\{[^\n]+\})",
            text,
        )
        rows.append(
            {
                "dataset": dataset,
                "pattern_scale": float(scale_tag.replace("p", ".")),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "canonicalization_candidate_cells": (
                    int(candidate_match.group(1)) if candidate_match else None
                ),
                "canonicalization_candidates": (
                    int(candidate_match.group(2)) if candidate_match else None
                ),
                "pattern_scaling_counts": (
                    scale_match.group(2) if scale_match else None
                ),
                "log": str(log),
            }
        )

aggregate = []
for scale_tag in scales:
    scale = float(scale_tag.replace("p", "."))
    selected = [row for row in rows if row["pattern_scale"] == scale]
    aggregate.append(
        {
            "pattern_scale": scale,
            "mean_precision": mean(row["precision"] for row in selected),
            "mean_recall": mean(row["recall"] for row in selected),
            "mean_f1": mean(row["f1"] for row in selected),
            "min_f1": min(row["f1"] for row in selected),
            "max_f1": max(row["f1"] for row in selected),
        }
    )

output = Path("results/agent_pattern_weight_grid")
with (output / "canonicalization_on_metrics.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
with (output / "canonicalization_on_aggregate.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=aggregate[0].keys())
    writer.writeheader()
    writer.writerows(aggregate)
(output / "canonicalization_on_summary.json").write_text(
    json.dumps({"runs": rows, "aggregate": aggregate}, indent=2),
    encoding="utf-8",
)

print(json.dumps({"runs": rows, "aggregate": aggregate}, indent=2))
