#!/usr/bin/env python3
"""Prepare aligned TableEG tables for CleanEM without altering source files."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DATASETS = {
    "beers": "beers",
    "hospital": "hospital",
    "flight": "flight",
    "Restaurant": "Restaurant",
}


def normalize(value):
    if pd.isna(value):
        return "<NA>"
    text = str(value).strip().lower()
    if text in {"", "nan", "na", "n/a", "null", "none"}:
        return "<NA>"
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else f"{number:.12g}"
    except ValueError:
        return re.sub(r"\s+", " ", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report = {}
    for dataset, prefix in DATASETS.items():
        source = args.downloads / dataset
        target = args.output / dataset.lower()
        target.mkdir(parents=True, exist_ok=True)
        dirty = pd.read_csv(source / "dirty.csv", dtype=object)
        clean = pd.read_csv(source / "clean.csv", dtype=object)
        annotations = pd.read_csv(source / f"{prefix}_annotation.csv", dtype=object)

        if "row_id" not in dirty or "row_id" not in clean:
            raise ValueError(f"{dataset}: missing row_id")
        if dirty["row_id"].duplicated().any() or clean["row_id"].duplicated().any():
            raise ValueError(f"{dataset}: duplicate row_id")

        clean = clean.set_index(clean["row_id"].astype(str), drop=False)
        dirty_ids = dirty["row_id"].astype(str).tolist()
        if set(dirty_ids) != set(clean.index):
            raise ValueError(f"{dataset}: dirty/clean row_id mismatch")
        clean = clean.loc[dirty_ids].reset_index(drop=True)

        aliases = {"beer-name": "beer_name", "brewery-name": "brewery_name"}
        clean = clean.rename(columns=aliases)
        if list(dirty.columns) != list(clean.columns):
            raise ValueError(
                f"{dataset}: columns differ after aliasing: "
                f"{list(dirty.columns)} vs {list(clean.columns)}"
            )

        row_map = dirty[["row_id"]].copy()
        row_map.insert(0, "row_index", range(len(row_map)))

        dirty_by_id = dirty.set_index(dirty["row_id"].astype(str), drop=False)
        clean_by_id = clean.set_index(clean["row_id"].astype(str), drop=False)
        annotation_rows = []
        for _, annotation in annotations.iterrows():
            item = annotation.to_dict()
            row_id = str(item["row_id"])
            column = str(item["column"])
            if row_id not in dirty_by_id.index or column not in dirty.columns:
                item["valid_key"] = False
                item["table_difference"] = False
                annotation_rows.append(item)
                continue
            dirty_value = dirty_by_id.loc[row_id, column]
            clean_value = clean_by_id.loc[row_id, column]
            item["valid_key"] = True
            item["table_difference"] = normalize(dirty_value) != normalize(clean_value)
            item["source_error_matches_dirty"] = (
                normalize(item.get("error_value")) == normalize(dirty_value)
            )
            item["source_right_matches_clean"] = (
                normalize(item.get("right_value")) == normalize(clean_value)
            )
            # Hospital's released annotation has the two value fields reversed.
            if dataset == "hospital":
                item["source_error_value"] = item.get("error_value")
                item["source_right_value"] = item.get("right_value")
                item["error_value"] = dirty_value
                item["right_value"] = clean_value
                item["annotation_orientation_fixed"] = True
            else:
                item["annotation_orientation_fixed"] = False
            annotation_rows.append(item)

        # row_id is an alignment key, not a data-quality target for CleanEM.
        dirty.drop(columns=["row_id"]).to_csv(target / "dirty.csv", index=False)
        clean.drop(columns=["row_id"]).to_csv(target / "clean.csv", index=False)
        row_map.to_csv(target / "row_map.csv", index=False)
        pd.DataFrame(annotation_rows).to_csv(target / "annotation_aligned.csv", index=False)

        report[dataset] = {
            "rows": len(dirty),
            "columns_for_cleanem": len(dirty.columns) - 1,
            "annotations": len(annotation_rows),
            "table_difference_annotations": sum(
                bool(item.get("table_difference")) for item in annotation_rows
            ),
            "orientation_fixed": dataset == "hospital",
        }

    with (args.output / "preparation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
