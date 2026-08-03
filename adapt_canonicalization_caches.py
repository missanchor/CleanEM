import json
from pathlib import Path

from main import _dataframe_fingerprint
from profiler import PandasProfiler


def adapt(source, destination, dirty_csv, rename=None):
    rename = rename or {}
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    df = PandasProfiler(dirty_csv).df
    reverse = {target: source_name for source_name, target in rename.items()}
    plans = {}
    for column in df.columns:
        source_column = reverse.get(column, column)
        if source_column not in payload["plans"]:
            raise KeyError(f"{source}: no plan for {column!r}")
        plan = dict(payload["plans"][source_column])
        plan["column"] = column
        plans[column] = plan
    adapted = {
        "version": 1,
        "dataframe_fingerprint": _dataframe_fingerprint(df),
        "columns": list(df.columns),
        "plans": plans,
        "adapted_from": source,
        "column_rename": rename,
    }
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(adapted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination, adapted["columns"])


adapt(
    "results/tableeg_experiment_b/cache/beers_canonicalization_plans.json",
    "results/agent_pattern_weight_grid/cache/beers_canonicalization_plans.json",
    "data/beers_error-01.csv",
)
adapt(
    "results/tableeg_experiment_b/cache/hospital_canonicalization_plans.json",
    "results/agent_pattern_weight_grid/cache/hospital_canonicalization_plans.json",
    "data/hospital_error-01.csv",
    rename={"Address1": "Address"},
)
