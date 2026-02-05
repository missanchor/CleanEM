from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .configuration import _resolve_path_like


class MockDataGenerator:
    """
    Utility that produces lightweight synthetic tabular data along with
    natural language descriptions for quick experiments and demos.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset_seed()

    def reset_seed(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def generate(self, dataset: str, num_samples: int) -> Tuple[List[List[Any]], List[str]]:
        dataset = dataset.lower()
        if dataset == "employee":
            return self._generate_employee_data(num_samples)
        if dataset == "sales":
            return self._generate_sales_data(num_samples)
        raise ValueError(f"Unsupported dataset type: {dataset}")

    def _generate_employee_data(self, num_samples: int) -> Tuple[List[List[Any]], List[str]]:
        departments = ["Engineering", "Sales", "Marketing", "HR", "Finance"]
        cities = ["San Francisco", "New York", "Boston", "Seattle", "Austin"]

        clean_rows: List[List[Any]] = []
        text_descriptions: List[str] = []

        for i in range(num_samples):
            row = [
                i + 1000,
                f"Employee_{i}",
                int(np.random.randint(22, 65)),
                str(np.random.choice(departments)),
                int(np.random.randint(40000, 150000)),
                str(np.random.choice(cities)),
            ]
            clean_rows.append(row)
            text_descriptions.append(self._employee_row_to_text(row))

        return clean_rows, text_descriptions

    def _generate_sales_data(self, num_samples: int) -> Tuple[List[List[Any]], List[str]]:
        products = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"]
        regions = ["North", "South", "East", "West"]

        clean_rows: List[List[Any]] = []
        text_descriptions: List[str] = []

        for i in range(num_samples):
            quantity = int(np.random.randint(1, 100))
            price = float(np.random.uniform(10.0, 2000.0))
            row = [
                f"ORD{i:05d}",
                str(np.random.choice(products)),
                quantity,
                round(price, 2),
                str(np.random.choice(regions)),
                f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            ]
            clean_rows.append(row)
            text_descriptions.append(self._sales_row_to_text(row))

        return clean_rows, text_descriptions

    @staticmethod
    def _employee_row_to_text(row: Iterable[Any]) -> str:
        employee_id, name, age, department, salary, city = row
        return (
            f"Employee record for {name} (ID {employee_id}). "
            f"{name} is {age} years old, works in the {department} department, "
            f"earns an annual salary of {salary} dollars, and is based in {city}."
        )

    @staticmethod
    def _sales_row_to_text(row: Iterable[Any]) -> str:
        order_id, product, quantity, price, region, date = row
        return (
            f"Sales order {order_id} for {quantity} unit(s) of {product} priced at "
            f"{price} dollars, fulfilled in the {region} region on {date}."
        )


def _parse_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    for converter in (int, float):
        try:
            return converter(text)
        except (ValueError, TypeError):
            continue
    return text


def _row_dict_to_text(row_dict: Dict[str, Any], row_idx: int, dataset_name: str) -> str:
    field_descriptions = []
    for key, raw_value in row_dict.items():
        value = str(raw_value).strip() if raw_value is not None else "N/A"
        if value == "":
            value = "N/A"
        field_descriptions.append(f"The attribute {key} has the value {value}")
    joined = ", ".join(field_descriptions)
    return f"{dataset_name} record {row_idx + 1}: {joined}."


def load_csv_dataset(
    data_path: Path,
    *,
    dataset_name: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Tuple[List[List[Any]], List[str], str, List[str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    with data_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"无法从 {data_path} 读取列名。")

        column_names = list(reader.fieldnames)
        dirty_rows: List[List[Any]] = []
        text_descriptions: List[str] = []
        resolved_name = dataset_name or data_path.stem

        for idx, row_dict in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            ordered_values = [row_dict.get(column, "") for column in column_names]
            parsed_row = [_parse_cell_value(value) for value in ordered_values]
            dirty_rows.append(parsed_row)
            text_descriptions.append(_row_dict_to_text(row_dict, idx, resolved_name))

    if not dirty_rows:
        raise ValueError(f"{data_path} 不包含任何可用数据。")

    return dirty_rows, text_descriptions, resolved_name, column_names


def load_data_from_config(
    exp_cfg: Dict[str, Any],
    *,
    default_seed: int,
    config_dir: Path,
    project_root: Path,
) -> Tuple[List[List[Any]], List[str], str, Optional[List[str]]]:
    if "data_path" in exp_cfg:
        data_path_value = exp_cfg["data_path"]
        data_path = Path(_resolve_path_like(data_path_value, config_dir, project_root))
        dataset_name = exp_cfg.get("dataset") or exp_cfg.get("dataset_name")
        max_rows = exp_cfg.get("max_rows")
        dirty_rows, text_descriptions, dataset_name, column_names = load_csv_dataset(
            data_path, dataset_name=dataset_name, max_rows=max_rows
        )
        return dirty_rows, text_descriptions, dataset_name, column_names

    mock_cfg = exp_cfg.get("mock_data")
    if mock_cfg is not None:
        generator_seed = mock_cfg.get("seed", default_seed)
        generator = MockDataGenerator(seed=generator_seed)
        dataset_type = mock_cfg.get("type", "employee")
        num_samples = mock_cfg.get("num_samples", exp_cfg.get("num_samples", 80))
        dataset_label = mock_cfg.get("name", f"mock-{dataset_type}")
        clean_rows, text_descriptions = generator.generate(dataset_type, num_samples)
        # For mock data, generate default column names
        if dataset_type == "employee":
            column_names = ["employee_id", "name", "age", "department", "salary", "city"]
        elif dataset_type == "sales":
            column_names = ["order_id", "product", "quantity", "price", "region", "date"]
        else:
            column_names = [f"col_{i}" for i in range(len(clean_rows[0]) if clean_rows else 0)]
        return clean_rows, text_descriptions, dataset_label, column_names

    raise ValueError("配置中未指定数据来源，请提供 `data_path` 或 `mock_data`。")


__all__ = [
    "MockDataGenerator",
    "_parse_cell_value",
    "_row_dict_to_text",
    "load_csv_dataset",
    "load_data_from_config",
]


