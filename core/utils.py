"""Centralized utilities for safe eval() dictionary and helper functions."""
import re
import pandas as pd
import numpy as np

try:
    from dateparser import parse
except ImportError:
    parse = None


def safe_float(value):
    """Safely convert value to float, returning None on failure."""
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = str(value).replace(',', '').strip()
        if not cleaned:
            return None
        return float(cleaned)
    except Exception:
        return None


def safe_not(pred, value, row=None) -> bool:
    """Negation with exception handling for rule evaluation."""
    try:
        result = pred(value, row) if callable(pred) else pred
        return not result
    except Exception:
        return True  # Exception counts as violation (not clean)


safe_dict = {
    "re": re,
    "str": str,
    "bool": bool,
    "pd": pd,
    "np": np,
    "float": float,
    "int": int,
    "len": len,
    "safe_float": safe_float,
    "parse": parse,
}
