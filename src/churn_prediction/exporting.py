"""Safe tabular export helpers for files opened in spreadsheet software."""

import pandas as pd


FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def safe_csv_bytes(data: pd.DataFrame) -> bytes:
    """Serialize a copy while neutralizing spreadsheet formula cells."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("CSV export input must be a pandas DataFrame.")

    sanitized = data.copy(deep=True)

    def neutralize(value):
        if not isinstance(value, str):
            return value

        if value.lstrip(" \t\r\n").startswith(FORMULA_PREFIXES):
            return "'" + value

        return value

    for column in sanitized.columns:
        if (
            pd.api.types.is_object_dtype(sanitized[column].dtype)
            or pd.api.types.is_string_dtype(sanitized[column].dtype)
        ):
            sanitized[column] = sanitized[column].map(neutralize)

    return sanitized.to_csv(index=False).encode("utf-8")
