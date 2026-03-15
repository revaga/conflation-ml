import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_columns(df: pd.DataFrame, expected_cols: list, name: str = "Parquet"):
    """Fail fast if expected columns are missing."""
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CRITICAL: {name} data missing expected columns: {missing}")

def read_parquet_safe(path: str, columns=None, expected_cols=None, **kwargs) -> pd.DataFrame:
    """
    Read parquet file with optional schema validation.
    Falls back to fastparquet if PyArrow raises 'Repetition level histogram size mismatch'.
    """
    try:
        if columns is not None:
            df = pd.read_parquet(path, columns=columns, **kwargs)
        else:
            df = pd.read_parquet(path, **kwargs)
    except OSError as e:
        if "histogram size mismatch" not in str(e) and "Repetition level" not in str(e):
            raise
        
        logger.warning(f"PyArrow error reading {path} (Repetition level mismatch). Falling back to fastparquet.")
        try:
            if columns is not None:
                df = pd.read_parquet(path, columns=columns, engine="fastparquet", **kwargs)
            else:
                df = pd.read_parquet(path, engine="fastparquet", **kwargs)
        except ImportError:
            raise RuntimeError(
                f"Reading {path} failed (PyArrow bug). Install fastparquet."
            ) from None

    if expected_cols:
        validate_columns(df, expected_cols, path)
    return df
