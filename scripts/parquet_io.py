"""
Safe Parquet I/O: work around PyArrow 'Repetition level histogram size mismatch'.
Use read_parquet_safe() instead of pd.read_parquet() for all parquet reads.
"""
import pandas as pd


def read_parquet_safe(path: str, columns=None, **kwargs) -> pd.DataFrame:
    """
    Read parquet file. Falls back to fastparquet if PyArrow raises
    'Repetition level histogram size mismatch'. Pass columns or other
    read_parquet kwargs as needed.
    """
    try:
        if columns is not None:
            return pd.read_parquet(path, columns=columns, **kwargs)
        return pd.read_parquet(path, **kwargs)
    except OSError as e:
        if "histogram size mismatch" not in str(e) and "Repetition level" not in str(e):
            raise
    try:
        if columns is not None:
            return pd.read_parquet(path, columns=columns, engine="fastparquet", **kwargs)
        return pd.read_parquet(path, engine="fastparquet", **kwargs)
    except ImportError:
        raise RuntimeError(
            "Reading parquet failed (PyArrow bug with this file). "
            "Install fastparquet: pip install fastparquet"
        ) from None
