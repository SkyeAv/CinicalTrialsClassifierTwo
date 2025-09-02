from __future__ import annotations
# ! annotations MUST BE AT THE TOP
from CTClassifier2.utils import root
from pathlib import Path
from typing import Any
import polars as pl
import fsspec

def _tablename_cols(df: pl.DataFrame, tablename: str) -> pl.DataFrame:
  return df.rename({col: f"{tablename}::{col}" for col in df.columns})

def _read_table(zips: str, tablename: str) -> pl.DataFrame:
  with fsspec.open(f"zip://{tablename}.txt::{zips}", mode="rt") as f:
    return _tablename_cols(pl.read_csv(f, has_header=True, separator="|"), tablename)

def generate_parquet(snapshot: dict[str, Any], nct_col: bool = False) -> None:
  version: int = snapshot["version"]
  zips: str = (snapshot["zip_directory"] / f"{version}.zip").as_posix()
  dfs: list[pl.DataFrame] = []
  parquet_p: Path = root() / "PARQUET" / f"{version}.parquet"
  parquet_p.parent.mkdir(parents=True, exist_ok=True)
  if not parquet_p.exists():
    for tablename in snapshot["tables"]:
      df = _read_table(zips, tablename)
      if not nct_col:
        df = df.rename({f"{tablename}::nct_id": "nct"})
        nct_col = True
      else:
        df = df.drop(f"{tablename}::nct_id")
      dfs.append(df)
    combined = pl.concat(dfs, how="horizontal")
    combined = combined.with_row_index("row_id")  # ! added index for joining
    combined.write_parquet(parquet_p)
  else:
    pass
  return None

