from __future__ import annotations
# ! annotations MUST BE AT THE TOP
from CTClassifier2.autoencoder import encoder
from CTClassifier2.utils import dtype_device
from CTClassifier2.utils import set_seed
from transformers import AutoTokenizer
from CTClassifier2.utils import root
from transformers import AutoModel
from functools import partial
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from functools import cache
from typing import Optional
from pathlib import Path
from typing import Any
from os import environ
import pyarrow as pa
import pandas as pd
import numpy as np
import tempfile
import joblib
import duckdb
import shutil
import torch

def _count_unique(con: Any, col: str) -> int:
  count_query: str = f"""\
SELECT COUNT(DISTINCT "{col}") AS n_unique FROM _t
"""
  return con.execute(count_query).fetchone()[0]

def _freetext_columns(
  parquet: str, 
  high_varation: int = 64, 
  enough_variation: int = 16
) -> tuple[list[str], list[str]]:
  with duckdb.connect() as con:
    load_query: str = f"""\
CREATE OR REPLACE TEMP VIEW _t AS SELECT * FROM parquet_scan("{parquet}")
"""
    con.execute(load_query)
    schema_query: str = """\
DESCRIBE SELECT * FROM _t
"""
    schema: pd.DataFrame = con.execute(schema_query).fetchdf()
    str_cols: list[str] = schema.loc[schema["column_type"] == "VARCHAR", "column_name"].tolist()
    if "nct" in str_cols:
      str_cols.remove("nct")
    cols: list[str] = [col for col in str_cols if high_varation > _count_unique(con, col) >= enough_variation]
    cols_complex: list[str] = [col for col in str_cols if _count_unique(con, col) >= high_varation]
    return cols, cols_complex

@cache
def _biobert() -> tuple[Any, Any]:
  biobert: str = environ["BIOBERT_DIR"]
  tokenizer = AutoTokenizer.from_pretrained(
    biobert,
    use_fast=True,
    local_files_only=True
  )
  model = AutoModel.from_pretrained(
    biobert,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=False,
    local_files_only=True
  ).eval()
  return model, tokenizer

def _arrow_init(
  parquet_embed: str,
  cols: list[str],
  cols_complex: list[str],
  cols_combined: list[str],
  ae_out_size: int,
  ae_out_size_complex: int,
) -> tuple[Any, Any, Any]:

  fields: list[Any] = [pa.field("row_id", pa.int64())]
  for col in cols_combined:
    if col in cols_complex:
      fields.append(pa.field(f"{col}::embed_complex", pa.list_(pa.float32(), list_size=ae_out_size_complex)))
    elif col in cols:
      fields.append(pa.field(f"{col}__embed", pa.list_(pa.float32(), list_size=ae_out_size)))
    else:
      pass

  schema = pa.schema(fields)
  writer = pq.ParquetWriter(parquet_embed, schema=schema, compression="zstd", use_dictionary=False)
  return writer, schema

def _make_scanner(parquet: str, cols: list[str], batch_len_rows: int) -> Any:
  dataset = ds.dataset(parquet, format="parquet")
  return dataset.scanner(columns=cols, batch_size=batch_len_rows)

@torch.inference_mode()
def _batched_pooler_out(
  vals: list[Optional[str]],
  model: Any,
  tokenizer: Any,
  max_len: int,
  batch_size: int = 128
) -> torch.Tensor:
  embeddings: list[torch.Tensor] = []
  device: torch.device = next(model.parameters()).device
  for idx in range(0, len(vals), batch_size):
    batch: list[Optional[str]] = [x if x else "" for x in vals[idx:idx+batch_size]]
    inputs = tokenizer(
      batch,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=max_len
    ).to(device)
    outputs = model(**inputs, output_hidden_states=False, output_attentions=False)
    pooler_out = getattr(outputs, "pooler_output", None)
    if pooler_out is None:
      raise RuntimeError(f"PY-CODE:7 | BioBert Failed To Produce a Pooler Out... {batch}")
    embeddings.append(pooler_out.detach().cpu().float())
  return torch.cat(embeddings, dim=0)

def _embedding_matrix(
  model: Any,
  tokenizer: Any,
  col: str,
  max_len: int,
  model_p: Path,
  parquet: str,
  scanner_cols: list[str],
  batch_len_rows: int
) -> torch.Tensor:
  cache_p: Path = model_p.parent / "POOLER_OUT.pkl"
  if not cache_p.exists():
    embdedded_columns: list[torch.Tensor] = []
    scanner = _make_scanner(parquet, scanner_cols, batch_len_rows)
    for batch in scanner.to_batches():
      vals = batch.column(col).to_pylist()
      embeddings: torch.Tensor = _batched_pooler_out(vals, model, tokenizer, max_len)
      embdedded_columns.append(embeddings)
    pooler_matrix = torch.cat(embdedded_columns, dim=0)
    joblib.dump(pooler_matrix, cache_p)
    return pooler_matrix
  else:
    return joblib.load(cache_p)

def _write_embeddings(
  writer: Any,
  schema: Any,
  ae_idx: dict[str, tuple[Path, int, int]],
  batch_len_rows: int
) -> None:
  try:
    mmaps = {c: (np.memmap(p, mode="r", dtype=np.float32, shape=(N, L)), L) for c, (p, N, L) in ae_idx.items()}
    n_rows_set: set[int] = {N for (_, N, _) in mmaps.values()}
    if len(n_rows_set) != 1:
      raise RuntimeError(f"PY-CODE:9 | Inconsistent row counts across columns... {n_rows_set}")
    
    def _2D_list_array(arr: Any, size: int) -> Any:
      flattened: Any = pa.array(arr.reshape(-1), type=pa.float32())
      return pa.FixedSizeListArray.from_arrays(flattened, size)
    
    n_rows: int = n_rows_set.pop()
    idx: int = 0
    while idx < n_rows:
      end: int = min(idx + batch_len_rows, n_rows)

      arrays: list[Any] = []
      for field in schema:

        name: str = field.name
        if name == "row_id":
          arrays.append(pa.array(np.arange(idx, end, dtype=np.int64)))

        if name.endswith("__embed_complex"):
          col_key: str = name[:-len("__embed_complex")]
        else:
          col_key = name[:-len("__embed")]
        
        Z, _, L = mmaps[col_key]
        arrays.append(_2D_list_array(np.asarray(Z[idx:end]), L))

      batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
      writer.write_batch(batch)
      idx = end
  finally:
    writer.close()
  return None

def _embed_texts(
  version: int,
  model: Any,
  tokenizer: Any,
  parquet: str,
  parquet_embed: str,
  cols: list[str],
  cols_complex: list[str],
  ae_out_size: int = 64,
  ae_out_size_complex: int = 256,
  max_len: int = 256,
  batch_len_rows: int = 16_384,
  bert_out_size: int = 768
) -> None:
  set_seed()
  dtype, device = dtype_device()
  cols_combined: list[str] = cols + cols_complex
  writer, schema = _arrow_init(
    parquet_embed,
    cols,
    cols_complex,
    cols_combined,
    ae_out_size,
    ae_out_size_complex,
  )
  ae = partial(
    encoder,
    bert_out_size=bert_out_size,
    device=device,
    dtype=dtype
  )
  mm_d: Path = Path(tempfile.mkdtemp())
  ae_idx: dict[str, tuple[Path, int, int]] = {}
  try:
    for col in cols_combined:
      model_p: Path = root() / "AUTOENCODER" / str(version) / col / "MODEL.pt"
      model_p.parent.mkdir(parents=True, exist_ok=True)
      is_complex: bool = col in cols_complex
      L: int = ae_out_size_complex if is_complex else ae_out_size
      # * predefine dimensionality reduction
      dr = partial(ae, complex_embed=True) if is_complex else partial(ae)
      biobert_embedding = _embedding_matrix(
        model,
        tokenizer,
        col,
        max_len,
        model_p,
        parquet,
        cols_combined,
        batch_len_rows
      )
      z: torch.Tensor = dr(
        pooler_out=biobert_embedding,
        ae_out_size=L,
        model_p=model_p
      )
      n_rows, out_dim = z.shape
      # ! use memmaps so RAM doesn't explode
      mm_p: Path = mm_d / str(version) / col / "ENCODED.mm"
      mm_p.parent.mkdir(parents=True, exist_ok=True)
      np.memmap(mm_p, mode="w+", dtype=np.float32, shape=(n_rows, out_dim))[:] = z.numpy()
      ae_idx[col] = (mm_p, n_rows, out_dim)
    _write_embeddings(writer, schema, ae_idx, batch_len_rows)
  finally:
    shutil.rmtree(mm_d, ignore_errors=True)
  return None

def embed_parquet(snapshot: dict[str, Any]) -> None:
  version: int = snapshot["version"]
  parquet_p: Path = root() / "PARQUET" / str(version)
  parquet: str = (parquet_p / "RAW.parquet").as_posix()
  cols, cols_complex = _freetext_columns(parquet)
  model, tokenizer = _biobert()
  parquet_embed: str = (parquet_p / "EMBEDDED.parquet").as_posix()
  _embed_texts(version, model, tokenizer, parquet, parquet_embed, cols, cols_complex)
  pf = pq.ParquetFile(parquet_embed)
  print("rows:", pf.metadata.num_rows, "cols:", pf.metadata.num_columns)
  print(pf.schema_arrow)
  return None
