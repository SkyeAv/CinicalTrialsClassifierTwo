from __future__ import annotations
# ! annotations MUST BE AT THE TOP
from torch_frame.nn import LinearEmbeddingEncoder
from CTClassifier2.utils import dtype_device
from torch_frame.utils import infer_df_stype
from torch_frame.nn import EmbeddingEncoder
from torch.nn.utils import clip_grad_norm_
from CTClassifier2.utils import set_seed
from torch_frame.nn.models import Trompt
from torch_frame.nn import LinearEncoder
from torch_frame.data import DataLoader
from torch_frame.data import Dataset
from CTClassifier2.utils import root
from dataclasses import dataclass
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.amp import autocast
from torch_frame import stype
from loguru import logger
from pathlib import Path
from typing import Self
from typing import Any
from torch import nn
import pandas as pd
import numpy as np
import duckdb
import torch
import math
import re

_IGNORE: int = -100  # ! for CrossEntropyLoss(ignore_index=_IGNORE)

def _split_lables(arr: Any, split_str: str = "[negative]") -> tuple[Any, Any]:
  # ! using any here because I hate what mypy does with np.ndarray
  parts: Any = np.char.partition(arr, split_str)
  untrust: Any = np.char.strip(parts[:, 0])
  trust: Any = np.char.strip(parts[:, 2])
  return trust, untrust

def _load_text(txt_p: Path) -> Any:
  with txt_p.open("rt", encoding="utf-8") as f:
    return np.array([line.rstrip("\n") for line in f])

def _label_frame(gold_p: Path, pseudo_p: Path) -> tuple[list[str], pd.DataFrame]:
  gold_arr: Any = _load_text(gold_p)
  pseudo_arr: Any = _load_text(pseudo_p)

  gold_trust, gold_untrust = _split_lables(gold_arr)
  pseudo_trust, pseudo_untrust = _split_lables(pseudo_arr)
  labels: list[str] = np.concatenate((gold_trust, gold_untrust, pseudo_trust, pseudo_untrust)).tolist()

  def _make(arr: Any, label: int, is_gold: bool) -> pd.DataFrame: 
    # * helper fn I put here bc it would be weird out of scope... idk... it just felt off :skull:
    return pd.DataFrame({
      "nct": arr,
      "label": label,
      "gold": is_gold
    })
  
  return (
    labels,
    pd.concat([
      _make(gold_trust, 0, True),
      _make(gold_untrust, 1, True),
      _make(pseudo_trust, 0, False),
      _make(pseudo_untrust, 1, False),
    ],
    ignore_index=True
    )
  )

def _load_features(parquet_p: Path, labels: list[str], max_threads: int = 32) -> pd.DataFrame:
  parquet: str = (parquet_p / "RAW.parquet").as_posix()
  parquet_embed: str = (parquet_p / "EMBEDDED.parquet").as_posix()
  describe_query: str = f"""\
DESCRIBE SELECT * FROM parquet_scan(?)
"""
  with duckdb.connect() as con:
    con.execute(f"PRAGMA threads={max_threads}")
    con.execute("PRAGMA enable_object_cache=true")
    embed_cols: list[str] = con.execute(describe_query, parquet_embed).fetchdf()["column_name"].tolist()
    corresponding_raw_cols: list[str] = [re.sub(r"::embed(?:_complex)?$", "", col) for col in embed_cols if "::embed" in col]
    exclude_clause: str = ", ".join([f'"{col}"' for col in corresponding_raw_cols])
    con.execute("CREATE TEMP TABLE ncts(nct VARCHAR)")
    con.execute("INSERT INTO ncts SELECT * FROM UNNEST(?::VARCHAR[])", labels)
    load_query: str = f"""\
WITH base AS (
  SELECT * EXCLUDE(?)
  FROM parquet_scan(?)
  WHERE nct IN (SELECT nct FROM ncts)
),
keys AS (
  SELECT DISTINCT row_id FROM base
),
trial_filter AS (
  SELECT *
  FROM parquet_scan(?)
  WHERE row_id IN (SELECT row_id FROM keys)
)
SELECT a.*, b.* EXCLUDE("row_id")
FROM base a
LEFT JOIN trial_filter b USING ("row_id")
"""
    return con.execute(load_query, exclude_clause, parquet, parquet_embed).fetchdf()

def _label_features(ff: pd.DataFrame, lf: pd.DataFrame) -> pd.DataFrame:
  return ff.merge(lf, on="nct", how="inner").drop(columns=["nct"])

def _add_masks_and_targets(df: pd.DataFrame) -> pd.DataFrame:
  df["mask_gold"] = df["gold"].astype(bool)
  df["mask_pseudo"] = ~df["mask_gold"]
  df["y_gold"] = np.where(df["mask_gold"], df["label"].astype(np.int64), _IGNORE)
  df["y_pseudo"] = np.where(df["mask_pseudo"], df["label"].astype(np.int64), _IGNORE)
  return df

def _attach_split(
  df: pd.DataFrame,
  val_ratio: float = 0.15,
  test_ratio: float = 0.15
) -> pd.DataFrame:
  n: int = len(df)
  n_test: int = int(n * val_ratio)
  n_val: int = int(n * test_ratio)
  perm: np.ndarray = np.random.permutation(n)
  split: np.ndarray = np.empty(n, dtype=np.int64)
  split[perm[:n_test]] = 2  # test
  split[perm[n_test:(n_test + n_val)]] = 1  # val
  split[perm[(n_test + n_val):]] = 0  # train
  df["split"] = split
  return df

def _get_stypes(df: pd.DataFrame) -> dict[str, Any]:
  infered = infer_df_stype(df)
  return {col: stype.embedding if any(x in col for x in ["::embed", "::embed_complex"]) else _stype for col, _stype in infered.items()}

def _load_dataset(df: pd.DataFrame) -> Any:
  aux_cols: list[str] = ["gold", "mask_gold", "mask_pseudo", "y_gold", "y_pseudo"]
  df = df.drop(columns=aux_cols)
  col_to_stype: dict[str, Any] = _get_stypes(df)
  dataset = Dataset(df, col_to_stype=col_to_stype, target="label", split_col="split")
  dataset.materialize()
  return dataset

def _indexes_from_split(df: pd.DataFrame) -> dict[str, list[int]]:
  idx_train: list[str] = np.flatnonzero(df["split"].to_numpy() == 0).tolist()
  idx_val: list[str] = np.flatnonzero(df["split"].to_numpy() == 1).tolist()
  idx_test: list[str] = np.flatnonzero(df["split"].to_numpy() == 2).tolist()
  return {"train": idx_train, "val": idx_val, "test": idx_test}

def _sidecar_tensors(df: pd.DataFrame, device: torch.device) -> dict[str, torch.Tensor]:
  y_gold = torch.as_tensor(df["y_gold"].to_numpy(), dtype=torch.long, device=device)
  y_pseudo = torch.as_tensor(df["y_pseudo"].to_numpy(), dtype=torch.long, device=device)
  mask_gold = torch.as_tensor(df["mask_gold"].to_numpy(), dtype=torch.bool, device=device)
  mask_pseudo = torch.as_tensor(df["mask_pseudo"].to_numpy(), dtype=torch.bool, device=device)
  return {
    "y_gold": y_gold,
    "y_pseudo": y_pseudo,
    "mask_gold": mask_gold,
    "mask_pseudo": mask_pseudo,
  }

def _train_priors(df: pd.DataFrame, dtype: Any, device: torch.device) -> dict[str, torch.Tensor]:
  m_train: Any = (df["split"].to_numpy() == 0)
  m_gold: Any = m_train & df["mask_gold"].to_numpy()
  m_pseu: Any = m_train & df["mask_pseudo"].to_numpy()

  def _priors(mask: np.ndarray, dtype: Any, device: torch.device) -> torch.Tensor:
    y: Any = df.loc[mask, "label"].to_numpy()
    counts = np.bincount(y, minlength=2).astype(np.float64)
    total = counts.sum().clip(min=1.0)
    return torch.tensor(counts / total, dtype=dtype).to(device)
  
  return {
    "prior_gold": _priors(m_gold, dtype, device),
    "prior_pseudo": _priors(m_pseu, dtype, device),
  }

def _tf_loader(
  ds: Dataset,
  idxs: list[str],
  shuffle: bool = False,
  batch_size: int = 1024,
  num_workers: int = 4,  # 4 workers per GPU
) -> Any:
  # * idx generator
  loader = DataLoader(
    ds,
    num_workers=num_workers,
    shuffle=shuffle,
    batch_size=batch_size,
    shuffle=shuffle
  )
  
  for idx, (batch, _) in enumerate(loader):
    start: int = idx * batch_size
    end: int = start + len(batch)
    batch_idxs = idxs[start:end]
    yield batch, torch.as_tensor(batch_idxs, dtype=torch.long)

class _TromptTwo(nn.Module):
  def __init__(
    self: Self,
    trompt: nn.Module,
    out_channels: int,
    num_layers: int,
    hidden: int,
    dropout: float
  ) -> None:
    super().__init__()
    self.trompt = trompt

    init = torch.linspace(-0.2, 0.2, steps=num_layers)
    init[-1] += 0.5  # favors final layer
    self.alpha = nn.Parameter(init)

    # * RESNET calibration
    self.calib = nn.Sequential(
      nn.Linear(out_channels, hidden),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, out_channels),
    )

    nn.init.zeros_(self.calib[-1].weight)
    nn.init.zeros_(self.calib[-1].bias)
    return None

  def forward(self: Self, tf: Any) -> torch.Tensor:
    layers: torch.Tensor = self.trompt(tf)
    _, L, _ = layers.shape
    w: torch.Tensor = self.alpha.softmax(dim=0).view(1, L, 1)
    logits: torch.Tensor = (layers * w).sum(dim=1)
    logits = logits + self.calib(logits)  # * RESNET implimentation
    return logits

class TemperatureScaler(nn.Module):
  def __init__(self: Self) -> None:
    super().__init__()
    self.log_temp = nn.Parameter(torch.zeros(()))
    self.eps = 1e-10  # ! prevents divide by zero erro
    return None
  
  def forward(self: Self, logits: torch.Tensor) -> torch.Tensor:
    return logits / (self.log_temp.exp() + self.eps)

class _TromptTwoMultiHeaded(nn.Module):
  def __init__(
    self: Self,
    trompt_two: nn.Module,
    out_channels: int,
    hidden: int,
    dropout: float
  ) -> None:
    super().__init__()
    self.backbone = trompt_two
    self.head_pseudo = nn.Sequential(
      nn.Linear(out_channels, hidden),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, out_channels)
    )

    self.scaler_gold = TemperatureScaler()
    self.scaler_pseudo = TemperatureScaler()
    return None

  def forward(self: Self, tf: Any) -> tuple[torch.Tensor, torch.Tensor]:
    logits_gold: torch.Tensor = self.backbone(tf)
    logits_pseudo: torch.Tensor = self.head_pseudo(logits_gold.detach())

    cal_logits_gold: torch.Tensor = self.scaler_gold(logits_gold)
    cal_logits_pseudo: torch.Tensor = self.scaler_pseudo(logits_pseudo)
    return cal_logits_gold, cal_logits_pseudo

def _init_model(
  dataset: Any,
  channels: int = 256,  # because embed_complex == 256 and I don't want it projected
  out_channels: int = 2,
  num_prompts: int = 16,
  num_layers: int = 6,
  hidden: int = 64,
  dropout: float = 0.1
) -> Any:
  stype_encoder_dict: dict[Any, Any] = {
    stype.numerical: LinearEncoder(),
    stype.categorical: EmbeddingEncoder(),
    stype.embedding: LinearEmbeddingEncoder()
  }
  trompt = Trompt(
    channels=channels,
    out_channels=out_channels,
    num_prompts=num_prompts,
    num_layers=num_layers,
    col_stats=dataset.col_stats,
    col_names_dict=dataset.tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict
  )
  trompt_two = _TromptTwo(
    trompt,
    out_channels,
    num_layers,
    hidden,
    dropout
  )
  return _TromptTwoMultiHeaded(
    trompt_two,
    out_channels,
    hidden,
    dropout
  )

def _init_log(model_p: Path) -> logger:
  log_dir: str = model_p.parent.as_posix()
  logger.remove()
  log: str = f"{log_dir}/TRAIN_{{time:YYYYMMDD}}.jsonl"
  logger.add(
    log,
    rotation="100 MB",
    enqueue=True,
    serialize=True,
    level="INFO"
  )
  return logger

@dataclass
class _AutoStopTransformer:
  patience: int = 7
  min_delta: float = 1e-4
  best: float = math.inf
  wait: int = 0
  stopped_at: int = 0
  checkpoint: Path = ...

  def _step(
    self: Self,
    loss: float,
    model: _TromptTwoMultiHeaded
  ) -> bool:
    improved: bool = (self.best - loss) > self.min_delta

    if improved:
      self.best = loss
      self.wait = 0
      torch.save(model.state_dict(), self.checkpoint)
    else:
      self.wait += 1
  
    return self.wait > self.patience

def _optimizer_scheduler(
  model: _TromptTwoMultiHeaded,
  epochs: int,
  device: torch.device,
  model_decay: float = 1e-4,
  non_model_decay: float = 0.0,
  scaler_lr: float = 5e-4,
  alpha_lr: float = 5e-3,
  calib_lr: float = 3e-3,
  pseudo_lr: float = 3e-3,
  backbone_lr: float = 1e-3,
  beta_one: float = 0.9,
  beta_two: float = 0.99,
  eta_min: float = 1e-5,
  start_factor: float = 0.1,
  warmup_iter: int = 5
) -> Any:
  backbone = model.backbone
  calib = model.backbone.calib  # * RESNET
  scalers = [model.scaler_gold, model.scaler_pseudo]

  param_groups = [
    {
      "params": backbone.trompt.parameters(),
      "lr": backbone_lr,
      "weight_decay": model_decay
    },
    {
      "params": [backbone.alpha],
      "lr": alpha_lr,
      "weight_decay": non_model_decay
    },
    {
      "params": calib.parameters(),
      "lr": calib_lr,
      "weight_decay": model_decay
    },
    {
      "params": model.head_pseudo.parameters(),
      "lr": pseudo_lr,
      "weight_decay": model_decay
    },
    {
      "params": [param for s in scalers for param in s.parameters()],
      "lr": scaler_lr,
      "weight_decay": non_model_decay
    }
  ]

  if device.type != "cuda":
    opt = torch.optim.AdamW(param_groups, betas=(beta_one, beta_two))
  else:
    opt = torch.optim.AdamW(param_groups, betas=(beta_one, beta_two), fused=True)

  warmup = torch.optim.lr_scheduler.LinearLR(
    opt,
    start_factor=start_factor,
    total_iters=warmup_iter
  )
  cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=epochs,
    eta_min=eta_min
  )
  sched = torch.optim.lr_scheduler.ChainedScheduler([warmup, cosine])
  return opt, sched

_GOLD_W: float = 1.0
_PSEUDO_W: float = 0.1

def _masked_loss(
  lg: torch.Tensor,
  lp: torch.Tensor,
  y_gold: torch.Tensor,
  y_pseudo: torch.Tensor,
  ce_reduction: str = "mean"
) -> torch.Tensor:
  mg: torch.Tensor = (y_gold != _IGNORE).any()
  mp: torch.Tensor = (y_pseudo != _IGNORE).any()
  fallback: torch.Tensor = torch.tensor(0.0, device=lg.device, dtype=lg.dtype)

  ce_loss = F.CrossEntropyLoss(ignore_index=_IGNORE, reduction=ce_reduction)

  gold_loss = torch.where(mg, ce_loss(lg, y_gold), fallback)
  pseudo_loss = torch.where(mp, ce_loss(lp, y_pseudo), fallback)
  return (_GOLD_W * gold_loss) + (_PSEUDO_W * pseudo_loss)

def _entropy_reg(alpha: Any, strength: int = 1e-3, eps: float = 1e-10) -> torch.Tensor:
  w = alpha.softmax(dim=0)
  H = -(w * w.clamp_min(eps).log()).sum()
  return -H * strength

def _logit_adjust(logits: torch.Tensor, priors: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
  return logits - (tau * priors.log())

def _train_epoch(
  model: _TromptTwoMultiHeaded,
  loader: Any,
  sidecar_tensors: torch.Tensor,
  opt: Any,
  scaler: Any,
  dtype: Any,
  device: torch.device,
  tau_gold: float,
  tau_pseudo: float,
  max_norm: float = 2.0
) -> float:
  model.train()
  total: float = 0.0
  n: int = 0

  for batch, idx in loader:
    y_gold = sidecar_tensors["y_gold"][idx]
    y_pseudo = sidecar_tensors["y_pseudo"][idx]
    opt.zero_grad(set_to_none=True)
    if "cuda" not in device.type:
      batch = batch.to(device, dtype=dtype, non_blocking=True)
      lg, lp = model(batch)
      lg = _logit_adjust(lg, sidecar_tensors["prior_gold"], tau=tau_gold)
      lp = _logit_adjust(lp, sidecar_tensors["prior_pseudo"], tau=tau_pseudo)
      loss: torch.Tensor = _masked_loss(lg, lp, y_gold, y_pseudo)
      loss = loss + _entropy_reg(model.backbone.alpha, strength=1e-3)
      loss.backward()
      clip_grad_norm_(model.parameters(), max_norm=max_norm)
      opt.step()
    else:
      batch = batch.to(device, non_blocking=True)
      with autocast(dtype=dtype):
        lg, lp = model(batch)
        lg = _logit_adjust(lg, sidecar_tensors["prior_gold"], tau=tau_gold)
        lp = _logit_adjust(lp, sidecar_tensors["prior_pseudo"], tau=tau_pseudo)
        loss = _masked_loss(lg, lp, y_gold, y_pseudo)
      scaler.scale(loss).backward()
      scaler.unscale_(opt)
      clip_grad_norm_(model.parameters(), max_norm=max_norm)
      scaler.step(opt)
      scaler.update()
    total += float(loss.detach().cpu())
    n += 1

  return total / max(1, n)

@torch.no_grad()
def _eval_epoch(
  model: _TromptTwoMultiHeaded,
  loader: Any,
  sidecar_tensors: torch.Tensor,
  dtype: Any,
  device: torch.device,
  tau_gold: float,
  tau_pseudo: float,
) -> float:
  model.eval()
  total: float = 0.0
  n: int = 0

  for batch, idx in loader:
    y_gold = sidecar_tensors["y_gold"][idx]
    y_pseudo = sidecar_tensors["y_pseudo"][idx]
    if "cuda" not in device.type:
      batch = batch.to(device, dtype=dtype, non_blocking=True)
      lg, lp = model(batch)
      lg = _logit_adjust(lg, sidecar_tensors["prior_gold"], tau=tau_gold)
      lp = _logit_adjust(lp, sidecar_tensors["prior_pseudo"], tau=tau_pseudo)
      loss = _masked_loss(lg, lp, y_gold, y_pseudo)
    else:
      batch = batch.to(device, non_blocking=True)
      with autocast(dtype=dtype):
        lg, lp = model(batch)
        lg = _logit_adjust(lg, sidecar_tensors["prior_gold"], tau=tau_gold)
        lp = _logit_adjust(lp, sidecar_tensors["prior_pseudo"], tau=tau_pseudo)
        loss = _masked_loss(lg, lp, y_gold, y_pseudo)
    total += float(loss.detach().cpu())
    n += 1

  return total / max(1, n)

def _training_loop(
  model_p: Path,
  dataset: Any,
  sidecar_tensors: dict[str, torch.Tensor],
  split_idx: dict[str, list[int]],
  dtype: Any,
  device: torch.device,
  epochs: int = 500,
  tau_gold: float = 1.25,  # how strongly the correction scales
  tau_pseudo: float = 0.75,
) -> None:
  logger = _init_log(model_p)
  stopper = _AutoStopTransformer(model_p)

  model: _TromptTwoMultiHeaded = _init_model(dataset).to(device=device, dtype=dtype)
  opt, sched = _optimizer_scheduler(model, epochs, device)
  scaler = GradScaler()

  train_loader = _tf_loader(
    dataset.get_split("train"),
    split_idx["train"],
  )
  val_loader = _tf_loader(
    dataset.get_split("val"),
    split_idx["val"]
  )
  test_loader = _tf_loader(
    dataset.get_split("test"),
    split_idx["test"]
  )

  for epoch in range(1, epochs + 1):
    train_loss: float = _train_epoch(model, train_loader, sidecar_tensors, opt, scaler, dtype, device, tau_gold, tau_pseudo)
    val_loss: float = _eval_epoch(model, val_loader, sidecar_tensors, dtype, device, tau_gold, tau_pseudo)
    logger.info(f"LOG-CODE:3 | Epoch:{epoch:.4f} | TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | Best:{stopper.best:.4f} | Wait:{stopper.wait}")
    sched.step()
    if stopper._step(val_loss, model):
      break
  
  if not model_p.exists():
    checkpoint: str = model_p.as_posix()
    raise FileNotFoundError(f"PY-CODE:8 | Autoencoder Checkpoint Not Found... {checkpoint}")

  test_loss: float = _eval_epoch(model, test_loader, sidecar_tensors, dtype, device, tau_gold, tau_pseudo)
  logger.info(f"LOG-CODE:4 | TestLoss:{test_loss:.4f}")
  return None

def fit_model(snapshot: dict[str, Any]) -> None:
  set_seed()
  dtype, device = dtype_device()
  version: int = snapshot["version"]
  parquet_p: Path = root() / "PARQUET" / str(version)
  lables, lf = _label_frame(snapshot["gold_labels"], snapshot["pseudo_lables"])
  ff: pd.DataFrame = _load_features(parquet_p, lables)
  df: pd.DataFrame = _label_features(ff, lf)
  df = _add_masks_and_targets(df)
  df = _attach_split(df)
  dataset: Any = _load_dataset(df)
  sidecar_tensors: dict[str, torch.Tensor] = _sidecar_tensors(df, device)
  split_idx: dict[str, list[int]] = _indexes_from_split(df)
  priors: dict[str, torch.Tensor] = _train_priors(df, dtype, device)
  sidecar_tensors.update(priors)
  model_p: Path = snapshot["saveto"]
  model_p.parent.mkdir(parents=True, exist_ok=True)
  _training_loop(
    model_p,
    dataset,
    sidecar_tensors,
    split_idx,
    dtype,
    device
  )
  return None
