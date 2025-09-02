from __future__ import annotations
# ! annotations MUST BE AT THE TOP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.amp import autocast
from functools import partial
from loguru import logger
from pathlib import Path
from typing import Self
from typing import Any
from torch import nn
import numpy as np
import torch
import math

def _hidden_dims(indim: int, outdim: int, n_layers: int) -> list[int]:
  ratios = np.geomspace(indim, outdim, num=(n_layers + 2))
  return [int(round(r)) for r in ratios]

def _dynamic_sequential(
  hidden: list[int],
  complex_embed: bool,
  activation_fn: nn.Module,
  dropout: nn.Module
) -> Any:
  layers: list[nn.Module] = []

  for idx, h in enumerate(hidden):
    if idx == 0:
      prev = h
    elif idx == 1 and complex_embed:
      layers += [nn.Linear(prev, h), activation_fn, dropout]
      prev = h
    else:
      layers += [nn.Linear(prev, h), activation_fn]
      prev = h

  return nn.Sequential(*layers)

class _AutoEncoder(nn.Module):
  def __init__(
    self: Self,
    bert_out_size: int,
    ae_out_size: int,
    complex_embed: bool = False,
    n_layers: int = 3,
    activation_fn: nn.Module = nn.GELU(),
    dropout: nn.Module = nn.Dropout(0.1)
  ) -> None:
    super().__init__()
    hidden: list[int] = _hidden_dims(
      bert_out_size,
      ae_out_size,
      n_layers
    )
    _ae_sequential = partial(
      _dynamic_sequential,
      complex_embed=complex_embed,
      activation_fn=activation_fn,
      dropout=dropout
    ) 
    self.encoder = _ae_sequential(reversed(hidden))
    self.decoder = _ae_sequential(hidden)
    return None
  
  def _encode(self: Self, x: torch.Tensor) -> torch.Tensor:
    return self.encoder(x)
  
  def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
    z = self.encoder(x)
    x_hat = self.decoder(z) 
    return x_hat, z
  
def _ttv_split(
  n: int,
  train_ratio: float,
  val_ratio: float,
  test_ratio: float
) -> list[int]:
  _round_int = lambda x, y: int(round(x * y))
  split: list[int] = [
    _round_int(x, y) for x, y in zip(
      [n] * 3,
      [train_ratio, val_ratio, test_ratio]
    )
  ]
  diff: int = sum(split) - n
  if diff != 0:
    split[0] = split[0] - diff
    return split
  else:
    return split

def _training_loaders(
  dataset: Any,
  num_workers: int,
  pin_memory: bool,
  drop_last: bool,
  persistent_workers: bool,
  prefetch_factor: int,
  train_ratio: float = 0.7,
  val_ratio: float = 0.15,
  test_ratio: float = 0.15,
  batch_size: int = 64,
) -> tuple[Any, Any, Any]:
  n: int = len(dataset)
  split: list[int] = _ttv_split(n, train_ratio, val_ratio, test_ratio)
  train_ds, val_ds, test_ds = random_split(
    dataset,
    split,
    generator=torch.Generator()
  )
  _ae_data_loader = partial(
    DataLoader,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    drop_last=drop_last
  )
  train_loader = _ae_data_loader(train_ds, shuffle=True)
  val_loader = _ae_data_loader(val_ds, shuffle=False)
  test_loader = _ae_data_loader(test_ds, shuffle=False)
  return train_loader, val_loader, test_loader

def _ae_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  mse: torch.Tensor = F.mse(x_hat, x)
  cos: torch.Tensor = 1 - F.cosine_similarity(x_hat, x, dim=-1).mean()
  return mse + 0.1 * cos

@dataclass
class _AutoStopAutoencoder:
  patience: int = 7
  min_delta: float = 1e-4
  best: float = math.inf
  wait: int = 0
  stopped_at: int = 0
  checkpoint: Path = ...

  def _step(
    self: Self,
    loss: float,
    ae: _AutoEncoder
  ) -> bool:
    improved: bool = (self.best - loss) > self.min_delta

    if improved:
      self.best = loss
      self.wait = 0
      torch.save(ae.state_dict(), self.checkpoint)
    else:
      self.wait += 1
  
    return self.wait > self.patience
  
def _train_epoch(
  ae: _AutoEncoder,
  loader: Any,
  opt: Any,
  scaler: Any,
  device: torch.device,
  dtype: Any,
  max_norm: float = 1.0,
) -> float:
  ae.train()
  total: float = 0.0
  n: int = 0

  for x in loader:
    opt.zero_grad(set_to_none=True)
    if device.type != "cuda":
      x = x.to(device, dtype=dtype, non_blocking=True)
      x_hat, _ = ae(x)
      loss: torch.Tensor = _ae_loss(x_hat, x)
      loss.backward()
      clip_grad_norm_(ae.parameters(), max_norm=max_norm)
      opt.step()
    else:
      x = x.to(device, non_blocking=True)
      with autocast(dtype=dtype):
        x_hat, _ = ae(x)
        loss = _ae_loss(x_hat, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    bs = x.size(0)
    total += loss.item() * bs
    n += bs

  return total / max(n, 1)

@torch.no_grad()
def _eval_epoch(
  ae: _AutoEncoder,
  loader: Any,
  device: torch.device,
  dtype: Any,
) -> float:
  ae.eval()
  total: float = 0.0
  n: int = 0

  for x in loader:
    if device.type != "cuda":
      x = x.to(device, dtype=dtype, non_blocking=True)
      x_hat, _ = ae(x)
      loss: torch.Tensor = _ae_loss(x_hat, x)
    else:
      x = x.to(device, non_blocking=True)
      with autocast(dtype=dtype):
        x_hat, _ = ae(x)
        loss = _ae_loss(x_hat, x)
    bs = x.size(0)
    total += loss.item() * bs
    n += bs

  return total / max(n, 1)

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

def _training_loop(
  ae: _AutoEncoder,
  train_loader: Any,
  val_loader: Any,
  test_loader: Any,
  model_p: Path,
  device: torch.device,
  dtype: Any,
  scaler: Any,
  lr: float = 1e-4,
  weight_decay: float = 0.0,
  epochs: int = 500,
  sched_mode: str = "min",
  sched_factor: float = 0.5,
  sched_patience: int = 5,
  sched_min_delta: float = 1e-5,
  sched_min_lr: float = 1e-7
) -> None:
  logger = _init_log(model_p)
  opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
  sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode=sched_mode,
    factor=sched_factor,
    patience=sched_patience,
    threshold=sched_min_delta,
    min_lr=sched_min_lr,
  )
  stopper = _AutoStopAutoencoder(checkpoint=model_p)

  for epoch in range(1, epochs + 1):
    train_loss = _train_epoch(ae, train_loader, opt, scaler, device, dtype)
    val_loss = _eval_epoch(ae, val_loader, device, dtype)
    logger.info(f"LOG-CODE:1 | Epoch:{epoch} | TrainLoss:{train_loss} | ValLoss:{val_loss} | Best:{stopper.best:.4f} | Wait:{stopper.wait}")
    sched.step(val_loss)
    if stopper._step(val_loss, ae):
      break


  if not model_p.exists():
    checkpoint: str = model_p.as_posix()
    raise FileNotFoundError(f"PY-CODE:6 | Autoencoder Checkpoint Not Found... {checkpoint}")
  
  test_loss = _eval_epoch(ae, test_loader, opt, device, dtype)
  logger.info(f"LOG-CODE:2 | TestLoss:{test_loss}")
  return None

@torch.no_grad()
def _tranform_tensor(
  ae: _AutoEncoder,
  loader: Any,
  device: torch.device,
  dtype: Any,
) -> torch.Tensor:
  embed: list[torch.Tensor] = []
  for x in loader:
    if device.type != "cuda":
      x = x.to(device, dtype=dtype, non_blocking=True)
    else:
      x = x.to(device, non_blocking=True)
      with autocast(dtype=dtype):
        z = ae._encode(x)
    embed.append(z.cpu())
  return torch.cat(embed)

def encoder(
  pooler_out: torch.Tensor,
  bert_out_size: int,
  ae_out_size: int,
  model_p: Path,
  device: torch.device,
  dtype: Any,
  complex_embed: bool = False,
  training: bool = True,
  num_workers: int = 8,
  pin_memory: bool = True,
  drop_last: bool = True,
  persistent_workers: bool = True,
  prefetch_factor: int = 2,
) -> torch.Tensor:
  dataset = TensorDataset(pooler_out)
  if training:
    train_loader, val_loader, test_loader = _training_loaders(
      dataset,
      num_workers,
      pin_memory,
      drop_last,
      persistent_workers,
      prefetch_factor
    )
    ae = _AutoEncoder(
      bert_out_size,
      ae_out_size,
      complex_embed
    )
    scaler = GradScaler()
    _training_loop(
      ae,
      train_loader,
      val_loader,
      test_loader,
      model_p,
      device,
      dtype,
      scaler
    )
    return encoder(
      pooler_out,
      bert_out_size,
      ae_out_size,
      model_p,
      device,
      dtype,
      False,
      complex_embed,
      num_workers,
      pin_memory,
      drop_last
    )
  else:
    prod_loader = DataLoader(
      dataset,
      batch_size=128,
      num_workers=num_workers,
      shuffle=False,
      pin_memory=pin_memory,
      persistent_workers=persistent_workers,
      prefetch_factor=prefetch_factor,
      drop_last=drop_last
    )
    if not model_p.exists():
      model: str = model_p.as_posix()
      raise FileNotFoundError(f"PY-CODE:6 | Autoencoder Weights Not Found... {model}")
    ae = torch.load(model_p, map_location=device).eval()
    return _tranform_tensor(ae, prod_loader, device, dtype)
