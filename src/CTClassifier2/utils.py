from __future__ import annotations
# ! annotations MUST BE AT THE TOP
from ruamel.yaml.error import YAMLError
from ruamel.yaml import YAML
from functools import cache
from pathlib import Path
from typing import Any
from os import environ
import numpy as np
import random
import torch

@cache
def root(cwd: Path = Path.cwd()) -> Path:
  return cwd.resolve() / "CACHE"

yaml = YAML()

def load_yaml(config: str) -> Any:
    try:
        with open(config, "r") as f:
            return yaml.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"PY-CODE:2 | File not found... {config}")
    except PermissionError:
        raise RuntimeError(f"PY-CODE:3 | Permission denied.. {config}")
    except YAMLError as e:
        err: str = str(e)
        raise RuntimeError(f"PY-CODE:4 | YAML parsing error... {config}... {err}")
    
def set_seed(x: int = int(environ["PYTHONHASHSEED"])) -> None:
  torch.use_deterministic_algorithms(True, warn_only=True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.cuda.manual_seed_all(x)
  torch.cuda.manual_seed(x)
  torch.manual_seed(x)
  np.random.seed(x)
  random.seed(x)
  return None

def dtype_device() -> tuple[Any, torch.device]:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dtype = torch.float16 if device.type == "cuda" else torch.float32
  return dtype, device
