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

@cache(maxsize=1)
def root(utils_p: Path = Path(__file__), build: str = "pyproject.toml") -> Path:
  for p in [utils_p] + list(utils_p.parents):
    if (p / build).exists():
      return p.resolve() / "CACHE"
  utils: str = utils_p.as_posix()
  raise FileNotFoundError(f"PY-CODE:1 | Couldn't Locate Root... {utils}")

yaml = YAML()

def load_yaml(yaml: str) -> Any:
    try:
        with open(yaml, "r") as f:
            return yaml.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"PY-CODE:2 | File not found... {yaml}")
    except PermissionError:
        raise RuntimeError(f"PY-CODE:3 | Permission denied.. {yaml}")
    except YAMLError as e:
        err: str = str(e)
        raise RuntimeError(f"PY-CODE:4 | YAML parsing error... {yaml}... {err}")
    
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
