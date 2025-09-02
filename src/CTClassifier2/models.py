from __future__ import annotations
# ! annotations MUST BE AT THE TOP
from pydantic import ValidationError
from pydantic import BaseModel
from pydantic import Field
from pathlib import Path
from typing import Any

class _SnapshotModel(BaseModel):
  version: int = Field(...)
  zip_directory: Path = Field(...)
  tables: list[str] = Field(...)
  gold_labels: Path = Field(...)
  pseudo_lables: Path = Field(...)
  save_to: Path = Field(...)

def get_snapshot(parsed: Any) -> dict[str, Any]:
  try:
    return _SnapshotModel.model_validate(parsed).model_dump()
  except ValidationError as e:
    err: str = str(e)
    raise ValidationError(f"PY-CODE:5 | Configuration Validation Error... {e}")
