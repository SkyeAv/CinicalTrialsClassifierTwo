from __future__ import annotations
# ! annotations MUST BE AT THE TOP
from CTClassifier2.datalake import generate_parquet
from CTClassifier2.freetext import embed_parquet
from CTClassifier2.utils import load_yaml
from typing import Any
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def install_snapshot(
  yaml: str = typer.Option(..., "-c", "--snapshot-config", help="Path to the snapshot config YAML")
) -> None:
  """Fetches Features from a Clinical Traials Snapshot and Outputs a Cached Parquet"""
  parsed: Any = load_yaml(yaml)
  snapshot: dict[str, Any] = snapshot(parsed)
  generate_parquet(snapshot)
  return None

@app.command()
def embed_snapshot(
  yaml: str = typer.Option(..., "-c", "--snapshot-config", help="Path to the snapshot config YAML")
) -> None:
  """Embeds and Auto-Encodes Freetext Features from a Clinical Traials Snapshot and Outputs a Cached Parquet"""
  parsed: Any = load_yaml(yaml)
  snapshot: dict[str, Any] = snapshot(parsed)
  embed_parquet(snapshot)
  return None

@app.command()
def train_labels(
  yaml: str = typer.Option(..., "-c", "--snapshot-config", help="Path to the snapshot config YAML")
) -> None:
  """Trains a Trompt Transformer on a Pre-Intialized and Embedded Clinical Traials Snapshot with Labled Trials """
  parsed: Any = load_yaml(yaml)
  snapshot: dict[str, Any] = snapshot(parsed)
  return None

def main() -> None:
  app()
  return None
