from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from transformers import set_seed


@dataclass
class RuntimeInfo:
    device: torch.device
    device_name: str
    fallback: bool


def resolve_device(requested: str) -> RuntimeInfo:
    req = requested.lower()
    if req == "cuda" and torch.cuda.is_available():
        return RuntimeInfo(torch.device("cuda"), "cuda", False)
    if req == "mps" and torch.backends.mps.is_available():
        return RuntimeInfo(torch.device("mps"), "mps", False)
    if req == "cpu":
        return RuntimeInfo(torch.device("cpu"), "cpu", False)
    return RuntimeInfo(torch.device("cpu"), "cpu", True)


def prepare_run_dir(base_dir: str | Path, run_name: str | None) -> Path:
    base = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{run_name}_{timestamp}" if run_name else timestamp
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    set_seed(seed)
