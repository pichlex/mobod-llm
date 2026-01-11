from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from mobod_llm.config import apply_overrides, load_config
from mobod_llm.experiments.adapters import run_adapters
from mobod_llm.experiments.datasets import run_dataset_comparison
from mobod_llm.experiments.freeze_layers import run_freeze_layers
from mobod_llm.experiments.quant_lora import run_quant_lora
from mobod_llm.experiments.zero_shot import run_zero_shot
from mobod_llm.utils import prepare_run_dir, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all training experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Override config key=val.")
    parser.add_argument("--device", default=None, help="Override runtime.device (cuda/cpu/mps).")
    parser.add_argument("--run-name", default=None, help="Optional run name prefix.")
    parser.add_argument("--stages", default=None, help="Comma-separated list of stages.")
    return parser.parse_args()


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def build_summary(csv_paths: list[Path], output_path: Path) -> None:
    frames = []
    cols = [
        "stage",
        "approach",
        "data_format",
        "accuracy",
        "f1",
        "training_time_s",
        "trainable_params",
        "total_params",
        "device",
    ]
    for path in csv_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in cols:
            if col not in df.columns:
                df[col] = None
        frames.append(df[cols])
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.device:
        cfg.setdefault("runtime", {})["device"] = args.device
    runtime = cfg["runtime"]

    seed_everything(runtime["seed"])
    device_info = resolve_device(runtime["device"])
    if device_info.fallback:
        print(f"Requested device '{runtime['device']}' unavailable, using CPU.")

    run_dir = prepare_run_dir(runtime["output_dir"], args.run_name or runtime.get("run_name"))
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(run_dir / "run.log", level="INFO")
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    stages = (
        [s.strip() for s in args.stages.split(",")]
        if args.stages
        else ["zero_shot", "freeze_layers", "adapters", "datasets", "quant_lora", "summary"]
    )

    outputs = {}

    if "zero_shot" in stages:
        logger.info("Stage: zero_shot")
        df = run_zero_shot(cfg["zero_shot"])
        path = run_dir / "zero_shot.csv"
        write_csv(path, df)
        outputs["zero_shot"] = path

    if "freeze_layers" in stages:
        logger.info("Stage: freeze_layers")
        df = run_freeze_layers(cfg["freeze_layers"], device_info.device)
        path = run_dir / "freeze_layers.csv"
        write_csv(path, df)
        outputs["freeze_layers"] = path

    if "adapters" in stages:
        logger.info("Stage: adapters")
        df = run_adapters(cfg["adapters"], device_info.device)
        path = run_dir / "adapters.csv"
        write_csv(path, df)
        outputs["adapters"] = path

    if "datasets" in stages:
        logger.info("Stage: datasets")
        df = run_dataset_comparison(cfg["datasets"], device_info.device)
        path = run_dir / "datasets.csv"
        write_csv(path, df)
        outputs["datasets"] = path

    if "quant_lora" in stages:
        logger.info("Stage: quant_lora")
        df = run_quant_lora(cfg["quant_lora"], device_info.device)
        path = run_dir / "quant_lora.csv"
        write_csv(path, df)
        outputs["quant_lora"] = path

    if "summary" in stages:
        logger.info("Stage: summary")
        summary_path = run_dir / "summary.csv"
        build_summary(list(outputs.values()), summary_path)

    logger.info(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
