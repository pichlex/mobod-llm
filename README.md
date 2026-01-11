# mobod-llm

Experiments are moved out of the notebook into reproducible scripts with YAML configs and CLI overrides.

## Structure
- `configs/`: YAML configs for experiments.
- `scripts/`: CLI entrypoints (run all experiments).
- `src/mobod_llm/`: experiment code and utilities.
- `outputs/`: generated CSV artifacts per run (ignored by git).
- `notebooks/`: evaluation notebook that builds plots from CSVs.

## Quick start
```bash
uv run scripts/run_experiments.py --config configs/default.yaml
```

Override any config value:
```bash
uv run scripts/run_experiments.py --config configs/default.yaml \
  --override runtime.device=cpu \
  --override freeze_layers.batch_size=4
```

## Outputs
Each run creates `outputs/<run_name>_<timestamp>/` with:
- `zero_shot.csv`
- `freeze_layers.csv`
- `adapters.csv`
- `datasets.csv`
- `quant_lora.csv`
- `summary.csv`
- `config.resolved.yaml`

Use `notebooks/evaluation.ipynb` to load CSVs and build plots.
