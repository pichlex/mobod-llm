from __future__ import annotations

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mobod_llm.experiments.common import build_dataloaders, count_params, evaluate_classifier, train_classifier


def get_encoder_layers(model) -> list:
    base = getattr(model, model.base_model_prefix, None)
    if base is None:
        base = getattr(model, "base_model", None)
    if base is None:
        return []

    if hasattr(base, "encoder") and hasattr(base.encoder, "albert_layer_groups"):
        layers = []
        for group in base.encoder.albert_layer_groups:
            layers.extend(list(group.albert_layers))
        return layers

    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        return list(base.encoder.layer)

    if hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
        return list(base.transformer.layer)

    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        return list(base.transformer.h)

    return []


def freeze_layers(model, freeze_pct: int) -> int:
    layers = get_encoder_layers(model)
    if not layers:
        return 0

    total_layers = len(layers)
    n_freeze = int(total_layers * (freeze_pct / 100))
    for idx, layer in enumerate(layers):
        requires_grad = idx >= n_freeze
        for param in layer.parameters():
            param.requires_grad = requires_grad
    return n_freeze


def run_freeze_layers(cfg: dict, device: torch.device) -> pd.DataFrame:
    model_name = cfg["model_name"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    max_length = cfg["max_length"]
    layer_freezing = cfg["layer_freezing"]
    data_formats = cfg["data_formats"]

    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    rows = []
    for freeze_pct in layer_freezing:
        for precision in data_formats:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            frozen_layers = freeze_layers(model, freeze_pct)
            model.to(device)

            train_loader, eval_loader = build_dataloaders(train_dataset, eval_dataset, tokenizer, batch_size)

            train_time = train_classifier(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                precision=precision,
            )

            if precision == "int8":
                model_cpu = model.to("cpu")
                quantized_model = torch.quantization.quantize_dynamic(
                    model_cpu, {torch.nn.Linear}, dtype=torch.qint8
                )
                eval_loader_cpu = build_dataloaders(train_dataset, eval_dataset, tokenizer, batch_size)[1]
                acc, f1 = evaluate_classifier(quantized_model, eval_loader_cpu, torch.device("cpu"), "fp32")
                device_name = "cpu"
            else:
                acc, f1 = evaluate_classifier(model, eval_loader, device, precision)
                device_name = str(device)

            trainable_params, total_params = count_params(model)

            rows.append(
                {
                    "stage": "freeze_layers",
                    "approach": f"freeze_{freeze_pct}",
                    "frozen_layers_pct": freeze_pct,
                    "frozen_layers_count": frozen_layers,
                    "data_format": precision,
                    "accuracy": acc,
                    "f1": f1,
                    "training_time_s": train_time,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "device": device_name,
                }
            )

    return pd.DataFrame(rows)
