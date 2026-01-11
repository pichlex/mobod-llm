from __future__ import annotations

import pandas as pd
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mobod_llm.experiments.common import build_dataloaders, count_params, evaluate_classifier, train_classifier


def build_peft_config(adapter_name: str, model, cfg: dict) -> object:
    if adapter_name == "lora":
        lora_cfg = cfg["lora"]
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        )

    if adapter_name == "prompt":
        prompt_cfg = cfg["prompt"]
        return PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=prompt_cfg["num_virtual_tokens"],
            prompt_tuning_init=PromptTuningInit.RANDOM,
            tokenizer_name_or_path=model.config._name_or_path,
        )

    if adapter_name == "prefix":
        prefix_cfg = cfg["prefix"]
        return PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=prefix_cfg["num_virtual_tokens"],
            prefix_projection=prefix_cfg["prefix_projection"],
        )

    raise ValueError(f"Unknown adapter: {adapter_name}")


def run_adapters(cfg: dict, device: torch.device) -> pd.DataFrame:
    model_name = cfg["model_name"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    max_length = cfg["max_length"]
    adapters = cfg["adapters"]
    data_formats = cfg["data_formats"]

    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    rows = []
    for adapter_name in adapters:
        for precision in data_formats:
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            base_model.config.pad_token_id = tokenizer.pad_token_id
            peft_config = build_peft_config(adapter_name, base_model, cfg)
            model = get_peft_model(base_model, peft_config)
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

            acc, f1 = evaluate_classifier(model, eval_loader, device, precision)
            trainable_params, total_params = count_params(model)

            rows.append(
                {
                    "stage": "adapters",
                    "approach": adapter_name,
                    "data_format": precision,
                    "accuracy": acc,
                    "f1": f1,
                    "training_time_s": train_time,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "device": str(device),
                }
            )

    return pd.DataFrame(rows)
