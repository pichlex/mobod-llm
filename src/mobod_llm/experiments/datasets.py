from __future__ import annotations

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mobod_llm.experiments.common import build_dataloaders, count_params, evaluate_classifier, train_classifier


def prepare_squad(tokenizer, max_length: int):
    dataset = load_dataset("squad")

    def preprocess(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = [1] * len(examples["question"])
        return tokenized

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    tokenized.set_format("torch")
    return tokenized["train"], tokenized["validation"]


def prepare_rte(tokenizer, max_length: int):
    dataset = load_dataset("super_glue", "rte")

    def preprocess(examples):
        tokenized = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    tokenized.set_format("torch")
    return tokenized["train"], tokenized["validation"]


def prepare_cola(tokenizer, max_length: int):
    dataset = load_dataset("glue", "cola")

    def preprocess(examples):
        tokenized = tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    tokenized.set_format("torch")
    return tokenized["train"], tokenized["validation"]


def run_dataset_comparison(cfg: dict, device: torch.device) -> pd.DataFrame:
    model_name = cfg["model_name"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    max_length = cfg["max_length"]

    rows = []
    for dataset_name in ["SQuAD", "RTE", "CoLA"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if dataset_name == "SQuAD":
            train_dataset, eval_dataset = prepare_squad(tokenizer, max_length)
        elif dataset_name == "RTE":
            train_dataset, eval_dataset = prepare_rte(tokenizer, max_length)
        else:
            train_dataset, eval_dataset = prepare_cola(tokenizer, max_length)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)

        train_loader, eval_loader = build_dataloaders(train_dataset, eval_dataset, tokenizer, batch_size)

        train_time = train_classifier(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            precision="fp32",
        )

        acc, f1 = evaluate_classifier(model, eval_loader, device, "fp32")
        trainable_params, total_params = count_params(model)

        rows.append(
            {
                "stage": "datasets",
                "approach": dataset_name,
                "data_format": "fp32",
                "accuracy": acc,
                "f1": f1,
                "training_time_s": train_time,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "device": str(device),
            }
        )

    return pd.DataFrame(rows)
