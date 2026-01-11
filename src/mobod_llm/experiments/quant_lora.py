from __future__ import annotations

import time

import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from mobod_llm.experiments.common import count_params, train_classifier


def run_quant_lora(cfg: dict, device: torch.device) -> pd.DataFrame:
    model_name = cfg["model_name"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    epochs = cfg["epochs"]
    max_length = cfg["max_length"]
    lora_cfg = cfg["lora"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_dataset("glue", "sst2")
    test_dataset = dataset["validation"]

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=max_length, padding="max_length")

    tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_test.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(tokenized_test, batch_size=batch_size, shuffle=False, collate_fn=collator)

    base_model.eval()
    quantized_model = torch.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)

    def inference_time(model, device_name: str) -> float:
        model.to(device_name)
        model.eval()
        start = time.time()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device_name) for k, v in batch.items()}
                model(**batch)
        return time.time() - start

    original_time = inference_time(base_model, "cpu")
    quantized_time = inference_time(quantized_model, "cpu")

    original_params = sum(p.numel() for p in base_model.parameters())
    quantized_params = sum(p.numel() for p in quantized_model.parameters())

    lora_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=["q_lin", "v_lin"],
    )
    lora_model = get_peft_model(lora_model, lora)
    lora_model.to(device)

    train_dataset = dataset["train"]
    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_train.set_format("torch")
    train_loader = torch.utils.data.DataLoader(
        tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collator
    )

    train_time = train_classifier(
        model=lora_model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        precision="fp32",
        stage="quant_lora:lora",
    )
    trainable_params, total_params = count_params(lora_model)

    rows = [
        {
            "stage": "quant_lora",
            "approach": "original",
            "data_format": "fp32",
            "inference_time_s": original_time,
            "model_params": original_params,
        },
        {
            "stage": "quant_lora",
            "approach": "quantized_int8",
            "data_format": "int8",
            "inference_time_s": quantized_time,
            "model_params": quantized_params,
            "speedup": original_time / quantized_time if quantized_time else float("inf"),
        },
        {
            "stage": "quant_lora",
            "approach": "lora",
            "data_format": "fp32",
            "training_time_s": train_time,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "device": str(device),
        },
    ]

    return pd.DataFrame(rows)
