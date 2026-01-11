from __future__ import annotations

import contextlib
import time

import evaluate
import torch
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup


def build_dataloaders(train_dataset, eval_dataset, tokenizer, batch_size: int) -> tuple[DataLoader, DataLoader]:
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)
    return train_loader, eval_loader


def evaluate_classifier(model, data_loader: DataLoader, device: torch.device, precision: str) -> tuple[float, float]:
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    model.eval()
    preds, refs = [], []
    use_autocast = precision == "fp16" and device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast() if use_autocast else contextlib.nullcontext()

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx:
                outputs = model(**batch)
            logits = outputs.logits
            preds.append(torch.argmax(logits, dim=-1).cpu())
            refs.append(batch["labels"].cpu())

    preds = torch.cat(preds).numpy()
    refs = torch.cat(refs).numpy()

    acc = accuracy_metric.compute(predictions=preds, references=refs)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=refs, average="binary")["f1"]
    return acc, f1


def train_classifier(
    model,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    precision: str,
    stage: str | None = None,
) -> float:
    lr = float(learning_rate)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * num_training_steps)),
        num_training_steps=num_training_steps,
    )

    use_autocast = precision == "fp16" and device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast() if use_autocast else contextlib.nullcontext()
    scaler = torch.cuda.amp.GradScaler() if use_autocast else None

    model.train()
    start = time.time()
    for epoch_idx in range(1, epochs + 1):
        if stage:
            logger.info(f"{stage}: epoch {epoch_idx}/{epochs} start")
        for batch in tqdm(train_loader, desc=f"epoch {epoch_idx}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx:
                outputs = model(**batch)
                loss = outputs.loss
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if stage:
            logger.info(f"{stage}: epoch {epoch_idx}/{epochs} end")
    return time.time() - start


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
