from __future__ import annotations

import pandas as pd
from transformers import pipeline


def run_zero_shot(cfg: dict) -> pd.DataFrame:
    model_name = cfg["model_name"]
    prompts = cfg["prompts"]
    labels = cfg["labels"]
    hypothesis_template = cfg["hypothesis_template"]

    classifier = pipeline("zero-shot-classification", model=model_name)

    rows = []
    for prompt in prompts:
        result = classifier(prompt, candidate_labels=labels, hypothesis_template=hypothesis_template)
        for label, score in zip(result["labels"], result["scores"]):
            rows.append(
                {
                    "stage": "zero_shot",
                    "prompt": prompt,
                    "label": label,
                    "score": score,
                }
            )

    return pd.DataFrame(rows)
