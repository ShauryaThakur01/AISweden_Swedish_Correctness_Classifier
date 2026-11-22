"""Fine-tune both candidates independently, then select on supplied validation."""

import json
import os
import platform
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from predict import encode, scores
from prepare import DATA, SEED, read_rows, write_rows

BASE = "KBLab/bert-base-swedish-cased"
REVISION = "ce7c3424687f042f1320e0528293d492c82918c4"
RUNS = Path("runs")
EPOCHS = 5
BATCH = 8


def fit(name, tokenizer, valid, device):
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=1, use_safetensors=True
    ).to(device)
    count = sum(p.numel() for p in model.parameters())
    if count >= 1_300_000_000:
        raise ValueError("Model exceeds submission size limit.")
    train = read_rows(DATA / f"{name}.jsonl")
    loader = DataLoader(range(len(train)), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for indices in loader:
            batch = [train[i] for i in indices.tolist()]
            inputs = encode(tokenizer, [r["text"] for r in batch], device)
            target = torch.tensor([r["score"] for r in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(**inputs).logits.flatten().sigmoid()
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"{name} epoch {epoch + 1}/{EPOCHS}, training MSE {total / len(train):.5f}", flush=True)

    pred = scores(model, tokenizer, valid)
    truth = [r["score"] for r in valid]
    correlation = float(spearmanr(truth, pred).statistic)
    result = dict(spearman=correlation if len(set(pred)) > 1 else None,
                  auc=float(roc_auc_score([r["label"] == "good" for r in valid], pred)),
                  mae=float(mean_absolute_error(truth, pred)), parameters=count, train_rows=len(train))
    folder = RUNS / name
    model.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    write_rows(RUNS / f"{name}_validation.jsonl",
               (dict(**r, prediction=p) for r, p in zip(valid, pred)))
    return result


def rank(metrics):
    rho = metrics["spearman"]
    return (float("-inf") if rho is None else rho, metrics["auc"], -metrics["mae"])


def main():
    valid = read_rows(DATA / "validation.jsonl")
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    # Each invocation owns its output. An interrupted rerun cannot overwrite a winner.
    RUNS.mkdir(exist_ok=False)
    candidates = {name: fit(name, tokenizer, valid, device) for name in ("supplied", "generated")}
    winner = max(candidates, key=lambda name: rank(candidates[name]))
    summary = dict(winner=winner, candidates=candidates, validation_rows=len(valid),
                   base=BASE, revision=REVISION, seed=SEED, epochs=EPOCHS, batch_size=BATCH,
                   device=torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
                   os=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda)
    report = json.dumps(summary, indent=2, allow_nan=False)
    (RUNS / "comparison.json").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
