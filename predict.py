"""Submission entry point. The model emits one logit per text."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prepare import read_rows, write_rows

ROOT = Path(__file__).resolve().parent
MAX_LENGTH = 256


def encode(tokenizer, texts, device):
    return tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH,
                     return_tensors="pt").to(device)


@torch.inference_mode()
def scores(model, tokenizer, rows, batch_size=16):
    model.eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs = encode(tokenizer, [r["text"] for r in batch], model.device)
        predictions.extend(model(**inputs).logits.flatten().sigmoid().tolist())
    return predictions


def default_checkpoint():
    selection = ROOT / "runs" / "comparison.json"
    if selection.exists():
        winner = json.loads(selection.read_text(encoding="utf-8"))["winner"]
        return str(selection.parent / winner), None
    publication = ROOT / "model_ref.json"
    if publication.exists():
        ref = json.loads(publication.read_text(encoding="utf-8-sig"))
        return ref["model_id"], ref["revision"]
    raise SystemExit("Train first, configure model_ref.json, or pass --model.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", help="Override the selected checkpoint with a directory or Hub ID")
    args = parser.parse_args()
    source, revision = (args.model, None) if args.model else default_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(source, revision=revision)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_rows(args.input)
    predictions = scores(model, tokenizer, rows)
    write_rows(args.output, (dict(id=row["id"], score=s) for row, s in zip(rows, predictions)))


if __name__ == "__main__":
    main()
