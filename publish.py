"""Publish the exact experiment splits and winning checkpoint to the Hub."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()
    api = HfApi()
    experiment = json.loads(Path("runs/comparison.json").read_text(encoding="utf-8"))
    winner = experiment["winner"]
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        for split in ("supplied", "generated", "validation"):
            copyfile(f"prepared/{split}.jsonl", folder / f"{split}.jsonl")
        card = """---
language: sv
configs:
- config_name: supplied
  data_files:
  - split: train
    path: supplied.jsonl
  - split: validation
    path: validation.jsonl
- config_name: generated
  data_files:
  - split: train
    path: generated.jsonl
  - split: validation
    path: validation.jsonl
---
# Swedish grammaticality training data

The supplied configuration uses the assignment's scored examples, split by source group.
The generated configuration uses Swedish templates with seeded grammatical errors
and matches the supplied training set in size and class counts. Both configurations
use the same supplied validation rows. IDs and groups are local to their source.
Each row includes the text, a score in [0,1], a good/poor label and an optional error description.
Generated scores are heuristic targets: 0.95 for clean text, 0.45 for one error and 0.25 for two.
"""
        (folder / "README.md").write_text(card, encoding="utf-8")
        api.create_repo(args.dataset_id, repo_type="dataset", private=False, exist_ok=True)
        data_commit = api.upload_folder(repo_id=args.dataset_id, repo_type="dataset", folder_path=folder)
    model_dir = Path("runs") / winner
    card = f"""---
language: sv
library_name: transformers
base_model: {experiment['base']}
datasets:
- {args.dataset_id}
---
# Swedish grammaticality scorer

This KB-BERT sequence regressor was fine-tuned on the {winner} configuration
of {args.dataset_id} at revision {data_commit.oid}. Tokenize cased texts with
truncation at 256 tokens. Apply sigmoid to the single output logit to obtain
a grammaticality score between zero and one. The score is a regression estimate.
Validation comparison and training settings are in comparison.json.
"""
    (model_dir / "README.md").write_text(card, encoding="utf-8")
    copyfile("runs/comparison.json", model_dir / "comparison.json")
    api.create_repo(args.model_id, private=False, exist_ok=True)
    commit = api.upload_folder(repo_id=args.model_id, folder_path=model_dir)
    ref = dict(model_id=args.model_id, revision=commit.oid)
    Path("model_ref.json").write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    print(f"Model: {args.model_id} @ {commit.oid}\nDataset: {args.dataset_id} @ {data_commit.oid}")


if __name__ == "__main__":
    main()
