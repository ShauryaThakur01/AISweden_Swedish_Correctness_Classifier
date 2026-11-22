# Run the Swedish KB-BERT experiment

Use Windows PowerShell from the project folder. Each block assumes the preceding step succeeded. A CPU is sufficient. The first training run needs internet access for the public pretrained checkpoint. Allow several GB of disk space for the environment, download cache and two fine-tuned models.

## 1. Set up Python

This machine already has Python 3.11.7 in Anaconda. Use its executable directly to create the project environment. On another Windows machine with a registered Python 3.11 installation, you can use `py -3.11 -m venv .venv` instead. Using the virtual environment's executable directly also works when PowerShell blocks activation.

```powershell
cd "$HOME\Desktop\swedish_correctness_classifier"
& "$HOME\anaconda3\python.exe" -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py --version
& $py -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
& $py -m pip check
```

For an NVIDIA machine with a CUDA 12.4-compatible driver, substitute `https://download.pytorch.org/whl/cu124` for the CPU index in the torch installation command. Choose the build before installing the remaining requirements. Training and prediction automatically use CUDA when available. Record the hardware and use the same build, machine and package versions to reproduce the run. In a new PowerShell session, set `$py` again.

## 2. Prepare the two training sets

```powershell
$env:PYTHONHASHSEED = "42"
& $py prepare.py
```

This reads `data/train.jsonl` and prints each split's size and good count. In `prepared/`, `supplied.jsonl` is the training portion of the assignment data, `generated.jsonl` is the matched synthetic training set, and `validation.jsonl` holds entire groups from the supplied data. Preserve these files from this run for publication. The two test files are first needed in step 4.

## 3. Train both candidates and select the winner

```powershell
& $py train.py
Get-Content runs/comparison.json
& $py -m pip freeze | Set-Content -Encoding ascii environment-lock.txt
```

The command first fine-tunes the supplied-data model, then starts again from the identical base checkpoint for the generated-data model. A Transformers message about newly initialized classifier weights is expected. Both candidates use five epochs, batch size 8, learning rate 0.00002 and seed 42. The last checkpoint from each run is scored on the same validation texts. Highest Spearman wins, followed by highest ROC-AUC, lowest MAE, and finally the supplied candidate on an exact tie. Undefined Spearman ranks last.

The model and tokenizer folders are `runs/supplied/` and `runs/generated/`. The comparison records the winner, metrics, sample counts, parameter count and device. Each `*_validation.jsonl` includes the source labels and predictions for error analysis. To repeat training, rename the existing `runs` directory to an unused backup name first, including after an interrupted run. Preserve the environment lock alongside your results. Recreate that exact environment with the same Python and PyTorch wheel index, installing the recorded torch build first and then `pip install -r environment-lock.txt`.

## 4. Evaluate the selected model once on the supplied test

```powershell
& $py evaluate.py | Tee-Object -FilePath runs/test-results.txt
```

The unchanged starter evaluator calls the required predictor interface in a subprocess, compares `scores.jsonl` with `data/test_labels.jsonl`, and reports Spearman, ROC-AUC, elapsed seconds and samples/minute, including model loading. It uses the local winner automatically. Record the CPU or GPU model, RAM, and whether CUDA was used. This is the 120-row public test. The organizers run their separate hidden test after submission.

To try the selected model on the included example pair, run the following commands. This optional step writes a separate output file.

```powershell
& $py predict.py --input my_texts.jsonl --output my_scores.jsonl
Get-Content my_scores.jsonl
```

Each nonblank input line must contain an ID and Swedish text, for example `{"id":1,"text":"I går läste läraren ett nytt brev."}`. Each output line contains that ID and a numeric score.

## 5. Publish the model and training data on Hugging Face

1. Sign in to Hugging Face and find your account handle (or an organization where you can create repositories). The starting checkpoint is `KBLab/bert-base-swedish-cased`. Your submission needs a **new model ID** pointing to your trained winner and a **dataset ID** pointing to the exact training files.
2. Choose two unused public repository names under your handle. For example, if your handle is `anna123`, use `anna123/swedish-grammar-kbbert` and `anna123/swedish-grammar-data`. The script creates the repos. One dataset ID covers both candidates through the named configurations `supplied` and `generated`.
3. Create a Hugging Face access token with write permission for these repos. Run the login command and paste the token at the hidden prompt. The token is a credential and belongs outside your code and report. [Hugging Face token settings](https://huggingface.co/settings/tokens)
4. Replace `YOUR_HANDLE` below with your actual handle. Run this block from the same project directory after training.

```powershell
.\.venv\Scripts\huggingface-cli.exe login
& $py publish.py --model-id YOUR_HANDLE/swedish-grammar-kbbert --dataset-id YOUR_HANDLE/swedish-grammar-data
```

The script uploads both training splits and their common validation set, then the winner's weights, tokenizer, configuration and model card. It writes the published model ID and immutable commit to `model_ref.json`. Keep this file in GitHub. In a fresh checkout, the required predictor command downloads that exact winner. The dataset card exposes the two training configurations for readers. The publication script prints both repository IDs and revisions.

## 6. Submit and finish the paper

In this workspace Git currently resolves to a parent directory. Before staging files, run `git rev-parse --show-toplevel` from the project folder. If it prints a parent path, run `git init -b main` inside `swedish_correctness_classifier` to give the project its own repository, then verify the top-level path again. Connect that project repository to your intended GitHub remote.

Commit the source files, pinned requirements, environment lock, run guide, paper, supplied assignment files and `model_ref.json` to your GitHub repository. Push the commit to the repository you intend to submit. The virtual environment, checkpoints and prepared copies are already ignored. Check the model and dataset pages while signed out. In a fresh checkout with the environment from step 1, use the example input from step 4 to verify that default inference resolves the Hub model.

Submit the GitHub repository URL, your model ID, your dataset ID and the finished paper. To finish the paper, send the contents of `runs/comparison.json`, `runs/test-results.txt`, the hardware details, and both validation prediction files (or representative rows with text, target and prediction). Include any failed trials you actually attempted and the final repository IDs.
