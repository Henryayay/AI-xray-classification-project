# How to use this repository

This document explains how the project is organised, how to run the checks, how to run the experiments, and how the files those experiments write become the numbers, tables and figures in the paper. It is the long form of the short instructions in `README.md`.

The scientific questions, endpoints and statistical tests are fixed in `docs/analysis_plan.md` and `configs/analysis_plan.yaml`. This document does not change that plan. It tells you how to execute it.

## 1. What this repository is for

The study compares vision-language models with supervised image models on two binary chest X-ray tasks from POLCOVID.

| Task | Negative class | Positive class | Hugging Face dataset |
|---|---|---|---|
| `task1` | Normal | Pneumonia | `Henryayay/POLCOVID_task1_70_10_20` |
| `task2` | Non-COVID Pneumonia | COVID-19 | `Henryayay/POLCOVID_task2_70_10_20` |

Each task has 1,596 training images, 228 validation images and 456 test images. The class counts are balanced. The exact dataset revisions are pinned in `src/cxr/tasks.py`.

The models are:

- CNNs trained on the task: ImageNet ResNet-50, ImageNet DenseNet-121, and a chest-X-ray-pretrained DenseNet-121.
- Frozen-encoder linear probes: RAD-DINO, and an ImageNet ResNet-50 used only as a CPU-sized probe.
- Gemma-3-4B-it and MedGemma-4B-it, each under zero-shot prompting, in-context learning, and LoRA fine-tuning.

Every condition is described by one file in `configs/`. The runner `scripts/run_experiment.py` reads that file, scores images, and writes one folder of results per run. Later scripts turn those folders into `results/summary/`, then into LaTeX tables and figures. Nothing that appears as a number in the paper is typed by hand.

## 2. Which files feed the paper

Use this path for any number you intend to report:

```
configs/*.yaml
    -> scripts/run_experiment.py
    -> results/<task>/<experiment>/<run>/
    -> scripts/analyze.py
    -> results/summary/
    -> scripts/make_tables.py and scripts/make_figures.py
    -> paper/generated/ and paper/figures/
    -> paper/main.tex
```

The notebooks under `ICL/`, `CNN/`, `Fine-tune/` and `legacy/` are the earlier exploratory runs. `scripts/extract_legacy_results.py` copies their published summary numbers into `results/legacy/legacy_results.csv` for the legacy comparison table. They are not re-scored by the new pipeline, and they do not fill the main tables.

`notebooks/colab_runner.ipynb` is only a driver. It clones or uses this repository and calls the same scripts. The logic lives in `src/cxr/` and `scripts/`.

## 3. Layout

| Path | Role |
|---|---|
| `src/cxr/` | Library: data loading, prompts, scoring, training, metrics, statistics |
| `configs/` | One YAML file per experiment, plus `analysis_plan.yaml` |
| `scripts/run_experiment.py` | Trains or scores one experiment and writes its results |
| `scripts/analyze.py` | Turns prediction files into metrics and pre-registered comparisons |
| `scripts/make_tables.py` | Writes the LaTeX tables in `paper/generated/` |
| `scripts/make_figures.py` | Writes the figures in `paper/figures/` |
| `scripts/audit_dataset.py` | Pixel-level audit of the two datasets |
| `scripts/power.py` | Sample-size and precision calculations for the fixed test set |
| `scripts/gradcam.py` | Grad-CAM panels for a finished CNN run |
| `scripts/extract_legacy_results.py` | Reads the old notebooks into `results/legacy/` |
| `scripts/make_synthetic_results.py` | Fake predictions for testing the pipeline only |
| `tests/` | CPU unit tests |
| `results/` | Audit, power, legacy numbers, and (after you run them) experiment outputs |
| `paper/` | LaTeX manuscript. `paper/generated/` and `paper/figures/` are produced by scripts |
| `docs/analysis_plan.md` | The statistical plan, frozen before test-set evaluation |
| `notebooks/colab_runner.ipynb` | Colab driver for the GPU experiments |
| `legacy/` | Original notebooks, kept unchanged |

Checkpoints (`model.pt`, LoRA adapters) and the embedding cache are git-ignored. Prediction files and `meta.json` are meant to be kept and, for the paper, committed.

## 4. Two different kinds of "test"

The word "test" is used for two unrelated things. Keep them separate.

**Unit tests** check that the code behaves as written. They do not download the POLCOVID images and they do not produce paper numbers. Run them on a laptop.

**The test split** is the held-out 456 images per task. Scoring it is an experimental result. The repository treats that as a separate, logged step (`--final`). A development run scores only the validation split, stores that result, and leaves the test images unread.

## 5. Prepare the environment

### 5.1 Laptop, for unit tests and for building the paper

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

`requirements-dev.txt` is enough for `pytest`, the audit, the power analysis, `analyze.py`, and the table and figure scripts. It is not enough to train Gemma or MedGemma.

### 5.2 GPU machine, for the experiments

An NVIDIA A100 is the machine the Colab runner expects. On that machine:

```bash
pip install -r requirements.txt -r requirements-gpu.txt
```

In Colab, open `notebooks/colab_runner.ipynb`, then choose **Runtime → Change runtime type → A100 GPU**. The notebook installs those requirements for you.

### 5.3 Hugging Face token

MedGemma is a gated model. The account that downloads it must have accepted the Health AI Developer Foundations terms on the model page, and the run must present that account's access token.

Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). The label you give the token on that page can be anything. The name the code looks up is fixed:

- In Colab, add a secret named exactly `HF_TOKEN` and paste the token value. The runner reads it with `userdata.get("HF_TOKEN")`.
- On a local or other GPU kernel, set the environment variable `HF_TOKEN` to the same value, or run `huggingface-cli login`.

Do not paste the token into a notebook cell or commit it. The runner does not upload anything to the Hugging Face Hub.

Gemma-3 and the POLCOVID datasets can often be read without a token. The token is required for MedGemma, and it also raises the Hub rate limit for the other downloads.

### 5.4 Where the kernel runs

A notebook opened in Cursor still executes on whichever kernel is selected. The result files are written on that kernel's disk.

Run this in the notebook if you are unsure:

```python
import os, platform
print(os.name)
print(platform.platform())
print(os.getcwd())
```

`nt` and a path under `C:\Users\Henryayay\` means a local Windows kernel. `posix`, `Linux`, and a path such as `/content` means a remote machine. A local Windows kernel can run the unit tests. It cannot run the A100 experiments unless this PC has a suitable GPU and the GPU packages installed.

## 6. Run the unit tests

From the repository root, with the laptop environment active:

```powershell
pytest -q
```

This is what continuous integration runs on every push (`.github/workflows/ci.yml`), together with `ruff check src scripts tests`.

| File | What it checks |
|---|---|
| `tests/test_components.py` | Prompts are the same for training and inference, exemplar image counts, label parsing, subset sampling, perturbations |
| `tests/test_metrics_stats.py` | Metrics match scikit-learn; bootstrap, DeLong, Holm and McNemar behave on synthetic scores |
| `tests/test_vlm_scoring.py` | Likelihood scoring with and without the prefix cache, on a tiny random Gemma-3 |

The last file needs network access to build that tiny model. If the Hub cannot be reached, that test is skipped. The others should pass offline.

CI then runs a second check that does not need a GPU. It builds fake prediction files, runs the analysis, and confirms the table and figure scripts can read them:

```powershell
python scripts/make_synthetic_results.py --out $env:TEMP\synth
Copy-Item -Recurse results\audit $env:TEMP\synth\audit
python scripts/analyze.py --results $env:TEMP\synth --n-boot 200
python scripts/make_tables.py --results $env:TEMP\synth --out $env:TEMP\synth_tables
python scripts/make_figures.py --results $env:TEMP\synth --out $env:TEMP\synth_figs
```

The fake directory is marked with a file named `SYNTHETIC`. `make_tables.py` and `make_figures.py` refuse to write those numbers into `paper/`. Use this only to see that the pipeline runs. Do not cite it.

A small real training loop, still not a paper result, is:

```powershell
python scripts/run_experiment.py --config configs/smoke_cnn.yaml --limit 48 --final
```

`--limit 48` keeps only the first 48 images of each split and writes under `results/_smoke/`, which is git-ignored. `analyze.py` ignores any run whose `meta.json` has a non-zero `limit`.

## 7. Run the experiments

### 7.1 One experiment from a terminal

Development run, validation split only:

```bash
python scripts/run_experiment.py --config configs/cnn_resnet50.yaml
```

Paper run, validation and test:

```bash
python scripts/run_experiment.py --config configs/cnn_resnet50.yaml --final
```

Useful flags:

| Flag | Effect |
|---|---|
| `--final` | Also score the test split, and append one line to `results/test_access_log.jsonl` |
| `--results DIR` | Write under `DIR` instead of `results/` |
| `--cache DIR` | Embedding cache. The Colab runner uses a folder next to the results folder |
| `--tasks task1` | Run one task instead of both |
| `--only-runs seed0` | Run one seed or one shot setting |
| `--limit N` | Smoke test. Results go to `results/_smoke/` and cannot enter the paper |
| `--overwrite` | Recompute a run that already exists |
| `--device cpu` | Force CPU |

### 7.2 The whole study from Colab

Open `notebooks/colab_runner.ipynb` and run the cells from top to bottom.

1. Select an A100 runtime.
2. Add the `HF_TOKEN` secret before the cell that reads it.
3. The Drive cell is on by default (`USE_DRIVE = True`). It mounts Google Drive and sets the output folder to `/content/drive/MyDrive/cxr_results`. Leave this on. A Colab disk is deleted when the runtime stops; Drive is what keeps the files.
4. Set `FINAL` in the experiments cell.
5. Run the experiments cell, then Grad-CAM, then the analysis cell.

`FINAL = False` still stores every run. Each run gets `predictions.parquet` and `meta.json`, and trained models keep their checkpoints. Only the validation split is scored. The test split is not read, and `test_access_log.jsonl` gains no new line. Grad-CAM and the paper tables that need test-set metrics stay empty for those runs.

`FINAL = True` scores validation and test, writes the same files with `"final": true` inside `meta.json`, and appends the test-set access log. Set this only when you want numbers that can enter the paper. The analysis plan is supposed to be frozen before the first such run.

The Colab notebook currently ships with `FINAL = True` in the experiments cell. Change that line to `False` if you are still developing.

### 7.3 Order

Run experiments in this order. Later jobs reuse checkpoints and the RAD-DINO embedding cache built by earlier ones.

1. `cnn_resnet50`
2. `cnn_resnet50_data_efficiency`
3. `cnn_densenet121`
4. `cnn_xrv_densenet121`
5. `probe_rad_dino` (this also builds the cache used by retrieval ICL)
6. `probe_imagenet_resnet50` (CPU-sized; also listed in the runner)
7. `vlm_gemma3_zeroshot`, `vlm_medgemma_zeroshot`
8. `vlm_gemma3_icl_random`, `vlm_medgemma_icl_random`
9. `vlm_gemma3_icl_knn`, `vlm_medgemma_icl_knn`
10. `vlm_gemma3_lora`, `vlm_medgemma_lora`
11. `vlm_gemma3_lora_data_efficiency`

The Colab runner loops over this list. One config can take many hours because each trained model uses five seeds and each ICL setting uses five exemplar draws at each shot count.

### 7.4 What a config contains

Example, `configs/cnn_resnet50.yaml`: both tasks, ResNet-50, 30 epochs, seeds 0–4, learning rate chosen on a separate sweep seed from `{1e-5, 1e-4, 3e-4}`, and six image perturbations applied only when the test split is scored.

Example, `configs/vlm_gemma3_icl_random.yaml`: Gemma-3, random exemplars, shot counts 2, 4, 6, 8 and 10, seeds 0–4. Those seeds are independent exemplar draws, not training seeds.

Retrieval ICL (`vlm_*_icl_knn`) uses the same shot counts. The five seeds there are five exemplar orderings of a retrieved set, not five new retrievals.

Zero-shot MedGemma uses a single seed because there is no training and no exemplar draw.

### 7.5 Resuming

If `meta.json` already exists for a run, the runner skips it, with two exceptions:

- The existing run was validation-only (`"final": false`) and the new command passes `--final`. The runner scores again. Training is not repeated when `model.pt` or the LoRA adapter is already in the run folder.
- You pass `--overwrite`.

A failed run that never wrote `meta.json` is started again on the next launch. Keep the results folder on Drive, or copy it off the GPU machine before the runtime is deleted.

## 8. What gets stored

### 8.1 One run

```
results/<task>/<experiment>/<run>/
    predictions.parquet
    meta.json
    model.pt            CNN checkpoints only; git-ignored
    train_log.json      CNN and LoRA training history
    adapter/            LoRA weights; git-ignored
```

`<run>` looks like `seed0`, `seed2_frac0.25`, or `seed1_shots6`.

`predictions.parquet` has one row per scored image:

| Column | Meaning |
|---|---|
| `split` | `validation` or `test` |
| `index` | Position of the image in that split |
| `y_true` | 0 for the negative class, 1 for the positive class |
| `logit` | Model score. Higher favours the positive class. For a VLM this is a log-likelihood difference, not a generated word |
| `perturbation` | `none`, or a named corruption. Corruptions are computed on the test split of a `--final` run |
| `logit_cf` | Present for VLMs when a content-free score was computed; used for contextual calibration |

`meta.json` records the task, experiment, seed, shot count, whether `--final` was set, the config, a hash of the config, the git commit, the dataset revision, package versions, the GPU name, and the wall-clock time. CNN runs also record the learning rate chosen on the validation sweep.

### 8.2 Study-level files

| Path | Produced by | Used for |
|---|---|---|
| `results/audit/` | `scripts/audit_dataset.py` | Duplicate and image-statistic tables. Already in the repository |
| `results/power/` | `scripts/power.py` | Sample-size table. Already in the repository |
| `results/legacy/legacy_results.csv` | `scripts/extract_legacy_results.py` | Comparison with the original notebooks |
| `results/test_access_log.jsonl` | every `--final` run | Audit trail: when the test split was read, by which commit |
| `results/summary/run_metrics.csv` | `analyze.py` | One row per run, score variant and decision rule |
| `results/summary/condition_summary.csv` | `analyze.py` | Mean, SD, min and max across seeds |
| `results/summary/selections.json` | `analyze.py` | Which condition won each family on validation AUROC |
| `results/summary/comparisons.csv` | `analyze.py` | Primary comparisons with Holm adjustment, and secondary comparisons |
| `results/summary/robustness.csv` | `analyze.py` | Test AUROC under each perturbation |
| `results/summary/leakage_sensitivity.csv` | `analyze.py` | Test AUROC with near-duplicate training images removed |
| `results/summary/generation_vs_likelihood.csv` | `analyze.py` | Old free-text parsing versus likelihood scoring |
| `results/summary/summary.json` | `analyze.py` | All of the above in one file |
| `results/gradcam/` | `scripts/gradcam.py` | Localisation scores and example panels for CNN `seed0` |

`analyze.py` uses a test-split row only when `meta.json` says `"final": true` and `"limit": 0`. Validation metrics are still computed for final runs, because thresholds and calibration are fit on validation.

### 8.3 Paper files

`scripts/make_tables.py` reads `results/` and writes:

| File | Content |
|---|---|
| `paper/generated/tab_main_task1.tex`, `tab_main_task2.tex` | Main performance tables |
| `paper/generated/tab_icl.tex` | Shot curves |
| `paper/generated/tab_comparisons.tex` | Pre-registered comparisons |
| `paper/generated/tab_efficiency.tex` | Training-set size curves |
| `paper/generated/tab_robustness.tex`, `tab_shortcut.tex` | Perturbations |
| `paper/generated/tab_leakage.tex` | Near-duplicate sensitivity |
| `paper/generated/tab_generation.tex` | Generated text versus likelihood scores |
| `paper/generated/tab_audit.tex`, `tab_power.tex`, `tab_legacy.tex` | Audit, power, legacy notebooks |
| `paper/generated/numbers.tex` | Macros the prose of `main.tex` includes |

If an experiment has not been run, the corresponding table is a placeholder that says the result is pending. That is intentional.

`scripts/make_figures.py` writes, when the matching summary exists:

- `paper/figures/fig_icl_shots.pdf`
- `paper/figures/fig_roc_reliability.pdf`
- `paper/figures/fig_efficiency.pdf`
- `paper/figures/fig_robustness.pdf`
- copies of the audit pair images and any Grad-CAM panels

`paper/main.tex` includes these files. A missing figure is shown as a labelled placeholder rather than as a failed build.

## 9. From a finished GPU run to the manuscript

Do this after the `--final` runs you care about are on disk. If they were written to Google Drive, either pass that folder as `--results`, or copy it onto the repository's `results/` directory first. Copy the prediction folders and `test_access_log.jsonl`. You do not need `model.pt` for the tables. You do need `model.pt` if you still want Grad-CAM.

On the machine that has the results:

```bash
python scripts/analyze.py --results results
python scripts/gradcam.py --task task1 --experiment cnn_resnet50 --run seed0 --results results
python scripts/gradcam.py --task task2 --experiment cnn_resnet50 --run seed0 --results results
python scripts/make_tables.py --results results
python scripts/make_figures.py --results results
```

Repeat Grad-CAM for `cnn_densenet121` and `cnn_xrv_densenet121` if those checkpoints exist. The Colab runner does that loop for you.

Then build the PDF:

```bash
cd paper
latexmk -pdf main.tex
```

`paper/main.pdf` is git-ignored. The source of the numbers is `paper/generated/` and `results/summary/`. Commit those, not a hand-edited table.

CI checks that `paper/generated/` matches a fresh run of `make_tables.py` on the committed `results/`. If you change results and forget to regenerate the tables, that check fails.

### What you can quote

Quote a test-set number only when all of the following are true:

- The run's `meta.json` has `"final": true` and `"limit": 0`.
- The row appears in `results/summary/run_metrics.csv` with `split` equal to `test`.
- The table or macro in `paper/generated/` was produced by `make_tables.py` after that summary existed.
- The result directory is not the one created by `make_synthetic_results.py`.

Validation numbers are real and are stored, but they are for choosing learning rates, checkpoints, shot counts and ensemble weights. The primary endpoint in the analysis plan is test-set AUROC.

## 10. Work that is already done

These commands have already been run and their outputs are in the repository. Re-run them only if you change the audit or the power calculation.

```bash
python scripts/audit_dataset.py      # results/audit/
python scripts/power.py              # results/power/
python scripts/extract_legacy_results.py
```

The audit needs the datasets from the Hub. The power script does not; it uses the known test-set size of 456 images per task.

## 11. A practical sequence

1. On the laptop, create the environment and run `pytest -q`.
2. On Colab, set the A100 runtime and the `HF_TOKEN` secret. Accept the MedGemma terms while logged into the same Hugging Face account.
3. Leave `USE_DRIVE = True`.
4. For a first look, set `FINAL = False` and run one config, for example `cnn_resnet50`. Confirm that `predictions.parquet` and `meta.json` appear under the Drive folder.
5. When the plan is frozen and you want paper numbers, set `FINAL = True` and run the full list. A later `--final` pass reuses checkpoints from the validation pass.
6. Run the analysis cell in the Colab notebook, or the commands in section 9 on a copy of the Drive folder.
7. Build `paper/main.tex`.
8. Commit `results/` (without checkpoints), `paper/generated/`, and `paper/figures/`.

## 12. Checks before you trust a folder

Open `meta.json` and confirm:

- `final` is true for any test-set claim.
- `limit` is 0.
- `dataset_revision` matches `src/cxr/tasks.py`.
- `git_commit` is the commit you think you ran. A suffix `-dirty` means the working tree had uncommitted edits.
- `config_hash` matches the other seeds of the same experiment.

Open `results/test_access_log.jsonl`. Each line is one test-set evaluation, with the UTC time and the git commit. The analysis plan is meant to predate the first line. Compare `git log docs/analysis_plan.md` with that file if you need to show the order.

## 13. Troubleshooting

**The Colab cell says the token is missing.** The secret's name is not exactly `HF_TOKEN`, or notebook access for that secret is off.

**MedGemma returns 403 or asks you to accept a licence.** The token's account has not accepted the model terms, or the token is from a different account.

**`Saved predictions` shows a Windows path, but the file is not on the PC.** The kernel is Linux. The file is on that machine. With the Colab runner, look in Google Drive under `MyDrive/cxr_results` when `USE_DRIVE = True`.

**A re-run prints `skip ... (exists)`.** `meta.json` is already there. Use `--overwrite` to recompute it, or delete that run folder.

**Tables still say "pending".** `analyze.py` did not see a final, unlimited run for that experiment, or `make_tables.py` was pointed at a different `--results` folder than the one that contains the runs.

**Grad-CAM exits immediately.** The CNN run was not a `--final` run, or `model.pt` is missing from that run folder.

**Figures or tables look complete but the numbers are not from the study.** The results folder contains a file named `SYNTHETIC`. Those outputs must not be copied into `paper/`.
