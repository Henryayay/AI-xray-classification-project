# Vision-language models vs. supervised CNNs for chest X-ray classification (POLCOVID)

Code, data audit, pre-registered analysis plan and manuscript for a controlled comparison of
general-purpose (Gemma-3-4B-it) and medical (MedGemma-4B-it) vision-language models under
zero-shot prompting, in-context learning (ICL) and LoRA fine-tuning, against CNN and
frozen-encoder baselines, on two binary tasks:

| Task | Negative class | Positive class | Dataset (pinned revision in `src/cxr/tasks.py`) |
|---|---|---|---|
| `task1` | Normal | Pneumonia | [`Henryayay/POLCOVID_task1_70_10_20`](https://huggingface.co/datasets/Henryayay/POLCOVID_task1_70_10_20) |
| `task2` | Non-COVID Pneumonia | COVID-19 | [`Henryayay/POLCOVID_task2_70_10_20`](https://huggingface.co/datasets/Henryayay/POLCOVID_task2_70_10_20) |

Each task has 1,596 / 228 / 456 balanced train / validation / test images.

## What is in the repository

```
src/cxr/            library: data, prompts, VLM likelihood scoring + LoRA, CNNs, encoders,
                    ICL exemplar selection, calibration, metrics, statistics, audit, perturbations
configs/            one YAML per experiment, plus the machine-readable analysis plan
scripts/            run_experiment, analyze, audit_dataset, power, gradcam,
                    make_tables, make_figures, extract_legacy_results, make_synthetic_results
docs/analysis_plan.md   pre-registered statistical analysis plan (frozen before test runs)
results/            committed outputs: audit, power analysis, legacy numbers, experiment
                    predictions (per image), summaries; test_access_log.jsonl
paper/              LaTeX manuscript; paper/generated and paper/figures are produced by scripts
notebooks/colab_runner.ipynb   thin Colab driver for the GPU experiments
legacy/             the original notebooks, unchanged (preliminary results only)
tests/              CPU unit tests (run in CI)
```

## Method changes relative to the original notebooks

- **Likelihood scoring instead of free-text parsing.** Each class name (plus end-of-turn token)
  is scored as the assistant continuation; the positive-class logit is the log-likelihood
  difference. This yields a score for every image (AUROC, calibration), removes parse failures
  and substring bugs, and is exactly what LoRA training optimises.
- **One prompt** (`src/cxr/prompts.py`) for LoRA training and every evaluation (the original
  fine-tuning notebooks evaluated with a different prompt than they trained with).
- **Single BOS token.** The original few-shot notebooks tokenised the chat template with special
  tokens enabled, producing a duplicated `<bos>`.
- **Seeds and draws.** 5 seeds for every trained model; 5 independent exemplar draws per ICL
  setting (the original used one draw, so its shot curve conflated k with exemplar choice).
- **Validation-only model selection** and a logged, opt-in test evaluation (`--final`).
- **New strategies:** retrieval-augmented ICL (RAD-DINO nearest neighbours per class),
  contextual and validation-centred calibration, MedGemma vs. Gemma-3 under identical
  pipelines, CXR-pretrained and foundation-model baselines, data-efficiency curves, a
  validation-weighted VLM + supervised ensemble, robustness/shortcut perturbations, Grad-CAM.
- **Statistics:** DeLong and bootstrap CIs, paired bootstrap of seed-averaged differences,
  exact McNemar, Holm correction over the pre-registered primary family, and a power analysis
  for the fixed test set.

## Reproducing

### 1. CPU parts (audit, power, analysis, paper)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                   # unit tests
python scripts/audit_dataset.py             # -> results/audit/
python scripts/power.py                     # -> results/power/
python scripts/extract_legacy_results.py    # -> results/legacy/
python scripts/run_experiment.py --config configs/probe_imagenet_resnet50.yaml --final
```

### 2. GPU experiments (A100 recommended)

Open `notebooks/colab_runner.ipynb` in Colab (A100 runtime), or on any CUDA machine:

```bash
pip install -r requirements.txt -r requirements-gpu.txt
# Development: validation split only, no test access
python scripts/run_experiment.py --config configs/cnn_resnet50.yaml
# Final: also scores the test split and appends to results/test_access_log.jsonl
python scripts/run_experiment.py --config configs/cnn_resnet50.yaml --final
```

Recommended order (later configs reuse earlier choices/caches):
`cnn_resnet50` -> `cnn_resnet50_data_efficiency` -> `cnn_densenet121` -> `cnn_xrv_densenet121`
-> `probe_rad_dino` (also builds the RAD-DINO cache used for retrieval ICL) -> `vlm_*_zeroshot`
-> `vlm_*_icl_random` -> `vlm_*_icl_knn` -> `vlm_*_lora` -> `vlm_gemma3_lora_data_efficiency`.
Runs are resumable: finished runs are skipped, and model checkpoints are kept locally (they are
git-ignored), so a `--final` pass after a validation pass does not retrain.

MedGemma is distributed under the Health AI Developer Foundations terms; accept them on the
model page before use. Use `HF_TOKEN` from Colab secrets; nothing is pushed to the Hub.

### 3. Analysis and manuscript

```bash
python scripts/analyze.py          # -> results/summary/
python scripts/gradcam.py --task task1 --experiment cnn_resnet50 --run seed0
python scripts/make_tables.py      # -> paper/generated/*.tex
python scripts/make_figures.py     # -> paper/figures/
cd paper && latexmk -pdf main.tex
```

Tables and figures whose experiments have not been run render as "pending" placeholders; no
number in the manuscript is typed by hand. `scripts/make_synthetic_results.py` produces clearly
marked synthetic predictions for testing the pipeline; the table and figure scripts refuse to
write them into `paper/`.

## Integrity notes

- The analysis plan was committed before the first `--final` run; compare commit times in
  `git log docs/analysis_plan.md` with `results/test_access_log.jsonl`.
- All conditions that were run are reported, including negative results.
- The dataset carries no patient identifiers; see the audit and its limitations in the paper.
- This is research code, not a medical device.

## Citation

See `CITATION.cff`.
