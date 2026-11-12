# Reporting checklist

This maps the items of CLAIM 2024 (Tejani et al., *Radiol Artif Intell* 2024;6:e240300) and
TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378) to where they are addressed in
`paper/main.tex` or the repository. Items are paraphrased and grouped; consult the original
checklists for their exact wording. "N/A" items are explained. "Partial" marks items the data do
not allow us to satisfy fully; each is also listed under Limitations in the paper.

## Title and abstract

| Item | Status | Where |
|---|---|---|
| Identify the study as developing/evaluating AI models, state the task, data and model types | Done | Title, Abstract |
| Structured abstract with background, methods, results, conclusions | Done | Abstract |

## Introduction

| Item | Status | Where |
|---|---|---|
| Scientific and clinical background, intended use | Done | Introduction; Limitations (not for clinical use) |
| Study objectives and hypotheses | Done | Introduction (contributions); `docs/analysis_plan.md` section 1 (two-sided hypotheses) |

## Methods: study design and data

| Item | Status | Where |
|---|---|---|
| Study design (retrospective, model comparison) and pre-registration | Done | Methods "Statistical analysis"; `docs/analysis_plan.md` frozen before test access; `results/test_access_log.jsonl` |
| Data source, collection period, sites | Done | Data "Source" (POLCOVID, 15 Polish hospitals, 2020-2021) |
| Eligibility criteria / how cases were selected | Partial | Data "Limitations of the derivation": the selection of 1,140 images per class from POLCOVID is undocumented |
| Data pre-processing, de-identification | Done | Data; Methods (pad-to-square, resize, normalisation); images are de-identified at source |
| Data partitions, how assigned, level of partitioning | Partial | Data: published 70/10/20 splits; patient-level partitioning impossible (no IDs); audit in Results |
| Duplicates / leakage assessment | Done | Methods "Dataset audit"; Results; `scripts/audit_dataset.py`; `results/audit/` |
| Missing data | Done | No missing images or labels; all splits complete and balanced (Data) |
| Reference standard: definition, rationale, annotators | Partial | Labels from the source database; no reports or RT-PCR available (Limitations) |
| Class balance / prevalence | Done | Data (exactly balanced); Limitations (does not reflect clinical prevalence) |

## Methods: models

| Item | Status | Where |
|---|---|---|
| Detailed model description, inputs, outputs, architecture | Done | Methods "Supervised baselines", "Vision-language models" |
| Software, versions, pretrained weights and their sources | Done | `requirements*.txt`; pinned model and dataset revisions in `configs/` and `src/cxr/tasks.py`; per-run environment in `meta.json` |
| Initialisation, training procedure, hyperparameters and how they were chosen | Done | Methods; `configs/*.yaml`; `hparam_selection.json` per CNN |
| Model selection (checkpoints, k, ensemble weights) on validation only | Done | Methods "Selection"; `scripts/analyze.py` selections written to `results/summary/selections.json` |
| Ensembling | Done | Methods "Ensemble" |
| Prompts for generative models | Done | Methods "Prompt"; `src/cxr/prompts.py` |

## Methods: evaluation

| Item | Status | Where |
|---|---|---|
| Performance metrics and their justification | Done | Methods "Endpoints" (primary AUROC; secondary threshold and calibration metrics) |
| Statistical methods, uncertainty, multiplicity | Done | Methods "Inference"; Holm over 10 primary tests; paired bootstrap; DeLong; McNemar; Wilson |
| Sample size justification | Done | Methods "Sample size"; Table "power"; `scripts/power.py` |
| Robustness / sensitivity analyses | Done | Methods "Robustness and shortcuts"; leakage sensitivity |
| Explainability / interpretability | Done | Grad-CAM centre mass for CNNs; perturbation analysis |
| External validation | Not possible | Single source database; no external test set (Limitations) |
| Fairness / subgroup analyses | Not possible | No demographic or site metadata in the released datasets (Limitations) |
| Comparison with human readers | N/A | Not part of the study design |

## Results

| Item | Status | Where |
|---|---|---|
| Flow of data, numbers per split and class | Done | Data; Table "audit" |
| Baseline characteristics of the data | Partial | Per-class image statistics in `results/audit/class_summary.csv`; no patient demographics available |
| Performance with confidence intervals for all models, including negative/null results | Done | Main tables, comparisons table, `results/summary/run_metrics.csv` (every run) |
| Calibration | Done | ECE in main tables; reliability diagrams |
| Failure analysis | Done | ICL label collapse (positive-prediction rate); unparseable generations; robustness |

## Discussion and other information

| Item | Status | Where |
|---|---|---|
| Limitations, including bias and generalisability | Done | Limitations; Discussion |
| Implications for practice | Done | Limitations ("None of the models is suitable for clinical use") |
| Registration / protocol availability | Done | `docs/analysis_plan.md` and `configs/analysis_plan.yaml`, with commit history and deviations table |
| Code and data availability | Done | Ethics and data statement; repository; `README.md` |
| Funding, conflicts of interest | To complete | To be added by the authors at submission |
| Ethics approval | Done | Ethics and data statement (public de-identified data; approval described in the source publication) |
| Patient and public involvement | N/A | Secondary analysis of a public database |
