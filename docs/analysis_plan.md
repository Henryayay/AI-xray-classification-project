# Statistical analysis plan

Status: frozen before any `--final` (test-set) run. The machine-readable version is
[`configs/analysis_plan.yaml`](../configs/analysis_plan.yaml); where the two differ, the YAML is
authoritative because it is what `scripts/analyze.py` executes. Every change after freezing is
listed under "Deviations" with date, reason and whether it was made before or after test results
were seen. `results/test_access_log.jsonl` records every evaluation that touched the test split,
with the git commit used, so the order of plan and test evaluations can be audited.

## 1. Questions and hypotheses

On two binary chest-radiograph tasks from POLCOVID (T1: Normal vs. Pneumonia; T2: non-COVID
pneumonia vs. COVID-19):

- **Q1 (adaptation).** How do general-purpose (Gemma-3-4B-it) and medical (MedGemma-4B-it)
  vision-language models compare under zero-shot prompting, in-context learning (ICL) and LoRA
  fine-tuning with supervised image models (CNNs, frozen-encoder linear probes)?
- **Q2 (domain pretraining).** Does medical pretraining (MedGemma vs. Gemma-3, same architecture)
  improve discrimination zero-shot and after LoRA?
- **Q3 (exemplar selection).** Does retrieval of visually similar, class-balanced exemplars
  improve ICL over random exemplars?
- **Q4 (calibration of label bias).** Does correcting the label prior estimated on validation fix
  ICL's collapse onto one label (seen in the preliminary T2 runs)?

All hypotheses are two-sided; no direction is assumed.

## 2. Data

Published splits `Henryayay/POLCOVID_task{1,2}_70_10_20` at pinned revisions (`src/cxr/tasks.py`);
1,596 / 228 / 456 images (train / validation / test), exactly balanced. No patient identifiers
are available. The pixel-level audit (`scripts/audit_dataset.py`) was run before any model
evaluation; its results are reported regardless of outcome, and a leakage sensitivity analysis
(Section 6) is pre-specified.

## 3. Endpoints

- **Primary:** AUROC of the positive class (Pneumonia in T1, COVID-19 in T2) on the test set.
  AUROC is threshold-free and therefore unaffected by the decision-rule choices below.
- **Secondary:** balanced accuracy, sensitivity, specificity, F1, MCC, AUPRC, Brier score, ECE
  (10 equal-width bins; raw and after validation temperature scaling), positive-prediction rate.
- **Decision rules** for threshold metrics: (a) argmax, `logit >= 0`; (b) threshold maximising
  Youden's J on validation. Main tables use (b); calibration comparisons use (a), because (b)
  would itself absorb a constant label bias.

## 4. Conditions, seeds and selection

- Experiments are those in `configs/` (`cnn_*`, `probe_*`, `vlm_*`). Supervised and LoRA
  conditions use seeds 0-4; ICL uses 5 independent exemplar draws (random) or 5 exemplar
  orderings (retrieval) per k in {2, 4, 6, 8, 10}.
- Hyperparameters: CNN learning rate chosen from {1e-5, 1e-4, 3e-4} by minimum validation loss
  with a separate sweep seed; checkpoint chosen by minimum validation loss; linear-probe L2
  strength by validation NLL; LoRA settings fixed to the original recipe; ensemble weight by
  validation AUROC. No choice uses the test split.
- **Family-best selection:** where a comparison involves "the best" condition of a family, the
  condition with the highest mean validation AUROC (raw scores, full training data) is chosen;
  ties are broken alphabetically and then by smaller k.

## 5. Comparisons and inference

Primary comparisons (each in both tasks, so 10 tests; Holm-Bonferroni adjustment over all 10 at
family-wise alpha = 0.05):

| ID | A | B | Metric |
|---|---|---|---|
| best_vlm_vs_best_supervised | best of all VLM conditions | best supervised condition | AUROC |
| medgemma_vs_gemma3_lora | MedGemma LoRA | Gemma-3 LoRA | AUROC |
| medgemma_vs_gemma3_zeroshot | MedGemma zero-shot | Gemma-3 zero-shot | AUROC |
| icl_knn_vs_random_gemma3 | Gemma-3 ICL retrieval (best k) | Gemma-3 ICL random (best k) | AUROC |
| calibration_valcentered_vs_raw_gemma3_icl_random | val-centred scores | raw scores, same condition | balanced accuracy (argmax) |

Secondary (exploratory, unadjusted): the same ICL comparison for MedGemma, contextual
calibration vs. raw, val-centring for zero-shot, Gemma-3 LoRA vs. best supervised, and the
validation-weighted VLM + supervised ensemble against each component.

**Test statistic.** For conditions with several seeds/draws, the estimand is the difference in
the *seed-averaged* metric. Its 95% CI and two-sided p-value come from a paired,
class-stratified bootstrap over test images (10,000 replicates, seed 20261011): each replicate
resamples images once and recomputes every seed of both conditions on that resample. The p-value
is 2 x min(P(d <= 0), P(d >= 0)) with add-one correction. As supporting analyses we report
DeLong's test on seed-averaged scores and an exact McNemar test on the first seed of each
condition. Per-condition uncertainty: DeLong 95% CI for each run's AUROC, Wilson CI for
proportions, and mean +/- SD (with per-seed values in `run_metrics.csv`) across seeds.

## 6. Pre-specified sensitivity and descriptive analyses

- Leakage sensitivity: test AUROC recomputed after excluding test images with a highly similar
  (r >= 0.95) training image.
- Robustness: test AUROC under six fixed perturbations (border mask, centre/lung mask, two gamma
  changes, Gaussian noise, JPEG q=10). A centre-masked AUROC far above 0.5 is interpreted as
  evidence of non-pulmonary cues.
- Grad-CAM centre-mass for CNN checkpoints.
- Data efficiency: 10/25/50/100% class-stratified training subsets.
- Original protocol (generation + string matching) vs. likelihood scoring on the same images.

## 7. Sample size

The test set is fixed by the published split (n = 456). `scripts/power.py` reports what it can
resolve: e.g. at accuracy 0.85 the 95% CI half-width is about 3.3 points; at AUROC 0.85 and
inter-model correlation 0.5 the minimum difference detectable with 80% power is about 0.05
(0.066 at the Holm worst case). Comparisons whose observed difference is below the relevant MDE
are described as inconclusive rather than as evidence of equivalence.

## 8. Reporting rules

- All conditions that were run are reported, including failed or null results. Runs that crash
  are re-run with the same seed; persistent failures are reported as missing, not dropped.
- No result is excluded post hoc. Tables and figures are generated only by
  `scripts/make_tables.py` / `scripts/make_figures.py` from `results/`.
- Legacy notebook results (`legacy/`) are reported as preliminary only.

## Deviations

| Date | Change | Before/after test results | Reason |
|---|---|---|---|
| (none yet) | | | |
