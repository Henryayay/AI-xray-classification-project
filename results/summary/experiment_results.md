# Linear-probe experiment: results

This file summarises the runs that are saved in `results/task1` and `results/task2`. Both models are linear classifiers trained on frozen image features. Each number was recomputed from `predictions.parquet` with the same functions as `scripts/analyze.py` (`src/cxr/metrics.py`, `src/cxr/calibration.py`, `src/cxr/stats.py`) and checked against `results/summary/` before this file was written. 64 metric rows matched.

Only seed 0 was run, so there is no standard deviation across seeds. The interval next to each AUROC is the DeLong 95% confidence interval for that one fitted model. CNN, vision-language, and LoRA runs are not in this file because those predictions have not been saved.

## How the numbers are defined

- **Positive class.** Task 1: Pneumonia (negative: Normal). Task 2: COVID-19 (negative: non-COVID pneumonia). Each test set has 228 images of each class (456 in total). Each validation set has 114 of each class (228 in total).
- **AUROC** ranks images by the model's score. It does not use a cutoff. 0.5 is chance on a balanced set. 1 is a perfect ranking. **AUPRC** is the area under the precision-recall curve for the positive class.
- **Cutoff for the other metrics.** Two rules are reported. **Default** calls an image positive when the logit is at least 0. **Youden** uses the logit cutoff that maximised sensitivity + specificity − 1 on the validation set, then applies that cutoff to the test set. The validation images were not used to compute the test metrics.
- **Sensitivity** = true positives / 228 positive test images. **Specificity** = true negatives / 228 negative test images. **Precision** = true positives / images called positive. **Accuracy** = (true positives + true negatives) / 456. Because the two classes are the same size, accuracy and balanced accuracy are the same value once they are shown to three decimal places.
- **Brier score** is the mean squared error of the predicted probability (lower is better). **ECE** is the 10-bin expected calibration error (lower is better). The temperature-scaled ECE uses a temperature fitted on the validation set only.
- Displayed proportions are rounded half-up to three decimal places. Small differences are shown to four decimal places. The unrounded values are in `results/summary/run_metrics.csv`.

## 1. Runs

| Task | Model | Training fraction | Training images | Seed | L2 strength C | Validation NLL |
| --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 10% | 160 | 0 | 0.01 | 0.461 |
| Task 1 | RAD-DINO linear probe | 25% | 400 | 0 | 0.01 | 0.381 |
| Task 1 | RAD-DINO linear probe | 50% | 798 | 0 | 0.01 | 0.346 |
| Task 1 | RAD-DINO linear probe | 100% | 1596 | 0 | 0.01 | 0.338 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 10% | 160 | 0 | 0.01 | 0.520 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 25% | 400 | 0 | 0.01 | 0.470 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 50% | 798 | 0 | 0.01 | 0.433 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 100% | 1596 | 0 | 0.01 | 0.432 |
| Task 2 | RAD-DINO linear probe | 10% | 160 | 0 | 0.01 | 0.505 |
| Task 2 | RAD-DINO linear probe | 25% | 400 | 0 | 0.01 | 0.452 |
| Task 2 | RAD-DINO linear probe | 50% | 798 | 0 | 0.01 | 0.432 |
| Task 2 | RAD-DINO linear probe | 100% | 1596 | 0 | 0.01 | 0.394 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 10% | 160 | 0 | 0.001 | 0.632 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 25% | 400 | 0 | 0.001 | 0.571 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 50% | 798 | 0 | 0.001 | 0.548 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 100% | 1596 | 0 | 0.001 | 0.524 |

C is the inverse L2 penalty chosen by validation negative log-likelihood. A smaller C means stronger regularisation. Validation NLL is the score that was minimised to choose C (lower is better). It is not a test-set result.

## 2. Test set, all 1,596 training images

### Ranking (no cutoff)

| Task | Model | Test images | AUROC | AUROC 95% CI | AUPRC |
| --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 456 | 0.909 | 0.883 to 0.935 | 0.910 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 456 | 0.824 | 0.785 to 0.862 | 0.837 |
| Task 2 | RAD-DINO linear probe | 456 | 0.893 | 0.864 to 0.922 | 0.884 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 456 | 0.759 | 0.715 to 0.802 | 0.752 |

AUROC does not depend on the cutoff, so the default-cutoff rows have the same AUROC.

### Youden cutoff (chosen on validation, scored on the test set)

| Task | Model | Logit cutoff | TP | FN | TN | FP | Correct / 456 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | -0.2923 | 191 | 37 | 190 | 38 | 381/456 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 0.1101 | 163 | 65 | 191 | 37 | 354/456 |
| Task 2 | RAD-DINO linear probe | 0.0373 | 186 | 42 | 183 | 45 | 369/456 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 0.0578 | 152 | 76 | 166 | 62 | 318/456 |

TP: positive images called positive. FN: positive images called negative. TN: negative images called negative. FP: negative images called positive. TP + FN = 228 and TN + FP = 228 in every row.

| Task | Model | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | Called positive | Accuracy 95% CI (Wilson) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 0.836 | 0.838 | 0.833 | 0.834 | 0.836 | 0.671 | 0.502 | 0.799 to 0.867 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 0.776 | 0.715 | 0.838 | 0.815 | 0.762 | 0.557 | 0.439 | 0.736 to 0.812 |
| Task 2 | RAD-DINO linear probe | 0.809 | 0.816 | 0.803 | 0.805 | 0.810 | 0.618 | 0.507 | 0.771 to 0.843 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 0.697 | 0.667 | 0.728 | 0.710 | 0.688 | 0.395 | 0.469 | 0.654 to 0.738 |

These rates are the counts above divided by the matching denominator. Sensitivity = TP/228. Specificity = TN/228. Precision = TP/(TP+FP). Accuracy = (TP+TN)/456. The accuracy interval is a Wilson 95% interval.

| Task | Model | Brier | ECE | ECE after temperature scaling | Temperature |
| --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 0.121 | 0.038 | 0.033 | 1.069 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 0.174 | 0.086 | 0.076 | 1.208 |
| Task 2 | RAD-DINO linear probe | 0.133 | 0.058 | 0.056 | 1.040 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 0.200 | 0.035 | 0.056 | 0.777 |

### Default cutoff (logit at least 0)

| Task | Model | TP | FN | TN | FP | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 185 | 43 | 197 | 31 | 0.838 | 0.811 | 0.864 | 0.856 | 0.833 | 0.676 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 165 | 63 | 185 | 43 | 0.768 | 0.724 | 0.811 | 0.793 | 0.757 | 0.537 |
| Task 2 | RAD-DINO linear probe | 187 | 41 | 180 | 48 | 0.805 | 0.820 | 0.789 | 0.796 | 0.808 | 0.610 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 157 | 71 | 159 | 69 | 0.693 | 0.689 | 0.697 | 0.695 | 0.692 | 0.386 |

## 3. Difference between the two probes (full training set)

RAD-DINO minus the ImageNet ResNet-50 probe, on the same 456 test images. This comparison is descriptive. It is outside the pre-registered primary tests in `configs/analysis_plan.yaml`, which also need vision-language runs that have not been saved. The functions are the ones in that plan: DeLong's test for paired AUROCs, a paired class-stratified bootstrap with 10,000 replicates and seed 20261011, and an exact McNemar test on the Youden decisions. With one seed, the interval describes these two fitted models.

Both bootstrap p-values are 2/10,001 (about 0.000200). That is the smallest p-value this bootstrap can give, and it means every one of the 10,000 resamples had a positive AUROC difference.

| Task | AUROC difference | Bootstrap 95% CI | Bootstrap p | DeLong z | DeLong p |
| --- | --- | --- | --- | --- | --- |
| Task 1 | 0.0854 | 0.050 to 0.122 | 2.00e-04 | 4.5367 | 5.71e-06 |
| Task 2 | 0.1344 | 0.095 to 0.175 | 2.00e-04 | 6.5930 | 4.31e-11 |

| Task | Only RAD-DINO correct | Only ImageNet probe correct | McNemar p |
| --- | --- | --- | --- |
| Task 1 | 63 | 36 | 0.009 |
| Task 2 | 82 | 31 | 1.73e-06 |

McNemar uses the Youden cutoff of each model. "Only RAD-DINO correct" counts test images the RAD-DINO probe classified correctly and the ImageNet probe classified incorrectly.

## 4. Test set with less training data

Each subset keeps the same fraction of each training class, with the per-class count rounded to the nearest integer (`src/cxr/data.py`). That is why 10% is 160 images (80 per class) and 25% is 400 images (200 per class), while 50% is 798 and 100% is 1,596. The test set is the same 456 images in every row. AUROC does not use a cutoff. The other columns use the Youden cutoff from that run's own validation scores.

| Task | Model | Training images | AUROC | AUROC 95% CI | AUPRC | Accuracy | Sensitivity (TP/228) | Specificity (TN/228) | F1 | MCC | Brier | ECE (temperature-scaled) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 160 | 0.835 | 0.798 to 0.871 | 0.843 | 0.768 | 160/228 | 190/228 | 0.751 | 0.540 | 0.167 | 0.051 |
| Task 1 | RAD-DINO linear probe | 400 | 0.907 | 0.881 to 0.934 | 0.914 | 0.809 | 170/228 | 199/228 | 0.796 | 0.623 | 0.125 | 0.045 |
| Task 1 | RAD-DINO linear probe | 798 | 0.908 | 0.881 to 0.934 | 0.911 | 0.831 | 182/228 | 197/228 | 0.825 | 0.664 | 0.122 | 0.025 |
| Task 1 | RAD-DINO linear probe | 1596 | 0.909 | 0.883 to 0.935 | 0.910 | 0.836 | 191/228 | 190/228 | 0.836 | 0.671 | 0.121 | 0.033 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 160 | 0.764 | 0.720 to 0.807 | 0.758 | 0.697 | 130/228 | 188/228 | 0.653 | 0.408 | 0.200 | 0.070 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 400 | 0.851 | 0.816 to 0.886 | 0.849 | 0.761 | 153/228 | 194/228 | 0.737 | 0.531 | 0.158 | 0.040 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 798 | 0.840 | 0.804 to 0.877 | 0.838 | 0.724 | 126/228 | 204/228 | 0.667 | 0.476 | 0.164 | 0.060 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 1596 | 0.824 | 0.785 to 0.862 | 0.837 | 0.776 | 163/228 | 191/228 | 0.762 | 0.557 | 0.174 | 0.076 |
| Task 2 | RAD-DINO linear probe | 160 | 0.824 | 0.786 to 0.862 | 0.820 | 0.746 | 149/228 | 191/228 | 0.720 | 0.500 | 0.171 | 0.029 |
| Task 2 | RAD-DINO linear probe | 400 | 0.875 | 0.843 to 0.908 | 0.866 | 0.811 | 190/228 | 180/228 | 0.815 | 0.623 | 0.141 | 0.073 |
| Task 2 | RAD-DINO linear probe | 798 | 0.888 | 0.859 to 0.918 | 0.884 | 0.792 | 177/228 | 184/228 | 0.788 | 0.584 | 0.137 | 0.048 |
| Task 2 | RAD-DINO linear probe | 1596 | 0.893 | 0.864 to 0.922 | 0.884 | 0.809 | 186/228 | 183/228 | 0.810 | 0.618 | 0.133 | 0.056 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 160 | 0.688 | 0.640 to 0.737 | 0.661 | 0.643 | 155/228 | 138/228 | 0.655 | 0.286 | 0.225 | 0.035 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 400 | 0.740 | 0.695 to 0.786 | 0.728 | 0.686 | 159/228 | 154/228 | 0.690 | 0.373 | 0.207 | 0.059 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 798 | 0.740 | 0.695 to 0.785 | 0.730 | 0.660 | 165/228 | 136/228 | 0.680 | 0.323 | 0.207 | 0.057 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 1596 | 0.759 | 0.715 to 0.802 | 0.752 | 0.697 | 152/228 | 166/228 | 0.688 | 0.395 | 0.200 | 0.056 |

### Default cutoff at each training size

| Task | Model | Training images | Accuracy | Sensitivity (TP/228) | Specificity (TN/228) | Precision | F1 | Called positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 160 | 0.765 | 166/228 | 183/228 | 0.787 | 0.756 | 0.463 |
| Task 1 | RAD-DINO linear probe | 400 | 0.818 | 175/228 | 198/228 | 0.854 | 0.808 | 0.450 |
| Task 1 | RAD-DINO linear probe | 798 | 0.831 | 182/228 | 197/228 | 0.854 | 0.825 | 0.467 |
| Task 1 | RAD-DINO linear probe | 1596 | 0.838 | 185/228 | 197/228 | 0.856 | 0.833 | 0.474 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 160 | 0.684 | 144/228 | 168/228 | 0.706 | 0.667 | 0.447 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 400 | 0.781 | 171/228 | 185/228 | 0.799 | 0.774 | 0.469 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 798 | 0.789 | 173/228 | 187/228 | 0.808 | 0.783 | 0.469 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 1596 | 0.768 | 165/228 | 185/228 | 0.793 | 0.757 | 0.456 |
| Task 2 | RAD-DINO linear probe | 160 | 0.761 | 165/228 | 182/228 | 0.782 | 0.752 | 0.463 |
| Task 2 | RAD-DINO linear probe | 400 | 0.820 | 189/228 | 185/228 | 0.815 | 0.822 | 0.509 |
| Task 2 | RAD-DINO linear probe | 798 | 0.789 | 177/228 | 183/228 | 0.797 | 0.787 | 0.487 |
| Task 2 | RAD-DINO linear probe | 1596 | 0.805 | 187/228 | 180/228 | 0.796 | 0.808 | 0.515 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 160 | 0.651 | 144/228 | 153/228 | 0.658 | 0.644 | 0.480 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 400 | 0.680 | 150/228 | 160/228 | 0.688 | 0.673 | 0.478 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 798 | 0.689 | 151/228 | 163/228 | 0.699 | 0.680 | 0.474 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 1596 | 0.693 | 157/228 | 159/228 | 0.695 | 0.692 | 0.496 |

## 5. Validation set (full training data)

These numbers were used to choose C and the Youden cutoff. They are higher than the test numbers and are not the result of the experiment.

| Task | Model | Validation images | AUROC | AUROC 95% CI | Accuracy | Sensitivity | Specificity | Youden logit cutoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 228 | 0.926 | 0.889 to 0.964 | 0.890 | 0.904 | 0.877 | -0.2923 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 228 | 0.883 | 0.839 to 0.927 | 0.825 | 0.833 | 0.816 | 0.1101 |
| Task 2 | RAD-DINO linear probe | 228 | 0.907 | 0.868 to 0.947 | 0.851 | 0.868 | 0.833 | 0.0373 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 228 | 0.822 | 0.767 to 0.876 | 0.759 | 0.781 | 0.737 | 0.0578 |

## 6. Test-time perturbations (full training data)

Each cell is the test AUROC after changing the test images and leaving the trained probe unchanged. Clean is the unperturbed AUROC from section 2. Drop is clean minus perturbed (positive means the score fell).

The border mask blacks out the outer 12.5% of width and height. The centre mask blacks out the central 70% of width and height (rows and columns from 15% to 85%). Gamma 0.6 and 1.6 change contrast. Noise adds Gaussian noise with standard deviation 0.05 × 255. JPEG uses quality 10.

| Task | Model | Clean | Border hidden (drop) | Centre hidden (drop) | Gamma 0.6 (drop) | Gamma 1.6 (drop) | Gaussian noise (drop) | JPEG quality 10 (drop) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 0.909 | 0.896 (0.014) | 0.872 (0.037) | 0.913 (-0.004) | 0.875 (0.034) | 0.838 (0.071) | 0.800 (0.109) |
| Task 1 | ResNet-50 (ImageNet) linear probe | 0.824 | 0.754 (0.070) | 0.648 (0.175) | 0.822 (0.002) | 0.809 (0.015) | 0.732 (0.092) | 0.681 (0.143) |
| Task 2 | RAD-DINO linear probe | 0.893 | 0.882 (0.011) | 0.791 (0.102) | 0.872 (0.021) | 0.883 (0.009) | 0.867 (0.026) | 0.847 (0.046) |
| Task 2 | ResNet-50 (ImageNet) linear probe | 0.759 | 0.625 (0.134) | 0.560 (0.198) | 0.754 (0.005) | 0.760 (-0.001) | 0.710 (0.048) | 0.656 (0.103) |

### Perturbed AUROC at every training size

| Task | Model | Training images | Clean | Border hidden | Centre hidden | Gamma 0.6 | Gamma 1.6 | Gaussian noise | JPEG quality 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 160 | 0.835 | 0.830 | 0.830 | 0.842 | 0.809 | 0.750 | 0.708 |
| Task 1 | RAD-DINO linear probe | 400 | 0.907 | 0.897 | 0.872 | 0.902 | 0.879 | 0.844 | 0.803 |
| Task 1 | RAD-DINO linear probe | 798 | 0.908 | 0.894 | 0.873 | 0.913 | 0.887 | 0.845 | 0.795 |
| Task 1 | RAD-DINO linear probe | 1596 | 0.909 | 0.896 | 0.872 | 0.913 | 0.875 | 0.838 | 0.800 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 160 | 0.764 | 0.581 | 0.530 | 0.758 | 0.749 | 0.647 | 0.592 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 400 | 0.851 | 0.741 | 0.639 | 0.840 | 0.825 | 0.744 | 0.734 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 798 | 0.840 | 0.712 | 0.642 | 0.838 | 0.822 | 0.739 | 0.689 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 1596 | 0.824 | 0.754 | 0.648 | 0.822 | 0.809 | 0.732 | 0.681 |
| Task 2 | RAD-DINO linear probe | 160 | 0.824 | 0.807 | 0.738 | 0.814 | 0.831 | 0.826 | 0.773 |
| Task 2 | RAD-DINO linear probe | 400 | 0.875 | 0.865 | 0.786 | 0.860 | 0.876 | 0.864 | 0.830 |
| Task 2 | RAD-DINO linear probe | 798 | 0.888 | 0.875 | 0.808 | 0.872 | 0.881 | 0.865 | 0.834 |
| Task 2 | RAD-DINO linear probe | 1596 | 0.893 | 0.882 | 0.791 | 0.872 | 0.883 | 0.867 | 0.847 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 160 | 0.688 | 0.663 | 0.615 | 0.677 | 0.695 | 0.683 | 0.639 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 400 | 0.740 | 0.617 | 0.562 | 0.731 | 0.740 | 0.633 | 0.620 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 798 | 0.740 | 0.643 | 0.588 | 0.741 | 0.739 | 0.703 | 0.680 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 1596 | 0.759 | 0.625 | 0.560 | 0.754 | 0.760 | 0.710 | 0.656 |

## 7. Near-duplicate training images

Test AUROC after removing test images that the dataset audit paired with a training image at Pearson r ≥ 0.95 on normalised 256×256 pixels. The audit file is `results/audit/duplicates_task1.csv` and `duplicates_task2.csv`. Change is the AUROC without those images minus the AUROC on all test images.

| Task | Model | Training images | Test images removed | Test images kept | AUROC (all) | AUROC (kept) | Change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 160 | 8 | 448 | 0.835 | 0.836 | +0.0012 |
| Task 1 | RAD-DINO linear probe | 400 | 8 | 448 | 0.907 | 0.909 | +0.0020 |
| Task 1 | RAD-DINO linear probe | 798 | 8 | 448 | 0.908 | 0.908 | +0.0006 |
| Task 1 | RAD-DINO linear probe | 1596 | 8 | 448 | 0.909 | 0.911 | +0.0016 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 160 | 8 | 448 | 0.764 | 0.764 | +0.0005 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 400 | 8 | 448 | 0.851 | 0.850 | -0.0004 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 798 | 8 | 448 | 0.840 | 0.839 | -0.0010 |
| Task 1 | ResNet-50 (ImageNet) linear probe | 1596 | 8 | 448 | 0.824 | 0.823 | -0.0005 |
| Task 2 | RAD-DINO linear probe | 160 | 9 | 447 | 0.824 | 0.826 | +0.0013 |
| Task 2 | RAD-DINO linear probe | 400 | 9 | 447 | 0.875 | 0.873 | -0.0021 |
| Task 2 | RAD-DINO linear probe | 798 | 9 | 447 | 0.888 | 0.888 | -0.0005 |
| Task 2 | RAD-DINO linear probe | 1596 | 9 | 447 | 0.893 | 0.892 | -0.0006 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 160 | 9 | 447 | 0.688 | 0.683 | -0.0054 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 400 | 9 | 447 | 0.740 | 0.736 | -0.0042 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 798 | 9 | 447 | 0.740 | 0.735 | -0.0055 |
| Task 2 | ResNet-50 (ImageNet) linear probe | 1596 | 9 | 447 | 0.759 | 0.753 | -0.0051 |

## 8. Validation-set selection among the runs that exist

| Task | Highest validation AUROC | Validation AUROC | Candidates |
| --- | --- | --- | --- |
| Task 1 | RAD-DINO linear probe | 0.926 | 2 |
| Task 2 | RAD-DINO linear probe | 0.907 | 2 |

The rule in the analysis plan picks the run with the highest mean validation AUROC among supervised models, using raw scores and the full training set. Only these two probes have been run, so the choice is between those two. Vision-language families have no saved runs, and the pre-registered comparisons that need them are recorded as missing in `results/summary/comparisons.csv`.

