# Phantom AI 3.0 — Comprehensive Empirical Experiment Results

This document consolidates and synthesizes the empirical evaluations, benchmark scores, calibration curves, and ablation studies conducted throughout the development of Phantom AI 3.0.

---

## 1. Dataset Partitioning & Demographic Statistics (Phase 5.1)

To prevent identity leakage, data snooping, and circular evaluation, the CelebA dataset ($N = 202,599$ images across $10,177$ distinct identities) was deterministically partitioned at the **identity level**.

```
+---------------------------------------------------------------------------------------------------------+
|                                    CELEBA DATASET PARTITION BREAKDOWN                                   |
+-------------------+--------------+---------------+------------+-----------------------------------------+
| Partition         | Identities   | Images        | Proportion | Purpose                                 |
+-------------------+--------------+---------------+------------+-----------------------------------------+
| Split A (Total)   | 8,126        | 161,780       | 79.85%     | Generative Modeling (WGAN-GP)           |
|   - Split A1      | 6,500        | 130,063       | 64.20%     | Training (Generator + Critic)           |
|   - Split A2      | 1,626        | 31,717        | 15.65%     | Validation & Hyperparameter Selection   |
| Split B (Total)   | 980          | 19,313        | 9.53%      | Independent Forensic Evaluator          |
|   - Split B1      | 791          | 15,521        | 7.66%      | Evaluator Training                      |
|   - Split B2      | 189          | 3,792         | 1.87%      | Evaluator Calibration & Tuning          |
| Split C (Held-Out)| 1,071        | 21,506        | 10.62%     | Pure Benchmark (Untouched in Training)  |
+-------------------+--------------+---------------+------------+-----------------------------------------+
| Total             | 10,177       | 202,599       | 100.00%    | Complete CelebA Corpus                  |
+-------------------+--------------+---------------+------------+-----------------------------------------+
```

### Demographic & Attribute Distribution Skews
Analysis of the 20 target facial attributes across the dataset revealed substantial structural imbalances:
- **Gender:** 58.1% Female ($117,709$), 41.9% Male ($84,890$).
- **Age:** 77.9% Young ($157,750$), 22.1% Non-Young ($44,849$).
- **Facial Hair:** Goatee present in only 6.3% of images ($12,716$), Mustache in 4.1% ($8,400$).
- **Accessories:** Eyeglasses present in only 6.5% ($13,193$).
- **Hair Color:** Black Hair (23.9%), Brown Hair (20.5%), Blond Hair (14.8%), Gray Hair (4.2%).

---

## 2. Independent Forensic Evaluator Evaluation (Phase 5.3)

The multi-label `ForensicFeatureCNN` was trained on Split B1 for 10 epochs and evaluated on the independent Split B2 validation set ($3,792$ images).
*(Model Size: $4,824,020$ trainable parameters in `model.parameters()`; $4,828,381$ total tensors in checkpoint `state_dict` including $4,361$ BatchNorm running statistics).*

```
+-------------------------------------------------------------------------+
|                  FORENSIC EVALUATOR PERFORMANCE (SPLIT B2)              |
+-------------------------------------------------+-----------------------+
| Metric                                          | Measured Value        |
+-------------------------------------------------+-----------------------+
| Overall Macro F1 Score                          | 0.5141                |
| Overall Micro F1 Score                          | 0.5877                |
| Overall Classification Accuracy                 | 77.84%                |
| Mean Area Under ROC Curve (AUROC)               | 0.8533                |
+-------------------------------------------------+-----------------------+
```

### Attribute-Level Breakdown (Sample Highlights):
- **High-Accuracy Attributes (AUROC > 0.90):**
  - `Male`: Accuracy 93.4%, F1 0.912, AUROC 0.978
  - `Young`: Accuracy 84.1%, F1 0.887, AUROC 0.902
  - `Black_Hair`: Accuracy 86.2%, F1 0.741, AUROC 0.915
  - `Eyeglasses`: Accuracy 97.1%, F1 0.783, AUROC 0.981
- **Challenging / Low-Prevalence Attributes (AUROC 0.75 - 0.85):**
  - `Goatee`: Accuracy 94.2%, F1 0.482, AUROC 0.841
  - `Wavy_Hair`: Accuracy 68.4%, F1 0.581, AUROC 0.762
  - `Double_Chin`: Accuracy 95.1%, F1 0.312, AUROC 0.814

---

## 3. Conditional WGAN-GP Generative Quality (Phase 5.4)

The generator was trained on Split A1 ($130,063$ images) for 25 epochs. The final model was evaluated against the **Split C Held-Out Benchmark** ($21,506$ images, $1,071$ unseen identities).

```
+-----------------------------------------------------------------------------------+
|                        HELD-OUT GENERATIVE BENCHMARK (SPLIT C)                    |
+-------------------------------------------+-------------------+-------------------+
| Benchmark Metric                          | Measured Result   | Target / Baseline |
+-------------------------------------------+-------------------+-------------------+
| Custom Fréchet Distance (C-FD)            | 203.8529          | < 250.0           |
| Forensic Attribute Verification (FAV) F1  | 0.3562 (Macro)    | > 0.30            |
| FAV Micro F1 Score                        | 0.4954            | > 0.45            |
| FAV Macro Accuracy                        | 73.60%            | > 70.0%           |
| Mean Pairwise MS-SSIM                     | 0.3425            | < 0.40            |
| Visual Seed Diversity Index (1 - MS-SSIM) | 0.6575            | > 0.60            |
+-------------------------------------------+-------------------+-------------------+
```

### Analysis of Generative Metrics:
1. **Fréchet Distance (C-FD):** Evaluated in the feature space of `ForensicFeatureCNN`. A score of `203.85` indicates structural alignment with real face distributions at $64 \times 64$ resolution.
2. **FAV F1 (0.3562 Macro, 0.4954 Micro):** Verifies that when the generator is conditioned with $c_i = +1.0$, the independent evaluator recognizes the attribute in the synthetic output without direct adversarial collusion.
3. **Diversity (MS-SSIM 0.3425):** Mean structural similarity between composites generated with identical conditions but differing random seeds $z_1, z_2$ is $0.3425$, confirming that the mode-seeking penalty ($\lambda_{ms} = 1.0$) successfully prevented mode collapse.

---

## 4. Latent Space Semantic Editing & Entanglement (Phase 5.7 & 5.7.1)

Linear SVM hyperplanes were fitted in $\mathcal{Z} \in \mathbb{R}^{100}$ across 20 candidate facial attributes. Attribute selectivity was evaluated to quantify feature entanglement.

```
+---------------------------------------------------------------------------------------+
|                    LATENT DIRECTION SELECTIVITY & ENTAILMENT AUDIT                    |
+--------------------+-------------------+---------------+------------------------------+
| Attribute Name     | Selectivity Ratio | Final Status  | Primary Entanglement Note    |
+--------------------+-------------------+---------------+------------------------------+
| hair_bangs         | 2.16x             | ACCEPTED      | Clean forehead occlusion     |
| hair_gray          | 1.41x             | ACCEPTED      | Age drift within tolerance   |
| glasses            | 1.36x             | ACCEPTED      | High visual localization     |
| hair_blond         | 1.34x             | ACCEPTED      | Distinct color shift         |
| facial_goatee      | 1.23x             | ACCEPTED      | Chin-localized modification  |
| Young              | 1.08x             | REJECTED      | Drastic gender/skin warp     |
| Pale_Skin          | 1.04x             | REJECTED      | Lighting/contrast artifact   |
| Male               | 1.02x             | REJECTED      | Complete facial re-synthesis |
| Oval_Face          | 0.98x             | REJECTED      | Background collapse          |
| Chubby             | 0.95x             | REJECTED      | Uncontrolled jaw distortion  |
| ... (10 others)    | < 1.15x           | REJECTED      | Low selectivity (< 1.15x)    |
+--------------------+-------------------+---------------+------------------------------+
```

### Acceptance Summary:
- **Accepted Latent Directions (5):** Only directions achieving $\ge 1.15\times$ selectivity without corrupting orthogonal identity features were deployed.
- **Continuous GCIT Fallback (9 Features):** Rejected structural attributes (e.g., face shape, nose width, hairline) were reassigned to continuous GCIT slider modulation, avoiding latent hyperplane collapse.

---

## 5. NLP Extraction Pipeline Evaluation (Phases 3.5, 3.6 & 3.7)

```
+------------------------------------------------------------------------------------------+
|                                NLP PROGRESSION ACROSS PHASES                             |
+-----------------------+---------------------+---------------------+----------------------+
| Evaluation Metric     | Phase 3.5 (Audit)   | Phase 3.6 (Audit)   | Phase 3.7 (Indep.)   |
+-----------------------+---------------------+---------------------+----------------------+
| Corpus Size           | 60 synthetic        | 60 calibrated       | 100 independent      |
| Benchmark Origin      | Regex-aligned       | Rule-reconciled     | Real-world syntax    |
| Evidence Slots        | 600 slots           | 600 slots           | 1,000 slots          |
| Exact Span Matches    | 68 / 85             | 76 / 85             | 56 / 94              |
| Strict Span Precision | 0.8125              | 0.8636              | 0.6292               |
| Strict Span Recall    | 0.8000              | 0.8941              | 0.5600               |
| Strict Span F1 Score  | 0.8062              | 0.8786              | 0.5926               |
| Generalization Delta  | Baseline            | +7.24% (Tuned)      | -28.60% (Domain Drop)|
+-----------------------+---------------------+---------------------+----------------------+
```

### Confidence Calibration & Reliability Analysis (Phase 3.7)
Evaluating sequence-level model probabilities across the independent corpus demonstrated substantial probability overconfidence:
- **Expected Calibration Error (ECE):** `0.4865`
- **Maximum Calibration Error (MCE):** `0.7812`

```
+-------------------------------------------------------------------------+
|                  CONFIDENCE CALIBRATION (ECE = 0.4865)                  |
+-------------------+-----------------+-----------------+-----------------+
| Confidence Bin    | Predicted Conf. | Empirical Acc.  | Calibration Gap |
+-------------------+-----------------+-----------------+-----------------+
| 0.00 - 0.20       | 0.12            | 0.08            | +0.04           |
| 0.20 - 0.40       | 0.31            | 0.19            | +0.12           |
| 0.40 - 0.60       | 0.52            | 0.26            | +0.26           |
| 0.60 - 0.80       | 0.73            | 0.38            | +0.35           |
| 0.80 - 1.00       | 0.94            | 0.45            | +0.49 (Severe)  |
+-------------------+-----------------+-----------------+-----------------+
```

#### Forensic Mandate:
Because the BiLSTM-CRF model assigns $>90\%$ probability to spans that achieve only $45\%$ empirical accuracy under natural syntax, **heuristic NLP extraction confidence is strictly suppressed from investigator display and recorded as `null`**. Investigators must verify raw text spans directly rather than relying on misleading numerical confidence indicators.

