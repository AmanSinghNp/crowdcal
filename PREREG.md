# Crowdcal — Pre-registration

This document becomes binding when it is committed and tagged `prereg-v1`, before any real model call. `crowdcal run` refuses every real arm until that tag exists. The machine-readable copy is `config/prereg.json` (analysis) and `config/arms.json` (model pins). If this document and those files disagree, this document wins and the discrepancy is logged as an amendment.

- **Registered:** 2026-09-30
- **Data state at registration:** splits built (`data/splits/`, ids only). No model has been run on any split, and no real model output exists anywhere.

## 1. Primary endpoint

The mean soft Brier, (p − q)², between each arm's **temperature-recalibrated** P(entailment) and q, the share of ChaosNLI's human annotators who chose entailment. It is computed over the ChaosNLI SNLI + MNLI-m test items, after exclusions (§6). For prompted arms, it is averaged per item over the 3 frozen wordings (§3), then over items.

## 2. Hypotheses

- **H1 (primary):** `laya-ft@30000` has lower recalibrated soft Brier than `jev`. H1 is supported if the 95% paired-bootstrap CI of (laya-ft@30000 − jev) lies entirely below 0. It is refuted if the CI lies entirely above 0, and inconclusive otherwise.
- **H2 (secondary):** one family, Holm-corrected at α = 0.05, using two-sided bootstrap p-values.
  - H2a–d: `laya-ft@N` ≠ `mbert-ce@N` for N ∈ {100, 1000, 10000, 30000}. The direction is reported, and the expectation is laya-ft lower.
  - H2e: `laya-ft@30000` ≠ `laya-base`, with laya-ft expected to be lower.
- **Crossover (descriptive, answers Q2):** the smallest N at which the 95% CI of (laya-ft@N − best-hosted) lies entirely below 0. Best-hosted is the arm with the lowest recalibrated soft Brier **on `calib`** among {jev, deepseek, laya-base}, fixed before test metrics are computed.
- **Q1 (descriptive):** reliability of each arm against q, using 10 equal-mass bins and ECE, raw and recalibrated. There is no hypothesis test.
- **Q3 (descriptive):** cost per 1,000 decisions (§8).
- **Exploratory:** everything else, labelled as such on the results page. That includes raw (un-recalibrated) comparisons, isotonic recalibration, JS distance, per-wording and per-subset (SNLI vs. MNLI) results, deepseek vs. jev, and any 3-way NLI analysis.

## 3. Frozen question wordings

Context template: `Premise: {premise}\nHypothesis: {hypothesis}` (`config/wordings.json`).

1. Given the premise, is the hypothesis definitely true?
2. Does the premise entail the hypothesis?
3. If the premise is true, must the hypothesis also be true?

The target is binary: entailment = yes, and neutral or contradiction = no.

## 4. Arms and pinned versions

| Arm | Pin | Elicitation |
|---|---|---|
| `jev` | `typesafe/jev-1.13` via OpenRouter Decisions API (provider TypeSafe) | native Noul P(yes) |
| `deepseek` | `deepseek/deepseek-v4.1-flash`, OpenRouter provider **DeepSeek** (first-party; supports `logprobs` + `top_logprobs`), no fallbacks, reasoning off, temperature 0 | P(yes) = yes-token mass ÷ (yes + no) from top-20 logprobs of the first token |
| `laya-base` | `convaiinnovations/laya` @ `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, English root checkpoint | native Noul P(yes), as shipped |
| `laya-ft` | initialized from `laya-base` pin; RLCD via `NandhaKishorM/laya` @ `9d955671415fc19f069b9cc998928075c1f255ec`, notebook `laya_finetune_typed_decisions_2xT4_kaggle.ipynb` defaults | native Noul P(yes) |
| `mbert-ce` | `answerdotai/ModernBERT-large` @ `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` + 2-class head | softmax P(entailment) on the (premise, hypothesis) pair |

**Version drift:** each API response's served model string is stored. If an arm is served by more than one version, the analysis refuses to run. The affected arm is rerun in full on one version, and the rerun is logged as an amendment.

**Laya's shipped calibration:** `laya-base` "raw" means the probabilities exactly as the package returns them, including any temperature bundled with the checkpoint. Our recalibration is applied on top, the same as for every other arm.

## 5. Training and sample sizes

- **Test:** all ChaosNLI SNLI (1,514) + MNLI-m (1,599) = 3,113 items, each with 100 annotations, minus exclusions.
- **Calib:** 4,108 MNLI dev-matched items (5 annotations each). Dev has 4,108 items and is used only for checkpoint selection. Both exclude every ChaosNLI item and every text pair that appears in test.
- **Training sizes:** 100, 1000, 10000 and 30000. These are nested stratified subsets of MNLI train (seed 0), trained on binary hard labels.
- **Seeds:** 0, 1 and 2 at sizes 100 and 1000; seed 0 at 10000 and 30000. The bootstrap samples a seed per resample.
- **Checkpoint selection:** evaluate on `dev` after each epoch and keep the lowest raw soft Brier against the 5-annotator q. Test is never evaluated during training.
  - `mbert-ce`: AdamW, lr 2e-5, batch 32, at most 3 epochs or 300 steps (whichever is larger), warmup 10%, max length 256.
  - `laya-ft`: the pinned notebook's hyperparameters unchanged, apart from data size and seed.
- **Repeats (API arms):** the pilot (Phase 3) runs each wording twice on 50 calib items. If every repeated p is identical to 4 decimals, the arm uses 1 repeat. Otherwise it uses 3, and p is the mean. The decision is logged as an amendment.
- **Precision (informative, not binding):** with about 3.1k paired items and a per-item SD of the Brier difference around 0.03, the SE of the paired difference is about 0.0005. That means differences of about 0.0015 or larger are detectable.

## 6. Exclusion rules

- A call is retried 3 times with exponential backoff. An item that still has no valid probability in any arm is excluded from **all** arms.
- Unparseable output counts as a failure. For `deepseek`, that means neither yes nor no appears in the top-20 logprobs. There is no manual correction.
- An arm with more than 2% of items failing is flagged on the results page, not dropped.
- **DeepSeek fallback:** if the pilot shows that more than 2% of deepseek calls fail the logprobs parse, `deepseek` switches to verbalized probability (bare number from 0 to 100) for the whole run, and the switch is logged as an amendment. The switch is decided on the pilot only, never after test data is seen.

## 7. Recalibration

- **Primary:** temperature scaling on logit(p), with one parameter T per arm or checkpoint. For prompted arms, wordings are pooled. T is fitted on `calib` by minimizing soft Brier against the 5-annotator entailment share, with log T bounded to (−5, 5).
- **Exploratory:** isotonic regression, fitted on the same data.

## 8. Cost reference

- **API arms:** mean `usage.cost` per decision, as reported by OpenRouter, × 1,000.
- **Local arms:** measured inference seconds per 1,000 decisions on a T4 × **$0.35 per T4 GPU-hour**. That is the Google Cloud Compute Engine on-demand T4 GPU price in us-central1, GPU only and excluding the host VM, as listed September 2026.
- **Fine-tuning:** measured GPU-hours × the same rate, reported separately and not amortized.
- **Budget:** the runner stops hard at $20 of cumulative API spend per arm. The expected total for both API arms is under $5: about 7.2k items × 3 wordings × up to 3 repeats, at ≈150 input tokens each.

## 9. Analysis code

The analysis is `crowdcal report` at the commit tagged `prereg-v1`, run on `config/prereg.json`. Any later change to `crowdcal/analysis.py`, `metrics.py`, `calib.py` or `stats.py` that alters a reported number is an amendment. Bug fixes are allowed but must be listed as amendments, with a before and after of the affected numbers.

## 10. Planned follow-ups (not part of v1.0 claims)

- v1.1: civil_comments, using the toxicity share as a second soft-label domain, with the same arms and pipeline.

## Amendments

_None yet. Each entry must give the date, what changed, why, and confirm that no test-split model output had been examined._
