# Crowdcal — Pre-registration (DRAFT)

Status: **draft**. It becomes binding when committed and tagged `prereg-v1`, which must happen before any real model call. Fill in every `TODO` before tagging.

## 1. Primary endpoint

The mean soft Brier, (p − q)², between each arm's **temperature-recalibrated** P(entailment) and the ChaosNLI human entailment share q. It is computed over the ChaosNLI SNLI + MNLI-m test items and averaged over the 3 frozen wordings (§3) for prompted arms.

## 2. Hypotheses

- **H1 (primary):** `laya-ft@30k` has lower recalibrated soft Brier than `jev`. Paired bootstrap, 95% CI of the difference entirely below 0.
- **H2 (secondary, Holm family, α = 0.05):**
  - H2a–d: `laya-ft@N` < `mbert-ce@N` for N ∈ {100, 1k, 10k, 30k}.
  - H2e: `laya-ft@30k` < `laya-base`.
- **Crossover (descriptive):** the smallest N where `laya-ft@N − best-hosted` has a 95% CI < 0. Best hosted = lowest recalibrated soft Brier on `calib` among {jev, deepseek, laya-base}.
- Everything else is exploratory: raw (un-recalibrated) scores, isotonic, JS distance, per-wording results, subset splits, 3-way NLI.

## 3. Frozen question wordings

Context template: `Premise: {premise}\nHypothesis: {hypothesis}`

1. Given the premise, is the hypothesis definitely true?
2. Does the premise entail the hypothesis?
3. If the premise is true, must the hypothesis also be true?

## 4. Arms and pinned versions

| Arm | Pin |
|---|---|
| jev | model version: TODO (record from API response) |
| deepseek | model: DeepSeek V4.1 Flash, provider: TODO, reasoning: off, temperature: 0 |
| laya-base / laya-ft init | `convaiinnovations/laya` @ TODO (commit SHA) |
| mbert-ce init | `answerdotai/ModernBERT-large` @ TODO (commit SHA) |

## 5. Sample sizes

- Test: all ChaosNLI SNLI + MNLI-m items (~3.1k) after exclusions.
- Training sizes: 100, 1k, 10k, 30k (nested, stratified, seed 0). 3 seeds at 100 and 1k; 1 seed at 10k and 30k.
- Repeats for API arms: 1 if the pilot shows identical outputs across repeats, otherwise 3 (averaged).

## 6. Exclusion rules

- A call is retried 3× with backoff. An item still failing in any arm is excluded from **all** arms.
- An arm with more than 2% of items failing is reported with a flag, and not dropped.
- Unparseable outputs count as failures. There is no manual correction.

## 7. Recalibration

Temperature scaling on logit(p), one parameter per arm or checkpoint, fitted on `calib` against the 5-annotator entailment share by minimizing soft Brier. Isotonic is exploratory.

## 8. Cost reference

Reference T4 price: TODO $/GPU-hour (source + date).

## 9. Planned follow-ups (not part of v1.0 claims)

- v1.1: civil_comments toxicity share as a second soft-label domain, using the same arms and pipeline.

## Amendments

_None yet. Each entry must give the date, what changed, why, and whether any test data had been seen (the answer must be no)._
