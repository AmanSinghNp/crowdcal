# Crowdcal — Spec v1.0

> Does 70% mean 70% of people agree? A calibration benchmark for typed decision models, scored against the full human label distribution.

Status: frozen at tag `prereg-v1`. Where this spec and `PREREG.md` differ, `PREREG.md` wins.

## 1. Questions

1. **Calibration vs. crowd:** does each model's P(entailment) match the share of human annotators who chose entailment?
2. **Data efficiency:** how many labelled examples does fine-tuned Laya need to beat the hosted models? Does Laya's RLCD pipeline beat plain cross-entropy fine-tuning of the same backbone?
3. **Cost:** what does each option cost per 1,000 decisions, including GPU time for fine-tuning?

## 2. Scope

| In v1.0 | Deferred |
|---|---|
| ChaosNLI (SNLI + MNLI-m subsets), binary entailment | civil_comments (v1.1, named in PREREG as planned follow-up) |
| 5 arms, raw + recalibrated | 3-way NLI (exploratory only, arms that emit 3 probs) |
| Headline figure, per-arm tables, paired comparisons, cost table | Latency benchmark |
| | Hand-labelled realistic decision set (stretch) |
| | SST-2 (dropped) |

## 3. Task

- **Input:** premise + hypothesis.
- **Question (prompted arms):** a yes/no question from a frozen set of 3 wordings (see PREREG §3).
- **Model output:** P(yes) = P(entailment) ∈ [0, 1].
- **Target:** q = share of ChaosNLI's 100 annotators who chose `entailment` (neutral and contradiction both count as "no").

## 4. Arms

| ID | Model | Source | Probability from | Training |
|---|---|---|---|---|
| `jev` | TypeSafe Jev | OpenRouter | native yes/no probability | none |
| `deepseek` | DeepSeek V4.1 Flash, reasoning off | OpenRouter, pinned provider | yes/no token logprobs; fallback: verbalized probability (the pilot decides) | none |
| `laya-base` | `convaiinnovations/laya` @ pinned SHA | HF, local / Kaggle | native yes/no probability | none |
| `laya-ft` | same checkpoint + RLCD | Kaggle 2×T4 | native | MNLI subsets |
| `mbert-ce` | `answerdotai/ModernBERT-large` + 2-class head | Kaggle 2×T4 | softmax | MNLI subsets, cross-entropy |

`mbert-ce` uses Laya's own backbone, so the comparison is "Laya pipeline vs. off-the-shelf fine-tuning". It is **not** a pure test of the loss function, because Laya also had its own decision pre-training. Say this in the write-up.

Every arm is reported **raw** and **recalibrated**. The recalibration step is identical for all arms (§7).

## 5. Data and splits

| Split | Source | Size | Use |
|---|---|---|---|
| `train@{100,1k,10k,30k}` | MNLI train, nested stratified subsets (100 ⊂ 1k ⊂ 10k ⊂ 30k), fixed seed | 30k max | fine-tuning, hard labels |
| `dev` | half of (MNLI dev-matched − ChaosNLI-MNLI items) | ~4.1k | checkpoint selection only |
| `calib` | other half | ~4.1k | recalibration fitting (5-annotator soft labels); pilot draws 50 items from here |
| `test` | ChaosNLI SNLI (~1,514) + MNLI-m (~1,599) | ~3.1k | final numbers only |

- **Leakage guard:** a test asserts that no `test` item ID (or premise/hypothesis pair) appears in `train`, `dev` or `calib`. It must pass before any run.
- **Checks to confirm in Phase 1:** that ChaosNLI-SNLI comes from SNLI dev and ChaosNLI-MNLI from MNLI dev-matched. The assertion test holds either way.
- **Pinning:** record the dataset revisions (HF dataset SHA or file checksum) in `data/splits/manifest.json`.
- **Prior baseline:** predict the mean entailment rate of `train@30k` for every item.

## 6. Metrics

For item i: model prob p_i, human share q_i, annotator count n_i.

| Metric | Definition | Role |
|---|---|---|
| Soft Brier | mean (p_i − q_i)² | **primary** |
| JS distance | Jensen–Shannon distance between Bernoulli(p_i) and Bernoulli(q_i), base 2 | secondary |
| Per-item gap | p_i − q_i | diagnostics, reliability plot |
| Noise ceiling | mean q_i(1 − q_i)/(n_i − 1) | the soft Brier a perfect model would still get from annotator sampling noise |
| Crowd reliability | bin p into 10 equal-mass bins; mean p vs. mean q per bin; ECE vs. q | answers Q1 |
| Bernoulli noise-floor test | per arm: is soft Brier − noise ceiling > 0? Bootstrap CI on the excess | "is the miscalibration real, or annotator noise?" |

Prompted arms: each metric is averaged over the 3 wordings, then over items. Per-wording numbers are reported too.

## 7. Recalibration

- **Primary:** temperature scaling on the logit of p, with one parameter T per arm, per checkpoint for fine-tuned arms. T is fitted on `calib` by minimizing soft Brier against the 5-annotator shares.
- **Exploratory:** isotonic regression, fitted the same way.
- Identical code path for every arm. Parameters are saved to `data/calib/{arm}.json`.

## 8. Statistics

- **CIs:** paired bootstrap over test items, 10,000 resamples, 95% percentile.
- **Seed variance:** for 100 and 1k, 3 training seeds; the bootstrap samples a seed per resample, so bands include training noise. 10k and 30k use 1 seed.
- **Primary comparison:** `laya-ft@30k` − `jev`, recalibrated soft Brier (paired).
- **Crossover:** the smallest training size at which the paired difference `laya-ft@N − best-hosted` has its 95% CI entirely below 0. "Best hosted" is whichever of {jev, deepseek, laya-base} has the lowest recalibrated soft Brier **on `calib`**. It's chosen before test data is seen.
- **Secondary family (Holm, α = 0.05):** `laya-ft` vs. `mbert-ce` at each of 4 sizes, and `laya-ft@30k` vs. `laya-base` (5 tests).
- Everything else is labelled exploratory.

## 9. Runs and caching

- **Cache:** append-only JSONL, one file per (arm, config-hash), in `data/raw/`.
- **Row schema:**
  ```json
  {"key": "sha256", "arm": "jev", "model_version": "...", "provider": "...",
   "dataset": "chaosnli-mnli", "dataset_rev": "...", "item_id": "...",
   "wording_id": 2, "wording_sha": "...", "repeat": 0, "p_yes": 0.71,
   "raw": {...}, "usage": {"in": 212, "out": 1, "cost_usd": 0.000008},
   "status": "ok", "ts": "..."}
  ```
- **Cache key:** sha256 of (arm, model_version, provider, dataset, dataset_rev, item_id, wording_sha, repeat, elicitation params). A cache hit means the call is skipped.
- **No source text** is stored in rows, only item IDs, which avoids redistributing dataset text.
- **The report refuses to mix config hashes** within an arm. A mismatch is a hard error, not a warning.
- **Repeats:** the pilot measures variation between repeated calls per API arm. If answers are identical, 1 repeat; otherwise 3, and p is averaged.
- **Failures:** retry 3× with backoff. An item that still fails is dropped from **all** arms, which keeps the comparison paired. An arm with more than 2% failures is flagged in the report.
- **Budget:** a hard stop at a cumulative $20 of API spend, read from `usage.cost_usd`.
- **Freezing:** after the main run, write `data/raw/SHA256SUMS`. The report verifies it.

## 10. Cost (Q3)

- **API arms:** mean `usage.cost_usd` per decision × 1,000.
- **Local arms:** measured T4 inference throughput × a reference T4 $/hr fixed in PREREG.
- **Fine-tuning:** GPU-hours per training size × the same rate. Shown as a separate column, not amortized.

## 11. Outputs

- **Headline figure (`docs/headline.png`):**
  - x-axis: training examples (100, 1k, 10k, 30k), log scale.
  - y-axis: recalibrated soft Brier, lower is better.
  - Curves for `laya-ft` and `mbert-ce`, with 95% bands.
  - Flat lines with bands for `jev`, `deepseek`, `laya-base` and the prior baseline.
  - A dashed line for the noise ceiling.
  - The crossover annotated.
- **Results page (`docs/index.html`, GitHub Pages):**
  - the headline figure;
  - an arm table (raw and recalibrated, all metrics);
  - reliability plots;
  - paired comparisons with Holm-adjusted p-values;
  - the cost table;
  - an "Exploratory" section, clearly separated.
- **Command:** `crowdcal report` rebuilds all of `docs/` from `data/raw/` + `data/splits/` + `data/calib/`. No network access.

## 12. Repo layout

```
crowdcal/
  data.py      splits, leakage guard, soft labels
  arms.py      openrouter runner (jev, deepseek), local HF runner (laya, mbert), fake arms
  cache.py     keyed JSONL cache
  metrics.py   soft Brier, JSD, noise ceiling, reliability
  calib.py     temperature + isotonic
  stats.py     paired bootstrap, Holm
  report.py    figure + HTML
  cli.py       crowdcal splits | run | calibrate | report
notebooks/     kaggle fine-tuning (laya-ft, mbert-ce), one per arm
data/          splits/, raw/, calib/, checkpoints manifest
PREREG.md
tests/         one test file per module with real logic
```

Stack: Python 3.12, uv, `datasets`, `transformers`, `numpy`, `scipy`, `matplotlib`, `httpx`. Code MIT, data CC BY 4.0.

## 13. Phases

| # | Phase | Week | Done when |
|---|---|---|---|
| 1 | Foundations | 1 | Pipeline runs end to end on **fake arms + synthetic soft labels** and produces the headline figure. Leakage test passes. Metric tests check against hand-computed values. |
| 2 | Pre-registration | 1–2 | `PREREG.md` complete (wordings, hypotheses, sizes, exclusions, T4 rate, pinned versions). Committed and tagged `prereg-v1` **before any real API call**. `crowdcal report` runs against the frozen config on fake data. |
| 3 | Pilot | 2 | 50 `calib` items × every arm: answers parse, failure rate < 2%, cost matches the estimate, DeepSeek reasoning confirmed off, logprobs available (or fallback triggered), repeat-variance measured. Changes go in `PREREG.md` §Amendments with date + reason. **No metrics looked at.** |
| 4 | Fine-tuning | 2–3 | 8 `laya-ft` runs and 8 `mbert-ce` runs (100 and 1k × 3 seeds, plus 10k and 30k × 1). Each checkpoint is saved with its config, dev score and GPU-hours. Selection uses `dev` only. |
| 5 | Main run | 3 | All arms × test × 3 wordings × repeats cached. `SHA256SUMS` written. |
| 6 | Analysis + write-up | 4 | `crowdcal report` output published. README findings section, LinkedIn post and awesome-list entry written. Tag `v1.0`. |

## 14. Definition of done (v1.0)

- Every number on the results page traces back to cached rows in `data/raw/`.
- `uv run crowdcal report` rebuilds the page from the raw files alone.
- The primary endpoint and comparison match `prereg-v1` exactly. Any deviation is listed as an amendment.
- Git history shows `prereg-v1` before the first commit containing real model output.
