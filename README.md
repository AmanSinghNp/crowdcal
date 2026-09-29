# Crowdcal

Does 70% mean 70% of people agree? Crowdcal is a calibration benchmark for typed decision models. It scores each model's P(entailment) against the full distribution of human labels (the share of ChaosNLI annotators who chose entailment), and asks how much labelled data a fine-tuned small model needs to beat hosted ones, and what each option costs per 1,000 decisions. See [SPEC.md](SPEC.md) and [PREREG.md](PREREG.md).

## Status

Phase 1: the pipeline runs end to end on fake arms and synthetic data and produces the headline figure. There are no real results yet.

## Quickstart

```
uv sync
uv run pytest
uv run crowdcal demo      # synthetic end-to-end run -> build/demo/site/
```

## Real runs

`crowdcal run` refuses to call any model unless the git tag `prereg-v1` exists. Complete the TODOs in `PREREG.md` and `config/`, commit, tag, then run.
