# Phase 4 fine-tuning notebooks (Kaggle)

| Notebook | Arm | Output dir per run |
|---|---|---|
| `finetune_mbert_kaggle.ipynb` | `mbert-ce` (ModernBERT-large + 2-class head, cross-entropy) | `mbert-ce@<size>-s<seed>/` |
| `finetune_laya_kaggle.ipynb` | `laya-ft` (Laya + RLCD, PREREG §4/§5) | `laya-ft@<size>-s<seed>/` |

Both clone crowdcal at commit `38cd6d871c5063bb2cf4f50670b2e66d6e4c76ea`, load only `train@*` and `dev` (never `calib` or `test`), pick the checkpoint with the lowest raw soft Brier on `dev`, and write to `/kaggle/working/checkpoints/<arm>@<size>-s<seed>/` with a `train_meta.json` (hyperparameters, per-epoch dev Brier, `gpu_hours`, `infer_sec_per_1k`, versions). `checkpoints/summary.json` lists the finished runs. `train_meta.json` is written last, so a run without it is unfinished and is redone.

## Kaggle setup (once)

1. Verify your phone number in Kaggle account settings. GPUs and Internet are unavailable until you do.
2. Import the notebook: **Create -> New Notebook -> File -> Import Notebook**, then upload the `.ipynb`.
3. In the right-hand settings panel: **Accelerator -> GPU T4 x2**, **Internet -> On**. `mbert-ce` uses both GPUs through `DataParallel`. The batch of 32 is split across them, so the effective batch stays 32.
4. Do not enable other accelerators. The notebooks assume CUDA fp16 (T4 has no bf16).

## Run order

1. Smoke test on the smallest run first: set `SELECT = [(100, 0)]` in the config cell and run all. Check that `checkpoints/mbert-ce@100-s0/train_meta.json` exists and that the printed per-epoch dev Brier goes down.
2. Run the rest. Order does not matter; runs are independent and a finished run is skipped when the notebook is rerun.
3. Both notebooks are independent and can run in different sessions or on different days.

## Splitting `RUNS` across sessions

`RUNS = [(100,0),(100,1),(100,2),(1000,0),(1000,1),(1000,2),(10000,0),(30000,0)]` is the full PREREG §5 list. Set `SELECT` in the config cell to a subset and leave `RUNS` alone. A suggested split per notebook:

| Session | `SELECT` |
|---|---|
| A | `[(100,0),(100,1),(100,2),(1000,0),(1000,1),(1000,2)]` |
| B | `[(10000,0),(30000,0)]` |

Use one notebook version (Save & Run All) per session so its Output tab holds exactly that session's runs.

## Time estimates (Kaggle T4 x2)

These are estimates from FLOP counts, not measurements. Expect them to be off by up to 2x, and replace them with the printed `gpu_hours` after the first run. Kaggle allows 12 hours per session and about 30 GPU hours per week, counted in wall-clock session time.

`mbert-ce` per run (includes about 1 minute for model load and about 1 minute for data load and save):

| Size | Optimizer steps | Epochs | Dev evals | Approx. wall-clock |
|---|---|---|---|---|
| 100 | 300 | 75 | 75 | 20 min (dev eval dominates) |
| 1,000 | 320 | 10 | 10 | 7 min |
| 10,000 | 939 | 3 | 3 | 12 min |
| 30,000 | 2,814 | 3 | 3 | 30 min |

The full 8-run set takes about 4–5 hours on one T4. DataParallel is disabled because it breaks ModernBERT on transformers 5.x, and batches are split into 4 micro-batches of 8 to fit in memory, which keeps the effective batch at 32. `gpu_hours` in `train_meta.json` counts training only (dev eval excluded), times the number of GPUs, so it is about 1 hour in total. `laya-ft` takes about 2.5–3.5 hours for all 8 runs. Scoring dev after each of its 4 epochs, one item at a time, dominates that time, so each run takes 15–35 minutes whatever its size. Both notebooks fit in one 12-hour session each.

## Disk: the ~20 GB `/kaggle/working` limit

`/kaggle/working` holds about 20 GB. Each `mbert-ce` checkpoint is about 1.6 GB (fp32 `safetensors`), so all 8 take about 13 GB. Only the selected epoch is saved. Check `du -sh /kaggle/working/checkpoints/*` between runs. If a session gets near the limit, download the finished runs, then start a new session and use `SELECT` for the remaining runs. Also, the Hugging Face cache and the MultiNLI download (about 0.3 GB) sit outside `/kaggle/working` and don't count.

## Downloading outputs into `data/checkpoints/<dir>/`

1. Run the notebook with **Save Version -> Save & Run All (Commit)**. Keep the browser tab open if you run it interactively instead.
2. Open the notebook's **Output** tab. Download each run directory, or use the Kaggle CLI: `kaggle kernels output <user>/<notebook-slug> -p kaggle_out`.
3. Move each run directory into your local checkout so the layout is `data/checkpoints/mbert-ce@100-s0/{config.json,model.safetensors,tokenizer*,train_meta.json}`. The directory name must be exactly `<arm>@<size>-s<seed>`, because `config/arms.json` and the analysis look it up by that name. `data/checkpoints/` is git-ignored.
4. Check each one loads: `crowdcal.arms.LocalArm("mbert-ce@100-s0", "data/checkpoints/mbert-ce@100-s0", "local", prompted=False).predict(item, None)` returns a probability.
5. Keep each session's `summary.json` if you like, but nothing reads it: the analysis reads every `train_meta.json`.

## Phase 5: score the checkpoints (`score_checkpoints_kaggle.ipynb`)

Run this after both training notebooks finish. Attach both of their outputs as inputs (Add Input → Notebooks), use GPU T4 x2 with Internet on, and choose Save & Run All. It takes about 1.5 hours, runs one checkpoint queue per GPU, and writes `crowdcal_phase5.zip`, which holds the cache JSONL files and each checkpoint's `train_meta.json`. Unzip it at the repo root, then run `crowdcal freeze` and `crowdcal report`.
