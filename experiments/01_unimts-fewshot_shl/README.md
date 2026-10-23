# 01 UniMTS few-shot on SHL

## Question

Does the pretrained UniMTS motion model (zero-shot, linear probe, full fine-tuning) beat the same architecture from
random init and a LightGBM feature baseline when only k = 1, 2, 5, 10 labelled 10 s windows per class are
available? Data: SHL User 1, hips phone.

*(Reconstructed from `COMMANDS.md` and `results/shl_fewshot.md` during the migration. `COMMANDS.md` has no explicit
question.)*

## Plan

These are the steps as run (see `COMMANDS.md`). Every script runs from this folder. Paths come from `shl_common.py`
(see Setup below).

0. **Inspect.** `cache_raw.py` caches the SHL User 1 hips acc and gyro data plus the coarse labels as one `.npz` per
   recording. `inspect_shl.py` produces sanity numbers and pure-window counts per day (`logs/window_counts.csv`).
1. **Check resampling.** `check_resample.py` compares UniMTS's FFT resampling with zero-phase FIR decimation.
2. **Windows.** `make_windows.py` cuts 10 s pure windows (no Null, no gap, no NaN) at 100 Hz and decimates them to
   20 Hz by FIR before windowing. There are three class sets:
   - A: still / walk / run
   - B: still / walk-or-run / vehicle
   - C: all 8 SHL classes

   Each set has two label wordings, w1 and w2. Split by day, single user:
   - fewshot pool: days 190517 and 250517
   - test: the other 73 days, capped at 300 windows per SHL class, seed 0

   Output goes to `logs/windows_log.txt`.
3. **Zero-shot.** `run_zeroshot.sh`:
   - Set A is evaluated over joints 1/2/5/6 × wordings w1/w2. The joint and wording are chosen on the **pool**
     (`make_zeroshot_table.py --pick-best`) and then reported on test.
   - Sets B and C use the chosen pair.
   - A cross-check runs the unmodified upstream `evaluate_custom.py`.
4. **Few-shot.**
   - `make_kshot.py` draws k ∈ {1, 2, 5, 10} windows per class × seeds 0–4 from the pool. All methods use the same
     windows.
   - `run_fewshot.sh` runs `finetune_seeded.py` on the upstream `finetune_custom.py` with sets B and C × k × seed ×
     {probe, full, random, random_gyro}. Settings: joint 1, wording w2, 20 Hz, 200 epochs, Apple GPU (MPS). That is
     160 runs.
5. **LightGBM baseline.** `baseline_lgbm.py` builds |acc| (and |gyro|) features at 20 or 100 Hz and trains on the
   same k-shot windows.
6. **Tables.** `make_table.py` writes `results/shl_fewshot.{md,csv}` and the plots. `make_zeroshot_table.py` writes
   `results/zeroshot.{md,csv}`.

Set A was run zero-shot only. Few-shot on A was optional and was not run.

## Setup (after migration from `UniMTS2`, 2026-10-23)

| What | Where |
|---|---|
| Upstream UniMTS | `../ext/UniMTS` at commit `656fcdab7eee17d5b4158acf98fa348f74ce29a6` (https://github.com/xiyuanzh/UniMTS), plus `patches/UniMTS.patch` |
| Checkpoint | `<data>/checkpoints/unimts/checkpoint/UniMTS.pth`, sha256 `a9858c0084d936655240407e30ff9db9adeded6a67dc5650e3f667578e93b220` |
| Upstream working dir | `<data>/checkpoints/unimts/`: `checkpoint/`, `config.json`, `.cache/huggingface/` (copied from the UniMTS2 root), `UniMTS_data -> ../../raw/unimts/UniMTS_data` |
| SHL raw | `<data>/raw/shl/` (User 1 hips, zip parts, `release/`, `doc/`, `LICENSE.txt`) |
| UniMTS pretraining/eval data | `<data>/raw/unimts/UniMTS_data/` |
| Processed windows | `<data>/processed/shl_u1_hips_w10s/{raw,hz20,hz100,kshot}/` |
| Result `.npz` files (predictions, not in git) | `<data>/processed/01_unimts-fewshot_shl/{zeroshot,fewshot}/` |
| Envs | `unimts` (conda, Python 3.9.25) for UniMTS; `tmd-lgbm` (conda, Python 3.11, LightGBM 4.6.0) for `baseline_lgbm.py` and `make_table.py`. Exports in `env/`. |

`<data>` is the data root, `/Users/ms/Projects/waiis/data` (see `CLAUDE.local.md`).

To reuse the clone, run `cd ../ext/UniMTS && git checkout . && git apply <this folder>/patches/UniMTS.patch`.

**About the patch.** It is the unchanged `git diff` of UniMTS2:

- Its substance is `.cuda()` → CPU and `map_location` in `torch.load`.
- Most of the `evaluate.py` hunk is editor reformatting.
- `evaluate_custom.py:48` and `finetune_custom.py:60,110` still lack `map_location`. `run_cpu.py` and
  `finetune_seeded.py` patch `torch.load` at runtime instead.

**Always call the env interpreter directly**, e.g. `/opt/homebrew/anaconda3/envs/unimts/bin/python`. Inside this repo,
`conda run -n unimts python` resolves to the uv `.venv` Python 3.11.

The upstream `*_custom.py` scripts download the checkpoint, `config.json` and `UniMTS_data.zip` (4.5 GB) from Hugging
Face into the cwd if `./checkpoint` or `./UniMTS_data` are missing (`finetune_custom.py:27-33`,
`evaluate_custom.py:24-30`). The run scripts therefore `cd` to `<data>/checkpoints/unimts` and set
`HF_HUB_OFFLINE=1`.

## Results

These results predate MLflow. This experiment's record is `results/shl_fewshot.csv` (aggregated over seeds) plus
`results/zeroshot.csv`. The per-run predictions are in `<data>/processed/01_unimts-fewshot_shl/`. Full tables, all
confusion matrices and plots are in `results/zeroshot.md` and `results/shl_fewshot.md`.

**Run status:**
- The few-shot grid finished on 2026-10-22 at 04:12: 160/160 UniMTS runs and 0 FAILED (`logs/run_fewshot.out`).
- LightGBM ran all 160 variants.
- The zero-shot cross-check through the unmodified `evaluate_custom.py` on set A test, j1 w2, gave acc 0.9378 and F1
  0.9366 (weighted). That matches `zeroshot_eval.py` (acc 0.938).

**Evaluation conventions:**
- The split is **by day for a single user** (SHL User 1). Every number is optimistic for new users.
- The joint and wording were chosen on the fewshot pool, never on test.
- No normalisation statistics are fitted.
- **Model size and latency per window are not reported** (open question 4).

### Zero-shot (test, 20 Hz, acc only)

The joint and wording were chosen on set A's pool: joint 1, w2 (pool macro-F1 0.825). Joint 5 w2 is tied within
0.002 (0.823).

| Set | joint | wording | n | acc | macro-F1 | per-class recall |
|---|---|---|---|---|---|---|
| A | 1 | w2 | 900 | 0.938 | 0.937 | still 0.993, walk 0.840, run 0.980 |
| B | 1 | w2 | 2400 | 0.507 | 0.542 | still 0.983, walk-or-run 0.922, vehicle 0.247 |
| C | 1 | w2 | 2400 | 0.343 | 0.295 | still 0.923, walk 0.230, run 0.967, bike 0.413, car 0.120, bus 0.083, train 0.010, subway 0.000 |

In set B, 1022 of 1500 vehicle windows are predicted as "standing still". In set C, most vehicle windows go to still,
and walk goes to cycling (197 of 300).

### Few-shot, macro-F1 (mean ± std over 5 seeds)

**Set B** (3 classes; test: still 300, walk-or-run 600, vehicle 1500)

| method | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|
| zeroshot (k=0) | 0.542 | | | |
| probe | 0.528 ± 0.035 | 0.602 ± 0.098 | 0.619 ± 0.056 | 0.676 ± 0.049 |
| full | 0.510 ± 0.066 | 0.502 ± 0.111 | 0.558 ± 0.031 | 0.626 ± 0.052 |
| random | 0.347 ± 0.055 | 0.367 ± 0.080 | 0.449 ± 0.065 | 0.526 ± 0.035 |
| random_gyro | 0.416 ± 0.067 | 0.419 ± 0.103 | 0.501 ± 0.088 | 0.581 ± 0.052 |
| lgbm_acc_hz20 | 0.457 ± 0.068 | 0.542 ± 0.093 | 0.691 ± 0.026 | 0.715 ± 0.033 |
| lgbm_accgyro_hz20 | 0.457 ± 0.068 | 0.552 ± 0.101 | 0.667 ± 0.049 | 0.706 ± 0.044 |
| lgbm_acc_hz100 | 0.486 ± 0.042 | 0.557 ± 0.105 | 0.672 ± 0.050 | 0.719 ± 0.038 |
| lgbm_accgyro_hz100 | 0.457 ± 0.045 | 0.535 ± 0.104 | 0.644 ± 0.057 | 0.722 ± 0.023 |

**Set C** (8 classes, 300 test windows each)

| method | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|
| zeroshot (k=0) | 0.295 | | | |
| probe | 0.398 ± 0.026 | 0.459 ± 0.036 | 0.486 ± 0.029 | 0.483 ± 0.018 |
| full | 0.385 ± 0.039 | 0.434 ± 0.023 | 0.438 ± 0.028 | 0.443 ± 0.051 |
| random | 0.241 ± 0.039 | 0.294 ± 0.029 | 0.318 ± 0.023 | 0.313 ± 0.027 |
| random_gyro | 0.281 ± 0.049 | 0.326 ± 0.031 | 0.331 ± 0.027 | 0.312 ± 0.039 |
| lgbm_acc_hz20 | 0.326 ± 0.046 | 0.381 ± 0.038 | 0.447 ± 0.046 | 0.486 ± 0.016 |
| lgbm_accgyro_hz20 | 0.266 ± 0.068 | 0.333 ± 0.057 | 0.427 ± 0.037 | 0.503 ± 0.018 |
| lgbm_acc_hz100 | 0.317 ± 0.060 | 0.370 ± 0.060 | 0.406 ± 0.041 | 0.509 ± 0.028 |
| lgbm_accgyro_hz100 | 0.273 ± 0.023 | 0.371 ± 0.048 | 0.413 ± 0.048 | 0.501 ± 0.030 |

**Probe (best pretrained at every k) minus each reference.** Values are macro-F1, paired by seed, mean ± std, n=5.

| Set | k | − random | − random_gyro | − lgbm_acc_hz20 | − lgbm_accgyro_hz100 |
|---|---|---|---|---|---|
| B | 1 | +0.181 ± 0.032 | +0.112 ± 0.068 | +0.071 ± 0.096 | +0.071 ± 0.071 |
| B | 2 | +0.235 ± 0.054 | +0.183 ± 0.052 | +0.060 ± 0.080 | +0.067 ± 0.101 |
| B | 5 | +0.170 ± 0.089 | +0.118 ± 0.082 | −0.072 ± 0.038 | −0.025 ± 0.057 |
| B | 10 | +0.150 ± 0.062 | +0.095 ± 0.062 | −0.039 ± 0.043 | −0.046 ± 0.030 |
| C | 1 | +0.157 ± 0.060 | +0.117 ± 0.062 | +0.072 ± 0.063 | +0.125 ± 0.042 |
| C | 2 | +0.166 ± 0.047 | +0.134 ± 0.032 | +0.078 ± 0.047 | +0.088 ± 0.022 |
| C | 5 | +0.169 ± 0.029 | +0.155 ± 0.037 | +0.039 ± 0.048 | +0.073 ± 0.063 |
| C | 10 | +0.170 ± 0.036 | +0.172 ± 0.027 | −0.002 ± 0.011 | −0.017 ± 0.029 |

Per-class recall and the k=10, seed 0 confusion matrices of every method are in `results/shl_fewshot.md`. Plots:
`results/shl_fewshot_B.png` and `results/shl_fewshot_C.png`.

### Deviations recorded before the migration (from `COMMANDS.md`)

- `run_zeroshot.sh` failed only at the final cross-check (`evaluate_custom.py:48`, `torch.load` without
  `map_location`). The fix was to add the `run_cpu.py` launcher and rerun only the remaining part.
- The wall time of the timing run `C_k10_s0_full` is invalid because the Mac slept from 19:19 to 21:36 (`pmset -g log`).
- Fine-tuning used MPS, not CPU. A 5-epoch check against CPU gave the same loss to 4 decimals and 99.7% identical
  predictions (`results/fewshot/validate_mps/`).
- From run `B_k1_s0_full` on, `finetune_seeded.py` keeps the best-loss checkpoint in memory instead of writing
  275 MB per save. Checked on `B_k1_s0_probe`: prediction agreement 1.0, loss difference 0.0
  (`validate_mps/B_k1_s0_probe_inmem.npz`).
- Upstream `finetune_custom.py` has no seed argument. `finetune_seeded.py` seeds python, numpy and torch before running
  it.

### Migration from `/Users/ms/Projects/waiis/UniMTS2` (2026-10-23)

**Commands run:**

```sh
cd UniMTS2 && git diff > experiments/01_unimts-fewshot_shl/patches/UniMTS.patch     # 292 lines, 7 files
git clone https://github.com/xiyuanzh/UniMTS.git ../ext/UniMTS && git checkout 656fcda && git apply UniMTS.patch
#   -> git diff in the clone is byte-identical to the patch; all 21 tracked files equal UniMTS2's (cmp)
mv UniMTS2/shl-data            <data>/raw/shl
mv UniMTS2/UniMTS_data         <data>/raw/unimts/UniMTS_data
mv UniMTS2/checkpoint          <data>/checkpoints/unimts/checkpoint
mv /Users/ms/data/shl_unimts   <data>/processed/shl_u1_hips_w10s
cp -p UniMTS2/config.json; cp -Rp UniMTS2/.cache  -> <data>/checkpoints/unimts/   (+ symlink UniMTS_data)
mv <processed>/shl_u1_hips_w10s/{logs/fewshot,logs/run_fewshot.out,window_counts.csv,windows_log.txt} -> logs/   (fewshot/ later moved out, see Logs)
cp -p experiments/shl/{COMMANDS.md,*.py,*.sh} -> .;  {requirements-lgbm.txt,lgbm-env-lock.txt} -> env/
rsync results/: *.npz (345) -> <data>/processed/01_unimts-fewshot_shl/, everything else (13) -> results/
conda env export -n unimts > env/unimts.yml; conda list --explicit > env/unimts-lock.txt; pip freeze > env/unimts-pip-freeze.txt
conda create -n tmd-lgbm --file env/lgbm-env-lock.txt     # explicit lock identical to the old .env-lgbm
conda env export -n tmd-lgbm > env/tmd-lgbm.yml; pip freeze > env/tmd-lgbm-pip-freeze.txt
```

**Logs.** Only the logs this README refers to are in git:

- `logs/run_fewshot.out`
- `logs/window_counts.csv`
- `logs/windows_log.txt`
- `results/fewshot/validate_mps/*.log`

The other 162 were moved to `<data>/processed/01_unimts-fewshot_shl/logs/`:

- `fewshot/<run>.log`: one per UniMTS run, 160 files
- `zeroshot_run.log`: was `results/zeroshot_run.log`
- `timing_C_k10_s0_full.log`: was `results/fewshot/`

Not migrated, left in UniMTS2: `UniMTS_data.zip`, `.idea/`, `__pycache__/`, `experiments/shl/.env-lgbm/` (replaced by
`tmd-lgbm`).

`/Users/ms/data/unimts/hf/` (an older download of the HF repo) was left untouched.

**Path changes.** Line numbers are in the new files; "was" gives the original.

| File:line | Change |
|---|---|
| `shl_common.py:1-17` | was `:1-5`. The two absolute paths `RAW` and `OUT` are replaced by paths derived from the project root: `DATA`, `EXT`, `UNIMTS_HOME`, `CKPT`, `SHL_RAW`, `OUT`, `RAW`, `RUNS`, `LOGS`. |
| `cache_raw.py:8-11` | was `:8-14`. `SRC` and `DST` are absolute paths → `from shl_common import SHL_RAW as SRC, RAW as DST`. |
| `inspect_shl.py:3,8,37` | `window_counts.csv` → `logs/` instead of the data dir (imports `LOGS`). |
| `make_windows.py:5-6` | Docstring paths. |
| `make_windows.py:13` | Imports `LOGS` and `EXT`. |
| `make_windows.py:148` | `sys.path` repo root `__file__/../..` → `EXT`. |
| `make_windows.py:155` | `windows_log.txt` → `logs/`. |
| `make_kshot.py:3` | Docstring path. |
| `zeroshot_eval.py:2-3` | Docstring. |
| `zeroshot_eval.py:6-7` | `sys.path.insert(0, os.getcwd())` → `EXT`. |
| `zeroshot_eval.py:16` | `DATA` → `OUT/'hz20'`. |
| `zeroshot_eval.py:24` | `--checkpoint` default `./checkpoint/UniMTS.pth` → `CKPT`. |
| `run_cpu.py:3,7,10,12` | `sys.path` and `runpy` target: cwd → `EXT`. |
| `finetune_seeded.py:9-10,17` | Docstring and imports. |
| `finetune_seeded.py:101` | `sys.path` cwd → `EXT`. |
| `finetune_seeded.py:105` | `runpy.run_path('finetune_custom.py')` → `EXT/finetune_custom.py`. The temporary `./checkpoint/<run_tag>` stays relative to cwd, which is now `<data>/checkpoints/unimts`. |
| `make_table.py:2-3` | Docstring. |
| `make_table.py:6,13,17-19` | **New required `--runs DIR`** (the `.npz` root) replaces the hard-coded input. |
| `make_table.py:27` | Label JSON from `OUT`. |
| `make_table.py:38,42` | Reads `{runs}/fewshot/runs` and `{runs}/zeroshot`. Outputs stay in `results/`. |
| `make_zeroshot_table.py:1-3` | Docstring. |
| `make_zeroshot_table.py:6,10,12-17` | **New required `--runs DIR`**. `--pick-best` moves to argparse. |
| `make_zeroshot_table.py:28` | Reads `{runs}/zeroshot/*.npz`. |
| `make_zeroshot_table.py:32` | Label JSON from `OUT`. |
| `make_zeroshot_table.py:42` | `a.pick_best`. |
| `make_zeroshot_table.py:48,60` | `zeroshot.{csv,md}` → `results/`, was `results/zeroshot/..`, so the location is unchanged. |
| `baseline_lgbm.py:5-6` | Docstring. |
| `baseline_lgbm.py:8,12,14-16` | `D` → `shl_common.OUT`. Output dir is the new optional `--out`, default `RUNS/fewshot/runs`; it was `results/fewshot/runs`. |
| `run_zeroshot.sh:2` | Usage line. |
| `run_zeroshot.sh:6-11` | `cd ../..` (repo root) → derive `DATA`, `D` and `RUNS`, then `cd <data>/checkpoints/unimts`. |
| `run_zeroshot.sh:12` | `PY="conda run -n unimts python"` → the direct interpreter (see Setup). |
| `run_zeroshot.sh:13` | `OUT` → `$RUNS/zeroshot`. |
| `run_zeroshot.sh:18,23,27,33,38` | `experiments/shl/` → `$HERE/`. `--runs $RUNS` is passed to `make_zeroshot_table.py`. |
| `run_zeroshot.sh:34-35` | `/Users/ms/data/shl_unimts` → `$D`. |
| `run_fewshot.sh:2` | Usage line. |
| `run_fewshot.sh:7-11` | Same header as `run_zeroshot.sh`. |
| `run_fewshot.sh:14-15` | `D`, `RES` and `LOG` → `$RUNS/fewshot/runs` and `$RUNS/logs/fewshot`. |
| `run_fewshot.sh:25` | `experiments/shl/` → `$HERE/`. |

These files are unchanged: `check_resample.py`, `COMMANDS.md` (historical record; its paths are the old ones), and
`env/requirements-lgbm.txt` (its comment still names `.env-lgbm`).

**Other changes:**
- `../../.gitignore`: added exceptions so `experiments/*/env/`, `experiments/*/logs/**` and
  `experiments/*/results/**/*.log` are tracked. Without them, `ENV/` matches `env/` on case-insensitive macOS, and
  `*.log` is ignored. `CLAUDE.local.md` is now ignored.
- `../../LICENSES.md`: added SHL and UniMTS entries.

## Open questions

1. **UniMTS licence source.** The GitHub repo has no LICENSE file and no licence metadata. Apache-2.0 comes from the
   Hugging Face model card.
2. **Naming of the processed data.** It is one folder, `shl_u1_hips_w10s/`, because it holds `raw` (100 Hz cache),
   `hz20`, `hz100` and `kshot`, and the scripts address the rates as subfolders. Splitting it per rate would mean
   changing every `{OUT}/{hz}` path in `make_windows.py`, `make_kshot.py` and `baseline_lgbm.py`.
3. **Size of `results/shl_fewshot.csv`.** It holds aggregates only. Per-seed numbers are recomputed from the `.npz`
   files outside git.
4. **Missing metrics.** Model size and latency per window (median and p95, single thread) are not measured yet, as
   CLAUDE.md requires.
5. ~~Should the conclusion become a `DECISIONS.md` entry?~~ Answered: yes. The draft entry is "UniMTS is not adopted
   as the on-phone model". The size used there was measured on the checkpoint:
   - IMU encoder (`acc`): 5,179,900 params (4.94 Mi) = 20.7 MB fp32
   - CLIP text encoder: 63.1M params
   - checkpoint total: 68.6M params, 274 MB

## Conclusion

**Draft — not approved** (copied from `results/conclusion.md`, written before the migration)

Pretrained UniMTS clearly beats the same architecture trained from random init. The linear probe was the best
pretrained variant at every k in both sets. Paired by seed, it scores +0.15 to +0.24 macro-F1 above random in set B
and +0.16 to +0.17 in set C. Adding gyro to random init (random_gyro) closes at most about a third of that gap, and
none of it in set C at k=10. Full fine-tuning is worse than the probe everywhere, at 200 epochs on so few windows.

Against LightGBM on the same windows, pretraining helps only at the smallest k:

| Set | k | Probe minus lgbm_acc_hz20 | Seeds won by probe |
|---|---|---|---|
| B | 1 | +0.07 | 4/5 |
| B | 2 | +0.06 | 4/5 |
| B | 5 | −0.07 | 0/5 |
| B | 10 | −0.04 | 1/5 |
| C | 1 | +0.07 | 4/5 |
| C | 2 | +0.08 | 5/5 |
| C | 5 | +0.04 | 4/5 |
| C | 10 | 0.00 | 2/5 |

At k=10, the best LightGBM variants reach 0.72 in set B and 0.50–0.51 in set C, which matches or beats the probe.

The pretrained advantage comes from the motion classes: walk/run in B, and walk, run and bike in C. For example, in C
at k=10 the probe recalls run 0.98 and bike 0.78. LightGBM gets 0.76 and 0.75 for those classes.

The pretrained model is weak exactly where we care most, on vehicles.

- **Probe:** it confuses vehicles with still and with each other. In C at k=10 its recall is car 0.30, bus 0.29,
  train 0.24 and subway 0.18.
- **LightGBM (C, k=10):** recall is car 0.36–0.40, bus 0.28–0.45, train 0.23–0.28 and subway 0.36–0.40.
- **Car windows in set B:** the share predicted as "vehicle" is 0.72 for the probe and 0.88 for full fine-tuning at
  k=10. For LightGBM it is 0.90–0.91.

Zero-shot (k=0) is a reasonable start only for still and walk/run. It reaches 0.54 in B and 0.30 in C, with vehicle
recall of 0.25 in B and 0.00–0.12 per vehicle class in C.

These numbers have three limits:

- All results come from one user (SHL User 1), one placement (hips) and a split by day. Train and test therefore share
  the person, the phone and the routes, so every number is optimistic for new users.
- Each k uses only 5 seeds, and the paired standard deviations (0.01–0.10) are often as large as the differences
  against LightGBM.
- Probe and full fine-tuning use only the accelerometer, because the released checkpoint has no gyro weights.

Set A (still/walk/run) was evaluated zero-shot only. It was optional for few-shot and was not run.
