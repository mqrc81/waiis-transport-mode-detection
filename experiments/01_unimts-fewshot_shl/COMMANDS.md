# Commands run (SHL x UniMTS)

All from `experiments/shl/`, env `unimts`. Output data under `<data>/processed/shl_u1_hips_w10s/` (`<data>` =
`/Users/ms/Projects/waiis/data`; `OUT` in `shl_common.py:12`). The commands originally wrote to
`/Users/ms/data/shl_unimts/`, which was moved there (README, migration); paths below updated 2026-10-28 (experiment
04). The logs `logs/run_fewshot.out`, `window_counts.csv` and `windows_log.txt` were later moved to this folder's
`logs/`.

## Step 0 / 1: inspection
```sh
conda run -n unimts python cache_raw.py 010317          # timing test, one recording
conda run -n unimts python cache_raw.py > <data>/processed/shl_u1_hips_w10s/raw/cache_log.txt 2>&1   # all recordings -> raw/*.npz
conda run -n unimts python inspect_shl.py 010317        # sanity numbers + window_counts.csv
conda run -n unimts python check_resample.py 010317
conda run -n unimts python check_resample.py 010617
```
conda run -n unimts python inspect_shl.py 010317        # rerun after full cache -> all-day window table
# ad-hoc checks (python -c): unlabeled days 180417/200317; dt/NaN over all 90 recordings
```

## Step 2: windows (data moved to UniMTS2/shl-data; cache verified: first/last timestamps of all 90 recordings match)
```sh
conda run -n unimts python make_windows.py              # -> <data>/processed/shl_u1_hips_w10s/{hz20,hz100}/{A,B,C}/, windows_log.txt
conda run -n unimts python inspect_shl.py 010317        # after shl_common refactor: totals unchanged
```

## Step 3: zero-shot
```sh
WANDB_MODE=disabled HF_HUB_OFFLINE=1 conda run -n unimts python experiments/shl/zeroshot_eval.py \
  --set A --split test --joint 5 --wording w1 --out <scratch>   # timing test (from repo root): 77 s
bash experiments/shl/run_zeroshot.sh > experiments/shl/results/zeroshot_run.log 2>&1   # all of Step 3
```
# run_zeroshot.sh failed only at the final cross-check (evaluate_custom.py:48 torch.load has no map_location);
# added run_cpu.py launcher, then re-ran only the remaining part:
conda run -n unimts python experiments/shl/run_cpu.py evaluate_custom.py --checkpoint ./checkpoint/UniMTS.pth \
  --X_path <data>/processed/shl_u1_hips_w10s/hz20/A/X_test.npy --y_path <data>/processed/shl_u1_hips_w10s/hz20/A/y_test.npy \
  --config_path <data>/processed/shl_u1_hips_w10s/hz20/A/labels_w2.json --joint_list 1 --original_sampling_rate 20
conda run -n unimts python experiments/shl/make_zeroshot_table.py
```

## Step 4: few-shot (prep + timing)
```sh
conda run -n unimts python make_kshot.py                # -> <data>/processed/shl_u1_hips_w10s/kshot/{A,B,C}/k{1,2,5,10}_s{0..4}/
# timing run (from repo root); INVALID wall time: Mac slept 19:19-21:36 during it (pmset -g log)
WANDB_MODE=disabled HF_HUB_OFFLINE=1 conda run -n unimts python experiments/shl/finetune_seeded.py --seed 0 \
  --out experiments/shl/results/fewshot/timing_C_k10_s0_full.npz -- --mode full --run_tag shl_timing_C_k10_s0_full \
  --num_class 8 --joint_list 1 --original_sampling_rate 20 \
  --X_train_path $D/kshot/C/k10_s0/X_hz20.npy --y_train_path $D/kshot/C/k10_s0/y.npy \
  --X_test_path $D/hz20/C/X_test.npy --y_test_path $D/hz20/C/y_test.npy --config_path $D/hz20/C/labels_w2.json
# micro-benchmarks (scratchpad, not part of results): bench.py (CPU threads 2/4/8/10, random data),
# bench_real.py (real data, flush_denormal), bench_mps.py (MPS), gyro=1 model build check (python -c)
pmset -g log | grep ...                                  # sleep/wake events during the timing run
```
# MPS validation (5 epochs, set C k10 s0 full): cpu vs mps vs mps again
for run in cpu mps mps2; do caffeinate -i $PY experiments/shl/finetune_seeded.py --seed 0 --device ${run%2} \
  --out experiments/shl/results/fewshot/validate_mps/C_k10_s0_full_5ep_$run.npz -- --mode full --run_tag shl_val_$run \
  --num_epochs 5 ...same data args as the timing run...; done
# full grid (resumable)
mkdir -p <data>/processed/shl_u1_hips_w10s/logs/fewshot
caffeinate -is bash experiments/shl/run_fewshot.sh > <data>/processed/shl_u1_hips_w10s/logs/run_fewshot.out 2>&1
```
# finetune_seeded.py: best-loss checkpoint kept in memory from run B_k1_s0_full on (B_k1_s0_probe was written to disk);
# check that results are identical (pred agreement 1.0, loss diff 0.0):
caffeinate -i $PY experiments/shl/finetune_seeded.py --seed 0 --device mps \
  --out experiments/shl/results/fewshot/validate_mps/B_k1_s0_probe_inmem.npz -- --mode probe ...B k1 s0 args...
```

## Step 5: LightGBM baseline (separate env, unimts unchanged)
```sh
conda create -y -q -p experiments/shl/.env-lgbm -c conda-forge --file experiments/shl/requirements-lgbm.txt
conda list -p experiments/shl/.env-lgbm --explicit > experiments/shl/lgbm-env-lock.txt
experiments/shl/.env-lgbm/bin/python experiments/shl/baseline_lgbm.py      # 8 s, all variants
```

## Step 6: results (regenerates everything from the run files)
```sh
experiments/shl/.env-lgbm/bin/python experiments/shl/make_table.py
# grid finished 2026-10-22 04:12: 160/160 UniMTS runs, 0 FAILED (log <data>/processed/shl_u1_hips_w10s/logs/run_fewshot.out)
# the conclusion paragraph is hand-written in results/conclusion.md and included in shl_fewshot.md by make_table.py
```

## Post-run checks
```sh
grep -c FAILED <data>/processed/shl_u1_hips_w10s/logs/run_fewshot.out                     # 0
ls experiments/shl/results/fewshot/runs | grep -vc lgbm                             # 160
# car / bus / train / subway recall as "vehicle" in set B, and seeds where the probe beats lgbm_acc_hz20: inline python
# over results/fewshot/runs/*.npz + <data>/processed/shl_u1_hips_w10s/hz20/B/meta_test.csv (shl_label)
```
