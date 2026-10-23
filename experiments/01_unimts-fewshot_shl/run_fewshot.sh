#!/usr/bin/env bash
# Step 4: few-shot UniMTS on SHL. Run: bash run_fewshot.sh   (resumable: finished runs are skipped)
# Sets B C x k 1 2 5 10 x seeds 0-4 x {probe, full, random, random_gyro}; joint 1, wording w2, 20 Hz, 200 epochs.
# Train = our pre-sampled k-shot file (make_kshot.py), no --k. Apple GPU via finetune_seeded.py --device mps.
# Keep the Mac plugged in with the lid open; caffeinate -i prevents idle sleep.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$(cd "$HERE/../../../data" && pwd)                 # data root, see CLAUDE.local.md; paths as in shl_common.py
D=$DATA/processed/shl_u1_hips_w10s
RUNS=$DATA/processed/$(basename "$HERE")
cd "$DATA/checkpoints/unimts"                          # upstream scripts expect checkpoint/, config.json, UniMTS_data in cwd
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY=/opt/homebrew/anaconda3/envs/unimts/bin/python     # direct, so logs are not buffered by conda run
RES=$RUNS/fewshot/runs
LOG=$RUNS/logs/fewshot
mkdir -p $RES $LOG

for seed in 0 1 2 3 4; do
  for set in B C; do
    for k in 1 2 5 10; do
      for method in probe full random random_gyro; do
        name=${set}_k${k}_s${seed}_${method}
        [ -f $RES/$name.npz ] && continue
        mode=${method%_gyro}; gyro=0; [ "$method" = random_gyro ] && gyro=1
        echo "$(date +%T) $name"
        caffeinate -i $PY $HERE/finetune_seeded.py --seed $seed --device mps --out $RES/$name.npz -- \
          --mode $mode --gyro $gyro --run_tag shl_$name --num_class $([ $set = B ] && echo 3 || echo 8) --joint_list 1 \
          --original_sampling_rate 20 --num_epochs 200 \
          --X_train_path $D/kshot/$set/k${k}_s${seed}/X_hz20.npy --y_train_path $D/kshot/$set/k${k}_s${seed}/y.npy \
          --X_test_path $D/hz20/$set/X_test.npy --y_test_path $D/hz20/$set/y_test.npy \
          --config_path $D/hz20/$set/labels_w2.json > $LOG/$name.log 2>&1 \
          && grep -E "^acc|TIMES" $LOG/$name.log || echo "FAILED $name (see $LOG/$name.log)"
      done
    done
  done
done
