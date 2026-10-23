#!/usr/bin/env bash
# Step 3: zero-shot UniMTS on SHL (20 Hz windows from make_windows.py). Run: bash experiments/shl/run_zeroshot.sh
# Stage 1: set A, joints 1 2 5 6 x wordings w1 w2, on fewshot_pool (selection) and test (report).
# Stage 2: sets B and C with the joint/wording that has the best macro-F1 on the POOL for set A.
set -euo pipefail
cd "$(dirname "$0")/../.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
PY="conda run -n unimts python"
OUT=experiments/shl/results/zeroshot

for split in fewshot_pool test; do
  for j in 1 2 5 6; do
    for w in w1 w2; do
      $PY experiments/shl/zeroshot_eval.py --set A --split $split --joint $j --wording $w --out $OUT
    done
  done
done

read -r J W < <($PY experiments/shl/make_zeroshot_table.py --pick-best)
echo "best on pool (set A): joint $J wording $W"
for s in B C; do
  for split in fewshot_pool test; do
    $PY experiments/shl/zeroshot_eval.py --set $s --split $split --joint $J --wording $W --out $OUT
  done
done

# cross-check: unmodified evaluate_custom.py on one variant must print the same acc / F1
# (run_cpu.py only adds map_location='cpu' to torch.load; evaluate_custom.py:48 lacks it)
$PY experiments/shl/run_cpu.py evaluate_custom.py --checkpoint ./checkpoint/UniMTS.pth \
  --X_path /Users/ms/data/shl_unimts/hz20/A/X_test.npy --y_path /Users/ms/data/shl_unimts/hz20/A/y_test.npy \
  --config_path /Users/ms/data/shl_unimts/hz20/A/labels_$W.json --joint_list $J --original_sampling_rate 20 \
  | grep -E "^acc" | tee $OUT/crosscheck_evaluate_custom_A_test_j${J}_$W.txt

$PY experiments/shl/make_zeroshot_table.py
