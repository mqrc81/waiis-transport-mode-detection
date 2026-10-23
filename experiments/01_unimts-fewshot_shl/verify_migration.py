"""Step 3 of the migration: compare reruns from the new layout (<runs>/verify/) with the original run files and CSVs.
Usage: python verify_migration.py      (after the verify runs listed in README, Results > Verification)
"""
import glob, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from shl_common import HERE, RUNS

V = RUNS / 'verify'


def compare(old, new):
    a, b = np.load(old), np.load(new)
    assert np.array_equal(a['y'], b['y']), 'test labels differ'
    f1 = [f1_score(d['y'], d['pred'], average='macro') for d in (a, b)]
    r = {'f1_old': f1[0], 'f1_new': f1[1], 'pred_agree': np.mean(a['pred'] == b['pred'])}
    if 'logits' in a:
        r['max_abs_logit_diff'] = float(np.abs(a['logits'] - b['logits']).max())
    if 'losses' in a:
        r['final_loss_old'], r['final_loss_new'] = float(a['losses'][-1]), float(b['losses'][-1])
        r['max_abs_loss_diff'] = float(np.abs(a['losses'] - b['losses']).max())
    return r


zs = pd.read_csv(HERE / 'results' / 'zeroshot.csv')
fs = pd.read_csv(HERE / 'results' / 'shl_fewshot.csv')

print('== zero-shot, set B, test, joint 1, w2')
row = zs[(zs.set == 'B') & (zs.split == 'test') & (zs.joint == 1) & (zs.wording == 'w2')].iloc[0]
print(f'zeroshot.csv: acc {row.acc:.4f} macro_f1 {row.macro_f1:.4f}')
print(compare(RUNS / 'zeroshot' / 'B_test_j1_w2.npz', V / 'zeroshot' / 'B_test_j1_w2.npz'))

print('== few-shot, set B, k=1, seed 0, probe (MPS)')
row = fs[(fs.set == 'B') & (fs.method == 'probe') & (fs.k == 1)].iloc[0]
print(f'shl_fewshot.csv (mean over {row.n_seeds} seeds, not seed 0): f1_mean {row.f1_mean:.4f} f1_std {row.f1_std:.4f}')
print(compare(RUNS / 'fewshot' / 'runs' / 'B_k1_s0_probe.npz', V / 'fewshot' / 'B_k1_s0_probe.npz'))

print('== LightGBM, all variants')
new = sorted(glob.glob(f'{V}/lgbm/*.npz'))
same = [np.array_equal(np.load(f)['pred'], np.load(RUNS / 'fewshot' / 'runs' / os.path.basename(f))['pred']) for f in new]
print(f'{sum(same)}/{len(new)} runs with identical predictions')
row = fs[(fs.set == 'B') & (fs.method == 'lgbm_acc_hz20') & (fs.k == 10)].iloc[0]
print(f'shl_fewshot.csv B lgbm_acc_hz20 k=10 (mean over seeds): f1_mean {row.f1_mean:.4f}')
print('B_k10_s0_lgbm_acc_hz20:', compare(RUNS / 'fewshot' / 'runs' / 'B_k10_s0_lgbm_acc_hz20.npz',
                                          V / 'lgbm' / 'B_k10_s0_lgbm_acc_hz20.npz'))
