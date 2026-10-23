"""Step 6: few-shot results table, plots and confusion matrices.
Usage: experiments/shl/.env-lgbm/bin/python experiments/shl/make_table.py
Reads results/fewshot/runs/*.npz (UniMTS + LightGBM) and results/zeroshot/{set}_test_j1_w2.npz (k=0).
Writes results/shl_fewshot.csv, results/shl_fewshot.md, results/shl_fewshot_{set}.png
"""
import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f'{HERE}/results'
D = '/Users/ms/data/shl_unimts'
METHODS = ['zeroshot', 'probe', 'full', 'random', 'random_gyro',
           'lgbm_acc_hz20', 'lgbm_accgyro_hz20', 'lgbm_acc_hz100', 'lgbm_accgyro_hz100']
PRETRAINED = ['probe', 'full']
LIKE_FOR_LIKE = 'lgbm_acc_hz20'


def names(s):
    return [' '.join(v) for v in json.load(open(f'{D}/hz20/{s}/labels_w1.json'))['label_dictionary'].values()]


def md_table(df, index=True):
    df = df.reset_index() if index else df
    fmt = lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |', '|' + '---|' * len(df.columns)] +
                     ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in df.itertuples(index=False)])


runs = []
for f in glob.glob(f'{RES}/fewshot/runs/*.npz'):
    m = re.match(r'([A-C])_k(\d+)_s(\d)_(.+)\.npz', os.path.basename(f))
    runs.append((m[1], int(m[2]), int(m[3]), m[4], f))
for s in ['B', 'C']:
    runs.append((s, 0, 0, 'zeroshot', f'{RES}/zeroshot/{s}_test_j1_w2.npz'))

rows, preds = [], {}
for s, k, seed, method, f in runs:
    d = np.load(f)
    y, p, cls = d['y'].astype(int), d['pred'].astype(int), names(s)
    cm = confusion_matrix(y, p, labels=range(len(cls)))
    preds[(s, k, seed, method)] = (y, p)
    rows.append({'set': s, 'method': method, 'k': k, 'seed': seed, 'macro_f1': f1_score(y, p, average='macro'),
                 **{f'rec_{c}': r for c, r in zip(cls, cm.diagonal() / cm.sum(1))}})
df = pd.DataFrame(rows)

md = ['# Few-shot UniMTS vs random init vs LightGBM on SHL hips (User 1)', '',
      '- Test: 73 days not in the pool, capped at 300 windows per SHL class (seed 0). Pool: days 190517 + 250517.',
      '- k windows per class drawn from the pool per seed (make_kshot.py); every method uses the same windows.',
      '- UniMTS: joint 1, wording w2, 20 Hz, 200 epochs, best training loss; probe/full use the released checkpoint, '
      'which is acc-only (--gyro 0); random is the same architecture from random init (acc only), random_gyro adds gyro '
      '(--gyro 1, 132 inputs). Fine-tuning ran on the Apple GPU (MPS); 5-epoch check vs CPU: same loss to 4 decimals, '
      '99.7% identical predictions.',
      '- k=0 is zero-shot (pretrained, text labels, no training).',
      '- LightGBM: |acc| (and |gyro|) features, 20 Hz or 100 Hz copies of the same windows; lgbm_acc_hz20 is like-for-like with UniMTS.',
      '- Split is by day for a single user, so all numbers are optimistic compared with new users.', '']
if os.path.exists(f'{RES}/conclusion.md'):   # hand-written interpretation, kept in its own file
    md += [open(f'{RES}/conclusion.md').read().strip(), '']
out = []
for s in ['B', 'C']:
    sub = df[df.set == s]
    if sub.empty:
        continue
    cls = names(s)
    n_test = np.bincount(preds[(s, 0, 0, 'zeroshot')][0], minlength=len(cls))
    g = sub.groupby(['method', 'k'])
    t = g.macro_f1.agg(['mean', 'std', 'count']).rename(columns={'mean': 'f1_mean', 'std': 'f1_std', 'count': 'n_seeds'})
    t = t.join(g[[f'rec_{c}' for c in cls]].mean()).reset_index()
    t['method'] = pd.Categorical(t.method, METHODS, ordered=True)
    t = t.sort_values(['method', 'k'])
    t.insert(0, 'set', s)
    t['macro_f1'] = [f'{m:.3f} ± {0 if np.isnan(sd) else sd:.3f}' for m, sd in zip(t.f1_mean, t.f1_std)]
    out.append(t.assign(**{f'n_test_{c}': n for c, n in zip(cls, n_test)}))
    md += [f'## Set {s}', '', 'n_test per class: ' + ', '.join(f'{c} {n}' for c, n in zip(cls, n_test)), '',
           md_table(t[['method', 'k', 'n_seeds', 'macro_f1'] + [f'rec_{c}' for c in cls]], index=False), '']

    # paired by seed: best pretrained (probe or full, chosen per k by mean F1) minus random / LightGBM
    piv = sub.pivot_table(index=['k', 'seed'], columns='method', values='macro_f1')
    cmp = []
    for k in sorted(sub.k.unique()):
        if k == 0 or k not in piv.index.get_level_values(0):
            continue
        pk = piv.loc[k]
        avail = [m for m in PRETRAINED if m in pk and pk[m].notna().any()]
        if not avail:
            continue
        best = max(avail, key=lambda m: pk[m].mean())
        r = {'k': k, 'best_pretrained': best, 'f1': pk[best].mean()}
        for ref in ['random', 'random_gyro', LIKE_FOR_LIKE, 'lgbm_accgyro_hz100']:
            if ref in pk and pk[ref].notna().any():
                dlt = (pk[best] - pk[ref]).dropna()
                r[f'minus_{ref}'] = f'{dlt.mean():+.3f} ± {dlt.std():.3f} (n={len(dlt)})'
        cmp.append(r)
    if cmp:
        md += ['Best pretrained UniMTS minus each reference, macro-F1, paired by seed (mean ± std):', '',
               md_table(pd.DataFrame(cmp), index=False), '']

    # plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    zs = t[t.method == 'zeroshot'].f1_mean.iloc[0]
    for method in [m for m in METHODS if m != 'zeroshot']:
        tm = t[t.method == method]
        if tm.empty:
            continue
        ks, mu, sd = list(tm.k), list(tm.f1_mean), list(tm.f1_std.fillna(0))
        if method in PRETRAINED:
            ks, mu, sd = [0] + ks, [zs] + mu, [0] + sd
        ax.errorbar(ks, mu, yerr=sd, marker='o', capsize=3, label=method,
                    linestyle='-' if not method.startswith('lgbm') else '--')
    ax.axhline(1 / len(cls), color='grey', lw=0.8, ls=':', label='chance')
    ax.set(xlabel='k (windows per class); k=0 = zero-shot', ylabel='macro-F1 (mean ± std over seeds)',
           title=f'SHL set {s} ({len(cls)} classes)', xticks=[0, 1, 2, 5, 10], ylim=(0, 1))
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f'{RES}/shl_fewshot_{s}.png', dpi=130)
    plt.close(fig)
    md += [f'![macro-F1 vs k, set {s}](shl_fewshot_{s}.png)', '']

    for method in METHODS[1:]:
        if (s, 10, 0, method) in preds:
            y, p = preds[(s, 10, 0, method)]
            cm = pd.DataFrame(confusion_matrix(y, p, labels=range(len(cls))), index=[f'true {c}' for c in cls], columns=cls)
            md += [f'Confusion matrix, set {s}, {method}, k=10, seed 0 (rows true, cols predicted)', '', md_table(cm), '']

if out:
    pd.concat(out).drop(columns='macro_f1').to_csv(f'{RES}/shl_fewshot.csv', index=False, float_format='%.4f')
open(f'{RES}/shl_fewshot.md', 'w').write('\n'.join(md))
print(f'wrote {RES}/shl_fewshot.md, .csv, .png ({len(df)} runs)')
