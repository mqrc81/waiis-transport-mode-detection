"""Step 3 table: macro-F1, per-class recall and confusion matrices from <runs>/zeroshot/*.npz.
Usage: python make_zeroshot_table.py --runs <data>/processed/01_unimts-fewshot_shl [--pick-best]
Writes results/zeroshot.csv and results/zeroshot.md.
--pick-best prints "<joint> <wording>" with the best macro-F1 on the set-A fewshot_pool and exits.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from shl_common import OUT

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
p = argparse.ArgumentParser()
p.add_argument('--runs', required=True, help='directory with zeroshot/*.npz')
p.add_argument('--pick-best', action='store_true')
a = p.parse_args()


def md_table(df, index=True):
    """Markdown table without the tabulate dependency (unimts env stays unchanged)."""
    df = df.reset_index() if index else df
    fmt = lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |', '|' + '---|' * len(df.columns)] +
                     ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in df.itertuples(index=False)])


rows, cms = [], {}
for f in sorted(glob.glob(f'{a.runs}/zeroshot/*.npz')):
    d = np.load(f)
    s, split, j, w = os.path.basename(f)[:-4].replace('fewshot_pool', 'fewshotpool').split('_')
    y, p, names = d['y'], d['pred'], list(d['labels'])
    short = list(json.load(open(f'{OUT}/hz20/{s}/labels_w1.json'))['label_dictionary'].values())
    short = [' '.join(v) for v in short]   # w1 names as column keys so both wordings share columns
    cm = confusion_matrix(y, p, labels=range(len(names)))
    rec = cm.diagonal() / cm.sum(1)
    rows.append({'set': s, 'split': split.replace('fewshotpool', 'fewshot_pool'), 'joint': int(j[1:]), 'wording': w,
                 'n': len(y), 'acc': (p == y).mean(), 'macro_f1': f1_score(y, p, average='macro'),
                 **{f'rec_{n}': r for n, r in zip(short, rec)}})
    cms[(s, rows[-1]['split'], j, w)] = pd.DataFrame(cm, index=[f'true {n}' for n in names], columns=names)
df = pd.DataFrame(rows)

if a.pick_best:
    b = df[(df.set == 'A') & (df.split == 'fewshot_pool')].sort_values(['macro_f1', 'joint', 'wording'],
                                                                        ascending=[False, True, True]).iloc[0]
    print(b.joint, b.wording)
    sys.exit()

df.to_csv(f'{RES}/zeroshot.csv', index=False, float_format='%.4f')
md = ['# Zero-shot UniMTS on SHL hips (User 1), 20 Hz, acc only (--gyro 0)', '',
      'Selection of joint/wording on fewshot_pool (set A), reported on test. Chance macro-F1 ~ 1/n_classes.', '']
for s in ['A', 'B', 'C']:
    sub = df[df.set == s].dropna(axis=1, how='all')
    if sub.empty:
        continue
    md += [f'## Set {s}', '', md_table(sub.sort_values(['split', 'joint', 'wording']), index=False), '']
    for (cs, split, j, w), cm in cms.items():
        if cs == s and split == 'test':
            md += [f'Confusion matrix, set {s}, test, joint {j[1:]}, {w} (rows true, cols predicted)', '',
                   md_table(cm), '']
open(f'{RES}/zeroshot.md', 'w').write('\n'.join(md))
print('\n'.join(md))
