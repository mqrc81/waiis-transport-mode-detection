"""Step 3: zero-shot UniMTS on SHL windows; same loop as evaluate_custom.py:50-71, but saves predictions.
Run from the repo root (imports data.py / contrastive.py).
Usage: python experiments/shl/zeroshot_eval.py --set A --split test --joint 5 --wording w1 --out results/zeroshot
"""
import argparse, os, sys
sys.path.insert(0, os.getcwd())   # repo root
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from data import load_custom_data
from contrastive import ContrastiveModule

DATA = '/Users/ms/data/shl_unimts/hz20'

p = argparse.ArgumentParser()
p.add_argument('--set', required=True)
p.add_argument('--split', required=True, choices=['test', 'fewshot_pool'])
p.add_argument('--joint', type=int, required=True)
p.add_argument('--wording', required=True, choices=['w1', 'w2'])
p.add_argument('--out', required=True)
p.add_argument('--checkpoint', default='./checkpoint/UniMTS.pth')
a = p.parse_args()

d = f'{DATA}/{a.set}'
inputs, masks, labels, label_list, all_text = load_custom_data(
    f'{d}/X_{a.split}.npy', f'{d}/y_{a.split}.npy', f'{d}/labels_{a.wording}.json', [a.joint], 20, padding_size=200, split='test')
loader = DataLoader(TensorDataset(inputs, masks, labels), batch_size=64, shuffle=False)

args = argparse.Namespace(gyro=0, stft=0, stage='evaluation')
model = ContrastiveModule(args).to('cpu')
model.model.load_state_dict(torch.load(a.checkpoint, map_location='cpu'))
model.eval()

preds, logits = [], []
with torch.no_grad():
    for input, mask, label in loader:
        b, t, c = input.shape
        indices = np.array([range(i, i + 3) for i in range(0, c, 6)]).flatten()   # --gyro 0: acc only
        input = input[:, :, indices]
        b, t, c = input.shape
        input = input.reshape(b, t, 22, -1).permute(0, 3, 1, 2).unsqueeze(-1)
        logits_per_imu, _ = model(input, all_text)
        logits.append(logits_per_imu.numpy())
        preds.append(logits_per_imu.argmax(-1).numpy())

preds, logits, y = np.concatenate(preds), np.concatenate(logits), labels.numpy().astype(int)
os.makedirs(a.out, exist_ok=True)
name = f'{a.set}_{a.split}_j{a.joint}_{a.wording}'
np.savez(f'{a.out}/{name}.npz', y=y, pred=preds, logits=logits, labels=np.array(label_list))
print(f'{name}: acc {np.mean(preds == y):.4f} macro-F1 {f1_score(y, preds, average="macro"):.4f}')
