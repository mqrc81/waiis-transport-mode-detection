"""Run the unmodified finetune_custom.py with fixed seeds, on CPU (or Apple GPU), and keep its test predictions.

- seeds python/numpy/torch before the script runs (it has no seed argument)
- torch.load gets map_location='cpu' (finetune_custom.py:60,110 lack it)
- sklearn.metrics.f1_score is wrapped to record (y_true, y_pred) of the final test evaluation
- times training epochs, checkpoint writes and test evaluation
- keeps the best-loss checkpoint in memory (same selection, no 275 MB writes); deletes ./checkpoint/<run_tag>/ afterwards
- --device mps: redirects the script's .to("cpu") calls (model, inputs, labels) to the Apple GPU
Usage (cwd shl_common.UNIMTS_HOME): python <exp>/finetune_seeded.py --seed S --out x.npz -- <finetune_custom.py args>
(finetune_custom.py is taken from shl_common.EXT; it needs checkpoint/, config.json and UniMTS_data in the cwd)
The finetune args must include a unique --run_tag and no --k (the k-shot file is the whole train set).
"""
import argparse, builtins, os, random, runpy, shutil, sys, time
import numpy as np
import torch
import sklearn.metrics
from shl_common import EXT

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--out', required=True)
p.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
a, rest = p.parse_known_args()
rest = [r for r in rest if r != '--']
assert '--k' not in rest and '--run_tag' in rest
run_tag = rest[rest.index('--run_tag') + 1]

random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)

timers = {'ckpt_save_s': 0.0, 'n_ckpt_saves': 0}
_save, _load = torch.save, torch.load
ckpt_dir = os.path.abspath(f'./checkpoint/{run_tag}')
in_memory = {}   # best-loss checkpoints of this run, kept in RAM instead of a 275 MB file per save


def timed_save(obj, f, *args, **kw):
    t0 = time.time()
    if isinstance(f, str) and os.path.abspath(f).startswith(ckpt_dir + os.sep):
        in_memory[os.path.abspath(f)] = {k: v.detach().clone() for k, v in obj.items()}
    else:
        _save(obj, f, *args, **kw)
    timers['ckpt_save_s'] += time.time() - t0
    timers['n_ckpt_saves'] += 1


def load(f, *args, **kw):
    if isinstance(f, str) and os.path.abspath(f) in in_memory:
        return in_memory[os.path.abspath(f)]
    return _load(f, *args, map_location='cpu', **{k: v for k, v in kw.items() if k != 'map_location'})


torch.save, torch.load = timed_save, load

if a.device == 'mps':
    _tensor_to, _module_to = torch.Tensor.to, torch.nn.Module.to

    def _redirect(args, kw):
        if args and isinstance(args[0], str) and args[0] == 'cpu':
            args = ('mps',) + tuple(args[1:])
        if kw.get('device') == 'cpu':
            kw['device'] = 'mps'
        return args, kw

    def tensor_to(self, *args, **kw):
        args, kw = _redirect(args, kw)
        return _tensor_to(self, *args, **kw)

    def module_to(self, *args, **kw):
        args, kw = _redirect(args, kw)
        return _module_to(self, *args, **kw)

    torch.Tensor.to, torch.nn.Module.to = tensor_to, module_to

captured = {}
_f1 = sklearn.metrics.f1_score


def f1_capture(y_true, y_pred, *args, **kw):
    captured['y'], captured['pred'] = np.asarray(y_true), np.asarray(y_pred)
    return _f1(y_true, y_pred, *args, **kw)


sklearn.metrics.f1_score = f1_capture

# epoch timestamps from the script's own "Epoch [i/N]" prints
marks, losses = [], []
_print = builtins.print


def stamped_print(*args, **kw):
    if args and isinstance(args[0], str) and args[0].startswith('Epoch ['):
        marks.append(time.time())
        losses.append(float(args[0].rsplit('Loss: ', 1)[1]))
    _print(*args, **kw)


builtins.print = stamped_print

sys.path.insert(0, str(EXT))
sys.argv = ['finetune_custom.py'] + rest
t_start = time.time()
try:
    runpy.run_path(os.path.join(EXT, 'finetune_custom.py'), run_name='__main__')
finally:
    builtins.print = _print
    shutil.rmtree(f'./checkpoint/{run_tag}', ignore_errors=True)
t_end = time.time()

# marks[0] is after epoch 1, so setup (data load, CLIP load) + epoch 1 sits before it
times = {'total_s': t_end - t_start, 'train_epochs_s': marks[-1] - marks[0] if marks else 0.0,
         'n_epochs_timed': max(len(marks) - 1, 0), 'setup_plus_epoch1_s': marks[0] - t_start if marks else 0.0,
         'test_eval_s': t_end - marks[-1] if marks else 0.0, **timers}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez(a.out, y=captured['y'], pred=captured['pred'], seed=a.seed, device=a.device, argv=np.array(rest), losses=np.array(losses), **times)
print('TIMES', {k: round(v, 1) for k, v in times.items()})
