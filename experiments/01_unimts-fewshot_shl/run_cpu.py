"""Run an unmodified UniMTS script on CPU: makes torch.load default to map_location='cpu'
(the repo's CPU patch missed some torch.load calls, e.g. evaluate_custom.py:48), then runs it as __main__.
Usage (from repo root): python experiments/shl/run_cpu.py evaluate_custom.py [args...]
"""
import functools, os, runpy, sys
import torch

torch.load = functools.partial(torch.load, map_location='cpu')
sys.path.insert(0, os.getcwd())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
