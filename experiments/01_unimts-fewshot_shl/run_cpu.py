"""Run an unmodified UniMTS script on CPU: makes torch.load default to map_location='cpu'
(the repo's CPU patch missed some torch.load calls, e.g. evaluate_custom.py:48), then runs it as __main__.
Usage (cwd shl_common.UNIMTS_HOME): python <exp>/run_cpu.py evaluate_custom.py [args...]   (script taken from shl_common.EXT)
"""
import functools, os, runpy, sys
import torch
from shl_common import EXT

torch.load = functools.partial(torch.load, map_location='cpu')
sys.path.insert(0, str(EXT))
sys.argv = sys.argv[1:]
runpy.run_path(os.path.join(EXT, sys.argv[0]), run_name='__main__')
