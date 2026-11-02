"""waiis_bench_v1 stays frozen while tmd.bench grows (experiment 09).

- The v1 data on disk match the committed reference manifest (every file's sha256), read through `tmd.bench` with
  version v1.
- 06's dev out-of-fold probabilities that the reorganisation gate reproduced bit for bit (05 B and F_fb, 06 A, G_v2,
  H1_v2; seed 0) are still the files on disk. The recomputation itself is
  `experiments/09_frozen-benchmark_es-shl-nor-collecty/gate_v1.py` (a scratch rebuild of v1 plus two CV runs; too
  slow for pytest); this test catches any later change to the files it matched.
Skipped when the data are not on this machine.
"""
import pytest

from tmd import bench
from tmd.paths import PROCESSED

V1 = "waiis_bench_v1"
OOF_SHA256 = {  # recorded at the gate (09 README, stop-point 1 follow-up), after the bit-for-bit match
    ("05_acc-baseline_waiis-bench", "B_seed0"): "9d683627132f0bb950a9d06cbf047f14213014ff43e49684685b6b0518e16abd",
    ("05_acc-baseline_waiis-bench", "F_fb_seed0"): "e473b7b461ba401b4f3e96f0d89a03c686e4e2b950f2b2e72ca732e3d8ca9936",
    ("06_gps-acc-baseline_waiis-bench", "A_seed0"): "d9f4df2725979f202faccb336ae28a7629648096b552aaf996333efb8aeeffd0",
    ("06_gps-acc-baseline_waiis-bench", "G_v2_seed0"): "5cd5ef82436d241d04427ea9145432b5cd9d59b4f8c83ca235a352426f11c944",
    ("06_gps-acc-baseline_waiis-bench", "H1_v2_seed0"): "11175ad1df3ff85a354c196c5f720e13ef3848e133972c28d6eb2c80331ecb0b",
}


@pytest.mark.skipif(not bench.root(V1).exists(), reason="waiis_bench_v1 not on this machine")
def test_v1_matches_its_reference_manifest():
    assert bench.verify(V1) == []


@pytest.mark.parametrize("exp,name", sorted(OOF_SHA256))
def test_06_dev_oof_files_unchanged(exp, name):
    p = PROCESSED / exp / "oof" / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"{p} not on this machine")
    assert bench.sha256(p) == OOF_SHA256[(exp, name)]
