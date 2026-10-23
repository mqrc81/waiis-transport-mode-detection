# Decisions

## 2026-10-23: UniMTS is not adopted as the on-phone model

**Draft — not approved**

**Decision.** We will not use pretrained UniMTS (zero-shot, linear probe or fine-tuned) as the production
transport-mode model.

**Why.**
- **Not better than LightGBM once there are a few labels.** On the same k-shot windows, the best pretrained variant
  (linear probe) behaves like this against LightGBM (|acc| features, 20 Hz):
  - Set B (still / walk-or-run / vehicle): below from k=5 (−0.07 macro-F1 at k=5, −0.04 at k=10).
  - Set C (8 classes): ahead at k=5 (+0.04) and level at k=10 (−0.00).

  The probe's advantage at k ≤ 2 (+0.06 to +0.08) comes from the motion classes. On vehicles it is weaker than
  LightGBM. For example, in set C at k=10 the probe recalls car 0.30, bus 0.29, train 0.24 and subway 0.18.
- **Too large.** The IMU encoder alone has 5.18M parameters (4.94 Mi), which is about 20 MB in fp32 (≈ 5 MB int8). The
  ceiling is 2 MB. The released checkpoint also contains the CLIP text encoder (63.1M parameters, 274 MB in total);
  zero-shot needs it, or text embeddings precomputed from it.

**Limits of the evidence.** There is one user (SHL User 1, hips), the split is by day, and there are 5 seeds per k.
The paired standard deviations are often as large as the differences against LightGBM. Latency was not measured.

**Evidence.** `experiments/01_unimts-fewshot_shl/` (README Results and Conclusion, and `results/shl_fewshot.md`).
