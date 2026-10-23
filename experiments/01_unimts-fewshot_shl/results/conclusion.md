## Conclusion

Pretrained UniMTS clearly beats the same architecture trained from random init. The linear probe was the best pretrained variant at every k in both sets. Paired by seed, it scores +0.15 to +0.24 macro-F1 above random in set B and +0.16 to +0.17 in set C. Adding gyro to random init (random_gyro) closes at most about a third of that gap, and none of it in set C at k=10. Full fine-tuning is worse than the probe everywhere, at 200 epochs on so few windows.

Against LightGBM on the same windows, pretraining helps only at the smallest k:

| Set | k | Probe minus lgbm_acc_hz20 | Seeds won by probe |
|---|---|---|---|
| B | 1 | +0.07 | 4/5 |
| B | 2 | +0.06 | 4/5 |
| B | 5 | −0.07 | 0/5 |
| B | 10 | −0.04 | 1/5 |
| C | 1 | +0.07 | 4/5 |
| C | 2 | +0.08 | 5/5 |
| C | 5 | +0.04 | 4/5 |
| C | 10 | 0.00 | 2/5 |

At k=10, the best LightGBM variants reach 0.72 in set B and 0.50–0.51 in set C, which matches or beats the probe.

The pretrained advantage comes from the motion classes: walk/run in B, and walk, run and bike in C. For example, in C at k=10 the probe recalls run 0.98 and bike 0.78. LightGBM gets 0.76 and 0.75 for those classes.

The pretrained model is weak exactly where we care most, on vehicles.

- **Probe:** it confuses vehicles with still and with each other. In C at k=10 its recall is car 0.30, bus 0.29, train 0.24 and subway 0.18.
- **LightGBM (C, k=10):** recall is car 0.36–0.40, bus 0.28–0.45, train 0.23–0.28 and subway 0.36–0.40.
- **Car windows in set B:** the share predicted as "vehicle" is 0.72 for the probe and 0.88 for full fine-tuning at k=10. For LightGBM it is 0.90–0.91.

Zero-shot (k=0) is a reasonable start only for still and walk/run. It reaches 0.54 in B and 0.30 in C, with vehicle recall of 0.25 in B and 0.00–0.12 per vehicle class in C.

These numbers have three limits:

- All results come from one user (SHL User 1), one placement (hips) and a split by day. Train and test therefore share the person, the phone and the routes, so every number is optimistic for new users.
- Each k uses only 5 seeds, and the paired standard deviations (0.01–0.10) are often as large as the differences against LightGBM.
- Probe and full fine-tuning use only the accelerometer, because the released checkpoint has no gyro weights.

Set A (still/walk/run) was evaluated zero-shot only. It was optional for few-shot and was not run.
