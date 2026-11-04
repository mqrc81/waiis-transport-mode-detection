"""Stop-point 2, step 0: 10's saved fold models reproduce 10's saved out-of-fold probabilities (acc, GPS), and fusion
+ HMM reproduce `raw` and `p`, on ExtraSensory dev and SHL User 1, for E2 and E1, seeds 0-2. No training.

    uv run python check_models.py            # log: logs/check_models.log
"""
import logging

import numpy as np

import dev

log = logging.getLogger()


def main():
    logging.basicConfig(filename=dev.HERE / "logs" / "check_models.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    worst = {}
    for source in ("shl_user1", "extrasensory"):
        df = dev.frame(source)
        x_acc, x_gps = df[dev.ACC].to_numpy(np.float32), df[dev.GPS].to_numpy(np.float32)
        cov, fold = df.covered.to_numpy(), df.fold.to_numpy()
        for seed in dev.SEEDS:
            p_gps = dev.predict_oof(dev.fold_models("A0_gps", seed), x_gps, fold)
            for name, m in dev.MODELS.items():
                p_acc = dev.predict_oof(dev.fold_models(name, seed), x_acc, fold)
                raw, p = dev.fuse_smooth(m, df, p_acc, p_gps, cov)
                d = {k: float(np.nanmax(np.abs(v - dev.saved(name, seed, df, k))))
                     for k, v in (("acc", p_acc), ("gps", p_gps), ("raw", raw), ("p", p))}
                f_new = dev.six_f1(df.y.to_numpy()[df.official], p[df.official.to_numpy()])
                f_old = dev.six_f1(df.y.to_numpy()[df.official], dev.saved(name, seed, df, "p")[df.official.to_numpy()])
                msg = f"{source} {name} seed {seed}: max |diff| " + " ".join(f"{k} {v:.2e}" for k, v in d.items()) + \
                    f"; six-class F1 official {f_new:.4f} (saved {f_old:.4f})"
                log.info(msg)
                print(msg, flush=True)
                for k, v in d.items():
                    worst[k] = max(worst.get(k, 0.0), v)
    ok = all(v < 1e-6 for v in worst.values())
    msg = f"worst max |diff| {worst}; reproduced (< 1e-6): {ok}"
    log.info(msg)
    print(msg)


if __name__ == "__main__":
    main()
