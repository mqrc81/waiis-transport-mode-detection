# CLAUDE.md

## Project
Classify transport mode (still, walk, run, bike, car, bus, train, metro) from phone
accelerometer + gyroscope. Current production rate is 25 Hz; the model must eventually run on a
phone, so size and latency matter as much as accuracy. Work is in Python, over files.

## Layout
- `src/tmd`: shared code. Anything a second experiment would reuse goes here.
- `experiments/NN_question_dataset/`: one folder per experiment. `README.md` holds Question / Plan / Results / Conclusion. Add `patches/`, `env/` or `results/` only when there is something to put in them.
- `mlflow.db`, `mlartifacts/`: MLflow tracking store and run artifacts (git-ignored).
- `DECISIONS.md`, `../LICENSES.md`: project-wide.
- `log/`: my daily logs. **Do not edit.**
- Outside git: data in `../data/` (`raw/`, `processed/`, `checkpoints/`), upstream clones in `../ext/`. Paths are relative to project root.

## Working on an experiment
- Follow the README's Plan, including its stop-points.
- At every stop-point, update the Results section: commands run, numbers, deviations, open questions. A fresh session must be able to resume from the README alone.
- At the end, draft the Conclusion and, if it changes direction, a `DECISIONS.md` entry. Write them from the Results section and the exported `runs.csv` only, and mark both **Draft — not approved**. I approve them by removing the marker.

## Rules
- Data, checkpoints and environments never go in git.
- Edit files in `../../ext` only when needed. Save them with git diff > experiments/NN_name/patches/<repo>.patch, and record the upstream commit hash in the README. Before another experiment uses the same clone, reset it (`git checkout .`) and apply that experiment's patch.
- Ask before deleting files or installing anything system-wide. Adding packages with `uv add` is fine.
- Any deviation from an upstream README or from the Plan is recorded in Results, with file and line.
- Third-party environments (not managed by uv) are exported to the experiment's `env/` folder (`pip freeze` or `conda env export`).
- Files in `processed/` are named by their parameters, e.g. `shl_u1_hips_20hz_w10s.npz`.

## Evaluation conventions
- Split by user; by day only when there is a single user, and say so. Never random window splits.
- Normalisation statistics and hyperparameter choices come from training data only, never from a set reported as test.
- Report macro-F1, per-class recall, the confusion matrix, model size and latency per window (single thread, after warm-up, median and p95).

## Logging results
- Log every run through `src/tracking.py`: `with run(experiment, params):`, then `mlflow.log_metrics` and `mlflow.log_figure`. The experiment name is the experiment folder name.
- Params include model, dataset, rate_hz, class_set, seed, and anything the Plan varies.
- The helper tags the git commit and `dirty`. Log dirty runs anyway, and mention it in Results.
- When the experiment finishes, export its runs once with `uv run python src/tracking.py export <experiment>`. Commit the export (`results/runs.csv` in the experiment folder) with the final Results tables, which are generated from it. Never edit the export by hand. Until then, MLflow is the only record.
- To inspect runs, use `mlflow.search_runs` in a short script and print only what you need.
- Experiment 01 predates MLflow and keeps its own CSV.

## Keeping context small
- Scripts print a short summary. Full output goes to a log file in the experiment folder.
- On failure, read the tail of the log, not all of it.
- Use a subagent to read large upstream code. Bring back a short summary, written to the README if it's worth keeping.
