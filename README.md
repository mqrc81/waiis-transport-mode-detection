# WAIIS Transport mode detection

> **Parked (2026-11-05).** This repository is no longer developed. Its findings are resumed in a company-internal
> repository. Everything here stays as the record of experiments 01–12: results, decisions and code are frozen at the
> last commit. Where things stand and what comes next: [STATUS.md](STATUS.md).

Classify a person's transport mode (e.g. still, walk, run, bike, car, bus, train,
metro) from smartphone sensor data. The phone records and uploads the data; the model runs on an edge server.
Work is in Python, organised as one folder per experiment under `experiments/`, with shared code in `src/tmd`.

- Final status, results, reference model and next steps: [STATUS.md](STATUS.md)
- Project-wide decisions: [DECISIONS.md](DECISIONS.md)
- Conventions this repository was worked under: [CLAUDE.md](CLAUDE.md)
