# Orbit Wars Submission

The repository is split by responsibility:

- `agents/`: Smith, 1039, 1200, Best Orbit, and Light Intruder opponents.
- `data/`: training runs, pressure experiments, lab records, comparisons, and replays.
- `notebooks/`: source and reference notebooks.
- `kaggle_training/`: locally prepared Kaggle training kernel.
- `main.py`: the development agent.
- `submission.py`: the standalone competition submission.

Run a local simulation and save/watch an MP4:

```bash
python watch_match.py --players mine starter
```

Run the Agent Smith 1v1:

```bash
python watch_match.py --players mine smith
```

Run the 1039 launch-safety agent 1v1:

```bash
python watch_match.py --players mine 1039
```

Run the 1200 PPO-strategy agent 1v1:

```bash
python watch_match.py --players mine 1200
```

Run two comparison agents against each other:

```bash
python watch_match.py --players smith 1039
```

```bash
python watch_match.py --players smith 1200
```

```bash
python watch_match.py --players 1039 1200
```

Run a 4-player comparison:

```bash
python watch_match.py --players mine smith 1039 1200
```

Run a 4-player comparison without yours:

```bash
python watch_match.py --players smith 1039 1200 random
```

Videos are stored in `data/replays/`.

Run instrumented games and collect tactical training data:

```bash
python agent_lab.py --players mine 1200 --games 10 --seed 20260650
```

See `RL_LAB.md` for the full data-collection workflow, tendency reports, tactical weights, and progression tracking.

Run a 100-with/100-without tactical training split:

```bash
python train_tactical_model.py --with-mine-games 100 --without-mine-games 100 --workers 4
```

Prepare the same workload as a private Kaggle notebook:

```bash
python scripts/prepare_kaggle_training.py
```

This only creates `kaggle_training/training.ipynb`; it does not upload or start a Kaggle kernel. Review `kaggle_training/README.md` before any future `kaggle kernels push`.
