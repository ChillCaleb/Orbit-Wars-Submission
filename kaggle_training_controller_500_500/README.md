# Kaggle Training Kernel

Generate the self-contained notebook from the current repository:

```bash
python scripts/prepare_kaggle_training.py
```

The generated notebook embeds the runtime source, all training opponents, and the newest available warm-start model. Its default workload is 100 games with the development agent and 100 games without it.

The current embedded opponent pool includes `smith`, `1039`, `1200`, `best`, and `intruder`.

Nothing is uploaded by the preparation command. A future upload would be a separate, explicit action:

```bash
kaggle kernels push -p kaggle_training
```

Do not run that command until the notebook and metadata have been reviewed.

Monitor a launched kernel every five minutes, then download and validate its
outputs automatically:

```bash
python scripts/monitor_kaggle_training.py
```

The monitor writes timestamped status checks, downloaded artifacts, and a
validated metrics summary under `data/kaggle_runs/<timestamp>/`.

For a single status check:

```bash
python scripts/monitor_kaggle_training.py --once
```

More Kaggle resource notes live in `docs/KAGGLE_RESOURCES.md`.
