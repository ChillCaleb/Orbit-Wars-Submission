# Kaggle Training Kernel

Generate the self-contained notebook from the current repository:

```bash
python scripts/prepare_kaggle_training.py
```

The generated notebook embeds the runtime source, all training opponents, and the newest available warm-start model. Its default workload is 100 games with the development agent and 100 games without it.

Nothing is uploaded by the preparation command. A future upload would be a separate, explicit action:

```bash
kaggle kernels push -p kaggle_training
```

Do not run that command until the notebook and metadata have been reviewed.
