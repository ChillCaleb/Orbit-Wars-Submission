# Orbit Wars Submission

Terminal runbook for the current Orbit Wars agent, local spars, tactical/controller training, Kaggle training, artifact collection, and submission.

## Layout

- `agents/`: opponent agents used for spars and training.
- `agents/imported/`: notebook/archive agents imported for training variety. These are excluded from submission bundles.
- `data/training_runs/`: local and promoted training run artifacts.
- `data/kaggle_runs/`: downloaded Kaggle kernel outputs.
- `data/lab/`: temporary comparisons, backups, and candidate tests.
- `data/replays/`: MP4 replays from local match runs.
- `docs/`: longer notes and Kaggle resource details.
- `kaggle/bundles/`: generated private Kaggle training notebooks.
- `notebooks/`: source/reference notebooks pulled from Kaggle.
- `notebooks/agents/`: source notebooks used to import additional training agents.
- `main.py`: active development agent.
- `model_weights.npz`: active tactical model weights loaded by `main.py`.
- `controller_weights.npz`: active controller selector weights loaded by `main.py`.
- `submission.py`: generated Kaggle submission wrapper.
- `submission.zip`: generated zipped submission bundle.

## Agent Aliases

Use these names with `watch_match.py`, `agent_lab.py`, `train_tactical_model.py`, and `train_controller_model.py` lineups.

| Alias | Agent |
| --- | --- |
| `mine` | Current development agent in `main.py` |
| `smith` | Smith controller baseline |
| `1039` | Launch-safety heuristic agent |
| `1200` | PPO-strategy public agent |
| `best` | Best Orbit Wars notebook agent |
| `intruder` | Light Intruder 1200+ agent |
| `starter` | Kaggle starter bot |
| `random` | Kaggle random bot |

Imported agents under `agents/imported/` are auto-discovered for training lineups. Current imported names include `candidate_0615_k_l0014_2p_producer_style`, `i_the_orbit`, `orbit_wars`, `orbit_wars_exp50`, `orbit_wars_i_m_stronger`, `risk_aware_wave_control_stable`, and `smooth_intruder_complement`.

## Quick Spars

Run three games without saving video:

```bash
python watch_match.py --players mine best --games 3 --no-save --no-open
```

Run three games and save every replay:

```bash
python watch_match.py --players mine intruder --games 3 --save-all --no-open
```

Common 1v1 checks:

```bash
python watch_match.py --players mine smith --games 3 --no-save --no-open
python watch_match.py --players mine 1039 --games 3 --no-save --no-open
python watch_match.py --players mine 1200 --games 3 --no-save --no-open
python watch_match.py --players mine best --games 3 --no-save --no-open
python watch_match.py --players mine intruder --games 3 --no-save --no-open
```

Benchmark agents against each other:

```bash
python watch_match.py --players best intruder --games 3 --no-save --no-open
python watch_match.py --players smith 1039 --games 3 --no-save --no-open
python watch_match.py --players 1200 intruder --games 3 --no-save --no-open
```

Run a 4-player comparison:

```bash
python watch_match.py --players mine smith best intruder --games 3 --no-save --no-open
```

## Instrumented Lab Runs

Collect per-turn tactical reports:

```bash
python agent_lab.py --players mine intruder --games 10 --seed 20260650
python agent_lab.py --players mine smith best intruder --games 5 --seed 20260800
```

Replay videos and lab outputs are generated under `data/`. See `docs/RL_LAB.md` for deeper lab workflow notes.

## Import Notebook Agents

Put source notebooks in `notebooks/agents/`, then import or refresh the local training agents:

```bash
python scripts/import_notebook_agents.py
```

The importer writes generated agents to `agents/imported/`.

## Local Tactical Training

Small tactical train/test split:

```bash
python train_tactical_model.py \
  --with-mine-games 10 \
  --without-mine-games 10 \
  --lineup-preset all \
  --training-mode outcome \
  --workers 4 \
  --progress-every 5
```

Larger local tactical run:

```bash
python train_tactical_model.py \
  --with-mine-games 100 \
  --without-mine-games 100 \
  --lineup-preset all \
  --training-mode outcome \
  --workers 4 \
  --progress-every 20
```

Use `--lineup-preset new-agents` to emphasize recently imported agents, or `--lineup-preset opportunity` for the older opportunity-focused mix.

## Local Controller Training

Small controller train/test split:

```bash
python train_controller_model.py \
  --with-mine-games 50 \
  --without-mine-games 50 \
  --lineup-preset all \
  --target-mode future-quality \
  --epochs 12 \
  --batch-size 2048 \
  --lr 0.035 \
  --progress-every 5
```

Controller labels:

- `future-quality`: trains toward decisive enemy payloads, pressure coverage, and low-churn movement.
- `winner-source`: trains toward matching the final winner's proposal source style.

## Kaggle Training Bundles

Prepare the current `200 with mine / 200 without mine` tactical bundle:

```bash
python scripts/prepare_kaggle_training.py \
  --job tactical \
  --with-mine-games 200 \
  --without-mine-games 200 \
  --lineup-preset all \
  --training-mode outcome \
  --epochs 12 \
  --batch-size 2048 \
  --lr 0.04 \
  --seed 2026197000 \
  --workers 4 \
  --progress-every 25 \
  --output-dir kaggle/bundles/tactical_200_200
```

Prepare the current `200 with mine / 200 without mine` controller bundle:

```bash
python scripts/prepare_kaggle_training.py \
  --job controller \
  --with-mine-games 200 \
  --without-mine-games 200 \
  --lineup-preset all \
  --target-mode future-quality \
  --epochs 12 \
  --batch-size 2048 \
  --lr 0.035 \
  --seed 2026198000 \
  --workers 4 \
  --progress-every 25 \
  --output-dir kaggle/bundles/controller_200_200
```

If you create a new bundle directory, copy the matching kernel metadata from a previous bundle:

```bash
cp kaggle/bundles/tactical_200_200/kernel-metadata.json kaggle/bundles/tactical_NEW/kernel-metadata.json
cp kaggle/bundles/controller_200_200/kernel-metadata.json kaggle/bundles/controller_NEW/kernel-metadata.json
```

## Launch Kaggle Runs

Launch tactical training:

```bash
kaggle kernels push -p kaggle/bundles/tactical_200_200
```

Launch controller training:

```bash
kaggle kernels push -p kaggle/bundles/controller_200_200
```

Kernel links:

```text
https://www.kaggle.com/code/calebbanks/orbit-wars-tactical-training
https://www.kaggle.com/code/calebbanks/orbit-wars-controller-training
```

## Check Kaggle Status

```bash
kaggle kernels status calebbanks/orbit-wars-tactical-training
kaggle kernels status calebbanks/orbit-wars-controller-training
```

One-shot monitor check:

```bash
python scripts/monitor_kaggle_training.py \
  --kernel calebbanks/orbit-wars-tactical-training \
  --manifest kaggle/bundles/tactical_200_200/bundle_manifest.json \
  --once
```

Continuous monitor with five-minute checks:

```bash
python scripts/monitor_kaggle_training.py \
  --kernel calebbanks/orbit-wars-controller-training \
  --manifest kaggle/bundles/controller_200_200/bundle_manifest.json \
  --interval 300 \
  --timeout-hours 12 \
  --output-dir data/kaggle_runs/controller-200x200
```

## Collect Kaggle Outputs

Create a collection directory and download tactical outputs:

```bash
mkdir -p data/kaggle_runs/collect-tactical-200x200
kaggle kernels output calebbanks/orbit-wars-tactical-training \
  -p data/kaggle_runs/collect-tactical-200x200 \
  --force
```

Create a collection directory and download controller outputs:

```bash
mkdir -p data/kaggle_runs/collect-controller-200x200
kaggle kernels output calebbanks/orbit-wars-controller-training \
  -p data/kaggle_runs/collect-controller-200x200 \
  --force
```

Inspect downloaded metrics:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("data/kaggle_runs/collect-tactical-200x200/training_output"),
    Path("data/kaggle_runs/collect-controller-200x200/training_output"),
]
for root in paths:
    for metrics in sorted(root.glob("*/metrics.json")) + sorted(root.glob("*/controller_metrics.json")):
        print("\n==", metrics, "==")
        print(json.dumps(json.loads(metrics.read_text()), indent=2))
PY
```

Archive a completed Kaggle run into `data/training_runs/`:

```bash
cp -a data/kaggle_runs/collect-tactical-200x200/training_output/RUN_DIR data/training_runs/
cp -a data/kaggle_runs/collect-controller-200x200/training_output/RUN_DIR data/training_runs/
```

Replace `RUN_DIR` with the timestamped directory created by the trainer, such as `20260619-071528-tactical-outcome`.

## Promote Candidate Weights

Back up the current root weights before promotion:

```bash
mkdir -p data/lab/pre-promotion-backup
cp model_weights.npz data/lab/pre-promotion-backup/model_weights.npz
cp controller_weights.npz data/lab/pre-promotion-backup/controller_weights.npz
```

Promote tactical weights:

```bash
cp data/training_runs/RUN_DIR/model_weights.npz model_weights.npz
```

Promote controller weights:

```bash
cp data/training_runs/RUN_DIR/controller_weights.npz controller_weights.npz
```

Quick candidate smoke test:

```bash
python watch_match.py --players mine best --games 3 --no-save --no-open
```

## Build Submission

Build both `submission.zip` and `submission.py` from the active root weights:

```bash
python scripts/build_submission.py
```

Build using explicit candidate weights:

```bash
python scripts/build_submission.py \
  --model-weights data/training_runs/RUN_DIR/model_weights.npz \
  --controller-weights data/training_runs/RUN_DIR/controller_weights.npz
```

Verify the generated submission file compiles:

```bash
python -m py_compile submission.py
```

Verify the bundle contains active weights and excludes imported training agents:

```bash
python - <<'PY'
import zipfile

with zipfile.ZipFile("submission.zip") as zf:
    imported = [name for name in zf.namelist() if name.startswith("agents/imported/")]
    weights = [name for name in zf.namelist() if name.endswith("_weights.npz") or name == "model_weights.npz"]
    print("imported_in_submission", len(imported))
    print("weights", weights)
PY
```

## Submit To Kaggle

Submit the generated single-file wrapper:

```bash
kaggle competitions submit -c orbit-wars -f submission.py -m "Tactical and controller update"
```

Submit the zip bundle if you want Kaggle to use the packaged artifact directly:

```bash
kaggle competitions submit -c orbit-wars -f submission.zip -m "Zip tactical and controller update"
```

If submitting from a specific notebook version is required:

```bash
kaggle competitions submit -c orbit-wars -f submission.py -k calebbanks/orbit-wars-tactical-training -v 10 -m "Notebook version submission"
```

## Replay Analysis

Analyze one downloaded Kaggle replay without continuous polling:

```bash
python scripts/analyze_kaggle_replay.py path/to/episode-replay.json --player 0
```

See `docs/KAGGLE_RESOURCES.md` for replay datasets and elite-trajectory notes.

## Common Status Commands

Check whether local training is running:

```bash
ps -ef | grep train_tactical_model.py
ps -ef | grep train_controller_model.py
```

Check current Git worktree:

```bash
git status --short
```

List recent training runs:

```bash
find data/training_runs -maxdepth 2 -type f | sort
```

List Kaggle bundles:

```bash
find kaggle/bundles -maxdepth 2 -type f | sort
```
