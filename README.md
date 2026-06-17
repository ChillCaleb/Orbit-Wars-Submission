# Orbit Wars Submission

Quick terminal guide for running the current agent, benchmark opponents, training, and replay analysis.

## Layout

- `agents/`: local opponent agents and imported benchmark agents.
- `data/`: generated runs, Kaggle outputs, comparison reports, and MP4 replays.
- `docs/`: longer project notes and Kaggle resource notes.
- `kaggle_training/`: self-contained private Kaggle training kernel.
- `notebooks/`: source/reference notebooks pulled from Kaggle.
- `main.py`: current development agent.
- `submission.py`: standalone Kaggle competition submission file.

## Agent Aliases

Use these names with `watch_match.py`, `agent_lab.py`, and `train_tactical_model.py` lineups:

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

## Quick Matches

Run without saving video:

```bash
python watch_match.py --players mine intruder --games 3 --no-save --no-open
```

Run and save every video:

```bash
python watch_match.py --players mine intruder --games 3 --save-all --no-open
```

Videos are written to `data/replays/`.

## Common Spars

```bash
python watch_match.py --players mine smith --games 3 --no-save --no-open
python watch_match.py --players mine 1039 --games 3 --no-save --no-open
python watch_match.py --players mine 1200 --games 3 --no-save --no-open
python watch_match.py --players mine best --games 3 --no-save --no-open
python watch_match.py --players mine intruder --games 3 --no-save --no-open
```

Compare benchmark agents directly:

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

Collect tactical reports and per-turn data:

```bash
python agent_lab.py --players mine intruder --games 10 --seed 20260650
python agent_lab.py --players mine smith best intruder --games 5 --seed 20260800
```

See `docs/RL_LAB.md` for the full lab workflow.

## Local Training

Run 100 games with mine and 100 without mine:

```bash
python train_tactical_model.py --with-mine-games 100 --without-mine-games 100 --workers 4
```

The current training lineups include `best` and `intruder`.

## Kaggle Training

Prepare a private Kaggle notebook:

```bash
python scripts/prepare_kaggle_training.py \
  --with-mine-games 100 \
  --without-mine-games 100 \
  --workers 4
```

Launch it:

```bash
kaggle kernels push -p kaggle_training
```

Monitor and validate it:

```bash
python scripts/monitor_kaggle_training.py
```

Recent kernel link:

```text
https://www.kaggle.com/code/calebbanks/orbit-wars-tactical-training
```

## Replay Analysis

Analyze one downloaded Kaggle replay without continuous polling:

```bash
python scripts/analyze_kaggle_replay.py path/to/episode-replay.json --player 0
```

See `docs/KAGGLE_RESOURCES.md` for replay datasets and elite-trajectory integration notes.
