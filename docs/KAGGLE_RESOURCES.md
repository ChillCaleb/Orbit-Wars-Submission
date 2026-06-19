# Kaggle Resources

## Recommended Order

1. Continue balanced local/Kaggle spars against Smith, 1039, 1200, Best Orbit,
   and Light Intruder.
2. Analyze selected real ladder replays with `scripts/analyze_kaggle_replay.py`.
3. Add the CC0 `orbit-wars-elite-trajectories` value data as an auxiliary
   controller signal after the current pressure-conditioned model is stable.
4. Consider the larger policy and daily replay datasets only after the smaller
   value integration proves useful.

## Useful Public Data

- `pawanmali/orbit-wars-elite-trajectories`
  - `elite_positions.npz`: 2,198,874 state snapshots, 24 features, eventual
    winner label, and launch-volume statistics.
  - `elite_policy.npz`: 742,361 expert decision states with source/target
    policy labels.
  - Best immediate fit: pretrain or calibrate a state-value/urgency signal.
- `kaggle/orbit-wars-episodes-index`
  - Small manifest for the official daily replay datasets.
- `kaggle/orbit-wars-episodes-YYYY-MM-DD`
  - Raw public episodes, generally 1.3-1.6 GB per day.
  - Valuable later, but too large and unfiltered for the first integration.

## Useful Public Techniques

- Replay-to-Parquet pipelines expose episode, player, tick economy, action,
  planet topology, and planet-state tables.
- Learned value search predicts whether a simulated future state is winning.
- Conservative shot validators reject bad controller actions instead of
  replacing the controller.
- Public analyses consistently associate stronger play with higher launch
  frequency, not merely larger fleet sizes.

## Current Direction

The current framework should keep its heuristic perception and Smith-derived
controller. The next useful learned layer is not a wholesale policy replacement;
it is a pressure/urgency value signal that ranks the controller's moves and
rejects actions that ignore an actively collapsing area.

## Useful Commands

Prepare and launch a 100-with/100-without Kaggle training run:

```bash
python scripts/prepare_kaggle_training.py \
  --with-mine-games 100 \
  --without-mine-games 100 \
  --workers 4

kaggle kernels push -p kaggle/bundles/training
```

Monitor the run:

```bash
python scripts/monitor_kaggle_training.py
```

Recent kernel:

```text
https://www.kaggle.com/code/calebbanks/orbit-wars-tactical-training
```
