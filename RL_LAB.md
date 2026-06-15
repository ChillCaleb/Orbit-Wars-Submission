# Orbit Wars RL Lab

This lab is the first data layer for the hybrid plan:

```text
tactical labels + enemy tendency map + learned selector + Smith-style controller
```

The goal is not to replace the controller yet. The goal is to record useful games, label the board in our language, and build progression data that a learning system can use.

## Files

- `tactical_features.py`
  - Quadrant labels: `Q0_SE`, `Q1_SW`, `Q2_NW`, `Q3_NE`
  - Planet labels: static/rotating, big/small, corner node
  - Role scores: anchor, feeder, sweeper, strike-stage
  - Quadrant arrays for model input
  - Action target inference and capture events

- `agent_lab.py`
  - Runs instrumented 2-player or 4-player games
  - Wraps each agent so every action is recorded
  - Writes per-turn JSONL records, summaries, reports, and progression rows

- `data/lab/progression.csv`
  - Long-term tendency/progression table across runs

- `data/lab/<run_id>/tendency_report.md`
  - Human-readable tendency report for one run

- `data/lab/<run_id>/summary.json`
  - Machine-readable aggregate summary

- `data/lab/<run_id>/game_###.jsonl`
  - Per-turn tactical records

- `data/lab/<run_id>/weights_snapshot.json`
  - Label weights used for that run

## Commands

Run yours against the 1200 strategy and record data:

```bash
python agent_lab.py --players mine 1200 --games 10 --seed 20260650
```

Run Smith against 1039:

```bash
python agent_lab.py --players smith 1039 --games 10 --seed 20260700
```

Run the 4-player mix:

```bash
python agent_lab.py --players mine smith 1039 1200 --games 5 --seed 20260800
```

Save the full board in every JSONL row for heavier training data:

```bash
python agent_lab.py --players mine 1200 --games 20 --seed 20260900 --full-board
```

Skip per-turn JSONL files when you only want progression stats:

```bash
python agent_lab.py --players smith 1039 --games 25 --seed 20261000 --no-records
```

## Current Label Weights

These are not neural model weights yet. They are tactical-label weights used to score candidates:

- `anchor`: static, big, corner position, nearby small support, safety, production, ships
- `feeder`: established quadrant, static, meridian access, surplus, safety, production
- `sweeper`: small static, inner position, equator access, safety, ships
- `strike_stage`: surplus, frontier pressure, safety, production

Each run stores the exact weights in `weights_snapshot.json`.

## What We Are Measuring

The lab tracks:

- Launch volume and ships launched
- Static vs rotating target preference
- Big/small central rotating target preference
- Neutral/enemy/friendly target preference
- Source and target quadrants
- Captures and losses
- Who captures from whom
- First established quadrant timing
- Final production and ship totals
- Role candidates every turn
- Numeric quadrant arrays for training

This is the beginning of the enemy tendency map: where they attack, what they value, how they transfer, and who they harvest.

## Initial Read

Small sample only, but the first runs already show useful differences:

- `1200` launches many more fleets than ours and uses heavy friendly transfers.
- `smith` beat `1039` in the initial 2-player sample and attacked enemies more directly.
- In the first 4-player sample, `smith` and `1039` split wins, while `1039` showed strong establishment behavior.
- Our current bot tends to stay concentrated in its opening side and needs better transition/pressure recognition.

## Next Training Step

The next concrete layer is a learned high-level selector. It should not output raw angle/ships yet.

Suggested high-level actions:

```text
COMPLETE_ESTABLISHMENT
SELECT_ANCHOR
SELECT_FEEDER
FEED_QUADRANT
DENY_LEADER
PUNISH_FINISHER
STRIKE_STAGE_NODE
REINFORCE_ANCHOR
HOLD_BATTERY
EXPAND_SAFE_ZONE
```

The selector reads the quadrant arrays, role scores, and enemy tendency features. A Smith-style controller then executes the chosen action with real geometry and timing.
