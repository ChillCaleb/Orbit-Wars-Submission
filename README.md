# Orbit-Wars-Submission

Run a local simulation and save/watch an MP4:

```bash
.venv312/bin/python watch_match.py --players mine starter
```

Run the Agent Smith 1v1:

```bash
.venv312/bin/python watch_match.py --players mine smith
```

Run the 1039 launch-safety agent 1v1:

```bash
.venv312/bin/python watch_match.py --players mine 1039
```

Run the 1200 PPO-strategy agent 1v1:

```bash
.venv312/bin/python watch_match.py --players mine 1200
```

Run two comparison agents against each other:

```bash
.venv312/bin/python watch_match.py --players smith 1039
```

```bash
.venv312/bin/python watch_match.py --players smith 1200
```

```bash
.venv312/bin/python watch_match.py --players 1039 1200
```

Run a 4-player comparison:

```bash
.venv312/bin/python watch_match.py --players mine smith 1039 1200
```

Run a 4-player comparison without yours:

```bash
.venv312/bin/python watch_match.py --players smith 1039 1200 random
```

Videos are stored in `MP4/` as `replay 1.mp4`, `replay 2.mp4`, and so on.

Run instrumented games and collect tactical training data:

```bash
.venv312/bin/python agent_lab.py --players mine 1200 --games 10 --seed 20260650
```

See `RL_LAB.md` for the full data-collection workflow, tendency reports, tactical weights, and progression tracking.
