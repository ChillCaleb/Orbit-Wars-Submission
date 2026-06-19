# V7 Tactical 500/500 Evaluation

Collected on 2026-06-18.

## Kaggle Collection

- Tactical kernel: `calebbanks/orbit-wars-tactical-training`
- Status at collection: `COMPLETE`
- Local collection: `data/kaggle_runs/collect-20260618-tactical-500x500`
- Staged run: `data/training_runs/20260617-235507-tactical-outcome`
- Controller kernel: `calebbanks/orbit-wars-controller-training`
- Controller status at last check: `RUNNING`
- Partial controller artifacts: unavailable

## Tactical Metrics

Run: `data/training_runs/20260617-235507-tactical-outcome`

- Games: 500 with mine, 500 without mine
- Training mode: outcome
- Train samples: 20,442
- Test samples: 4,608
- Train accuracy: 0.7286
- Test accuracy: 0.7201
- With-mine test accuracy: 0.7160
- Without-mine test accuracy: 0.7238

## Spar Results

All local spars used:

```bash
ORBIT_MODEL_WEIGHTS=data/training_runs/20260617-235507-tactical-outcome/model_weights.npz
```

For the baseline tactical runs:

```bash
ORBIT_DISABLE_CONTROLLER=1
```

For smoke-controller comparisons:

```bash
ORBIT_CONTROLLER_WEIGHTS=data/training_runs/20260617-110124-controller-selector/controller_weights.npz
```

| Run | Opponent | Controller | Record | Enemy Launches | Enemy Avg Ships | Enemy >=50 | Enemy >=80 | Captured From Opponent |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260618-072945-mine-best` | Best | off | 0-3 | 98 | 45.1 | 29 | 9 | 35 |
| `20260618-073205-mine-best` | Best | smoke | 0-3 | 100 | 39.3 | 23 | 6 | 45 |
| `20260618-073503-mine-intruder` | Intruder | off | 0-3 | 63 | 43.3 | 21 | 6 | 37 |
| `20260618-073720-mine-intruder` | Intruder | smoke | 0-3 | 56 | 44.9 | 17 | 5 | 38 |
| `20260618-075332-mine-smith` | Smith | off | 0-3 | 137 | 36.8 | 28 | 6 | 98 |
| `20260618-075332-mine-1039` | 1039 | off | 2-1 | 199 | 46.4 | 61 | 25 | 98 |

## Read

The 500/500 tactical model is materially more willing to attack than the prior v6 tactical model. Against Best, enemy launches increased from 46 in the older v6 eval to 98 in the v7 tactical baseline, and heavy enemy launches increased from 3 to 9.

The gap is no longer simply "does not attack." The gap is commitment quality. Best and Intruder still send much larger enemy-facing payloads. Best averaged roughly 77 ships per enemy launch in the v7 baseline comparison while our agent averaged roughly 45. Intruder averaged roughly 63 while our agent averaged roughly 43.

Smith exposes a different issue: the agent is extremely busy but still spends too much motion on friendly movement. Smith made 411 enemy-targeted launches in the 3-game set while our agent made 137. Our agent captured a lot from Smith, but Smith's pressure volume overwhelmed us.

The smoke future-quality controller is not ready to promote. It helped captures against Best but reduced heavy enemy payload quality and hurt enemy target share against Intruder. The full 500/500 controller run should be evaluated before deciding whether to integrate any controller weights.

## Current Recommendation

- Keep the v7 tactical weights staged but do not overwrite root weights yet.
- Do not promote the smoke controller.
- When the full controller Kaggle run finishes, collect it and rerun the same Best and Intruder A/B.
- If the full controller does not improve enemy payload quality, the next target should be controller-side commitment: fewer friendly maintenance moves when not pressured, and stronger multi-source enemy attack assembly.
