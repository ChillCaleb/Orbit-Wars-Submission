# Tactical Training Run

Run directory: `20260617-013005-tactical-outcome`
Seed: `20260617`
Games: without_mine=500, with_mine=500
Train ratio: 0.8
Sample stride: every 10 environment steps
Training mode: outcome

## Dataset
- Train samples: 19218
- Test samples: 4997
- Feature dimension: 201
- Category counts: {'without_mine_train': 400, 'without_mine_test': 100, 'with_mine_train': 400, 'with_mine_test': 100}
- Lineups: {'best smith': 50, 'intruder smith': 50, 'best intruder': 50, 'intruder 1200': 50, 'best 1200': 50, 'best 1039': 50, 'intruder 1039': 50, 'best intruder smith 1200': 50, 'best intruder smith 1039': 50, 'best intruder 1200 1039': 50, 'mine best': 100, 'mine intruder': 100, 'mine best intruder smith': 100, 'mine best intruder 1039': 100, 'mine best intruder 1200': 100}

## Progression
- Warm start requested: True
- Warm start used: True
- Warm start model: `/kaggle/working/orbit_training/model_weights.npz`
- Warm start note: initialized_from_existing_model
- History merged from: []

## Model
- Model type: numpy logistic action-value predictor
- Target: shaped action quality using final winner, sun/flight penalties, and production opportunity cost
- Inputs: board quadrant state, role label, phase label, target context, recent tactical tendencies, and action penalty features
- This is a role-conditioned action scorer layered onto the heuristic controller.

## Metrics
### train
- samples=19218 positive_rate=0.5330 loss=0.6239 accuracy=0.7365
- with_mine: samples=10235 positive_rate=0.5226 loss=0.6268 accuracy=0.7319
- without_mine: samples=8983 positive_rate=0.5445 loss=0.6207 accuracy=0.7416
### test
- samples=4997 positive_rate=0.5307 loss=0.6336 accuracy=0.7194
- with_mine: samples=2797 positive_rate=0.5265 loss=0.6310 accuracy=0.7296
- without_mine: samples=2200 positive_rate=0.5358 loss=0.6368 accuracy=0.7069

## Artifacts
- `dataset.npz`: train/test arrays
- `model_weights.npz`: learned weights and normalization
- `training_progress.csv`: epoch log
- `game_log.csv`: per-game lineup/winner log
- `samples_meta.csv`: per-sample metadata
- `metrics.json`: final metrics
