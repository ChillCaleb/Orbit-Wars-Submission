# Tactical Training Run

Run directory: `20260617-000126-tactical-outcome`
Seed: `20260616`
Games: without_mine=100, with_mine=100
Train ratio: 0.8
Sample stride: every 10 environment steps
Training mode: outcome

## Dataset
- Train samples: 3798
- Test samples: 1013
- Feature dimension: 201
- Category counts: {'without_mine_train': 80, 'without_mine_test': 20, 'with_mine_train': 80, 'with_mine_test': 20}
- Lineups: {'intruder smith': 10, 'best smith': 10, 'best intruder': 10, 'best 1039': 10, 'best 1200': 10, 'intruder 1039': 10, 'intruder 1200': 10, 'best intruder smith 1039': 10, 'best intruder smith 1200': 10, 'best intruder 1200 1039': 10, 'mine best': 20, 'mine intruder': 20, 'mine best intruder smith': 20, 'mine best intruder 1200': 20, 'mine best intruder 1039': 20}

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
- samples=3798 positive_rate=0.5263 loss=0.7545 accuracy=0.6486
- with_mine: samples=2024 positive_rate=0.5207 loss=0.7620 accuracy=0.6274
- without_mine: samples=1774 positive_rate=0.5325 loss=0.7462 accuracy=0.6720
### test
- samples=1013 positive_rate=0.5377 loss=0.7682 accuracy=0.6472
- with_mine: samples=621 positive_rate=0.5396 loss=0.7575 accuracy=0.6468
- without_mine: samples=392 positive_rate=0.5347 loss=0.7845 accuracy=0.6478

## Artifacts
- `dataset.npz`: train/test arrays
- `model_weights.npz`: learned weights and normalization
- `training_progress.csv`: epoch log
- `game_log.csv`: per-game lineup/winner log
- `samples_meta.csv`: per-sample metadata
- `metrics.json`: final metrics
