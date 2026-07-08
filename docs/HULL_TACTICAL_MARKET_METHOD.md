# Hull Tactical Market Method

This document explains the methodology from the Kaggle writeup:

```text
[4th Place] Technical Model, No Learning: Short Term Reversal
Hull Tactical Market Prediction
Author: YannFb
Published: 2026-07-03
```

Source:

```text
https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/4th-place-technical-model-no-learning-short-te
```

This is a standalone automation pattern document. It is not tied to any game or
other competition in this repository.

Important scope note:

```text
The writeup does not disclose the exact alpha formula.
The writeup does not provide runnable source code or shell commands.
The writeup does describe the full system philosophy and portfolio methods.
```

So this file does two things:

1. Explains the methods that are actually described in the writeup.
2. Gives local Python command shapes for implementing this kind of deterministic
   automation pipeline yourself.

This is educational documentation, not financial advice.

## Core Idea

The solution is not a machine learning solution.

It is a deterministic quantitative pipeline:

```text
pre-existing alpha signal
+ auxiliary stabilizing signals
+ inverse-volatility weighting
+ volatility targeting
+ leverage clipping
+ walk-forward validation
+ anti-overfit discipline
= competitive automated decision system
```

The author's central framing was:

```text
Portfolio = Alpha + Risk Management
```

The main lesson is that automation does not always require a learned model.
Sometimes the higher-value path is:

```text
understand the domain
write deterministic rules
validate through time
control risk
avoid over-optimization
```

## What Type Of Automation This Is

This is algorithmic automation.

It is not:

```text
feed data into a neural network
train for many epochs
submit learned predictions
```

It is:

```text
define a signal
transform features consistently
combine signals with robust rules
scale exposure based on risk
validate without leaking the future
submit deterministic allocations
```

The "model" is the whole system, even though there is no learned model.

## Explicit Methods In The Writeup

The writeup describes these methods:

1. Short-horizon mean-reversion alpha.
2. Walk-forward validation.
3. Minimal hyperparameter tuning.
4. Portfolio construction as the main optimization surface.
5. Inverse-volatility signal weighting.
6. Auxiliary low-alpha stabilizing signals.
7. Volatility targeting overlay.
8. Leverage clipping to respect competition constraints.
9. Feature transformation pipeline.
10. Feature selection based on allocation stability, not only prediction.
11. Avoidance of covariance optimization.
12. Avoidance of grid search and Bayesian optimization.
13. Diagnostics through equity curves, drawdowns, rolling Sharpe, and volatility.
14. Explicit acknowledgement of regime limitations.

## Method 1: No Learning

The author emphasizes that the final solution used no machine learning model.

That means:

```text
no neural network
no gradient boosting model
no fitted regression for final weights
no Bayesian hyperparameter optimization
no competition-specific alpha mining
```

The pipeline is deterministic. Given the same historical data and parameters,
it should produce the same allocation series.

Why this matters:

- It reduces the number of ways to overfit.
- It makes behavior interpretable.
- It makes debugging easier.
- It makes validation more honest.
- It forces the system to rely on domain logic and risk control.

This is a valid path to automation:

```text
automation through algorithms instead of automation through learned weights
```

## Method 2: Pre-Existing Alpha

The alpha was not discovered inside the competition data.

The author says the core signal came from prior personal quantitative research.
It belongs to the family of short-horizon mean-reversion indicators and had
already been validated outside the competition environment.

This is important.

The workflow was not:

```text
search competition data until something scores well
```

It was:

```text
bring a signal with an economic rationale
avoid tuning it heavily to the competition
focus on portfolio construction
```

That is a major anti-overfit choice.

## Method 3: Short-Horizon Mean Reversion

The writeup does not reveal the exact alpha formula.

It does identify the alpha family:

```text
short-horizon mean reversion
```

The general idea of mean reversion:

```text
if price or return moves unusually far in one short-term direction,
expect some partial reversal over the next short horizon
```

A generic mean-reversion signal often has this shape:

```text
recent_return = price_t / price_{t-k} - 1
signal_t = -zscore(recent_return)
```

or:

```text
signal_t = -(price_t - rolling_mean_t) / rolling_std_t
```

This is only illustrative. It is not the author's disclosed alpha.

The important method is not the exact formula. The important method is:

```text
use a simple signal with a prior economic reason to exist
then spend most effort on allocation and risk control
```

## Method 4: Walk-Forward Validation

Walk-forward validation respects time.

You do not randomly shuffle rows in a market strategy. Random shuffling leaks
future information into the past and creates unrealistic validation.

Walk-forward validation usually works like this:

```text
train_or_calibrate_window_1 -> test_next_period_1
train_or_calibrate_window_2 -> test_next_period_2
train_or_calibrate_window_3 -> test_next_period_3
...
combine all test periods into one out-of-sample history
```

For a deterministic strategy, "train" may mean:

```text
compute rolling statistics
set volatility estimates
set signal weights
calibrate fixed parameters
```

The rule:

```text
At time t, use only information available at or before time t.
```

## Method 5: Portfolio Construction Over Forecast Chasing

The author stopped treating the competition primarily as a forecasting problem.

Instead of focusing on:

```text
Can I predict the next return slightly better?
```

the system focused on:

```text
Given a signal, how do I turn it into a robust allocation?
```

That shift matters because the metric rewarded return while penalizing
excessive volatility. In that situation, risk engineering can dominate marginal
forecast improvements.

The automation design becomes:

```text
raw signal -> stable portfolio exposure -> controlled volatility -> clipped allocation
```

## Method 6: Auxiliary Stabilizing Signals

The main alpha was sparse. It was not active all the time.

Sparse signals can create unstable realized volatility estimates because the
portfolio may spend long periods near zero exposure and then suddenly become
active.

To stabilize the allocation, the author combined the main alpha with a few
additional low-alpha signals.

The goal of these auxiliary signals was not necessarily to predict returns well.
Their goal was to make the exposure profile smoother.

This is one of the most transferable ideas in the writeup:

```text
A feature or signal can be useful even if it is weak as a standalone predictor.
```

It may help by:

- reducing variance
- stabilizing exposure
- making rolling estimates more reliable
- smoothing allocation changes
- improving risk targeting

## Method 7: Inverse-Volatility Weighting

The author did not fit signal weights with regression.

Instead, each signal was weighted inversely to its rolling volatility.

Formula:

```text
w_i = (1 / sigma_i) / sum_j(1 / sigma_j)
```

Where:

```text
w_i      = weight for signal i
sigma_i  = rolling volatility of signal i
```

Practical implementation:

```text
signal_return_i,t = signal_i,t-1 * market_return_t
sigma_i,t = rolling_std(signal_return_i, window)
raw_weight_i,t = 1 / max(sigma_i,t, epsilon)
weight_i,t = raw_weight_i,t / sum(raw_weight_j,t)
```

Why this works:

- unstable signals get less weight
- stable signals get more weight
- no covariance matrix is needed
- no optimizer is needed
- the rule adapts across regimes

This is a robust algorithmic weighting method.

## Method 8: Why Not Mean-Variance Optimization

The writeup explicitly rejects a more complex optimization like:

```text
maximize: mu^T w / sqrt(w^T Sigma w)
```

Where:

```text
mu     = expected returns
Sigma  = covariance matrix
w      = portfolio weights
```

The reasons:

1. There were only a few signals.
2. Covariance estimates from short financial histories are noisy.
3. Estimation error can overwhelm theoretical benefits.
4. A simpler inverse-volatility rule was more reliable.

This is a major automation lesson:

```text
The mathematically fancier solution is not always the more robust solution.
```

## Method 9: Volatility Targeting

After combining the signals, the author rescaled portfolio exposure to target a
desired realized volatility.

Formula:

```text
L_t = sigma_target / sigma_hat_portfolio,t
```

Where:

```text
L_t                     = leverage multiplier at time t
sigma_target            = desired volatility level
sigma_hat_portfolio,t   = estimated realized portfolio volatility
```

Then:

```text
final_weight_t = clip(L_t * raw_weight_t, 0, 2)
```

Why volatility targeting helps:

- increases exposure when realized volatility is too low
- decreases exposure when realized volatility is too high
- creates a more consistent risk profile
- aligns the strategy with risk-aware scoring

The author used a relatively long volatility window and updated periodically.
That prevented the system from overreacting to short-lived noise.

## Method 10: Leverage Clipping

The final allocation was clipped to respect the allowed leverage range.

Generic version:

```text
final_allocation = clip(scaled_allocation, min_leverage, max_leverage)
```

In the writeup, the allocation was clipped between:

```text
0 and 2
```

Clipping matters because automated systems need hard safety bounds.

Without clipping:

- a low volatility estimate could create excessive leverage
- a bad volatility estimate could produce unstable allocations
- the strategy could violate competition constraints

Risk controls should be part of the algorithm, not an afterthought.

## Method 11: Feature Transformation Pipeline

The competition provided many features.

The author did not manually inspect every feature one by one. Instead, he
applied a consistent transformation pipeline to candidate features before
evaluating them.

A generic transformation pipeline might include:

```text
raw feature
-> lag
-> difference or return transform
-> rolling mean
-> rolling standard deviation
-> z-score
-> clipping/winsorization
-> smoothing
-> signal normalization
```

The actual transformations are not fully disclosed in the writeup.

The transferable method is:

```text
standardize feature treatment
evaluate systematically
discard most features
keep only features that improve the whole strategy
```

## Method 12: Most Features Were Discarded

The author kept only a small number of engineered features.

The selection criterion was not only:

```text
does this feature predict returns by itself?
```

It was also:

```text
does this feature improve portfolio stability?
does this feature reduce estimation noise?
does this feature improve allocation behavior?
```

That is a broader view of feature value.

Some features are useful because they improve the control system, not because
they are strong standalone predictors.

## Method 13: Minimal Hyperparameter Optimization

The author reports very little tuning:

- no grid search
- no Bayesian optimization
- no cross-validation
- no final ensemble
- only a small number of full experiments

This was intentional.

In financial time series, over-optimization is dangerous because the historical
sample is finite, noisy, and regime-dependent.

The method was:

```text
prefer robust priors
prefer simple parameters
prefer walk-forward checks
avoid searching until the leaderboard improves by accident
```

## Method 14: Diagnostics

The writeup uses diagnostic plots and metrics:

- equity curve comparison
- drawdown profile
- rolling Sharpe ratio
- realized volatility
- volatility limit comparison
- performance summary table

These diagnostics are part of the method.

An automated strategy should not only output a final score. It should show:

```text
when it wins
when it loses
how volatile it is
how deep drawdowns get
whether risk stays controlled
whether behavior changes across regimes
```

## Method 15: Regime Awareness

The author states the strategy has a natural limitation.

Because it is based on short-term mean reversion, it can underperform in
persistent low-volatility trending markets.

It is expected to do better in:

```text
volatile markets
range-bound markets
markets with short-term reversals
```

It is expected to struggle in:

```text
smooth trending markets
low-volatility bull regimes
```

This is another important automation principle:

```text
Every strategy has a regime where it is naturally weak.
```

Good documentation should say that explicitly.

## Reported Performance Summary

The writeup reports these summary values:

| Configuration | Total Return | Annual Return | Annual Volatility | Sharpe | Max Drawdown | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla signal, no volatility targeting | 258.7% | 5.2% | 7.7% | 0.68 | -13.6% | 59.6% |
| Competition signal with volatility targeting | 776.9% | 9.4% | 14.3% | 0.66 | -25.8% | 59.6% |
| Buy and hold benchmark | 215.4% | 6.3% | 19.2% | 0.33 | -63.8% | 53.9% |

The key interpretation:

```text
volatility targeting increased total and annual return,
kept volatility below buy-and-hold,
and accepted a larger drawdown than the unscaled vanilla signal.
```

The hit rate did not change between the vanilla signal and volatility-targeted
version. That supports the writeup's point: the improvement came from portfolio
construction, not from better directional prediction.

## The Full Algorithmic Pipeline

A generic version of the Hull-style pipeline:

```text
1. Load time series data.
2. Clean and align timestamps.
3. Compute returns.
4. Build main alpha signal.
5. Build auxiliary stabilizing signals.
6. Transform and normalize all signals consistently.
7. Estimate each signal's rolling volatility.
8. Compute inverse-volatility signal weights.
9. Combine signals into a raw portfolio exposure.
10. Estimate realized portfolio volatility.
11. Compute volatility-targeting leverage multiplier.
12. Apply leverage multiplier to raw exposure.
13. Clip exposure to allowed bounds.
14. Run walk-forward backtest.
15. Write diagnostics.
16. Export final deterministic submission.
```

This is automation without a learned model.

## Pseudocode

```python
prices = load_prices()
returns = prices.pct_change()

main_signal = build_short_term_reversal_signal(prices)
aux_signals = build_auxiliary_stabilizers(features)

signals = normalize_signals([main_signal, *aux_signals])

signal_returns = signals.shift(1) * returns
signal_vol = rolling_std(signal_returns, window=vol_window)

inv_vol = 1.0 / clip_lower(signal_vol, epsilon)
signal_weights = inv_vol / inv_vol.sum(axis=1)

raw_exposure = (signal_weights * signals).sum(axis=1)

portfolio_returns = raw_exposure.shift(1) * returns
portfolio_vol = rolling_std(portfolio_returns, window=portfolio_vol_window)

leverage = target_vol / clip_lower(portfolio_vol, epsilon)
leverage = update_periodically(leverage, every=n_days)

final_allocation = clip(leverage * raw_exposure, 0.0, 2.0)

backtest = run_walk_forward(final_allocation, returns)
write_diagnostics(backtest)
export_submission(final_allocation)
```

This pseudocode is not the author's exact code. It is a general implementation
shape based on the methods described.

## Local Folder Layout For This Method

If implementing this style of system locally, use a layout like:

```text
data/hull_method/
  raw/
  processed/
  features/
  signals/
  runs/
  submissions/
scripts/
  hull_method_validate_data.py
  hull_method_build_features.py
  hull_method_build_signals.py
  hull_method_walk_forward.py
  hull_method_build_portfolio.py
  hull_method_backtest.py
  hull_method_plot_diagnostics.py
  hull_method_export_submission.py
docs/
  HULL_TACTICAL_MARKET_METHOD.md
```

Generated data should stay out of git. Source scripts and docs should be
committed.

## Local Environment Commands

The writeup is tagged Python on Kaggle, and Python is a natural fit for this
kind of pipeline.

Basic setup:

```bash
cd /workspaces/Orbit-Wars-Submission
```

```bash
python --version
```

Create a virtual environment if needed:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install core packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy scikit-learn pyarrow
```

Optional packages:

```bash
python -m pip install seaborn statsmodels polars
```

For reproducibility, freeze dependencies:

```bash
python -m pip freeze > requirements-hull-method.txt
```

## Project Setup Commands

Create local folders:

```bash
mkdir -p data/hull_method/raw
mkdir -p data/hull_method/processed
mkdir -p data/hull_method/features
mkdir -p data/hull_method/signals
mkdir -p data/hull_method/runs
mkdir -p data/hull_method/submissions
```

Check git status:

```bash
git status --short
```

Add generated folders to `.gitignore` before running large experiments:

```text
data/hull_method/
```

Do not commit raw market data or generated submissions unless there is a clear
reason.

## Data Validation Command Shape

Target command:

```bash
python scripts/hull_method_validate_data.py \
  --input data/hull_method/raw/train.csv \
  --out data/hull_method/processed/data_report.json
```

The validation script should check:

- timestamp order
- missing values
- duplicate rows
- feature columns
- target/return columns
- impossible values
- train/test boundary
- leakage-prone columns

Expected output:

```text
data/hull_method/processed/data_report.json
```

Example report fields:

```json
{
  "rows": 100000,
  "columns": 120,
  "missing_cells": 0,
  "duplicate_timestamps": 0,
  "date_min": "2000-01-01",
  "date_max": "2026-01-01",
  "status": "ok"
}
```

## Feature Build Command Shape

Target command:

```bash
python scripts/hull_method_build_features.py \
  --input data/hull_method/raw/train.csv \
  --out data/hull_method/features/features.parquet \
  --config configs/hull_features.json
```

The feature builder should:

- sort by time
- compute returns
- lag features
- build rolling statistics
- standardize features
- clip outliers
- avoid future leakage

Output:

```text
data/hull_method/features/features.parquet
```

## Signal Build Command Shape

Target command:

```bash
python scripts/hull_method_build_signals.py \
  --features data/hull_method/features/features.parquet \
  --out data/hull_method/signals/signals.parquet \
  --config configs/hull_signals.json
```

The signal builder should produce:

```text
main_alpha
aux_signal_1
aux_signal_2
aux_signal_3
```

The main alpha would be the pre-existing domain signal. Auxiliary signals should
be judged by whether they improve allocation stability, not only by standalone
return.

## Walk-Forward Validation Command Shape

Target command:

```bash
python scripts/hull_method_walk_forward.py \
  --signals data/hull_method/signals/signals.parquet \
  --prices data/hull_method/processed/prices.parquet \
  --out data/hull_method/runs/walk-forward-v1 \
  --train-window 1260 \
  --test-window 63 \
  --step-size 21
```

The walk-forward script should:

- use only past data for each test period
- recompute rolling volatility from past data
- compute signal weights
- produce out-of-sample allocations
- write fold metrics
- write combined out-of-sample equity curve

Expected files:

```text
data/hull_method/runs/walk-forward-v1/config.json
data/hull_method/runs/walk-forward-v1/folds.jsonl
data/hull_method/runs/walk-forward-v1/allocations.parquet
data/hull_method/runs/walk-forward-v1/metrics.json
```

## Portfolio Construction Command Shape

Target command:

```bash
python scripts/hull_method_build_portfolio.py \
  --signals data/hull_method/signals/signals.parquet \
  --returns data/hull_method/processed/returns.parquet \
  --out data/hull_method/runs/portfolio-v1 \
  --signal-vol-window 252 \
  --epsilon 1e-8
```

The portfolio script should:

1. Compute signal returns.
2. Compute rolling signal volatility.
3. Convert volatility to inverse-volatility weights.
4. Normalize weights to sum to 1.
5. Combine signals into raw exposure.

Expected outputs:

```text
data/hull_method/runs/portfolio-v1/signal_weights.parquet
data/hull_method/runs/portfolio-v1/raw_exposure.parquet
data/hull_method/runs/portfolio-v1/metrics.json
```

## Volatility Targeting Command Shape

Target command:

```bash
python scripts/hull_method_apply_vol_target.py \
  --raw-exposure data/hull_method/runs/portfolio-v1/raw_exposure.parquet \
  --returns data/hull_method/processed/returns.parquet \
  --out data/hull_method/runs/vol-target-v1 \
  --target-vol 0.14 \
  --portfolio-vol-window 252 \
  --update-every 21 \
  --min-allocation 0.0 \
  --max-allocation 2.0
```

The volatility targeting script should:

1. Estimate realized portfolio volatility.
2. Compute leverage multiplier.
3. Update multiplier periodically.
4. Scale raw exposure.
5. Clip final exposure.

Expected outputs:

```text
data/hull_method/runs/vol-target-v1/final_allocation.parquet
data/hull_method/runs/vol-target-v1/leverage.parquet
data/hull_method/runs/vol-target-v1/metrics.json
```

The exact target volatility should be chosen through robust validation, not by
leaderboard probing.

## Backtest Command Shape

Target command:

```bash
python scripts/hull_method_backtest.py \
  --allocation data/hull_method/runs/vol-target-v1/final_allocation.parquet \
  --returns data/hull_method/processed/returns.parquet \
  --out data/hull_method/runs/backtest-v1
```

The backtest should compute:

- total return
- annual return
- annual volatility
- Sharpe ratio
- max drawdown
- hit rate
- turnover
- average exposure
- exposure distribution
- rolling volatility
- rolling Sharpe

Expected outputs:

```text
data/hull_method/runs/backtest-v1/metrics.json
data/hull_method/runs/backtest-v1/equity_curve.parquet
data/hull_method/runs/backtest-v1/drawdown.parquet
```

## Diagnostics Plot Command Shape

Target command:

```bash
python scripts/hull_method_plot_diagnostics.py \
  --backtest data/hull_method/runs/backtest-v1 \
  --out data/hull_method/runs/backtest-v1/plots
```

Plots should include:

```text
equity_curve.png
drawdown.png
rolling_sharpe.png
rolling_volatility.png
allocation.png
leverage.png
signal_weights.png
```

The writeup relied on plots to explain the behavior of the strategy. That is
part of the method.

## Export Command Shape

Target command:

```bash
python scripts/hull_method_export_submission.py \
  --allocation data/hull_method/runs/vol-target-v1/final_allocation.parquet \
  --template data/hull_method/raw/sample_submission.csv \
  --out data/hull_method/submissions/submission-v1.csv
```

The export script should:

- match the required submission schema
- enforce allocation bounds
- check for missing timestamps
- check for NaN values
- write a summary

Expected outputs:

```text
data/hull_method/submissions/submission-v1.csv
data/hull_method/submissions/submission-v1.summary.json
```

## Full Local Workflow Command Sequence

This is the complete command flow for implementing this style of system.

```bash
cd /workspaces/Orbit-Wars-Submission
```

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn pyarrow
```

```bash
mkdir -p data/hull_method/raw
mkdir -p data/hull_method/processed
mkdir -p data/hull_method/features
mkdir -p data/hull_method/signals
mkdir -p data/hull_method/runs
mkdir -p data/hull_method/submissions
```

```bash
python scripts/hull_method_validate_data.py \
  --input data/hull_method/raw/train.csv \
  --out data/hull_method/processed/data_report.json
```

```bash
python scripts/hull_method_build_features.py \
  --input data/hull_method/raw/train.csv \
  --out data/hull_method/features/features.parquet \
  --config configs/hull_features.json
```

```bash
python scripts/hull_method_build_signals.py \
  --features data/hull_method/features/features.parquet \
  --out data/hull_method/signals/signals.parquet \
  --config configs/hull_signals.json
```

```bash
python scripts/hull_method_build_portfolio.py \
  --signals data/hull_method/signals/signals.parquet \
  --returns data/hull_method/processed/returns.parquet \
  --out data/hull_method/runs/portfolio-v1 \
  --signal-vol-window 252 \
  --epsilon 1e-8
```

```bash
python scripts/hull_method_apply_vol_target.py \
  --raw-exposure data/hull_method/runs/portfolio-v1/raw_exposure.parquet \
  --returns data/hull_method/processed/returns.parquet \
  --out data/hull_method/runs/vol-target-v1 \
  --target-vol 0.14 \
  --portfolio-vol-window 252 \
  --update-every 21 \
  --min-allocation 0.0 \
  --max-allocation 2.0
```

```bash
python scripts/hull_method_backtest.py \
  --allocation data/hull_method/runs/vol-target-v1/final_allocation.parquet \
  --returns data/hull_method/processed/returns.parquet \
  --out data/hull_method/runs/backtest-v1
```

```bash
python scripts/hull_method_plot_diagnostics.py \
  --backtest data/hull_method/runs/backtest-v1 \
  --out data/hull_method/runs/backtest-v1/plots
```

```bash
python scripts/hull_method_export_submission.py \
  --allocation data/hull_method/runs/vol-target-v1/final_allocation.parquet \
  --template data/hull_method/raw/sample_submission.csv \
  --out data/hull_method/submissions/submission-v1.csv
```

Again, these scripts are command targets. They would need to be implemented if
this method were turned into a local project.

## Configuration File Shape

Use config files instead of hard-coding parameters.

Example:

```json
{
  "signal_vol_window": 252,
  "portfolio_vol_window": 252,
  "vol_update_every": 21,
  "target_vol": 0.14,
  "min_allocation": 0.0,
  "max_allocation": 2.0,
  "epsilon": 1e-8,
  "feature_clip": 5.0
}
```

Why:

- repeatability
- easier comparison
- less accidental tuning
- clearer audit trail

Every run should copy its config into the run folder.

## Run Folder Shape

Each run should write:

```text
data/hull_method/runs/<run-name>/
  config.json
  metrics.json
  folds.jsonl
  allocations.parquet
  signal_weights.parquet
  leverage.parquet
  equity_curve.parquet
  drawdown.parquet
  plots/
  run_summary.md
```

The run summary should answer:

```text
What signal version was used?
What windows were used?
What target volatility was used?
What bounds were used?
What changed from the last run?
Did performance improve out of sample?
Did risk increase?
```

## Metrics To Track

Core metrics:

```text
total_return
annual_return
annual_volatility
sharpe
sortino
max_drawdown
hit_rate
turnover
average_allocation
max_allocation
min_allocation
realized_volatility
```

Validation metrics:

```text
fold_count
mean_fold_return
median_fold_return
worst_fold_return
positive_fold_rate
fold_volatility
fold_drawdown
```

Risk metrics:

```text
drawdown_duration
tail_loss
volatility_breach_count
allocation_clip_count
leverage_change_frequency
```

## Anti-Overfit Rules

The writeup is strongly anti-overfit.

Use these rules:

1. Do not tune many parameters on one leaderboard or one backtest.
2. Do not use random train/test splits for time series.
3. Do not add features just because they improve one period.
4. Do not optimize covariance matrices unless there is enough data.
5. Do not trust a strategy that only works in one regime.
6. Do not hide drawdowns behind final return.
7. Do not change the signal repeatedly after seeing leaderboard feedback.
8. Prefer fewer parameters with stronger rationale.

The discipline is part of the method.

## Automation Lessons

This writeup is useful beyond finance because it shows a second path to
automation.

The two broad paths are:

```text
learned automation:
    model learns behavior from data or rewards

algorithmic automation:
    human defines rules, validation, and risk controls
```

The Hull method is algorithmic automation.

It shows that a high-performing automated system can come from:

- a simple signal
- robust aggregation
- conservative validation
- explicit risk constraints
- simple math
- low tuning

## When This Method Is A Good Fit

This method is a good fit when:

- there is a known domain signal
- the signal is noisy but plausible
- overfitting risk is high
- the metric penalizes volatility or instability
- risk control matters as much as raw prediction
- interpretability matters
- data is time-ordered
- future leakage is dangerous

Examples:

- allocation systems
- scheduling systems
- inventory systems
- bidding systems
- risk-controlled decision systems
- signal-combination systems
- automated rule-based operations

## When This Method Is Not Enough

This method may not be enough when:

- there is no useful domain signal
- the relationship is too complex for simple rules
- there is abundant labeled data and low leakage risk
- the objective requires perception or language understanding
- the system must adapt to many hidden states
- the environment changes faster than rules can be updated

In those cases, learned models may be more appropriate.

## Final Summary

The Hull solution is an example of disciplined algorithmic automation.

Its core pattern:

```text
use a validated signal
avoid overfitting the signal
combine signals with inverse-volatility weighting
scale exposure with volatility targeting
clip output to hard constraints
validate through time
inspect risk diagnostics
keep the system simple
```

The key lesson:

```text
Automation does not always require learning.
Sometimes the winning system is a deterministic pipeline with better validation
and better risk control.
```

