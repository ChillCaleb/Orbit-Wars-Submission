# General Model Training Roadmap

This is the general-purpose step-by-step roadmap for starting a training
project.

It points to the deeper docs that already exist in this repo, then gives a
broader map for other kinds of model training. Use this when you are asking:

```text
I want to train a model. What do I need to build, decide, run, measure, and save?
```

This file is intentionally a map, not a full encyclopedia. Some training types
need their own deeper guide before you should implement them seriously. Those
places are marked clearly.

## Existing Deep Docs In This Repo

These are already documented in detail:

| File | What it covers |
| --- | --- |
| `docs/SELF_PLAY_TRAINING_METHOD.md` | The Orbit Wars self-play reinforcement learning method, including the Python PPO starter and step-by-step construction. |
| `docs/CPP_SELF_PLAY_LOCAL_RUN.md` | The optional future C++ simulator/rollout backend, local build commands, validation, profiling, and run workflow. |
| `docs/RL_LAB.md` | Tactical lab workflow, recorded features, tactical labels, and local Orbit Wars data collection. |
| `docs/KAGGLE_RESOURCES.md` | Kaggle replay/notebook resources and integration notes. |

If the task is Orbit Wars self-play, start with:

```text
docs/SELF_PLAY_TRAINING_METHOD.md
```

If the task is making self-play faster with C++, use:

```text
docs/CPP_SELF_PLAY_LOCAL_RUN.md
```

This document is broader. It explains how to think about training projects in
general.

## The Universal Training Shape

Almost every model training project has the same skeleton:

```text
goal -> data/environment -> action/output -> loss/reward -> training loop -> evaluation -> checkpoint -> deployment/use
```

Different training styles change the details, but not the skeleton.

Examples:

| Training type | Input | Output | Feedback |
| --- | --- | --- | --- |
| Supervised classifier | rows/images/text | class label | correct label |
| Regression model | features | number | target number |
| LLM fine-tune | prompt | response | target response or preference |
| Image model | image/text/noise | image/label/embedding | target image, caption, or denoising loss |
| Self-play RL | game state | action | win/loss/reward |
| Offline RL | logged states | action policy | logged returns/rewards |
| Ranking/recommendation | user/item/context | ranked list | clicks, conversions, ratings |

The first question is always:

```text
What exact behavior am I trying to improve?
```

## Step 1: Define The Task

Write one sentence:

```text
Train a model that takes <input> and produces <output> so that <metric> improves.
```

Examples:

```text
Train a model that takes an Orbit Wars board and produces launch decisions so that win rate improves.
```

```text
Train a classifier that takes a support ticket and predicts priority so that macro F1 improves.
```

```text
Fine-tune a language model that takes a customer question and produces a grounded answer so that human preference score improves.
```

If you cannot fill in the sentence, do not start coding yet.

## Step 2: Choose The Training Family

Use this decision table:

| If the feedback is... | Training family |
| --- | --- |
| Known correct answers exist. | Supervised learning. |
| You have examples of good outputs but not exact numeric rewards. | Supervised fine-tuning or imitation learning. |
| A simulator/game/environment gives rewards after actions. | Reinforcement learning or self-play. |
| Agents compete against each other. | Self-play reinforcement learning. |
| You have logs of past decisions and rewards but cannot safely explore live. | Offline reinforcement learning or contextual bandits. |
| You need a model to prefer one output over another. | Preference modeling or reward modeling. |
| You need to adapt a large language model to a style/task. | LLM fine-tuning, LoRA, or retrieval-augmented generation. |
| You need to generate or adapt images/audio/video. | Generative model fine-tuning or embedding/model adaptation. |
| You need to rank items. | Learning to rank or recommendation training. |

If the task is Orbit Wars:

```text
training family = self-play reinforcement learning
deep doc = docs/SELF_PLAY_TRAINING_METHOD.md
```

## Step 3: Define The Evaluation Before Training

Training without evaluation is just producing artifacts.

Write down:

```text
primary_metric:
secondary_metrics:
baseline:
test_set_or_eval_opponents:
minimum_improvement_to_keep_model:
```

Examples for Orbit Wars:

```text
primary_metric: win rate against held-out agents
secondary_metrics: average reward, survival, production lead, entropy
baseline: current main.py agent
test_set_or_eval_opponents: mine, smith, best, intruder, older checkpoints
minimum_improvement_to_keep_model: beats baseline by a stable margin over enough games
```

Examples for supervised classification:

```text
primary_metric: macro F1
secondary_metrics: precision, recall, calibration, latency
baseline: logistic regression or current rules
test_set_or_eval_opponents: fixed holdout split
minimum_improvement_to_keep_model: +2% macro F1 without worse latency
```

Need a deeper guide:

```text
Evaluation/export for Orbit Wars checkpoints still needs its own dedicated doc.
```

## Step 4: Define The Data Or Environment Contract

Every training project needs a contract.

For data-based training:

```text
one example = input fields + target fields + metadata
```

For RL/self-play:

```text
one transition = observation + action + logprob + value + reward + done + metadata
```

For ranking:

```text
one query/session = context + candidate items + observed choices + labels/rewards
```

For LLM fine-tuning:

```text
one example = messages or prompt + target response + metadata
```

Document:

```text
schema:
source:
how generated:
version:
known limitations:
fields not allowed:
```

Do this before training. If the data contract keeps changing silently, training
results become impossible to compare.

## Step 5: Create A Reproducible Project Layout

Use a layout like this:

```text
project/
  data/
    raw/
    processed/
    training_runs/
    eval_runs/
  docs/
  scripts/
  src/
  tests/
  models/
  requirements.txt
  README.md
```

For this repo, generated Orbit Wars training artifacts already use folders like:

```text
data/lab/
data/self_play_runs/
data/training_runs/
data/kaggle_runs/
```

Generated data and checkpoints should usually be ignored by git. Source code and
docs should be committed.

## Step 6: Prepare The Local Environment

General Python setup:

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For this repo, if the environment is already prepared:

```bash
cd /workspaces/Orbit-Wars-Submission
python -m pip install -r requirements.txt
```

For future C++ work:

```bash
g++ --version
cmake --version
ninja --version
```

See:

```text
docs/CPP_SELF_PLAY_LOCAL_RUN.md
```

## Step 7: Build The Smallest Baseline

Before training a serious model, build a simple baseline.

Examples:

| Task | Baseline |
| --- | --- |
| Classification | Majority class, logistic regression, small tree. |
| Regression | Mean predictor, linear regression. |
| LLM response | Prompt-only baseline or retrieval baseline. |
| Orbit Wars | Current `main.py`, `smith`, `best`, `intruder`, or scripted policy. |
| Ranking | Popularity ranker or simple score formula. |
| RL/self-play | Random/scripted policy. |

The baseline tells you whether training helped.

For Orbit Wars:

```bash
python watch_match.py --players mine best --games 3 --no-save --no-open
python watch_match.py --players mine intruder --games 3 --no-save --no-open
```

## Step 8: Make A Smoke Test

A smoke test is the smallest possible end-to-end run.

It should prove:

1. Inputs load.
2. Model initializes.
3. One tiny batch or one tiny episode runs.
4. Loss/reward is computed.
5. Backprop or update runs.
6. Metrics are written.
7. A checkpoint is saved.

For Orbit Wars self-play:

```bash
python scripts/self_play_ppo.py \
  --updates 1 \
  --games-per-update 1 \
  --hidden-dim 32 \
  --batch-size 32 \
  --ppo-epochs 1 \
  --max-targets 3 \
  --run-dir data/self_play_runs/smoke-semantic-ppo
```

For a generic supervised trainer, the equivalent command should look like:

```bash
python scripts/train_supervised.py \
  --data data/processed/train.jsonl \
  --eval data/processed/valid.jsonl \
  --epochs 1 \
  --batch-size 8 \
  --limit-train-examples 64 \
  --run-dir data/training_runs/smoke
```

That script may not exist in this repo. This is the target shape.

## Step 9: Log Every Run

Every training run should write:

```text
config.json
metrics.jsonl
checkpoints/
README.md or run_summary.md
```

Recommended `metrics.jsonl` shape:

```json
{"step":1,"train_loss":0.91,"eval_metric":0.42,"lr":0.0003}
```

For RL/self-play:

```json
{"update":1,"episodes":16,"transitions":4200,"win_rate":0.53,"entropy":7.8}
```

If you cannot compare two run folders later, the training process is not
reproducible enough.

## Step 10: Save Checkpoints And Know How To Resume

A checkpoint should include:

```text
model weights
optimizer state
training step/update
arguments/config
feature dimensions
label/action mapping
metrics snapshot
random seed if relevant
```

Every trainer should have:

```bash
--run-dir
--resume
--resume-from
```

Orbit Wars self-play example:

```bash
python scripts/self_play_ppo.py \
  --resume \
  --updates 100 \
  --games-per-update 16 \
  --hidden-dim 192 \
  --batch-size 512 \
  --ppo-epochs 3 \
  --max-targets 8 \
  --opponent-mode pool \
  --run-dir data/self_play_runs/semantic-ppo-v1
```

## Step 11: Separate Training, Evaluation, And Export

Do not mix these into one mystery command.

Use three stages:

```text
train -> evaluate -> export/use
```

Training answers:

```text
Can the model optimize the training objective?
```

Evaluation answers:

```text
Is the model actually better on held-out tests/opponents?
```

Export answers:

```text
Can this model be used by the app, bot, or submission system?
```

Current Orbit Wars status:

```text
Python self-play training exists.
C++ option docs exist.
Dedicated checkpoint evaluation/export still needs deeper implementation and documentation.
```

## Step 12: Watch For Overfitting And Fake Progress

Common fake-progress signals:

- Training loss improves but validation gets worse.
- Self-play win rate improves only against weak or stale opponents.
- The model learns to exploit a simulator bug.
- Reward improves but real task quality declines.
- Metrics improve because the data split leaked answers.
- The model gets better on average but fails the important edge cases.

Countermeasures:

- Holdout data or held-out opponents.
- Fixed evaluation seeds.
- Baseline comparisons.
- Ablations.
- Manual inspection of examples.
- Periodic replay/viewer checks for game agents.

## Step 13: Scale Only After The Small Loop Works

Scaling choices:

| Need | Tooling |
| --- | --- |
| More CPU rollouts | multiprocessing, Ray, C++, Rust, distributed workers. |
| Faster simulator | C++, Rust, Numba, JAX, vectorized environment. |
| Faster neural training | GPU, mixed precision, larger batches. |
| More experiments | config files, run registry, experiment tracker. |
| Larger models | distributed training, gradient accumulation, checkpoint sharding. |

For Orbit Wars:

```text
The Python self-play loop proves the method.
The C++ backend is the optional scale path.
```

See:

```text
docs/CPP_SELF_PLAY_LOCAL_RUN.md
```

## Step 14: Decide What Needs A Deeper Guide

The rest of this file briefly explains major training families. Each one needs
its own deeper guide before serious implementation.

Use this rule:

```text
If money, leaderboard score, production behavior, or user trust depends on it,
write the deeper guide first.
```

## Training Family: Supervised Learning

Use when:

```text
You have examples with correct labels or target values.
```

Examples:

- Classify tickets.
- Predict churn.
- Detect fraud.
- Score tactical board states from labeled outcomes.
- Predict a numeric future value.

Minimum build steps:

1. Define input columns.
2. Define target column.
3. Split train/validation/test without leakage.
4. Build a baseline.
5. Train a simple model.
6. Evaluate on holdout.
7. Save model and preprocessing.
8. Export predictions.

Command shape:

```bash
python scripts/train_supervised.py \
  --train data/processed/train.jsonl \
  --valid data/processed/valid.jsonl \
  --target label \
  --model mlp \
  --epochs 20 \
  --batch-size 256 \
  --lr 0.001 \
  --run-dir data/training_runs/supervised-v1
```

Needs deeper explanation:

```text
data splitting, leakage prevention, preprocessing, calibration, class imbalance,
model selection, feature importance, and deployment.
```

## Training Family: Self-Play Reinforcement Learning

Use when:

```text
An agent can act in an environment and receive rewards.
```

Use especially when:

```text
The task is competitive and agents can play against themselves.
```

Examples:

- Orbit Wars.
- Board games.
- Strategy games.
- Multi-agent simulations.
- Automated bidding games.

Minimum build steps:

1. Define environment.
2. Define action space.
3. Define policy model.
4. Run episodes.
5. Store transitions.
6. Assign rewards.
7. Update policy.
8. Save checkpoints.
9. Evaluate against held-out opponents.
10. Scale rollouts.

Orbit Wars command:

```bash
python scripts/self_play_ppo.py \
  --updates 100 \
  --games-per-update 16 \
  --hidden-dim 192 \
  --batch-size 512 \
  --ppo-epochs 3 \
  --max-targets 8 \
  --opponent-mode pool \
  --run-dir data/self_play_runs/semantic-ppo-v1
```

Deep docs already exist:

```text
docs/SELF_PLAY_TRAINING_METHOD.md
docs/CPP_SELF_PLAY_LOCAL_RUN.md
```

Still needs deeper explanation:

```text
checkpoint evaluation/export, PFSP league management, reward shaping,
vectorized simulation, and trained-model submission packaging.
```

## Training Family: Offline Reinforcement Learning

Use when:

```text
You have logs of past decisions and outcomes but cannot safely let a model
explore freely.
```

Examples:

- Medical or financial decisions.
- Logged recommender choices.
- Operations policies from historical data.
- Game replays where you cannot run new simulations cheaply.

Minimum build steps:

1. Collect logged state/action/reward data.
2. Validate logging quality.
3. Check whether the logs cover enough possible actions.
4. Train a behavior model or value model.
5. Use conservative/offline RL methods.
6. Evaluate carefully with off-policy estimates or simulation.
7. Deploy only with guardrails.

Needs deeper explanation:

```text
off-policy evaluation, logged policy bias, conservative Q-learning,
counterfactual evaluation, and safety constraints.
```

## Training Family: Imitation Learning

Use when:

```text
You have examples of good behavior and want a model to copy them.
```

Examples:

- Learn from expert game replays.
- Copy a strong hand-written policy.
- Train from human demonstrations.

Minimum build steps:

1. Collect expert observations and actions.
2. Convert actions into labels.
3. Train supervised action prediction.
4. Evaluate in the actual environment.
5. Mix with self-play or RL if copying is not enough.

Needs deeper explanation:

```text
distribution shift, DAgger, expert data quality, action encoding,
and imitation-to-RL fine-tuning.
```

## Training Family: LLM Fine-Tuning

Use when:

```text
You need a language model to follow a specific style, format, domain, or task.
```

Examples:

- Customer support answers.
- Structured extraction.
- Domain-specific assistant behavior.
- Code transformation style.

Minimum build steps:

1. Decide whether fine-tuning is needed at all.
2. Build high-quality prompt/response examples.
3. Remove sensitive or prohibited data.
4. Split train/validation.
5. Run a tiny fine-tune.
6. Evaluate on held-out prompts.
7. Compare against prompt-only and retrieval baselines.
8. Add safety and refusal tests.

Command shape depends heavily on the provider or framework.

Needs deeper explanation:

```text
provider-specific API commands, chat/message formatting, eval harness,
safety tests, retrieval alternatives, cost control, and deployment.
```

Important note:

```text
For OpenAI, Anthropic, Hugging Face, or other current provider workflows,
check the current official docs before writing commands. Those APIs change.
```

## Training Family: LoRA Or Parameter-Efficient Fine-Tuning

Use when:

```text
You want to adapt a large model without training all weights.
```

Examples:

- Domain-specific LLM adaptation.
- Style adaptation.
- Image model adaptation.
- Small GPU fine-tunes.

Minimum build steps:

1. Choose base model.
2. Choose adapter method.
3. Format dataset.
4. Train adapter.
5. Evaluate against base model.
6. Save adapter and exact base model version.
7. Merge or load adapter at inference time.

Needs deeper explanation:

```text
model licensing, quantization, rank/alpha/dropout choices, tokenizer handling,
GPU memory planning, and export/serving.
```

## Training Family: Image, Audio, Or Video Models

Use when:

```text
The model consumes or generates media.
```

Examples:

- Image classification.
- Object detection.
- Segmentation.
- Speech recognition.
- Image generation fine-tuning.
- Video understanding.

Minimum build steps:

1. Define media format and resolution/sample rate.
2. Label or collect examples.
3. Split data carefully.
4. Add augmentation.
5. Train a baseline.
6. Evaluate with task-specific metrics.
7. Inspect samples manually.
8. Export model.

Needs deeper explanation:

```text
data licensing, augmentation, label quality, GPU memory, pretrained backbones,
media-specific metrics, and visual/audio inspection workflows.
```

## Training Family: Ranking And Recommendation

Use when:

```text
The model chooses or orders items for a user/context.
```

Examples:

- Search ranking.
- Product recommendations.
- Feed ranking.
- Candidate selection.

Minimum build steps:

1. Define user/context features.
2. Define item features.
3. Define positive/negative labels.
4. Avoid leakage from future events.
5. Train baseline ranker.
6. Evaluate offline.
7. Test online if possible.
8. Monitor feedback loops.

Needs deeper explanation:

```text
negative sampling, position bias, counterfactual evaluation, online A/B tests,
fairness, and feedback loops.
```

## Training Family: Reward Modeling And RLHF

Use when:

```text
You need a model that scores outputs according to human or preference feedback.
```

Examples:

- Rank answer quality.
- Train a helpfulness/safety reward model.
- Optimize a policy from preference pairs.

Minimum build steps:

1. Collect preference pairs or ratings.
2. Define label guidelines.
3. Train reward/preference model.
4. Evaluate agreement with held-out preferences.
5. Use reward model carefully in optimization.
6. Watch for reward hacking.

Needs deeper explanation:

```text
annotation guidelines, inter-rater agreement, preference data quality,
reward hacking, PPO/DPO variants, safety evaluation, and policy constraints.
```

## Training Family: Distributed Or Large-Scale Training

Use when:

```text
One machine is too slow or too small.
```

Examples:

- Multi-GPU neural training.
- Many rollout workers.
- Large model fine-tuning.
- Huge datasets.

Minimum build steps:

1. Make one-machine training correct.
2. Profile the bottleneck.
3. Add parallel data loading or rollout workers.
4. Add multi-GPU only when needed.
5. Make checkpoints robust.
6. Make runs resumable.
7. Track costs.

Needs deeper explanation:

```text
distributed data parallel, mixed precision, gradient accumulation,
checkpoint sharding, worker orchestration, cloud setup, and cost control.
```

## Training Family: Retrieval-Augmented Generation

Use when:

```text
The model needs access to changing or private knowledge.
```

This is often a better first step than LLM fine-tuning.

Minimum build steps:

1. Collect documents.
2. Chunk documents.
3. Embed chunks.
4. Store in a vector/search index.
5. Retrieve relevant chunks for each query.
6. Prompt the model with retrieved context.
7. Evaluate answer grounding.

Needs deeper explanation:

```text
chunking, embeddings, indexing, retrieval metrics, hallucination checks,
citations, freshness, and access control.
```

## Universal Commands Checklist

Before training:

```bash
git status --short
python --version
python -m pip install -r requirements.txt
```

For Python code:

```bash
python -m py_compile <script.py>
python <script.py> --help
```

For a smoke run:

```bash
python <train_script.py> \
  --epochs 1 \
  --batch-size 8 \
  --run-dir data/training_runs/smoke
```

For metrics:

```bash
tail -n 5 <run-dir>/metrics.jsonl
```

For generated files:

```bash
find <run-dir> -maxdepth 3 -type f | sort
```

For git:

```bash
git status --short
```

## Universal Definition Of Done

A training setup is usable when:

1. A new person can install dependencies.
2. A smoke run completes.
3. Metrics are written.
4. Checkpoints are written.
5. Training can resume.
6. Evaluation is separate from training.
7. A baseline exists.
8. A held-out test or opponent set exists.
9. The best model can be exported or used.
10. The docs explain the command path.

For Orbit Wars right now:

```text
Python self-play smoke training exists.
C++ backend option is documented but not implemented.
Evaluation/export is the next missing end-to-end piece.
```

## What To Write Next For This Repo

The next deeper docs that would make this repository more complete are:

1. `CHECKPOINT_EVALUATION_AND_EXPORT.md`
   - Load self-play checkpoints.
   - Evaluate against fixed agents.
   - Compare old/new checkpoints.
   - Export a trained policy into a usable Kaggle agent.

2. `SUPERVISED_TRAINING_GUIDE.md`
   - General supervised learning from CSV/JSONL.
   - Data splits.
   - Leakage checks.
   - Metrics.
   - Baselines.

3. `LLM_FINE_TUNING_GUIDE.md`
   - Provider-specific fine-tuning.
   - Dataset formatting.
   - Evaluation harness.
   - Cost/safety checks.

4. `OFFLINE_RL_GUIDE.md`
   - Logged decisions.
   - Off-policy evaluation.
   - Conservative methods.
   - Safety constraints.

This file tells you when those guides are needed. It does not replace them.

## Final Mental Model

Training is not just:

```text
pick model -> run fit()
```

It is:

```text
define goal
define data or environment
define feedback
build baseline
run tiny smoke test
train
evaluate
checkpoint
resume
export
compare
scale only after the loop works
```

For Orbit Wars, the deep path is already started in:

```text
docs/SELF_PLAY_TRAINING_METHOD.md
docs/CPP_SELF_PLAY_LOCAL_RUN.md
```

For other training projects, use this roadmap to choose the right family, then
write or find the deeper guide for that family before building the serious
version.

