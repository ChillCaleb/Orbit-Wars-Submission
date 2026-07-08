# Self-Play Training Method

This document explains the self-learning training method added in
`scripts/self_play_ppo.py`: what it is, why it matters for Orbit Wars, how it
relates to the high-ranking Kaggle writeups, and the exact commands to run it.

Short answer: yes, this is a training method. More specifically, it is
self-play reinforcement learning. The model is the thing being trained. The
self-play loop is the method that generates experience and improves the model.

## The Core Idea

In normal supervised learning, you train from examples:

```text
input -> correct answer
```

In self-play reinforcement learning, you train from consequences:

```text
state -> action -> game continues -> final result -> update policy
```

The model does not need a human to label every move. It needs:

1. An environment it can play in.
2. A set of legal or useful actions.
3. A reward signal that says how well the episode ended.
4. A training algorithm that nudges the model toward choices that led to better
   outcomes.
5. Enough repeated games for useful behavior to emerge.

For Orbit Wars, the environment is the Kaggle `orbit_wars` game. The actions
are semantic ship-launch decisions. The reward is win/loss at the end of a
match. The training algorithm in the starter script is PPO.

## What "Self-Learning" Means Here

The term "self-learning" can be a little slippery.

This is not unsupervised learning in the strict textbook sense. It is not just
finding patterns in unlabeled data.

It is also not supervised imitation learning. There is no fixed file of expert
moves that the model copies.

This setup is closer to:

```text
self-play reinforcement learning
```

The model learns by playing games against itself or against earlier versions of
itself. It receives outcome feedback and updates its policy. That makes it
"self-learning" in the practical Kaggle/game-AI sense: the data comes from the
agent interacting with the environment, not from a hand-labeled training set.

## Method Versus Model

It helps to separate four things:

| Layer | Meaning in this project |
| --- | --- |
| Environment | The Orbit Wars simulator that advances the game after actions are submitted. |
| Action space | The choices the model is allowed to make, such as "send all" or "hold with buffer." |
| Model | The neural network that scores those choices and estimates board value. |
| Training method | The loop that plays games, collects decisions, assigns rewards, and updates the model. |

The top Kaggle solutions were powerful because they had the method working at
scale. Their model architecture mattered, but the important pattern was:

```text
define actions + run huge self-play + update policy + keep stronger checkpoints
```

The diagrams and rules in the writeups usually describe the training system and
action interface. The learned behavior comes from running the loop many times.

## Why This Matters More Than Hand Tuning

A hand-written Orbit Wars bot has to make many fragile choices:

- Which neutral planet is worth taking?
- How many ships should launch?
- When should it defend instead of expand?
- When should it attack the leader?
- When is a transfer useful?
- How much overkill is acceptable?
- How does the answer change in 2-player versus 4-player games?

You can write rules for these decisions, but the rule set grows fast. Every new
rule interacts with the old rules.

Self-play attacks the problem differently. You define a controlled set of
choices, then let training discover which choices work in which situations.

That is the big difference between "I wrote a clever policy" and "I built a
system that can improve policies."

## How This Codebase Implements The Starter Version

The current starter implementation lives here:

```text
scripts/self_play_ppo.py
```

It is not yet medal-scale infrastructure. It is the first version of the method
inside this repository.

It gives the repo:

- A learnable policy model.
- A semantic action space.
- A self-play rollout loop.
- PPO updates.
- Checkpoints.
- Metrics.
- A place to attach a faster simulator or stronger league later.

The script intentionally reuses existing project logic from `main.py` instead
of inventing raw geometry from scratch.

Important borrowed functions include:

```text
main._parse
main._distance
main._quadrant
main._attack_measurement
main._planned_capture_need
main._offensive_capture_need
```

That means the model starts from the language of your current bot: planet
ownership, attack ETA, capture need, safety margin, source/target relationship,
and tactical ship counts.

## How To Make It Step By Step

This is the construction recipe. If you were making the self-play trainer again
from scratch, these are the pieces to build, in this order.

The important mental shift is this:

```text
do not start by asking "what neural network should I use?"
start by asking "what loop will create training experience?"
```

The neural network is only one part of the machine. The machine is:

```text
environment + action builder + policy + rollout collector + reward assignment + optimizer + checkpoint loop
```

### Step 1: Start From The Repo Root

Run everything from the project root:

```bash
cd /workspaces/Orbit-Wars-Submission
```

Check what is already changed:

```bash
git status --short
```

Make sure dependencies are available:

```bash
python -m pip install -r requirements.txt
```

The required packages are already listed in `requirements.txt`, especially:

```text
kaggle-environments
numpy
torch
```

### Step 2: Confirm The Existing Game Hooks

Before writing training code, find the helpers that already understand Orbit
Wars geometry and observations.

Commands:

```bash
grep -n "def _parse" main.py
grep -n "def _attack_measurement" main.py
grep -n "def _planned_capture_need" main.py
grep -n "def _offensive_capture_need" main.py
grep -n "def extract_observation" agent_lab.py
grep -n "def extract_reward" agent_lab.py
grep -n "def extract_status" agent_lab.py
```

Why this step matters:

- `main._parse` turns a Kaggle observation into planets, fleets, angular
  velocity, and comet data.
- `main._attack_measurement` tells whether a source can reach a target and gives
  angle/ETA information.
- `main._planned_capture_need` estimates ships needed for a neutral capture.
- `main._offensive_capture_need` estimates ships needed for an enemy capture.
- `agent_lab.extract_observation`, `extract_reward`, and `extract_status` make
  the Kaggle environment state easier to read.

This prevents the self-play model from learning raw geometry from zero. It gets
a better action interface on day one.

### Step 3: Create The Trainer File

The trainer lives here:

```text
scripts/self_play_ppo.py
```

If building it from scratch, create one script with these responsibilities:

1. Parse CLI arguments.
2. Build semantic action candidates.
3. Define the policy/value model.
4. Wrap the model as a Kaggle-compatible agent.
5. Run games.
6. Store transitions.
7. Assign rewards after the game ends.
8. Run PPO updates.
9. Save metrics and checkpoints.

The top of the file needs these import groups:

```python
import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kaggle_environments import make
from torch import nn
from torch.distributions import Categorical

import main as live_agent
from agent_lab import extract_observation, extract_reward, extract_status
from tactical_features import obs_get
```

The key design choice is importing `main as live_agent`. That gives the trainer
access to the same tactical geometry used by the hand-written bot.

### Step 4: Define Run Paths And Action Names

The trainer needs a consistent place to write outputs:

```python
ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "data" / "self_play_runs"
```

Then define the semantic actions:

```python
ACTION_KINDS = ("send_all", "sortie", "hold", "kill_at_arrival")
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_KINDS)}
```

This is the first major simplification. The model does not output raw arbitrary
actions. It chooses among a small set of meaningful move types.

### Step 5: Add Data Classes

The trainer needs simple containers for actions, decisions, stored training
data, and episode summaries.

Build these concepts:

```python
@dataclass(frozen=True)
class Candidate:
    source_id: int
    target_id: int
    kind: str
    ships: int
    angle: float
    eta: float
    features: tuple[float, ...]
```

```python
@dataclass
class SourceDecision:
    source_id: int
    selected_key: str | None
```

```python
@dataclass
class Transition:
    obs: dict[str, Any]
    player: int
    decisions: list[SourceDecision]
    old_logprob: float
    old_value: float
    reward: float = 0.0
```

```python
@dataclass
class EpisodeResult:
    seed: int
    player_count: int
    rewards: list[float]
    shaped_rewards: list[float]
    statuses: list[str]
    winners: list[int]
    steps: int
    transitions: int
```

Why these exist:

- `Candidate` is one possible move the model may choose.
- `SourceDecision` remembers which candidate was selected for each source.
- `Transition` is what PPO trains from later.
- `EpisodeResult` is for logging and debugging.

### Step 6: Build Feature Functions

The model needs numbers, not Python game objects.

Build feature groups in this order:

1. Global board features.
2. Per-planet features.
3. Source-target action features.

Global features should answer:

```text
What does the whole board look like for this player?
```

Examples:

```text
step / total_steps
player_count / 4
num_planets / max_planets
num_fleets / max_fleets
owned_planets / max_planets
neutral_planets / max_planets
enemy_planets / max_planets
own_ships / scale
enemy_ships / scale
production_difference / scale
```

Planet features should answer:

```text
What kind of planet is this?
```

Examples:

```text
ships
production
radius
x
y
is_static
is_big
is_owned_by_me
is_neutral
quadrant_one_hot
```

Edge features should answer:

```text
What would it mean to move from this source to this target with this action?
```

Examples:

```text
distance
eta
ships_sent
fraction_of_source_sent
capture_margin
same_quadrant
target_is_enemy
target_is_neutral
action_type_one_hot
```

This is where the training method benefits from your previous hand policy work.
The model is learning over tactical concepts, not blind coordinates.

### Step 7: Convert Semantic Actions Into Ship Counts

Each semantic action needs a rule that turns it into an actual ship count.

The starter logic is:

```text
send_all:
    send every available ship

kill_at_arrival:
    send roughly the estimated capture need

hold:
    send estimated capture need plus a buffer

sortie:
    send most ships but leave a reserve behind
```

This is important. Self-play does not mean "no rules." It means the rules define
the action language, then the model learns when to use each action.

### Step 8: Build Candidate Actions For Each Turn

For each observation:

1. Parse the board with `live_agent._parse`.
2. Find planets owned by the current player.
3. Find targets not owned by the current player.
4. For each source-target pair, call `live_agent._attack_measurement`.
5. Drop targets that are unreachable or too far away.
6. Rank the remaining targets.
7. Keep only `--max-targets` per source.
8. Create semantic candidates for each kept target.

The rough pseudocode is:

```python
for source in owned_planets:
    ranked_targets = []
    for target in non_owned_planets:
        measurement = live_agent._attack_measurement(...)
        if not measurement.clear:
            continue
        if measurement.eta > horizon:
            continue
        ranked_targets.append((measurement, target))

    for target in best_ranked_targets:
        for kind in ACTION_KINDS:
            ships = semantic_ship_count(kind, source, target, ...)
            if ships <= 0:
                continue
            features = candidate_features(...)
            candidates.append(Candidate(...))
```

This creates a small menu of good-enough possible actions. The model chooses
from the menu.

### Step 9: Define The Policy/Value Model

The starter model has three parts:

| Network | Job |
| --- | --- |
| `edge_net` | Score one source-target semantic action. |
| `noop_net` | Score doing nothing from a source. |
| `value_net` | Estimate how good the whole board is. |

The structure is an MLP:

```python
nn.Linear(input_dim, hidden_dim)
nn.Tanh()
nn.Linear(hidden_dim, hidden_dim)
nn.Tanh()
nn.Linear(hidden_dim, output_dim)
```

This is deliberately simple. The purpose is to prove the self-play loop before
spending complexity on architecture.

### Step 10: Select Actions During A Game

For each source planet:

1. Build one no-op score.
2. Build one score for every candidate action.
3. Turn scores into a categorical distribution.
4. Sample one option.
5. If the selected option is not no-op, emit a Kaggle move.
6. Add that decision's log probability to the total turn log probability.
7. Store which candidate key was selected.

Conceptually:

```python
logits = [noop_logit, candidate_1_logit, candidate_2_logit, ...]
dist = Categorical(logits=logits)
selected = dist.sample()
logprob += dist.log_prob(selected)
entropy += dist.entropy()
```

The trainer stores the sampled decisions so PPO can later recompute the new
probability of those same decisions.

### Step 11: Wrap The Model As A Kaggle Agent

Kaggle expects each agent to be callable. On every turn, the environment passes
an observation and configuration, and the agent returns actions.

The wrapper does this:

1. Extract the observation.
2. Extract the player id.
3. Call the policy selection function.
4. Store a `Transition` if this is a training-controlled player.
5. Return the selected moves.

The wrapper is the bridge between:

```text
Kaggle environment API
```

and:

```text
PyTorch policy training
```

### Step 12: Run One Episode

An episode is one complete game.

Implementation outline:

```python
env = make(
    "orbit_wars",
    configuration={"seed": seed, "randomSeed": seed},
    debug=False,
)
env.run(agents)
final_step = env.steps[-1]
```

After the game ends:

1. Read final rewards.
2. Determine winner or tie.
3. Assign shaped rewards.
4. Attach the shaped reward to every stored transition from that player.
5. Return episode metrics and transition rows.

Starter reward:

```text
unique winner: +1
losers in 2-player: -1
losers in 4-player: -1 / 3
ties: 0
```

This is simple and noisy, but it is enough to make the training loop real.

### Step 13: Implement PPO Update

The trainer collected:

```text
old_logprob
old_value
reward
observation
selected_decisions
```

PPO recomputes the probability of the same selected decisions under the updated
model.

Core math:

```text
advantage = reward - old_value
ratio = exp(new_logprob - old_logprob)
unclipped = ratio * advantage
clipped = clamp(ratio, 1 - clip, 1 + clip) * advantage
policy_loss = -mean(min(unclipped, clipped))
value_loss = mse(new_value, reward)
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

Then:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
```

That is the update that turns game outcomes into changed policy behavior.

### Step 14: Add Checkpoints And Metrics

A training run must write enough information to continue later and to debug
what happened.

Write:

```text
metrics.jsonl
episodes.jsonl
checkpoints/policy_update_000001.pt
README.md
```

Each metric row should include:

```text
update
episodes
transitions
mean_steps
slot0_win_rate
slot0_mean_reward
loss
policy_loss
value_loss
entropy
valid
optimizer_steps
```

Each checkpoint should include:

```text
model_state
optimizer_state
args
stats
feature dimensions
action names
```

This is the difference between a one-off experiment and a reusable training
method.

### Step 15: Add CLI Arguments

Expose the knobs that matter:

```text
--updates
--games-per-update
--players
--seed
--device
--hidden-dim
--lr
--batch-size
--ppo-epochs
--clip-coef
--value-coef
--entropy-coef
--max-grad-norm
--temperature
--horizon
--max-targets
--opponent-mode
--save-every
--run-dir
--resume
--resume-from
```

Do this before large training. If the knobs are hard-coded, every experiment
becomes harder to reproduce.

### Step 16: Ignore Generated Runs

Training will write metrics and checkpoints. Those should not be committed by
default.

Add this to `.gitignore`:

```text
data/self_play_runs/
```

Then check status:

```bash
git status --short
```

You should see the trainer and docs as source changes, but not checkpoint files
from smoke tests.

### Step 17: Compile-Check The Script

Before running games:

```bash
python -m py_compile scripts/self_play_ppo.py
```

This catches syntax errors without waiting for Kaggle environment startup.

### Step 18: Check The CLI

Run:

```bash
python scripts/self_play_ppo.py --help
```

The goal is to prove that:

1. Imports work.
2. Argument parsing works.
3. The script can start without training.

Kaggle/OpenSpiel may print warnings. That is normal in this environment.

### Step 19: Run The Smallest End-To-End Test

Do not start with a huge run. First prove the whole loop works:

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

This should produce:

```text
data/self_play_runs/smoke-semantic-ppo/README.md
data/self_play_runs/smoke-semantic-ppo/episodes.jsonl
data/self_play_runs/smoke-semantic-ppo/metrics.jsonl
data/self_play_runs/smoke-semantic-ppo/checkpoints/policy_update_000001.pt
```

Inspect metrics:

```bash
tail -n 1 data/self_play_runs/smoke-semantic-ppo/metrics.jsonl
```

The important fields are:

- `transitions` should be greater than zero.
- `optimizer_steps` should be greater than zero.
- `valid` should be greater than zero.
- A win is not required.

### Step 20: Run A Small Real Training Session

Once smoke passes:

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

Monitor:

```bash
tail -f data/self_play_runs/semantic-ppo-v1/metrics.jsonl
```

At this stage, you are not looking for a miracle. You are looking for a stable
training process that keeps producing transitions, checkpoints, and non-collapsed
entropy.

### Step 21: Resume Instead Of Restarting

To continue the same run:

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

The model size must match the checkpoint. Keep `--hidden-dim` the same.

### Step 22: Add Evaluation Next

The trainer is only half the system. The next file to make should evaluate
checkpoints against fixed opponents.

That evaluation script should:

1. Load a checkpoint.
2. Wrap it as an Orbit Wars agent.
3. Run matches against `mine`, `smith`, `best`, `intruder`, and older
   checkpoints.
4. Write win-rate tables.

Without this step, you only know that training is running. You do not know that
the model is improving.

### Step 23: Add Scale Last

Only after the loop and evaluator work should you chase scale.

Scale upgrades:

1. More parallel workers.
2. Faster simulator.
3. Checkpoint league.
4. PFSP opponent sampling.
5. Better reward shaping.
6. Export trained checkpoint into a Kaggle submission agent.

This order matters. Scaling a broken loop just produces bad data faster.

## The Training Loop

The high-level loop is:

```text
for each update:
    play N self-play games
    record the actions the policy chose
    wait until each game ends
    assign each recorded decision the final game reward
    run PPO updates on those decisions
    write metrics
    save a checkpoint
```

The actual starter script does this:

1. Create or load a policy model.
2. For each training update, run multiple Orbit Wars games.
3. During every turn, build candidate actions from the current board.
4. Ask the model to choose among those candidates.
5. Submit the chosen moves to the Kaggle environment.
6. Store the observation, selected decisions, log probability, and value
   estimate.
7. At the end of the game, assign terminal shaped rewards.
8. Use PPO to make selected winning decisions more likely and losing decisions
   less likely.
9. Save metrics and checkpoints.

## What The Model Sees

The policy does not receive pixels or a raw replay video. It receives structured
features.

The starter script builds three groups of features:

### Global Features

Examples:

- Current step.
- Player count.
- Number of planets.
- Number of fleets.
- Owned planet count.
- Neutral planet count.
- Enemy planet count.
- Own ship count.
- Enemy ship count.
- Production difference.

These help the value model answer:

```text
How good is this whole board for me?
```

### Planet Features

Examples:

- Ships.
- Production.
- Radius.
- X/Y position.
- Static versus rotating.
- Big versus small.
- Owned by me.
- Neutral.
- Quadrant.

These describe sources and targets.

### Edge Features

An edge means:

```text
source planet -> target planet using one action type
```

Examples:

- Distance.
- Estimated arrival time.
- Ships sent.
- Fraction of source ships used.
- Capture margin.
- Same quadrant or not.
- Target is enemy.
- Target is neutral.
- Action type one-hot.

This lets the model score a concrete tactical proposal:

```text
from planet 12, send 37 ships to planet 8 using hold-with-buffer
```

## The Semantic Action Space

The script does not ask the model to output arbitrary angles and arbitrary ship
counts directly. That would be a much harder learning problem.

Instead, it asks the model to choose among useful semantic actions that can be
converted into real Orbit Wars moves.

Current action kinds:

| Action | Meaning |
| --- | --- |
| `send_all` | Send all available ships from a source to a target. |
| `sortie` | Send most ships while keeping a local reserve. |
| `hold` | Send enough to capture plus an added holding buffer. |
| `kill_at_arrival` | Send roughly enough to capture or punish at arrival. |
| no-op | Do nothing from that source this turn. |

For every owned source planet, the script:

1. Finds reachable enemy or neutral targets.
2. Filters targets by an ETA horizon.
3. Keeps only the most promising targets.
4. Builds candidate semantic moves.
5. Lets the policy choose no-op or one candidate.

This is the same broad idea seen in strong self-play systems: do not make the
network learn every low-level mechanic from zero if you can provide a useful
action interface.

## Why PPO Is Used

PPO means Proximal Policy Optimization.

It is a reinforcement learning algorithm that updates a policy without changing
it too violently in one step.

The starter script stores:

- The old probability of the selected action.
- The old value estimate.
- The final return.
- The selected action decisions.

Then PPO compares:

```text
new probability of selected action
old probability of selected action
advantage from final reward - old value estimate
```

If a decision was part of a win, PPO tends to make similar decisions more
likely. If it was part of a loss, PPO tends to make similar decisions less
likely. The clipping term prevents one update from moving the policy too far.

This is a starter implementation, so it uses terminal rewards only. That is
simple but noisy. A stronger future version would add denser rewards or a better
value target.

## Opponent Modes

The script has two opponent modes.

### `self`

Every player uses the current model.

This is pure current-policy self-play:

```bash
--opponent-mode self
```

It is simple and useful at the beginning.

### `pool`

The training slot uses the current model and the other slots can use a recent
checkpoint sampled from the checkpoint folder:

```bash
--opponent-mode pool
```

This is the beginning of a league. It is not full PFSP yet, but it creates the
right place for it.

PFSP means Prioritized Fictitious Self Play. In a mature setup, the trainer does
not sample opponents randomly. It prefers opponents that are hard but beatable,
often based on measured win rates.

## Files Written By Training

Each run writes to:

```text
data/self_play_runs/<run-name>/
```

Files:

| File | Purpose |
| --- | --- |
| `README.md` | Snapshot of the run arguments and method. |
| `metrics.jsonl` | One JSON row per PPO update. |
| `episodes.jsonl` | One JSON row per played episode. |
| `checkpoints/policy_update_*.pt` | PyTorch model and optimizer checkpoints. |

The folder is ignored by git:

```text
data/self_play_runs/
```

That keeps local experiments and large checkpoints out of normal commits.

## Commands From A Clean Repo State

Run commands from the repository root:

```bash
cd /workspaces/Orbit-Wars-Submission
```

Install dependencies if the environment is not already prepared:

```bash
python -m pip install -r requirements.txt
```

Compile-check the trainer:

```bash
python -m py_compile scripts/self_play_ppo.py
```

Show available options:

```bash
python scripts/self_play_ppo.py --help
```

Expect Kaggle/OpenSpiel/LiteLLM import warnings in this environment. They are
noisy, but they do not mean the self-play script failed. The important thing is
that the command reaches the help text or training output.

## Smoke Test

Use a tiny run first. This proves that the script can:

- Start the Kaggle environment.
- Generate semantic actions.
- Play a full episode.
- Collect transitions.
- Run PPO.
- Write a checkpoint.

Command:

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

Check the produced files:

```bash
find data/self_play_runs/smoke-semantic-ppo -maxdepth 3 -type f | sort
```

Expected shape:

```text
data/self_play_runs/smoke-semantic-ppo/README.md
data/self_play_runs/smoke-semantic-ppo/checkpoints/policy_update_000001.pt
data/self_play_runs/smoke-semantic-ppo/episodes.jsonl
data/self_play_runs/smoke-semantic-ppo/metrics.jsonl
```

Inspect the latest metrics row:

```bash
tail -n 1 data/self_play_runs/smoke-semantic-ppo/metrics.jsonl
```

An example successful smoke run produced:

```json
{"entropy": 9.673886609881112, "episodes": 1, "loss": 0.3712713825568724, "mean_steps": 179.0, "optimizer_steps": 12.0, "policy_loss": -0.004454968518085694, "slot0_mean_reward": -1.0, "slot0_win_rate": 0.0, "transitions": 356, "update": 1, "valid": 356.0, "value_loss": 0.9449304503001524}
```

Do not over-interpret one smoke run. A loss in the smoke test is normal. The
goal is only to prove the loop works.

## First Real Local Run

After the smoke test, run a small but real local training session:

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

This is still small compared with top leaderboard training. It is a local
development run, not a final competition-scale run.

Monitor metrics while it runs:

```bash
tail -f data/self_play_runs/semantic-ppo-v1/metrics.jsonl
```

Watch for:

- `transitions`: should be nonzero.
- `optimizer_steps`: should be nonzero.
- `entropy`: should not collapse immediately to near zero.
- `slot0_win_rate`: noisy, but useful over many updates.
- `value_loss`: can be noisy because the starter version uses terminal rewards.

## Resume Training

Resume from the latest checkpoint in the same run directory:

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

Resume from a specific checkpoint:

```bash
python scripts/self_play_ppo.py \
  --resume \
  --resume-from data/self_play_runs/semantic-ppo-v1/checkpoints/policy_update_000050.pt \
  --updates 100 \
  --games-per-update 16 \
  --hidden-dim 192 \
  --batch-size 512 \
  --ppo-epochs 3 \
  --max-targets 8 \
  --opponent-mode pool \
  --run-dir data/self_play_runs/semantic-ppo-v1
```

Keep `--hidden-dim` consistent with the checkpoint.

## Useful Knobs

| Argument | What it controls | Practical effect |
| --- | --- | --- |
| `--updates` | Number of PPO update cycles. | More updates means longer training. |
| `--games-per-update` | Episodes collected before each PPO update. | More games gives more stable updates. |
| `--players` | 2-player or 4-player games. | 4-player is more chaotic and slower. |
| `--hidden-dim` | Width of the policy/value MLP. | Bigger model can learn more but runs slower. |
| `--batch-size` | Transitions per PPO minibatch. | Larger batches are steadier if enough data exists. |
| `--ppo-epochs` | Reuse collected data this many times. | More epochs can learn more but overfit each batch. |
| `--entropy-coef` | Exploration pressure. | Higher values keep action choices more random. |
| `--temperature` | Sampling temperature for decisions. | Higher values explore more; lower values exploit more. |
| `--horizon` | Max ETA for candidate targets. | Larger horizon gives more actions but more noise. |
| `--max-targets` | Targets kept per source planet. | Larger values increase action variety and compute. |
| `--opponent-mode` | Current self-play or checkpoint pool. | Pool is closer to stronger self-play practice. |
| `--save-every` | Checkpoint interval. | Lower values save more often. |

## How To Tell If It Is Learning

Do not judge from one game or one update.

Good signs over time:

- The policy keeps producing valid transitions.
- Entropy declines slowly, not instantly.
- Win rate improves against older checkpoints.
- Recent checkpoints beat earlier checkpoints in evaluation matches.
- The agent discovers recurring tactical patterns without those patterns being
  hard-coded.

Bad signs:

- Transitions are zero.
- Entropy collapses immediately.
- The model learns to do nothing.
- It overfits to one opponent checkpoint.
- It gets better in self-play but worse against fixed benchmark agents.

The most important evaluation is not the training reward. It is checkpoint
matches against held-out opponents.

## Evaluating Checkpoints

The current script trains and saves checkpoints, but it does not yet export a
checkpoint as a Kaggle submission agent. The next evaluation layer should:

1. Load a checkpoint.
2. Wrap it in an agent function.
3. Run it through `watch_match.py` or a dedicated evaluator.
4. Compare it against `mine`, `smith`, `best`, `intruder`, and older
   checkpoints.

Until that wrapper exists, the training metrics are useful for debugging the
method, not for claiming leaderboard strength.

## Why This Is Not Yet Medal-Scale

The starter trainer uses Kaggle's Python environment directly. That is correct
and convenient, but slow.

Top self-play systems usually need:

- Very fast simulation.
- Parallel rollout workers.
- A checkpoint league.
- Opponent sampling based on win rates.
- Automatic evaluation against fixed benchmarks.
- Model export into the competition submission format.
- Large numbers of games.

The biggest missing piece is scale. The method is correct in shape, but the
throughput is not yet close to the strongest writeups.

In plain terms:

```text
this repo now has the self-play loop
it does not yet have the self-play factory
```

## What Would Make This Stronger Next

Recommended next steps:

1. Add a checkpoint evaluation script.
2. Add export support so a trained checkpoint can become a Kaggle agent.
3. Add a faster local simulator or vectorized rollout path.
4. Add proper PFSP opponent sampling based on measured win rates.
5. Add denser reward signals for captures, production gain, leader denial, and
   survival.
6. Add 4-player-specific training and evaluation.
7. Compare trained checkpoints against the current hand policy, not only
   against themselves.

The first practical upgrade should be evaluation. Training without evaluation
can look busy while going nowhere. Evaluation tells us whether checkpoints are
actually getting stronger.

## Generalizing This Method To Other Tasks

This methodology can train models for other tasks if the task can be expressed
as an environment with actions and rewards.

Good fits:

- Board games.
- Strategy games.
- Simulated markets.
- Routing and scheduling.
- Resource allocation.
- Security attack/defense simulations.
- Negotiation games.
- Multi-agent planning problems.

Required ingredients:

| Ingredient | Question to answer |
| --- | --- |
| State | What does the model observe? |
| Action | What can the model choose? |
| Environment | What happens after an action? |
| Reward | How do we score the outcome? |
| Rollout budget | How many attempts can we afford? |
| Evaluation | How do we know the trained policy is actually better? |

The hard parts are almost always action design, simulator speed, and evaluation.
The neural network is usually not the first bottleneck.

## Mental Model

A concise way to think about it:

```text
Rules define the playground.
Actions define the language.
Rewards define pressure.
Self-play creates experience.
PPO turns experience into a better policy.
Scale turns a decent idea into a strong competitor.
```

For Orbit Wars, your old hand policies are still valuable. They provide the
tactical language and heuristics that make the action space smarter. Self-play
then searches over that language at a scale that manual tuning cannot match.
