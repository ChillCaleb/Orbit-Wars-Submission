# C++ Self-Play Local Run Option

This document is a local runbook for a future C++ self-play backend.

Current repo status:

```text
Python self-play trainer exists: scripts/self_play_ppo.py
C++ simulator/backend does not exist yet
```

So this file is intentionally an option plan. It explains exactly how the C++
path should be built, compiled, validated, and run once we add it.

The short version:

```text
Use C++ for fast game simulation and rollout generation.
Use Python/PyTorch for training at first.
Connect them with files or a small Python extension later.
```

That gives the most useful split:

- C++ handles speed and deterministic game stepping.
- Python handles PPO, checkpoints, metrics, and experiment control.

## Why C++ Is The Right Optional Backend

The Python/Kaggle environment is the official and easiest way to run Orbit Wars.
It is also slow for large-scale self-play.

The C++ backend is not a different training idea. It is a faster engine for the
same training idea:

```text
self-play reinforcement learning
```

The C++ backend should replace the slowest part:

```text
run one game -> run another game -> run another game -> repeat many times
```

It should not start by replacing everything.

Recommended first split:

| Layer | Language | Reason |
| --- | --- | --- |
| Orbit Wars simulator | C++ | Speed, deterministic stepping, easier profiling. |
| Rollout generation | C++ | Many games per second. |
| Policy inference | C++ first with simple policies, then optional TorchScript/ONNX later. |
| PPO training | Python/PyTorch | Already available, easiest to debug and iterate. |
| Metrics and checkpoints | Python | Already part of the repo flow. |

Full C++ training with LibTorch is possible, but it is not the best first
version. First make the fast simulator correct. Then connect it to training.

## Target Directory Layout

When the C++ backend is added, use this layout:

```text
cpp/
  CMakeLists.txt
  include/
    orbit_wars_cpp/
      action.hpp
      agent.hpp
      board.hpp
      config.hpp
      fleet.hpp
      game.hpp
      geometry.hpp
      json_io.hpp
      planet.hpp
      policy.hpp
      rollout.hpp
      rng.hpp
      trace.hpp
  src/
    action.cpp
    board.cpp
    game.cpp
    geometry.cpp
    json_io.cpp
    policy.cpp
    rollout.cpp
    trace.cpp
  tools/
    orbit_wars_smoke.cpp
    orbit_wars_replay_check.cpp
    orbit_wars_selfplay.cpp
    orbit_wars_benchmark.cpp
  tests/
    CMakeLists.txt
    test_geometry.cpp
    test_capture.cpp
    test_fleet_motion.cpp
    test_determinism.cpp
    test_json_roundtrip.cpp
```

Build output should go outside `cpp/`:

```text
build/cpp-debug/
build/cpp-release/
build/cpp-asan/
```

Generated C++ rollout data should go under:

```text
data/cpp_self_play_runs/
```

## What The First C++ Version Should Do

Do not start with PPO in C++.

The first C++ version should only prove these things:

1. It can load or create an Orbit Wars initial board.
2. It can step the game forward deterministically.
3. It can execute legal launch actions.
4. It can resolve fleet arrivals and captures.
5. It can finish a game.
6. It can write a rollout trace.
7. It can match the Python/Kaggle environment on small validation cases.
8. It can run much faster than the Python environment.

Only after that should it feed rollouts into training.

## Local Prerequisites

Check your compiler tools:

```bash
g++ --version
clang++ --version
cmake --version
ninja --version
python --version
```

You only need one compiler. `g++` or `clang++` is fine.

Recommended Linux packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3-dev
```

Optional but useful:

```bash
sudo apt-get install -y clang lldb valgrind linux-tools-common
```

On macOS:

```bash
xcode-select --install
brew install cmake ninja llvm
```

On Windows, use one of these:

```text
Visual Studio Build Tools + CMake
WSL2 Ubuntu + Linux commands above
```

For this repo, WSL2 is the cleaner path if you are on Windows.

## Recommended CMake Presets

The C++ backend should support these build types:

| Build | Purpose |
| --- | --- |
| Debug | Easy debugging, assertions, symbols. |
| Release | Fast rollout generation. |
| RelWithDebInfo | Fast-ish build with symbols for profiling. |
| ASAN | Detect memory bugs. |

Target commands:

```bash
cmake -S cpp -B build/cpp-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DORBIT_WARS_ENABLE_ASSERTS=ON
```

```bash
cmake -S cpp -B build/cpp-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake -S cpp -B build/cpp-relwithdebinfo -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DORBIT_WARS_ENABLE_ASSERTS=ON
```

```bash
cmake -S cpp -B build/cpp-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DORBIT_WARS_ENABLE_ASAN=ON \
  -DORBIT_WARS_ENABLE_ASSERTS=ON
```

Build:

```bash
cmake --build build/cpp-debug -j
cmake --build build/cpp-release -j
cmake --build build/cpp-asan -j
```

Run tests:

```bash
ctest --test-dir build/cpp-debug --output-on-failure
ctest --test-dir build/cpp-release --output-on-failure
ctest --test-dir build/cpp-asan --output-on-failure
```

## Minimal CMake Shape

The initial `cpp/CMakeLists.txt` should look conceptually like this:

```cmake
cmake_minimum_required(VERSION 3.20)
project(orbit_wars_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ORBIT_WARS_ENABLE_ASSERTS "Enable simulator asserts" ON)
option(ORBIT_WARS_ENABLE_ASAN "Enable address sanitizer" OFF)

add_library(orbit_wars_core
  src/action.cpp
  src/board.cpp
  src/game.cpp
  src/geometry.cpp
  src/json_io.cpp
  src/policy.cpp
  src/rollout.cpp
  src/trace.cpp
)

target_include_directories(orbit_wars_core PUBLIC include)

if(ORBIT_WARS_ENABLE_ASSERTS)
  target_compile_definitions(orbit_wars_core PUBLIC ORBIT_WARS_ENABLE_ASSERTS=1)
endif()

if(ORBIT_WARS_ENABLE_ASAN)
  target_compile_options(orbit_wars_core PUBLIC -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(orbit_wars_core PUBLIC -fsanitize=address)
endif()

add_executable(orbit_wars_smoke tools/orbit_wars_smoke.cpp)
target_link_libraries(orbit_wars_smoke PRIVATE orbit_wars_core)

add_executable(orbit_wars_replay_check tools/orbit_wars_replay_check.cpp)
target_link_libraries(orbit_wars_replay_check PRIVATE orbit_wars_core)

add_executable(orbit_wars_selfplay tools/orbit_wars_selfplay.cpp)
target_link_libraries(orbit_wars_selfplay PRIVATE orbit_wars_core)

add_executable(orbit_wars_benchmark tools/orbit_wars_benchmark.cpp)
target_link_libraries(orbit_wars_benchmark PRIVATE orbit_wars_core)

enable_testing()
add_subdirectory(tests)
```

This is not checked into the repo yet. It is the intended shape.

## First Local Smoke Build

After the C++ files exist, run:

```bash
cd /workspaces/Orbit-Wars-Submission
```

```bash
cmake -S cpp -B build/cpp-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DORBIT_WARS_ENABLE_ASSERTS=ON
```

```bash
cmake --build build/cpp-debug -j
```

```bash
build/cpp-debug/orbit_wars_smoke --seed 20260708 --players 2 --steps 20
```

Expected smoke output shape:

```text
seed=20260708
players=2
steps_requested=20
steps_completed=20
planets=<number>
fleets=<number>
status=ok
```

The first smoke binary should not involve neural networks. It should only prove
that the simulator can initialize and step without crashing.

## Determinism Check

The same seed and same actions must produce the same result every time.

Run:

```bash
build/cpp-debug/orbit_wars_smoke \
  --seed 20260708 \
  --players 2 \
  --steps 200 \
  --policy scripted \
  --hash-final-state
```

Run it again:

```bash
build/cpp-debug/orbit_wars_smoke \
  --seed 20260708 \
  --players 2 \
  --steps 200 \
  --policy scripted \
  --hash-final-state
```

Expected:

```text
final_state_hash=<same value both times>
```

If the hash changes, stop. Training on nondeterministic bugs is a bad time.

## Unit Tests

Run the debug tests first:

```bash
ctest --test-dir build/cpp-debug --output-on-failure
```

Run release tests:

```bash
ctest --test-dir build/cpp-release --output-on-failure
```

Run sanitizer tests:

```bash
ctest --test-dir build/cpp-asan --output-on-failure
```

Minimum test list:

```text
geometry.distance_matches_python
geometry.launch_angle_reaches_target
fleet_motion.position_after_n_steps
capture.neutral_capture_exact_need
capture.enemy_capture_exact_need
determinism.same_seed_same_hash
json_io.board_roundtrip
rollout.trace_roundtrip
```

These tests matter more than speed at first. If the C++ simulator is wrong, it
will train the wrong behavior very efficiently.

## Python/Kaggle Validation Data

The C++ simulator needs to be checked against the official Python environment.

Generate a small Python trace with full board data:

```bash
python agent_lab.py \
  --players mine best \
  --games 1 \
  --seed 20260708 \
  --full-board
```

The lab writes a run folder under:

```text
data/lab/
```

Find the newest game file:

```bash
find data/lab -name 'game_*.jsonl' | sort | tail -n 5
```

Use one trace for validation:

```bash
build/cpp-debug/orbit_wars_replay_check \
  --input data/lab/<run_id>/game_000.jsonl \
  --mode python-trace \
  --strict
```

Expected:

```text
loaded_turns=<number>
checked_turns=<number>
mismatches=0
status=ok
```

If mismatches exist, print the first mismatch:

```bash
build/cpp-debug/orbit_wars_replay_check \
  --input data/lab/<run_id>/game_000.jsonl \
  --mode python-trace \
  --strict \
  --first-mismatch
```

The validator should report:

```text
turn
entity type
entity id
field name
python value
c++ value
absolute error
relative error
```

## Validation Tolerances

Use strict tolerances for integers:

```text
owner: exact
ships: exact
planet id: exact
fleet id: exact
capture result: exact
winner: exact
```

Use small tolerances for floating point:

```text
x/y position: <= 1e-6 if possible
angle: <= 1e-6 if possible
eta: exact if integer-step based, otherwise <= 1e-6
```

If Kaggle uses float behavior that differs slightly, document the difference
and make the tolerance explicit. Do not silently accept large drift.

## First Rollout Run

After smoke tests and validation pass, run a tiny self-play rollout:

```bash
mkdir -p data/cpp_self_play_runs/smoke
```

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 10 \
  --players 2 \
  --seed 20260708 \
  --policy scripted \
  --out data/cpp_self_play_runs/smoke
```

Expected files:

```text
data/cpp_self_play_runs/smoke/config.json
data/cpp_self_play_runs/smoke/episodes.jsonl
data/cpp_self_play_runs/smoke/transitions.jsonl
data/cpp_self_play_runs/smoke/summary.json
```

Inspect outputs:

```bash
head -n 3 data/cpp_self_play_runs/smoke/episodes.jsonl
head -n 3 data/cpp_self_play_runs/smoke/transitions.jsonl
```

Expected `episodes.jsonl` row shape:

```json
{"episode":0,"seed":20260708,"players":2,"steps":179,"winner":1,"rewards":[-1,1],"transitions":356}
```

Expected `transitions.jsonl` row shape:

```json
{"episode":0,"step":12,"player":0,"obs_features":[...],"action_id":42,"logprob":-1.93,"value":0.12,"reward":-1.0}
```

The exact fields can change, but keep the file line-oriented JSON. It is easy
for Python to stream.

## Benchmark The Simulator

After correctness checks, benchmark speed:

```bash
build/cpp-release/orbit_wars_benchmark \
  --games 1000 \
  --players 2 \
  --seed 20260708 \
  --policy scripted
```

Expected output shape:

```text
games=1000
total_steps=<number>
elapsed_seconds=<number>
games_per_second=<number>
steps_per_second=<number>
```

Run the same benchmark for 4-player:

```bash
build/cpp-release/orbit_wars_benchmark \
  --games 1000 \
  --players 4 \
  --seed 20260708 \
  --policy scripted
```

Track benchmark results in the run folder:

```text
data/cpp_self_play_runs/benchmarks/
```

Suggested file:

```text
data/cpp_self_play_runs/benchmarks/YYYYMMDD-HHMMSS.json
```

## Feed C++ Rollouts Into Python Training

The first integration should be file-based. It is slower than a direct binding,
but easier to debug.

Target command:

```bash
python scripts/train_from_cpp_rollouts.py \
  --rollouts data/cpp_self_play_runs/smoke/transitions.jsonl \
  --run-dir data/self_play_runs/cpp-semantic-ppo-smoke \
  --hidden-dim 192 \
  --batch-size 512 \
  --ppo-epochs 3
```

That script does not exist yet. It should reuse as much of
`scripts/self_play_ppo.py` as possible:

- `SemanticPolicy`
- PPO update math
- checkpoint writing
- metrics writing

The only difference is where transitions come from:

```text
current Python trainer:
    transitions come from Kaggle env.run(...)

C++ rollout trainer:
    transitions come from data/cpp_self_play_runs/.../transitions.jsonl
```

## Better Integration Later

Once file-based integration works, upgrade to one of these:

| Integration | Pros | Cons |
| --- | --- | --- |
| JSONL files | Easiest to debug. | Disk IO overhead. |
| Binary files | Faster and smaller. | Harder to inspect. |
| pybind11 extension | Direct Python calls into C++. | More build complexity. |
| C API + ctypes | Simple ABI boundary. | Manual marshaling. |
| TorchScript/ONNX in C++ | C++ can run neural policy directly. | More export and runtime complexity. |
| Full LibTorch training | Everything in C++. | Heavy and slower to iterate. |

Recommended order:

```text
JSONL -> binary trace -> pybind11 -> optional C++ inference
```

Do not jump straight to full C++ training.

## Policy Options For C++ Rollouts

The C++ simulator can start with simple policy modes:

```text
random
scripted
current-hand-policy-port
checkpoint-policy
```

### `random`

Useful only for stress testing.

Command:

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 100 \
  --players 2 \
  --seed 20260708 \
  --policy random \
  --out data/cpp_self_play_runs/random-smoke
```

### `scripted`

Useful for deterministic validation.

Command:

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 100 \
  --players 2 \
  --seed 20260708 \
  --policy scripted \
  --out data/cpp_self_play_runs/scripted-smoke
```

### `current-hand-policy-port`

Useful once parts of `main.py` are ported to C++.

Command:

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 1000 \
  --players 2 \
  --seed 20260708 \
  --policy current-hand-policy-port \
  --out data/cpp_self_play_runs/hand-policy-port-v1
```

### `checkpoint-policy`

Useful after the Python trainer can export a model to C++ inference format.

Command:

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 1000 \
  --players 2 \
  --seed 20260708 \
  --policy checkpoint-policy \
  --checkpoint data/self_play_runs/semantic-ppo-v1/checkpoints/policy_update_000100.pt \
  --out data/cpp_self_play_runs/checkpoint-rollouts-v1
```

This command is conceptual until a checkpoint export format exists. A raw
PyTorch `.pt` checkpoint from Python is not automatically usable in a simple C++
binary.

## Model Export Options

When C++ needs neural policy inference, use one of these:

### Option A: TorchScript

Python export command target:

```bash
python scripts/export_self_play_policy.py \
  --checkpoint data/self_play_runs/semantic-ppo-v1/checkpoints/policy_update_000100.pt \
  --format torchscript \
  --out data/self_play_runs/semantic-ppo-v1/export/policy.pt
```

C++ rollout command target:

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 1000 \
  --players 2 \
  --seed 20260708 \
  --policy torchscript \
  --model data/self_play_runs/semantic-ppo-v1/export/policy.pt \
  --out data/cpp_self_play_runs/torchscript-v1
```

Pros:

- Native PyTorch path.
- Can keep model behavior close to Python.

Cons:

- Requires LibTorch in the C++ build.
- More setup work.

### Option B: ONNX Runtime

Python export command target:

```bash
python scripts/export_self_play_policy.py \
  --checkpoint data/self_play_runs/semantic-ppo-v1/checkpoints/policy_update_000100.pt \
  --format onnx \
  --out data/self_play_runs/semantic-ppo-v1/export/policy.onnx
```

C++ rollout command target:

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 1000 \
  --players 2 \
  --seed 20260708 \
  --policy onnx \
  --model data/self_play_runs/semantic-ppo-v1/export/policy.onnx \
  --out data/cpp_self_play_runs/onnx-v1
```

Pros:

- Good inference runtime.
- Language-neutral.

Cons:

- Export quirks.
- Dynamic candidate lists may need careful shape handling.

### Option C: Write Weights To Simple Arrays

Python export command target:

```bash
python scripts/export_self_play_policy.py \
  --checkpoint data/self_play_runs/semantic-ppo-v1/checkpoints/policy_update_000100.pt \
  --format npz \
  --out data/self_play_runs/semantic-ppo-v1/export/policy_weights.npz
```

C++ rollout command target:

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 1000 \
  --players 2 \
  --seed 20260708 \
  --policy mlp-arrays \
  --weights data/self_play_runs/semantic-ppo-v1/export/policy_weights.npz \
  --out data/cpp_self_play_runs/mlp-arrays-v1
```

Pros:

- Very transparent.
- Easy to inspect.
- Good fit for a small MLP.

Cons:

- You write the MLP forward pass in C++.
- More manual checking required.

For this project, simple array export may be the easiest to understand because
the starter policy is just MLP layers with `Tanh`.

## Full Local Workflow

Once the C++ backend exists, the full local loop should look like this:

### 1. Build C++

```bash
cmake -S cpp -B build/cpp-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-release -j
```

### 2. Validate C++

```bash
ctest --test-dir build/cpp-release --output-on-failure
```

```bash
python agent_lab.py \
  --players mine best \
  --games 1 \
  --seed 20260708 \
  --full-board
```

```bash
build/cpp-release/orbit_wars_replay_check \
  --input data/lab/<run_id>/game_000.jsonl \
  --mode python-trace \
  --strict
```

### 3. Benchmark C++

```bash
build/cpp-release/orbit_wars_benchmark \
  --games 1000 \
  --players 2 \
  --seed 20260708 \
  --policy scripted
```

### 4. Generate Rollouts

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 10000 \
  --players 2 \
  --seed 20260708 \
  --policy scripted \
  --out data/cpp_self_play_runs/rollouts-v1
```

### 5. Train In Python From Rollouts

```bash
python scripts/train_from_cpp_rollouts.py \
  --rollouts data/cpp_self_play_runs/rollouts-v1/transitions.jsonl \
  --run-dir data/self_play_runs/cpp-semantic-ppo-v1 \
  --hidden-dim 192 \
  --batch-size 512 \
  --ppo-epochs 3
```

### 6. Export Policy

```bash
python scripts/export_self_play_policy.py \
  --checkpoint data/self_play_runs/cpp-semantic-ppo-v1/checkpoints/policy_update_000100.pt \
  --format npz \
  --out data/self_play_runs/cpp-semantic-ppo-v1/export/policy_weights.npz
```

### 7. Generate New Rollouts With The Trained Policy

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 10000 \
  --players 2 \
  --seed 20260709 \
  --policy mlp-arrays \
  --weights data/self_play_runs/cpp-semantic-ppo-v1/export/policy_weights.npz \
  --out data/cpp_self_play_runs/rollouts-v2
```

### 8. Repeat

```text
rollouts -> train -> export -> rollouts -> train -> export
```

That is the C++ accelerated self-play loop.

## Debug Workflow

When something breaks, switch to debug:

```bash
cmake --build build/cpp-debug -j
```

Run one game:

```bash
build/cpp-debug/orbit_wars_selfplay \
  --games 1 \
  --players 2 \
  --seed 20260708 \
  --policy scripted \
  --out data/cpp_self_play_runs/debug-one-game \
  --trace-every-step
```

Inspect:

```bash
head -n 20 data/cpp_self_play_runs/debug-one-game/transitions.jsonl
head -n 20 data/cpp_self_play_runs/debug-one-game/trace.jsonl
```

Run under a debugger:

```bash
gdb --args build/cpp-debug/orbit_wars_selfplay \
  --games 1 \
  --players 2 \
  --seed 20260708 \
  --policy scripted \
  --out data/cpp_self_play_runs/gdb-one-game
```

Inside `gdb`:

```text
run
bt
frame 0
info locals
```

Run sanitizer:

```bash
build/cpp-asan/orbit_wars_selfplay \
  --games 10 \
  --players 2 \
  --seed 20260708 \
  --policy scripted \
  --out data/cpp_self_play_runs/asan-check
```

## Profiling Workflow

Build with symbols:

```bash
cmake -S cpp -B build/cpp-relwithdebinfo -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/cpp-relwithdebinfo -j
```

Simple timing:

```bash
time build/cpp-relwithdebinfo/orbit_wars_benchmark \
  --games 10000 \
  --players 2 \
  --seed 20260708 \
  --policy scripted
```

Linux `perf`:

```bash
perf record --call-graph dwarf \
  build/cpp-relwithdebinfo/orbit_wars_benchmark \
  --games 10000 \
  --players 2 \
  --seed 20260708 \
  --policy scripted
```

```bash
perf report
```

What to look for:

- Too much allocation inside the turn loop.
- Rebuilding candidate lists inefficiently.
- Repeated JSON parsing during simulation.
- Branch-heavy capture logic.
- Slow random number generation.
- Excessive copying of board state.

## Performance Rules For The C++ Backend

Keep these rules in mind:

1. Parse JSON at the boundary, not inside the hot loop.
2. Store planets and fleets in contiguous vectors.
3. Reuse buffers between turns.
4. Avoid heap allocation inside the inner step loop.
5. Keep deterministic RNG explicit.
6. Keep simulator state separate from trace-writing state.
7. Make debug correctness easy before optimizing.
8. Add benchmarks before major optimizations.

The first C++ version should be clear. The second version can be fast.

## Data Contract Between C++ And Python

The Python trainer needs enough information to run PPO.

Each transition should include:

```text
episode
step
player
observation features or raw observation
available action features
selected action key or index
old log probability if C++ policy sampled
old value if C++ policy estimated value
terminal reward
done flag
```

There are two possible contracts.

### Contract A: C++ Generates Features

C++ writes ready-to-train features:

```json
{
  "episode": 0,
  "step": 12,
  "player": 0,
  "global_features": [0.024, 0.5, 0.68],
  "source_features": [[...], [...]],
  "candidate_features": [[...], [...]],
  "selected": [3, 0],
  "reward": -1.0
}
```

Pros:

- Python training is simple.
- C++ and Python do not both implement features.

Cons:

- Feature bugs live in C++.
- Changing features requires C++ rebuilds.

### Contract B: C++ Generates Raw States

C++ writes raw board state and selected action:

```json
{
  "episode": 0,
  "step": 12,
  "player": 0,
  "planets": [...],
  "fleets": [...],
  "selected_action": {...},
  "reward": -1.0
}
```

Pros:

- Python feature code stays authoritative.
- Easier to experiment with features.

Cons:

- More Python preprocessing.
- Larger files.

Recommended first contract:

```text
Contract B for correctness and experimentation.
Contract A later for speed.
```

## What Not To Build First

Avoid these at the beginning:

- Full LibTorch training in C++.
- GPU inference inside the C++ simulator.
- Complex distributed worker orchestration.
- Custom binary trace format before JSONL works.
- Aggressive micro-optimization before validation.
- A simulator that is fast but not proven against Kaggle behavior.

The first win is correctness. The second win is speed. The third win is scale.

## Definition Of Done For The First C++ Option

The C++ option is real when all of these pass:

```bash
cmake -S cpp -B build/cpp-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DORBIT_WARS_ENABLE_ASSERTS=ON
cmake --build build/cpp-debug -j
ctest --test-dir build/cpp-debug --output-on-failure
```

```bash
cmake -S cpp -B build/cpp-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-release -j
ctest --test-dir build/cpp-release --output-on-failure
```

```bash
build/cpp-release/orbit_wars_benchmark \
  --games 1000 \
  --players 2 \
  --seed 20260708 \
  --policy scripted
```

```bash
build/cpp-release/orbit_wars_selfplay \
  --games 100 \
  --players 2 \
  --seed 20260708 \
  --policy scripted \
  --out data/cpp_self_play_runs/done-check
```

```bash
python scripts/train_from_cpp_rollouts.py \
  --rollouts data/cpp_self_play_runs/done-check/transitions.jsonl \
  --run-dir data/self_play_runs/cpp-done-check \
  --hidden-dim 64 \
  --batch-size 128 \
  --ppo-epochs 1
```

Expected result:

```text
C++ can produce valid rollout data.
Python can train from that data.
The outputs are reproducible from a seed.
The simulator has validation coverage against Python/Kaggle traces.
```

## Practical Starting Point

When we actually start implementing C++, begin with this order:

1. `geometry.hpp/cpp`
2. `planet.hpp`
3. `fleet.hpp`
4. `board.hpp/cpp`
5. `action.hpp/cpp`
6. `game.hpp/cpp`
7. `orbit_wars_smoke.cpp`
8. unit tests
9. Python trace validator
10. self-play rollout tool
11. benchmark tool
12. Python training-from-rollouts script

That path keeps each piece visible and testable.

## Summary

C++ is not required for self-play as a method.

C++ is useful because top-level self-play needs a huge number of games, and the
official Python environment is too slow for that scale.

The best first C++ option is:

```text
C++ simulator + C++ rollout generator + Python PPO trainer
```

That gives us speed without giving up the Python training workflow that already
works in this repo.

