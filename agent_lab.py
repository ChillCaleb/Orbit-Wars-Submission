import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from kaggle_environments import make

from tactical_features import (
    TACTICAL_WEIGHTS,
    action_features,
    build_quadrant_array,
    capture_events,
    numeric_quadrant_array,
    obs_get,
    planets_from_obs,
    quadrant_name,
    role_scores,
)
from watch_match import build_lineup


ROOT = Path(__file__).resolve().parent
LAB_DIR = ROOT / "LAB"
PROGRESSION_PATH = LAB_DIR / "progression.csv"


def extract_observation(state):
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def extract_status(state):
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def extract_reward(state):
    if isinstance(state, dict):
        value = state.get("reward", 0.0)
    else:
        value = getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def serialize_obs(obs):
    return {
        "player": obs_get(obs, "player", 0),
        "step": obs_get(obs, "step", 0),
        "angular_velocity": obs_get(obs, "angular_velocity", 0.0),
        "planets": obs_get(obs, "planets", []) or [],
        "fleets": obs_get(obs, "fleets", []) or [],
        "comet_planet_ids": obs_get(obs, "comet_planet_ids", []) or [],
    }


def starter_agent(obs, _rng=None):
    planets = planets_from_obs(obs)
    player = int(obs_get(obs, "player", 0) or 0)
    moves = []
    targets = [planet for planet in planets if planet.owner != player]
    if not targets:
        return moves
    for source in [planet for planet in planets if planet.owner == player]:
        target = min(targets, key=lambda planet: (math.hypot(source.x - planet.x, source.y - planet.y), planet.id))
        ships = max(int(target.ships) + 1, 20)
        if source.ships < ships:
            continue
        angle = math.atan2(target.y - source.y, target.x - source.x)
        moves.append([source.id, angle, ships])
    return moves


def random_agent(obs, rng):
    planets = planets_from_obs(obs)
    player = int(obs_get(obs, "player", 0) or 0)
    moves = []
    targets = [planet for planet in planets if planet.owner != player]
    if not targets:
        return moves
    for source in [planet for planet in planets if planet.owner == player]:
        if source.ships < 2 or rng.random() > 0.35:
            continue
        target = rng.choice(targets)
        ships = min(source.ships, max(1, int(target.ships) + 1))
        angle = math.atan2(target.y - source.y, target.x - source.x)
        moves.append([source.id, angle, ships])
    return moves


def clean_moves(action):
    if not action:
        return []
    cleaned = []
    for move in action:
        if move is None or len(move) < 3:
            continue
        try:
            source_id = int(move[0])
            angle = float(move[1])
            ships = int(move[2])
        except (TypeError, ValueError):
            continue
        if ships <= 0:
            continue
        cleaned.append([source_id, angle, ships])
    return cleaned


def call_agent(agent, obs, config, rng):
    if isinstance(agent, str):
        if agent == "starter":
            return starter_agent(obs, rng)
        if agent == "random":
            return random_agent(obs, rng)
        return []
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


def make_recording_agent(agent, player_index, label, calls_by_step, rng):
    def wrapped(obs, config=None):
        step = int(obs_get(obs, "step", 0) or 0)
        error = None
        try:
            moves = clean_moves(call_agent(agent, obs, config, rng))
        except Exception as exc:
            moves = []
            error = f"{type(exc).__name__}: {exc}"
        calls_by_step[step][player_index] = {
            "label": label,
            "player": player_index,
            "moves": moves,
            "error": error,
        }
        return moves

    return wrapped


def winners_from_step(final_step):
    rewards = [extract_reward(state) for state in final_step]
    best = max(rewards)
    return [idx for idx, reward in enumerate(rewards) if reward == best]


def build_step_records(env, calls_by_step, labels, include_full_board):
    records = []
    player_count = len(labels)
    step_count = len(env.steps)
    for step_index, step_states in enumerate(env.steps):
        board_obs = extract_observation(step_states[0])
        step = int(obs_get(board_obs, "step", step_index) or step_index)
        calls = calls_by_step.get(step, {})
        action_rows = []
        for player_index, label in enumerate(labels):
            player_obs = extract_observation(step_states[player_index])
            call = calls.get(player_index, {"moves": [], "error": None})
            move_features = [
                action_features(move, player_obs, player=player_index)
                for move in call.get("moves", [])
            ]
            action_rows.append(
                {
                    "player": player_index,
                    "label": label,
                    "move_count": len(move_features),
                    "ships_launched": sum(item["ships"] for item in move_features),
                    "error": call.get("error"),
                    "moves": move_features,
                }
            )

        next_obs = extract_observation(env.steps[step_index + 1][0]) if step_index + 1 < step_count else None
        record = {
            "step_index": step_index,
            "step": step,
            "status": [extract_status(state) for state in step_states],
            "reward": [extract_reward(state) for state in step_states],
            "actions": action_rows,
            "captures": capture_events(board_obs, next_obs) if next_obs is not None else [],
            "quadrant_array": build_quadrant_array(board_obs, player_count=player_count),
            "numeric_arrays": {
                labels[player]: numeric_quadrant_array(board_obs, player=player, player_count=player_count)
                for player in range(player_count)
            },
            "role_labels": {
                labels[player]: role_scores(extract_observation(step_states[player]), player=player, top_n=3)
                for player in range(player_count)
            },
        }
        if include_full_board:
            record["board"] = serialize_obs(board_obs)
        records.append(record)
    return records


def run_game(agents, labels, seed, include_full_board):
    calls_by_step = defaultdict(dict)
    rng = random.Random(seed)
    wrapped_agents = [
        make_recording_agent(agent, idx, labels[idx], calls_by_step, rng)
        for idx, agent in enumerate(agents)
    ]
    env = make(
        "orbit_wars",
        configuration={"seed": int(seed), "randomSeed": int(seed)},
        debug=False,
    )
    env.run(wrapped_agents)
    records = build_step_records(env, calls_by_step, labels, include_full_board)
    return env, records


def new_player_stats(label):
    return {
        "label": label,
        "wins": 0,
        "ties": 0,
        "launches": 0,
        "ships_launched": 0,
        "errors": 0,
        "target_owner": Counter(),
        "target_kind": Counter(),
        "target_size": Counter(),
        "target_quadrant": Counter(),
        "source_quadrant": Counter(),
        "central_rotating_big": 0,
        "central_rotating_small": 0,
        "captures": 0,
        "losses": 0,
        "capture_quadrants": Counter(),
        "loss_quadrants": Counter(),
        "captured_from": Counter(),
        "final_ships": 0,
        "final_production": 0,
        "quadrant_ship_samples": defaultdict(int),
        "quadrant_prod_samples": defaultdict(int),
        "quadrant_samples": defaultdict(int),
        "first_established": {},
    }


def update_control_samples(stats, record, labels):
    for row in record["quadrant_array"]:
        quadrant = row["quadrant"]
        for player_index, label in enumerate(labels):
            player_row = row["players"].get(str(player_index), {})
            stats[label]["quadrant_ship_samples"][quadrant] += int(player_row.get("ships", 0))
            stats[label]["quadrant_prod_samples"][quadrant] += int(player_row.get("production", 0))
            stats[label]["quadrant_samples"][quadrant] += 1
            if player_row.get("established") and quadrant not in stats[label]["first_established"]:
                stats[label]["first_established"][quadrant] = record["step"]


def summarize_game(env, records, labels, stats):
    final_step = env.steps[-1]
    winners = winners_from_step(final_step)
    for player_index, label in enumerate(labels):
        if player_index in winners and len(winners) == 1:
            stats[label]["wins"] += 1
        elif len(winners) > 1:
            stats[label]["ties"] += 1

    for record in records:
        update_control_samples(stats, record, labels)
        for action_row in record["actions"]:
            label = action_row["label"]
            if action_row.get("error"):
                stats[label]["errors"] += 1
            for move in action_row["moves"]:
                stats[label]["launches"] += 1
                stats[label]["ships_launched"] += int(move["ships"])
                stats[label]["target_owner"][move["target_owner_group"]] += 1
                stats[label]["target_kind"][move["target_kind"]] += 1
                stats[label]["target_size"][move["target_size"]] += 1
                stats[label]["target_quadrant"][move["target_quadrant"]] += 1
                if move["source"]:
                    stats[label]["source_quadrant"][move["source"]["quadrant"]] += 1
                if move["central_rotating_big"]:
                    stats[label]["central_rotating_big"] += 1
                if move["central_rotating_small"]:
                    stats[label]["central_rotating_small"] += 1

        for event in record["captures"]:
            to_owner = int(event["to_owner"])
            from_owner = int(event["from_owner"])
            if 0 <= to_owner < len(labels):
                to_label = labels[to_owner]
                stats[to_label]["captures"] += 1
                stats[to_label]["capture_quadrants"][event["quadrant"]] += 1
                if 0 <= from_owner < len(labels):
                    stats[to_label]["captured_from"][labels[from_owner]] += 1
                else:
                    stats[to_label]["captured_from"]["neutral"] += 1
            if 0 <= from_owner < len(labels):
                from_label = labels[from_owner]
                stats[from_label]["losses"] += 1
                stats[from_label]["loss_quadrants"][event["quadrant"]] += 1

    final_obs = extract_observation(final_step[0])
    final_planets = planets_from_obs(final_obs)
    for player_index, label in enumerate(labels):
        owned = [planet for planet in final_planets if planet.owner == player_index]
        stats[label]["final_ships"] += sum(planet.ships for planet in owned)
        stats[label]["final_production"] += sum(planet.production for planet in owned)

    return {
        "rewards": [extract_reward(state) for state in final_step],
        "status": [extract_status(state) for state in final_step],
        "winners": winners,
    }


def counter_dict(counter, limit=None):
    items = counter.most_common(limit)
    return {key: value for key, value in items}


def serializable_stats(stats):
    output = {}
    for label, data in stats.items():
        output[label] = {
            "label": label,
            "wins": data["wins"],
            "ties": data["ties"],
            "launches": data["launches"],
            "ships_launched": data["ships_launched"],
            "errors": data["errors"],
            "target_owner": counter_dict(data["target_owner"]),
            "target_kind": counter_dict(data["target_kind"]),
            "target_size": counter_dict(data["target_size"]),
            "target_quadrant": counter_dict(data["target_quadrant"]),
            "source_quadrant": counter_dict(data["source_quadrant"]),
            "central_rotating_big": data["central_rotating_big"],
            "central_rotating_small": data["central_rotating_small"],
            "captures": data["captures"],
            "losses": data["losses"],
            "capture_quadrants": counter_dict(data["capture_quadrants"]),
            "loss_quadrants": counter_dict(data["loss_quadrants"]),
            "captured_from": counter_dict(data["captured_from"]),
            "final_ships_total": data["final_ships"],
            "final_production_total": data["final_production"],
            "first_established": dict(data["first_established"]),
        }
    return output


def pct(part, whole):
    return 0.0 if whole <= 0 else round(100.0 * part / whole, 2)


def write_report(run_dir, summary, labels, games):
    lines = []
    lines.append("# Orbit Wars Agent Lab Report")
    lines.append("")
    lines.append(f"Run ID: `{summary['run_id']}`")
    lines.append(f"Lineup: {', '.join(labels)}")
    lines.append(f"Games: {games}")
    lines.append("")
    lines.append("## Results")
    for game in summary["games"]:
        winners = ", ".join(labels[idx] for idx in game["winners"])
        lines.append(
            f"- Game {game['game']}: seed={game['seed']} rewards={game['rewards']} winner={winners}"
        )
    lines.append("")
    lines.append("## Tactical Weights")
    lines.append(f"- Weights version: {TACTICAL_WEIGHTS['version']}")
    lines.append("- These are label weights, not neural model weights yet.")
    lines.append("")
    lines.append("## Agent Tendencies")

    for label in labels:
        stats = summary["players"][label]
        launches = stats["launches"]
        static_hits = stats["target_kind"].get("static", 0)
        rotating_hits = stats["target_kind"].get("rotating", 0)
        neutral_hits = stats["target_owner"].get("neutral", 0)
        enemy_hits = stats["target_owner"].get("enemy", 0)
        friendly_hits = stats["target_owner"].get("friendly", 0)
        avg_final_prod = round(stats["final_production_total"] / max(1, games), 2)
        avg_final_ships = round(stats["final_ships_total"] / max(1, games), 2)
        lines.append("")
        lines.append(f"### {label}")
        lines.append(f"- Record: wins={stats['wins']} ties={stats['ties']} over {games}")
        lines.append(f"- Launches: {launches}, ships launched: {stats['ships_launched']}")
        lines.append(
            f"- Target mix: static={pct(static_hits, launches)}%, rotating={pct(rotating_hits, launches)}%, "
            f"neutral={pct(neutral_hits, launches)}%, enemy={pct(enemy_hits, launches)}%, friendly={pct(friendly_hits, launches)}%"
        )
        lines.append(
            f"- Central rotating targets: big={stats['central_rotating_big']}, small={stats['central_rotating_small']}"
        )
        lines.append(f"- Captures/losses: {stats['captures']} / {stats['losses']}")
        lines.append(f"- Avg final production: {avg_final_prod}, avg final ships: {avg_final_ships}")
        lines.append(f"- Target quadrants: {stats['target_quadrant']}")
        lines.append(f"- Source quadrants: {stats['source_quadrant']}")
        lines.append(f"- Captured from: {stats['captured_from']}")
        lines.append(f"- First established quadrants: {stats['first_established']}")

    report_path = run_dir / "tendency_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def append_progression(summary, labels, games):
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    exists = PROGRESSION_PATH.exists()
    fieldnames = [
        "timestamp",
        "run_id",
        "lineup",
        "player",
        "wins",
        "ties",
        "games",
        "launches_per_game",
        "ships_launched_per_game",
        "static_target_pct",
        "rotating_big_targets_per_game",
        "neutral_target_pct",
        "enemy_target_pct",
        "captures_per_game",
        "losses_per_game",
        "avg_final_production",
        "avg_final_ships",
        "established_quadrants_seen",
    ]
    with PROGRESSION_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for label in labels:
            stats = summary["players"][label]
            launches = stats["launches"]
            writer.writerow(
                {
                    "timestamp": summary["timestamp"],
                    "run_id": summary["run_id"],
                    "lineup": " ".join(labels),
                    "player": label,
                    "wins": stats["wins"],
                    "ties": stats["ties"],
                    "games": games,
                    "launches_per_game": round(launches / max(1, games), 3),
                    "ships_launched_per_game": round(stats["ships_launched"] / max(1, games), 3),
                    "static_target_pct": pct(stats["target_kind"].get("static", 0), launches),
                    "rotating_big_targets_per_game": round(stats["central_rotating_big"] / max(1, games), 3),
                    "neutral_target_pct": pct(stats["target_owner"].get("neutral", 0), launches),
                    "enemy_target_pct": pct(stats["target_owner"].get("enemy", 0), launches),
                    "captures_per_game": round(stats["captures"] / max(1, games), 3),
                    "losses_per_game": round(stats["losses"] / max(1, games), 3),
                    "avg_final_production": round(stats["final_production_total"] / max(1, games), 3),
                    "avg_final_ships": round(stats["final_ships_total"] / max(1, games), 3),
                    "established_quadrants_seen": len(stats["first_established"]),
                }
            )


def slug(value):
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.lower())
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean.strip("-")


def parse_args():
    parser = argparse.ArgumentParser(description="Run instrumented Orbit Wars matches and record tactical data.")
    parser.add_argument("--players", nargs="+", required=True, help="Exactly 2 or 4 agents. Use mine, smith, 1039, 1200, random, starter, or .py files.")
    parser.add_argument("--games", type=int, default=3, help="Number of games to run.")
    parser.add_argument("--seed", type=int, default=20260602, help="Base seed.")
    parser.add_argument("--out", type=Path, default=LAB_DIR, help="Output directory.")
    parser.add_argument("--full-board", action="store_true", help="Save full board state in every JSONL row.")
    parser.add_argument("--no-records", action="store_true", help="Do not write per-turn JSONL files.")
    return parser.parse_args()


def main():
    args = parse_args()
    agents, labels = build_lineup(args.players)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug('-'.join(labels))}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "weights_snapshot.json").write_text(
        json.dumps(TACTICAL_WEIGHTS, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stats = {label: new_player_stats(label) for label in labels}
    games = []
    for game_idx in range(args.games):
        seed = args.seed + game_idx
        print(f"Game {game_idx + 1}/{args.games}: seed={seed} lineup={' '.join(labels)}")
        env, records = run_game(agents, labels, seed, include_full_board=args.full_board)
        game_summary = summarize_game(env, records, labels, stats)
        game_summary["game"] = game_idx + 1
        game_summary["seed"] = seed
        games.append(game_summary)

        if not args.no_records:
            record_path = run_dir / f"game_{game_idx + 1:03d}.jsonl"
            with record_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "lineup": labels,
        "games": games,
        "players": serializable_stats(stats),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = write_report(run_dir, summary, labels, args.games)
    append_progression(summary, labels, args.games)

    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Progression: {PROGRESSION_PATH}")


if __name__ == "__main__":
    main()
