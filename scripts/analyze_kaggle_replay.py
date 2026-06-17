import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tactical_features import (  # noqa: E402
    concentrated_pressure_profile,
    fleets_from_obs,
    infer_action_target,
    owner_control_scores,
    planets_from_obs,
)


def player_observation(step, player, step_index):
    observation = dict(step[player].get("observation") or step[0].get("observation") or {})
    observation["player"] = player
    observation["step"] = step_index
    return observation


def action_totals(replay):
    totals = []
    for player in range(len(replay.get("rewards", []))):
        launches = 0
        ships_launched = 0
        active_turns = 0
        for step in replay.get("steps", []):
            actions = step[player].get("action") or []
            active_turns += int(bool(actions))
            launches += len(actions)
            ships_launched += sum(int(action[2]) for action in actions if len(action) >= 3)
        totals.append(
            {
                "player": player,
                "reward": replay["rewards"][player],
                "launches": launches,
                "ships_launched": ships_launched,
                "active_turns": active_turns,
            }
        )
    return totals


def close_pressure_window(window, end_step, lost_steps):
    window["end_step"] = end_step
    window["duration"] = end_step - window["start_step"] + 1
    losses = lost_steps.get(window["target_id"], [])
    window["target_lost_step"] = next(
        (step for step in losses if step >= window["start_step"]),
        None,
    )
    window["target_eventually_lost"] = window["target_lost_step"] is not None
    return window


def analyze_replay(replay, player):
    player_count = len(replay.get("rewards", []))
    if player < 0 or player >= player_count:
        raise ValueError(f"player must be between 0 and {player_count - 1}")

    ownership_events = []
    lost_steps = {}
    pressure_windows = []
    current_window = None
    previous_owned = None
    peak_score = None

    for step_index, step in enumerate(replay.get("steps", [])):
        observation = player_observation(step, player, step_index)
        planets = planets_from_obs(observation)
        fleets = fleets_from_obs(observation)
        owned = {int(planet.id) for planet in planets if int(planet.owner) == player}
        actions = step[player].get("action") or []
        scores = owner_control_scores(planets, fleets, player_count)
        score = float(scores.get(player, 0.0))

        if peak_score is None or score > peak_score["score"]:
            peak_score = {"step": step_index, "score": round(score, 3), "owned_planets": len(owned)}

        if previous_owned is not None:
            gained = sorted(owned - previous_owned)
            lost = sorted(previous_owned - owned)
            if gained or lost:
                ownership_events.append(
                    {
                        "step": step_index,
                        "gained": gained,
                        "lost": lost,
                        "owned_planets": len(owned),
                        "control_score": round(score, 3),
                    }
                )
            for planet_id in lost:
                lost_steps.setdefault(planet_id, []).append(step_index)
        previous_owned = owned

        pressure = concentrated_pressure_profile(planets, fleets, player)
        pressure_key = int(pressure["target_id"]) if pressure["flagged"] else None
        current_key = current_window["target_id"] if current_window else None
        if current_window is not None and pressure_key != current_key:
            pressure_windows.append(close_pressure_window(current_window, step_index - 1, lost_steps))
            current_window = None

        if pressure["flagged"] and current_window is None:
            current_window = {
                "target_id": pressure_key,
                "quadrant": pressure["quadrant"],
                "start_step": step_index,
                "max_hostile_ships": 0,
                "max_defense_ratio": 0.0,
                "action_count": 0,
                "direct_reinforcement_count": 0,
                "action_steps": [],
                "direct_reinforcement_steps": [],
            }

        if current_window is not None:
            current_window["max_hostile_ships"] = max(
                current_window["max_hostile_ships"], int(pressure["hostile_ships"])
            )
            current_window["max_defense_ratio"] = max(
                current_window["max_defense_ratio"], float(pressure["defense_ratio"])
            )
            if actions:
                current_window["action_count"] += len(actions)
                current_window["action_steps"].append(step_index)
            for action in actions:
                target = infer_action_target(action, planets)
                if target is not None and int(target.id) == current_window["target_id"]:
                    current_window["direct_reinforcement_count"] += 1
                    current_window["direct_reinforcement_steps"].append(step_index)

    if current_window is not None:
        pressure_windows.append(
            close_pressure_window(current_window, len(replay.get("steps", [])) - 1, lost_steps)
        )

    player_actions = action_totals(replay)
    ours = player_actions[player]
    winner_reward = max(replay.get("rewards", []))
    winners = [row["player"] for row in player_actions if row["reward"] == winner_reward]
    return {
        "episode_id": replay.get("id"),
        "player": player,
        "result": "win" if player in winners else "loss",
        "rewards": replay.get("rewards", []),
        "steps": len(replay.get("steps", [])),
        "action_totals": player_actions,
        "our_action_totals": ours,
        "peak_control": peak_score,
        "ownership_events": ownership_events,
        "pressure_windows": pressure_windows,
        "pressure_summary": {
            "windows": len(pressure_windows),
            "flagged_steps": sum(window["duration"] for window in pressure_windows),
            "windows_with_actions": sum(window["action_count"] > 0 for window in pressure_windows),
            "windows_with_direct_reinforcement": sum(
                window["direct_reinforcement_count"] > 0 for window in pressure_windows
            ),
            "pressured_targets_lost": sum(window["target_eventually_lost"] for window in pressure_windows),
        },
    }


def markdown_report(analysis):
    ours = analysis["our_action_totals"]
    lines = [
        f"# Episode {analysis['episode_id']} Analysis",
        "",
        f"- Player: `{analysis['player']}`",
        f"- Result: **{analysis['result'].upper()}**",
        f"- Steps: {analysis['steps']}",
        f"- Launches: {ours['launches']} across {ours['active_turns']} active turns",
        f"- Ships launched: {ours['ships_launched']}",
        (
            f"- Peak control: {analysis['peak_control']['score']} at step "
            f"{analysis['peak_control']['step']}"
        ),
        "",
        "## Pressure Windows",
        "",
    ]
    if not analysis["pressure_windows"]:
        lines.append("No concentrated pressure windows were detected.")
    for window in analysis["pressure_windows"]:
        lines.extend(
            [
                (
                    f"- Steps {window['start_step']}-{window['end_step']}, target "
                    f"{window['target_id']} ({window['quadrant']}): "
                    f"{window['max_hostile_ships']} hostile ships, "
                    f"max defense ratio {window['max_defense_ratio']:.3f}"
                ),
                (
                    f"  Actions while flagged: {window['action_count']}; direct reinforcement: "
                    f"{window['direct_reinforcement_count']}; target lost: "
                    f"{window['target_eventually_lost']}"
                ),
            ]
        )
    lines.extend(["", "## Player Tempo", ""])
    for row in analysis["action_totals"]:
        lines.append(
            f"- Player {row['player']}: reward {row['reward']}, {row['launches']} launches, "
            f"{row['ships_launched']} ships, {row['active_turns']} active turns"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze one downloaded Orbit Wars Kaggle replay.")
    parser.add_argument("replay", type=Path)
    parser.add_argument("--player", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    replay_path = args.replay if args.replay.is_absolute() else ROOT / args.replay
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    analysis = analyze_replay(replay, args.player)
    output = args.output or replay_path.with_name(f"{replay_path.stem}-player-{args.player}-analysis.json")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    report = output.with_suffix(".md")
    report.write_text(markdown_report(analysis), encoding="utf-8")
    print(f"Analysis: {output}")
    print(f"Report: {report}")
    print(json.dumps(analysis["pressure_summary"], indent=2))


if __name__ == "__main__":
    main()
