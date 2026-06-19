import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL = "calebbanks/orbit-wars-tactical-training"
RUNS_DIR = ROOT / "data" / "kaggle_runs"
BUNDLES_DIR = ROOT / "kaggle" / "bundles"
TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCELLED"}


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class RunLogger:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message):
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run_command(command):
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def kernel_status(kernel):
    result = run_command(["kaggle", "kernels", "status", kernel])
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(output or f"status command exited {result.returncode}")
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    if not match:
        raise RuntimeError(f"could not parse Kaggle status: {output}")
    return match.group(1), output


def latest_bundle_manifest():
    candidates = sorted(BUNDLES_DIR.glob("*/bundle_manifest.json"), key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1]
    legacy = ROOT / "kaggle_training" / "bundle_manifest.json"
    return legacy if legacy.exists() else None


def expected_games(manifest_path=None):
    manifest = Path(manifest_path) if manifest_path else latest_bundle_manifest()
    if manifest is None:
        return None
    if manifest is not None and not manifest.is_absolute():
        manifest = ROOT / manifest
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return int(data["with_mine_games"]) + int(data["without_mine_games"])


def download_outputs(kernel, output_dir, logger):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_command(["kaggle", "kernels", "output", kernel, "-p", str(output_dir), "--force"])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"output download failed: {detail}")
    logger.write(f"Downloaded Kaggle outputs to {output_dir.relative_to(ROOT)}")


def validate_outputs(output_dir, expected, logger):
    metrics_paths = sorted(output_dir.glob("training_output/*/metrics.json"))
    game_logs = sorted(output_dir.glob("training_output/*/game_log.csv"))
    if not metrics_paths or not game_logs:
        raise RuntimeError("completed kernel did not produce metrics.json and game_log.csv")

    metrics_path = metrics_paths[-1]
    game_log_path = game_logs[-1]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with game_log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    categories = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1

    summary = {
        "validated_at": timestamp(),
        "run_directory": metrics_path.parent.name,
        "games": len(rows),
        "expected_games": expected,
        "categories": categories,
        "train": metrics.get("train", {}),
        "test": metrics.get("test", {}),
    }
    summary_path = output_dir / "monitor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    logger.write(
        "Validation: "
        f"{len(rows)}/{expected if expected is not None else '?'} games; "
        f"train accuracy={metrics['train']['accuracy']:.4f}, loss={metrics['train']['loss']:.4f}; "
        f"test accuracy={metrics['test']['accuracy']:.4f}, loss={metrics['test']['loss']:.4f}"
    )
    logger.write(f"Category counts: {json.dumps(categories, sort_keys=True)}")
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f"expected {expected} completed games but found {len(rows)}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Poll and validate an Orbit Wars Kaggle training kernel.")
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--interval", type=int, default=300, help="Seconds between status checks.")
    parser.add_argument("--once", action="store_true", help="Check once instead of waiting for completion.")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Bundle manifest used to validate expected game count.")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir or RUNS_DIR / run_id
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    logger = RunLogger(run_dir / "monitor.log")
    expected = expected_games(args.manifest)
    deadline = time.monotonic() + args.timeout_hours * 3600

    logger.write(
        f"Monitoring {args.kernel}; interval={args.interval}s; "
        f"expected_games={expected if expected is not None else 'unknown'}"
    )
    last_status = None
    while True:
        try:
            status, raw = kernel_status(args.kernel)
        except Exception as exc:
            logger.write(f"Status check failed: {exc}")
            if args.once:
                return 2
            status = None

        if status is not None and status != last_status:
            logger.write(f"Kaggle status: {status}")
            logger.write(f"Kaggle response: {raw}")
            last_status = status
        elif status is not None:
            logger.write(f"Kaggle status unchanged: {status}")

        if status == "COMPLETE":
            try:
                download_outputs(args.kernel, run_dir, logger)
                validate_outputs(run_dir, expected, logger)
            except Exception as exc:
                logger.write(f"Validation failed: {exc}")
                return 1
            logger.write("Training run completed and validated successfully.")
            return 0

        if status in TERMINAL_STATES:
            logger.write(f"Training ended unsuccessfully with status {status}.")
            return 1
        if args.once:
            return 0
        if time.monotonic() >= deadline:
            logger.write(f"Monitor timed out after {args.timeout_hours:g} hours.")
            return 1

        logger.write(f"Next check in {args.interval} seconds.")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
