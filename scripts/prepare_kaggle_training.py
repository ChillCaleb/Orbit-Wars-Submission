import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kaggle" / "bundles" / "training"
SOURCE_FILES = (
    "main.py",
    "tactical_features.py",
    "train_tactical_model.py",
    "train_controller_model.py",
    "agent_lab.py",
    "watch_match.py",
)


def newest_model():
    root_model = ROOT / "model_weights.npz"
    if root_model.exists():
        return root_model
    candidates = sorted((ROOT / "data" / "training_runs").glob("*/model_weights.npz"), reverse=True)
    return candidates[0] if candidates else None


def newest_controller_model():
    root_model = ROOT / "controller_weights.npz"
    if root_model.exists():
        return root_model
    candidates = sorted((ROOT / "data" / "training_runs").glob("*/controller_weights.npz"), reverse=True)
    return candidates[0] if candidates else None


def build_bundle():
    payload = io.BytesIO()
    manifest = []
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [ROOT / name for name in SOURCE_FILES]
        paths.extend(path for path in (ROOT / "agents").rglob("*") if path.is_file() and "__pycache__" not in path.parts)
        for path in paths:
            relative = path.relative_to(ROOT)
            archive.write(path, relative.as_posix())
            manifest.append(relative.as_posix())

        model_path = newest_model()
        if model_path is not None:
            archive.write(model_path, "model_weights.npz")
            manifest.append(f"model_weights.npz <- {model_path.relative_to(ROOT)}")
        controller_path = newest_controller_model()
        if controller_path is not None:
            archive.write(controller_path, "controller_weights.npz")
            manifest.append(f"controller_weights.npz <- {controller_path.relative_to(ROOT)}")

    return base64.b64encode(payload.getvalue()).decode("ascii"), manifest


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def build_notebook(
    bundle_b64,
    job,
    with_games,
    without_games,
    workers,
    lineup_preset,
    training_mode,
    target_mode,
    epochs,
    batch_size,
    lr,
    seed,
    progress_every,
):
    unpack = f"""import base64
import io
from pathlib import Path
import zipfile

BUNDLE_B64 = {bundle_b64!r}
WORKDIR = Path("/kaggle/working/orbit_training")
WORKDIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    archive.extractall(WORKDIR)
print("Prepared", WORKDIR)
"""
    if job == "controller":
        train = f"""import subprocess
import sys

command = [
    sys.executable,
    "train_controller_model.py",
    "--with-mine-games", "{with_games}",
    "--without-mine-games", "{without_games}",
    "--lineup-preset", "{lineup_preset}",
    "--target-mode", "{target_mode}",
    "--epochs", "{epochs}",
    "--batch-size", "{batch_size}",
    "--lr", "{lr}",
    "--seed", "{seed}",
    "--progress-every", "{progress_every}",
    "--out", "/kaggle/working/training_output",
]
controller = WORKDIR / "controller_weights.npz"
if controller.exists():
    command.extend(["--init-from-controller", str(controller)])
print("Running:", " ".join(command))
subprocess.run(command, cwd=WORKDIR, check=True)
"""
        summarize = """import json
from pathlib import Path
import shutil

runs = sorted(Path("/kaggle/working/training_output").glob("*"))
latest = runs[-1]
print(json.dumps(json.loads((latest / "controller_metrics.json").read_text()), indent=2))
archive = shutil.make_archive("/kaggle/working/orbit_training_output", "zip", latest)
print("Output archive:", archive)
"""
        title = "# Orbit Wars Controller Training\n\nSelf-contained proposal-controller training run."
    else:
        train = f"""import subprocess
import sys

command = [
    sys.executable,
    "train_tactical_model.py",
    "--with-mine-games", "{with_games}",
    "--without-mine-games", "{without_games}",
    "--lineup-preset", "{lineup_preset}",
    "--training-mode", "{training_mode}",
    "--epochs", "{epochs}",
    "--batch-size", "{batch_size}",
    "--lr", "{lr}",
    "--seed", "{seed}",
    "--workers", "{workers}",
    "--progress-every", "{progress_every}",
    "--out", "/kaggle/working/training_output",
]
model = WORKDIR / "model_weights.npz"
if model.exists():
    command.extend(["--init-from-model", str(model)])
print("Running:", " ".join(command))
subprocess.run(command, cwd=WORKDIR, check=True)
"""
        summarize = """import json
from pathlib import Path
import shutil

runs = sorted(Path("/kaggle/working/training_output").glob("*"))
latest = runs[-1]
print(json.dumps(json.loads((latest / "metrics.json").read_text()), indent=2))
archive = shutil.make_archive("/kaggle/working/orbit_training_output", "zip", latest)
print("Output archive:", archive)
"""
        title = "# Orbit Wars Tactical Training\n\nSelf-contained tactical training run."
    return {
        "cells": [
            markdown_cell(title),
            code_cell(unpack),
            code_cell(train),
            code_cell(summarize),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description="Build the self-contained Kaggle training notebook without uploading it.")
    parser.add_argument("--job", choices=("tactical", "controller"), default="tactical")
    parser.add_argument("--with-mine-games", type=int, default=100)
    parser.add_argument("--without-mine-games", type=int, default=100)
    parser.add_argument("--lineup-preset", choices=("all", "new-agents", "opportunity"), default="all")
    parser.add_argument("--training-mode", choices=("ranking", "outcome"), default="ranking")
    parser.add_argument("--target-mode", choices=("winner-source", "future-quality"), default="winner-source")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20261100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    bundle_b64, manifest = build_bundle()
    notebook = build_notebook(
        bundle_b64,
        args.job,
        args.with_mine_games,
        args.without_mine_games,
        args.workers,
        args.lineup_preset,
        args.training_mode,
        args.target_mode,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        args.progress_every,
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "training.ipynb"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    digest = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    manifest_data = {
        "notebook": str(notebook_path.relative_to(ROOT)),
        "sha256": digest,
        "job": args.job,
        "with_mine_games": args.with_mine_games,
        "without_mine_games": args.without_mine_games,
        "lineup_preset": args.lineup_preset,
        "training_mode": args.training_mode,
        "target_mode": args.target_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "workers": args.workers,
        "progress_every": args.progress_every,
        "files": manifest,
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {notebook_path}")
    print(f"SHA256 {digest}")
    print("No Kaggle upload was performed.")


if __name__ == "__main__":
    main()
