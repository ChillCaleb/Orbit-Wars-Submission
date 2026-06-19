#!/usr/bin/env python3
"""Import Kaggle notebook-style agents into the local training agent pool."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import re
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "agents" / "imported"
HELPER_SOURCE = ROOT / "agents" / "light_intruder" / "light_orbit_lite"

NOTEBOOK_SOURCES = (
    ROOT / "notebooks" / "agents" / "i-the-orbit.ipynb",
    ROOT / "notebooks" / "agents" / "orbit-wars-exp50.ipynb",
    ROOT / "notebooks" / "agents" / "orbit-wars-i-m-stronger.ipynb",
    ROOT / "notebooks" / "agents" / "orbit-wars.ipynb",
)

ARCHIVE_NOTEBOOKS = (
    ROOT / "notebooks" / "agents" / "orbit-wars-meta-snapshot-0618.ipynb",
    ROOT / "notebooks" / "best-orbit-wars-notebook.ipynb",
)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "imported_agent"


def notebook_cells(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cells", [])


def cell_source(cell) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def extract_writefile_main(path: Path) -> str | None:
    for cell in notebook_cells(path):
        source = cell_source(cell)
        if source.lstrip().startswith("%%writefile main.py"):
            return source.split("\n", 1)[1] if "\n" in source else ""
    return None


def extract_archives(path: Path) -> dict:
    for cell in notebook_cells(path):
        source = cell_source(cell)
        if "ARCHIVES" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ARCHIVES":
                    return ast.literal_eval(node.value)
    return {}


def rewrite_helper_imports(source: str, package_name: str) -> str:
    source = source.replace("from orbit_lite.", f"from {package_name}.")
    source = source.replace("import orbit_lite.", f"import {package_name}.")
    return source


def apply_source_compatibility(slug: str, source: str) -> str:
    if slug != "i_the_orbit" or "class Config:" not in source:
        return source
    marker = "    near_fraction:            float = 0.80   # 80% waves go to nearest targets\n"
    shim = '''\

    @property
    def max_offensive_targets(self) -> int:
        return self.max_targets

    @property
    def max_defensive_targets(self) -> int:
        return max(1, min(3, self.max_targets))

    @property
    def min_ships_to_launch(self) -> float:
        return self.min_ships

    @property
    def max_regroup_sources_per_lane(self) -> int:
        return self.max_sources

    @property
    def max_regroup_targets_per_source(self) -> int:
        return self.max_targets

    @property
    def max_regroup_time(self) -> float:
        return float(self.horizon)

    @property
    def regroup_pressure_delta_min(self) -> float:
        return 0.25

    @property
    def regroup_time_penalty_weight(self) -> float:
        return 1e-3
'''
    if "def max_offensive_targets" in source or marker not in source:
        return source
    return source.replace(marker, marker + shim, 1)


def copy_helper_package(destination: Path, package_name: str) -> None:
    package_dir = destination / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    shutil.copytree(HELPER_SOURCE, package_dir, ignore=shutil.ignore_patterns("__pycache__"))


def write_agent(slug: str, source: str, helper_members: dict[str, bytes] | None = None) -> Path:
    agent_dir = OUTPUT_ROOT / slug
    agent_dir.mkdir(parents=True, exist_ok=True)
    package_name = f"{slug}_orbit_lite"
    source = apply_source_compatibility(slug, source)
    source = rewrite_helper_imports(source, package_name)
    (agent_dir / "agent.py").write_text(source, encoding="utf-8")

    if helper_members:
        package_dir = agent_dir / package_name
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in helper_members.items():
            relative = Path(name)
            if relative.parts and relative.parts[0] == "orbit_lite":
                relative = Path(*relative.parts[1:])
            if not relative.parts or relative.suffix != ".py" or ".." in relative.parts:
                continue
            target = package_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    elif "orbit_lite" in source or package_name in source:
        copy_helper_package(agent_dir, package_name)

    return agent_dir / "agent.py"


def archive_payload_members(encoded: str) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)), mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if ".." in parts or Path(member.name).is_absolute():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            members[member.name] = extracted.read()
    return members


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    imported = []

    for path in NOTEBOOK_SOURCES:
        if not path.exists():
            continue
        source = extract_writefile_main(path)
        if not source:
            continue
        slug = slugify(path.stem)
        agent_path = write_agent(slug, source)
        imported.append((slug, agent_path))

    for path in ARCHIVE_NOTEBOOKS:
        if not path.exists():
            continue
        for name, meta in extract_archives(path).items():
            encoded = meta.get("b64")
            if not encoded:
                continue
            payload = archive_payload_members(encoded)
            main_py = payload.get("main.py")
            if not main_py:
                continue
            digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
            expected = meta.get("sha256")
            if expected and digest != expected:
                raise ValueError(f"{name} archive checksum mismatch")
            helper_members = {
                member_name: data
                for member_name, data in payload.items()
                if member_name.startswith("orbit_lite/") and member_name.endswith(".py")
            }
            slug = slugify(name)
            agent_path = write_agent(slug, main_py.decode("utf-8"), helper_members=helper_members)
            imported.append((slug, agent_path))

    for slug, agent_path in imported:
        print(f"{slug}: {agent_path.relative_to(ROOT)}")
    print(f"Imported {len(imported)} notebook/archive agents.")


if __name__ == "__main__":
    main()
