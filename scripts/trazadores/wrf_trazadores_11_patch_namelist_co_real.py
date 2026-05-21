#!/usr/bin/env python3
"""Patch namelist.input for a short real-CO wrfchemi WRF-Chem test."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


TIME_CONTROL_DEFAULTS = {
    "io_form_auxinput5": "2",
    "auxinput5_interval_m": "60",
}

CHEM_DEFAULTS = {
    "chem_opt": "100",
    "io_style_emissions": "2",
    "emiss_opt": "3",
    "bio_emiss_opt": "0",
    "gaschem_onoff": "1",
    "aerchem_onoff": "0",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch namelist.input for E_CO wrfchemi test.")
    p.add_argument("namelist", help="Path to namelist.input.")
    p.add_argument("--domain-token", default="d01", help="Domain token used in wrfchemi_d01.")
    p.add_argument("--run-hours", type=float, required=True)
    p.add_argument("--interval-min", type=float, default=60.0)
    p.add_argument("--io-style-emissions", choices=["1", "2"], default="2")
    p.add_argument("--backup-suffix", default=".before_co_real_wrfchemi")
    return p.parse_args()


def find_section(lines: list[str], section: str) -> tuple[int, int]:
    start = -1
    pat = re.compile(rf"^\s*&{re.escape(section)}\b", re.I)
    for idx, line in enumerate(lines):
        if pat.search(line):
            start = idx
            break
    if start < 0:
        raise SystemExit(f"Section &{section} not found")
    for idx in range(start + 1, len(lines)):
        if re.match(r"^\s*/\s*$", lines[idx]):
            return start, idx
    raise SystemExit(f"Section &{section} has no closing /")


def set_key(lines: list[str], start: int, end: int, key: str, value: str) -> None:
    pat = re.compile(rf"^(\s*){re.escape(key)}\s*=", re.I)
    new_line = f" {key:<24} = {value},\n"
    for idx in range(start + 1, end):
        if pat.search(lines[idx]):
            lines[idx] = new_line
            return
    lines.insert(end, new_line)


def patch_section(lines: list[str], section: str, updates: dict[str, str]) -> None:
    start, end = find_section(lines, section)
    for key, value in updates.items():
        set_key(lines, start, end, key, value)
        start, end = find_section(lines, section)


def main() -> None:
    args = parse_args()
    path = Path(args.namelist)
    backup = path.with_name(path.name + args.backup_suffix)
    if not backup.exists():
        shutil.copy2(path, backup)

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    frames = int(round(args.run_hours * 60.0 / args.interval_min)) + 1

    time_updates = dict(TIME_CONTROL_DEFAULTS)
    if args.io_style_emissions == "2":
        time_updates["auxinput5_inname"] = f"'wrfchemi_d<domain>'"
        time_updates["frames_per_auxinput5"] = str(frames)
    else:
        time_updates["auxinput5_inname"] = f"'wrfchemi_<hour>z_d<domain>'"
        time_updates["frames_per_auxinput5"] = "12"

    chem_updates = dict(CHEM_DEFAULTS)
    chem_updates["io_style_emissions"] = args.io_style_emissions

    patch_section(lines, "time_control", time_updates)
    patch_section(lines, "chem", chem_updates)
    path.write_text("".join(lines), encoding="utf-8")

    print(f"Patched {path}")
    print(f"Backup  {backup}")
    print(f"frames_per_auxinput5={time_updates['frames_per_auxinput5']}")
    print(f"auxinput5_inname={time_updates['auxinput5_inname']}")


if __name__ == "__main__":
    main()
