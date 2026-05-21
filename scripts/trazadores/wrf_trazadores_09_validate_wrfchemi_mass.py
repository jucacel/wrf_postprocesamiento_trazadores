#!/usr/bin/env python3
"""Validate physical CO mass represented by wrfchemi E_CO files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


MW_CO_KG_PER_MOL = 0.02801


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate E_CO mass in wrfchemi files.")
    p.add_argument("files", nargs="+", help="wrfchemi files.")
    p.add_argument("--interval-min", type=float, default=60.0)
    p.add_argument("--var", default="E_CO")
    return p.parse_args()


def get_area_km2(ds: Dataset) -> float:
    if "DX" in ds.ncattrs() and "DY" in ds.ncattrs():
        return float(ds.DX) * float(ds.DY) / 1e6
    raise ValueError("DX/DY global attributes not found; cannot compute physical mass.")


def read_times(ds: Dataset) -> tuple[str, str, int]:
    if "Times" not in ds.variables:
        n = len(ds.dimensions["Time"])
        return "NA", "NA", n
    raw = ds.variables["Times"][:]
    labels = ["".join(ch.decode() if isinstance(ch, bytes) else str(ch) for ch in row).strip() for row in raw]
    return labels[0], labels[-1], len(labels)


def main() -> None:
    args = parse_args()
    interval_hr = args.interval_min / 60.0
    total_kg = 0.0
    print("file,frames,start,end,max_E_CO,sum_E_CO,mass_kg,mass_ton")
    for item in args.files:
        path = Path(item)
        with Dataset(path) as ds:
            if args.var not in ds.variables:
                raise SystemExit(f"{path}: variable {args.var} not found")
            x = np.asarray(ds.variables[args.var][:], dtype=np.float64)
            area_km2 = get_area_km2(ds)
            mass_kg = float(np.nansum(x) * area_km2 * interval_hr * MW_CO_KG_PER_MOL)
            start, end, frames = read_times(ds)
            total_kg += mass_kg
            print(
                f"{path},{frames},{start},{end},"
                f"{float(np.nanmax(x)):.9g},{float(np.nansum(x)):.9g},"
                f"{mass_kg:.9g},{mass_kg / 1000.0:.9g}"
            )
    print(f"TOTAL,,,,,,{total_kg:.9g},{total_kg / 1000.0:.9g}")


if __name__ == "__main__":
    main()
