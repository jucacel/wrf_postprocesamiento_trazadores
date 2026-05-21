#!/usr/bin/env python3
"""Diagnose whether a wrfchemi CO pulse appears at the intended source in wrfout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wrfout", nargs="+", help="wrfout files to inspect")
    p.add_argument("--i", type=int, required=True, help="1-based west-east source index")
    p.add_argument("--j", type=int, required=True, help="1-based south-north source index")
    p.add_argument("--var", default="co", help="WRF output variable to inspect")
    p.add_argument("--radius", type=int, default=2, help="Horizontal radius around source for local max")
    return p.parse_args()


def wrf_time(ds: Dataset) -> str:
    chars = ds.variables.get("Times")
    if chars is None:
        return "unknown"
    arr = chars[:]
    if arr.ndim == 2:
        return b"".join(arr[0].astype("S1")).decode("ascii", errors="replace")
    return str(arr)


def main() -> None:
    args = parse_args()
    j0 = args.j - 1
    i0 = args.i - 1

    print(
        "time,src_col_max_ppmv,src_sfc_ppmv,local_col_max_ppmv,"
        "domain_max_ppmv,p50_ppmv,p95_ppmv,p99_ppmv,max_k,max_j,max_i"
    )
    for item in sorted(args.wrfout):
        path = Path(item)
        with Dataset(path) as ds:
            if args.var not in ds.variables:
                print(f"{path.name},NO_{args.var},,,,,,,,")
                continue
            data = np.asarray(ds.variables[args.var][0], dtype=float)
            nz, ny, nx = data.shape
            if not (0 <= j0 < ny and 0 <= i0 < nx):
                raise SystemExit(f"i/j fuera del dominio: i=1..{nx}, j=1..{ny}")
            r = max(int(args.radius), 0)
            js = slice(max(0, j0 - r), min(ny, j0 + r + 1))
            is_ = slice(max(0, i0 - r), min(nx, i0 + r + 1))

            src_col = data[:, j0, i0]
            local = data[:, js, is_]
            vals = data[np.isfinite(data)]
            max_idx = np.unravel_index(int(np.nanargmax(data)), data.shape)
            print(
                f"{wrf_time(ds)},"
                f"{float(np.nanmax(src_col)):.9g},"
                f"{float(data[0, j0, i0]):.9g},"
                f"{float(np.nanmax(local)):.9g},"
                f"{float(np.nanmax(vals)):.9g},"
                f"{float(np.nanpercentile(vals, 50)):.9g},"
                f"{float(np.nanpercentile(vals, 95)):.9g},"
                f"{float(np.nanpercentile(vals, 99)):.9g},"
                f"{max_idx[0] + 1},{max_idx[1] + 1},{max_idx[2] + 1}"
            )


if __name__ == "__main__":
    main()
