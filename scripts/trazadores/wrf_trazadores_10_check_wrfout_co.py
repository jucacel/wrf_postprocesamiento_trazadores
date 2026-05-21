#!/usr/bin/env python3
"""Check that a WRF-Chem CO run produced usable CO fields in wrfout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


CO_CANDIDATES = ("co", "CO", "CO_T", "co_t")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect CO-like variables in wrfout files.")
    p.add_argument("wrfout", nargs="+", help="wrfout files.")
    p.add_argument("--var", default="", help="Explicit CO variable name.")
    return p.parse_args()


def decode_time(ds: Dataset) -> str:
    if "Times" not in ds.variables:
        return "NA"
    raw = ds.variables["Times"][0]
    return "".join(ch.decode() if isinstance(ch, bytes) else str(ch) for ch in raw).strip()


def find_var(ds: Dataset, explicit: str) -> str | None:
    if explicit:
        return explicit if explicit in ds.variables else None
    lower = {name.lower(): name for name in ds.variables}
    for cand in CO_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def main() -> None:
    args = parse_args()
    print("file,time,var,shape,min,max,sum,units,description")
    found_any = False
    for item in args.wrfout:
        path = Path(item)
        with Dataset(path) as ds:
            varname = find_var(ds, args.var)
            if not varname:
                print(f"{path},{decode_time(ds)},NO_CO_VARIABLE,,,,,,")
                continue
            v = ds.variables[varname]
            x = np.asarray(v[:], dtype=np.float64)
            found_any = True
            print(
                f"{path},{decode_time(ds)},{varname},{x.shape},"
                f"{float(np.nanmin(x)):.9g},{float(np.nanmax(x)):.9g},{float(np.nansum(x)):.9g},"
                f"{getattr(v, 'units', '')},{getattr(v, 'description', '')}"
            )
    if not found_any:
        raise SystemExit("No CO-like variable was found in the inspected wrfout files.")


if __name__ == "__main__":
    main()
