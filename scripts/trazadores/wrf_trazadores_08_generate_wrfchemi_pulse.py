#!/usr/bin/env python3
"""
Generate a controlled real-CO WRF-Chem emission pulse as wrfchemi NetCDF.

The emitted species is E_CO in mol km-2 hr-1. The requested physical mass is
distributed over a WRF-grid patch and over the active emission time frames.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


MW_CO_KG_PER_MOL = 0.02801
DATE_FMT = "%Y-%m-%d_%H:%M:%S"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate wrfchemi E_CO pulse with a physically defined mass."
    )
    p.add_argument("--wrfinput", required=True, help="Path to wrfinput_d0x.")
    p.add_argument("--out-dir", required=True, help="Directory for wrfchemi outputs.")
    p.add_argument("--domain", default="d01", help="Output domain token, e.g. d01.")
    p.add_argument("--start", required=True, help="Run start: YYYY-MM-DD_HH:MM:SS.")
    p.add_argument("--run-hours", type=float, required=True, help="Total run length.")
    p.add_argument("--interval-min", type=float, default=60.0, help="Emission frame interval.")
    p.add_argument("--pulse-start", required=True, help="Pulse start: YYYY-MM-DD_HH:MM:SS.")
    p.add_argument("--pulse-hours", type=float, required=True, help="Pulse duration.")
    mass = p.add_mutually_exclusive_group(required=True)
    mass.add_argument("--mass-kg", type=float, help="Total emitted CO mass in kg.")
    mass.add_argument("--mass-ton", type=float, help="Total emitted CO mass in metric tons.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-latlon", nargs=2, type=float, metavar=("LAT", "LON"))
    src.add_argument("--source-ij", nargs=2, type=int, metavar=("I", "J"), help="1-based WRF indices.")
    p.add_argument("--patch-radius", type=int, default=1, help="0=1 cell, 1=3x3, 2=5x5.")
    p.add_argument(
        "--emissions-zdim",
        type=int,
        default=1,
        help="Number of wrfchemi emissions_zdim levels. Some mechanisms expect 2.",
    )
    p.add_argument(
        "--output-style",
        choices=["single", "dated", "split12"],
        default="single",
        help="single=wrfchemi_d01, dated=wrfchemi_<date>_d01, split12=00z/12z files.",
    )
    p.add_argument(
        "--copy-dated-start",
        action="store_true",
        help="Also copy the single file to wrfchemi_<start>_<domain>.",
    )
    p.add_argument("--include-zero-nox", action="store_true", help="Add zero E_NO and E_NO2.")
    p.add_argument("--include-zero-co2", action="store_true", help="Add zero E_CO2.")
    p.add_argument("--title", default="CO real pulse generated for WRF-Chem")
    return p.parse_args()


def dt(s: str) -> datetime:
    return datetime.strptime(s, DATE_FMT)


def time_chars(times: list[datetime]) -> np.ndarray:
    out = np.empty((len(times), 19), dtype="S1")
    for n, t in enumerate(times):
        out[n, :] = np.array(list(t.strftime(DATE_FMT)), dtype="S1")
    return out


def read_grid(wrfinput: Path) -> tuple[np.ndarray, np.ndarray, float, float, dict]:
    with Dataset(wrfinput) as ds:
        lat = np.asarray(ds.variables["XLAT"][0, :, :], dtype=np.float64)
        lon = np.asarray(ds.variables["XLONG"][0, :, :], dtype=np.float64)
        dx = float(getattr(ds, "DX"))
        dy = float(getattr(ds, "DY"))
        attrs = {name: getattr(ds, name) for name in ds.ncattrs()}
    return lat, lon, dx, dy, attrs


def source_index(args: argparse.Namespace, lat: np.ndarray, lon: np.ndarray) -> tuple[int, int]:
    ny, nx = lat.shape
    if args.source_ij:
        i1, j1 = args.source_ij
        i = i1 - 1
        j = j1 - 1
        if not (0 <= i < nx and 0 <= j < ny):
            raise SystemExit(f"--source-ij outside grid: i={i1}, j={j1}, nx={nx}, ny={ny}")
        return j, i
    src_lat, src_lon = args.source_latlon
    dist2 = (lat - src_lat) ** 2 + (lon - src_lon) ** 2
    return tuple(int(v) for v in np.unravel_index(np.argmin(dist2), dist2.shape))


def patch_indices(j0: int, i0: int, radius: int, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    j1 = max(0, j0 - radius)
    j2 = min(ny, j0 + radius + 1)
    i1 = max(0, i0 - radius)
    i2 = min(nx, i0 + radius + 1)
    jj, ii = np.meshgrid(np.arange(j1, j2), np.arange(i1, i2), indexing="ij")
    return jj.ravel(), ii.ravel()


def build_emissions(args: argparse.Namespace) -> tuple[np.ndarray, list[datetime], dict]:
    if args.emissions_zdim < 1:
        raise SystemExit("--emissions-zdim must be >= 1")

    wrfinput = Path(args.wrfinput)
    lat, lon, dx, dy, attrs = read_grid(wrfinput)
    ny, nx = lat.shape
    area_km2 = dx * dy / 1e6

    start = dt(args.start)
    interval = timedelta(minutes=args.interval_min)
    nframes = int(round(args.run_hours * 60.0 / args.interval_min)) + 1
    times = [start + n * interval for n in range(nframes)]

    pulse_start = dt(args.pulse_start)
    pulse_end = pulse_start + timedelta(hours=args.pulse_hours)
    active = [n for n, t in enumerate(times) if pulse_start <= t < pulse_end]
    if not active:
        raise SystemExit("Pulse does not overlap generated time frames.")
    actual_hours = len(active) * args.interval_min / 60.0

    j0, i0 = source_index(args, lat, lon)
    jj, ii = patch_indices(j0, i0, args.patch_radius, ny, nx)
    n_cells = len(jj)

    mass_kg = args.mass_kg if args.mass_kg is not None else args.mass_ton * 1000.0
    moles = mass_kg / MW_CO_KG_PER_MOL
    flux = moles / (n_cells * area_km2 * actual_hours)

    e_co = np.zeros((nframes, args.emissions_zdim, ny, nx), dtype=np.float64)
    for n in active:
        e_co[n, 0, jj, ii] = flux

    meta = {
        "lat": lat,
        "lon": lon,
        "dx": dx,
        "dy": dy,
        "area_km2": area_km2,
        "attrs": attrs,
        "source_j0": j0,
        "source_i0": i0,
        "source_lat": float(lat[j0, i0]),
        "source_lon": float(lon[j0, i0]),
        "patch_cells": n_cells,
        "active_frames": len(active),
        "actual_hours": actual_hours,
        "mass_kg": mass_kg,
        "flux": flux,
        "emissions_zdim": int(args.emissions_zdim),
    }
    return e_co, times, meta


def copy_global_attrs(src_attrs: dict, dst: Dataset, title: str) -> None:
    for k, v in src_attrs.items():
        try:
            setattr(dst, k, v)
        except Exception:
            pass
    dst.TITLE = title
    dst.CO_PULSE_GENERATOR = Path(__file__).name


def write_file(path: Path, e_co: np.ndarray, times: list[datetime], meta: dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    nframes, nz, ny, nx = e_co.shape

    def create_emission_var(ds: Dataset, name: str, description: str, data) -> None:
        v = ds.createVariable(name, "f8", ("Time", "emissions_zdim", "south_north", "west_east"))
        v.FieldType = 104
        v.MemoryOrder = "XYZ"
        v.description = description
        v.units = "mol km^-2 hr^-1"
        v.stagger = ""
        v.coordinates = "XLONG XLAT"
        v[:] = data

    with Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("emissions_zdim", nz)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createDimension("DateStrLen", 19)

        create_emission_var(ds, "E_CO", "carbon monoxide emissions", e_co)
        zero_emiss = np.zeros_like(e_co)

        if args.include_zero_co2:
            create_emission_var(ds, "E_CO2", "zero carbon dioxide emissions", zero_emiss)

        if args.include_zero_nox:
            for name in ("E_NO", "E_NO2"):
                create_emission_var(ds, name, "zero placeholder emissions", zero_emiss)

        tv = ds.createVariable("Time", "i4", ("Time",))
        tv[:] = np.arange(nframes, dtype=np.int32)
        latv = ds.createVariable("XLAT", "f4", ("south_north", "west_east"))
        lonv = ds.createVariable("XLONG", "f4", ("south_north", "west_east"))
        latv[:] = meta["lat"].astype("f4")
        lonv[:] = meta["lon"].astype("f4")
        ev = ds.createVariable("emissions_zdim", "i4", ("emissions_zdim",))
        ev[:] = np.arange(1, nz + 1, dtype=np.int32)
        timesv = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        timesv[:] = time_chars(times)

        copy_global_attrs(meta["attrs"], ds, args.title)
        ds.START_DATE = times[0].strftime(DATE_FMT)
        ds.SIMULATION_START_DATE = times[0].strftime(DATE_FMT)
        ds.CO_PULSE_MASS_KG = float(meta["mass_kg"])
        ds.CO_PULSE_MASS_TON = float(meta["mass_kg"] / 1000.0)
        ds.CO_PULSE_SOURCE_I_1BASED = int(meta["source_i0"] + 1)
        ds.CO_PULSE_SOURCE_J_1BASED = int(meta["source_j0"] + 1)
        ds.CO_PULSE_SOURCE_LAT = float(meta["source_lat"])
        ds.CO_PULSE_SOURCE_LON = float(meta["source_lon"])
        ds.CO_PULSE_PATCH_CELLS = int(meta["patch_cells"])
        ds.CO_PULSE_ACTIVE_FRAMES = int(meta["active_frames"])
        ds.CO_PULSE_ACTUAL_HOURS = float(meta["actual_hours"])
        ds.CO_PULSE_E_CO_FLUX = float(meta["flux"])
        ds.CO_PULSE_EMISSIONS_ZDIM = int(nz)


def output_paths(args: argparse.Namespace, times: list[datetime]) -> list[tuple[Path, slice]]:
    out = Path(args.out_dir)
    domain = args.domain
    if args.output_style == "single":
        return [(out / f"wrfchemi_{domain}", slice(None))]
    if args.output_style == "dated":
        return [(out / f"wrfchemi_{times[0].strftime(DATE_FMT)}_{domain}", slice(None))]

    paths: list[tuple[Path, slice]] = []
    for start_idx in range(0, len(times), 12):
        chunk = times[start_idx : start_idx + 12]
        if not chunk:
            continue
        suffix = "00z" if chunk[0].hour == 0 else "12z" if chunk[0].hour == 12 else chunk[0].strftime("%Hz")
        paths.append((out / f"wrfchemi_{suffix}_{domain}", slice(start_idx, start_idx + len(chunk))))
    return paths


def validate_mass(e_co: np.ndarray, meta: dict, interval_min: float) -> float:
    interval_hr = interval_min / 60.0
    moles = float(np.sum(e_co) * meta["area_km2"] * interval_hr)
    return moles * MW_CO_KG_PER_MOL


def main() -> None:
    args = parse_args()
    e_co, times, meta = build_emissions(args)
    written: list[Path] = []
    for path, slc in output_paths(args, times):
        write_file(path, e_co[slc], times[slc], meta, args)
        written.append(path)

    if args.copy_dated_start and args.output_style == "single":
        dated = Path(args.out_dir) / f"wrfchemi_{times[0].strftime(DATE_FMT)}_{args.domain}"
        write_file(dated, e_co, times, meta, args)
        written.append(dated)

    mass_check = validate_mass(e_co, meta, args.interval_min)
    print("CO wrfchemi generado")
    print(f"  masa objetivo kg     : {meta['mass_kg']:.6g}")
    print(f"  masa discretizada kg : {mass_check:.6g}")
    print(f"  fuente i,j 1-based   : {meta['source_i0'] + 1}, {meta['source_j0'] + 1}")
    print(f"  fuente lat,lon       : {meta['source_lat']:.7f}, {meta['source_lon']:.7f}")
    print(f"  celdas parche        : {meta['patch_cells']}")
    print(f"  emissions_zdim       : {meta['emissions_zdim']}")
    print(f"  frames activos       : {meta['active_frames']}")
    print(f"  duracion usada h     : {meta['actual_hours']:.3f}")
    print(f"  E_CO activo          : {meta['flux']:.6g} mol km-2 hr-1")
    for path in written:
        print(f"  archivo              : {path}")


if __name__ == "__main__":
    main()
