#!/usr/bin/env python3
"""Visualize isolated CO pulse in plan view as case minus no-emission control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrf_trazadores_00_core import add_figure_footer, build_gif_from_pngs, parse_wrf_times, read_var  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grafica en planta delta CO = caso con pulso - control sin emision local."
    )
    p.add_argument("--case-dir", required=True, help="Directorio con wrfout del caso con pulso.")
    p.add_argument("--control-dir", required=True, help="Directorio con wrfout del control.")
    p.add_argument("--outdir", required=True, help="Directorio de salida.")
    p.add_argument("--pattern", default="wrfout_*", help="Patron de wrfout dentro de cada directorio.")
    p.add_argument("--var", default="co", help="Variable de CO en wrfout.")
    p.add_argument("--i", type=int, default=56, help="Indice fuente WRF 1-based.")
    p.add_argument("--j", type=int, default=66, help="Indice fuente WRF 1-based.")
    p.add_argument(
        "--mode",
        choices=["colmax", "surface"],
        default="colmax",
        help="colmax=maximo vertical por celda; surface=nivel inferior.",
    )
    p.add_argument("--floor-ppbv", type=float, default=1.0e-3, help="Piso positivo para escala log.")
    p.add_argument("--vmax-ppbv", type=float, default=None, help="Maximo fijo para la escala de color.")
    p.add_argument(
        "--mask-threshold-ppbv",
        type=float,
        default=None,
        help="Umbral minimo visible antes de filtrar componentes. Si se omite usa --floor-ppbv.",
    )
    p.add_argument(
        "--component-filter",
        choices=["none", "max", "source"],
        default="none",
        help="none=todos los pixeles positivos; max=solo componente conectado al maximo; source=solo componente conectado a la fuente.",
    )
    p.add_argument("--pulse-mass-ton", type=float, default=1.0, help="Masa total emitida de CO en toneladas.")
    p.add_argument("--pulse-duration-hours", type=float, default=1.0, help="Duracion de liberacion del pulso.")
    p.add_argument("--pulse-start", default="2024-09-01_00:00:00", help="Inicio de liberacion del pulso.")
    p.add_argument("--release-type", default="pulso no continuo", help="Tipo de liberacion.")
    p.add_argument("--emissions-interval-min", type=float, default=60.0, help="Intervalo de wrfchemi.")
    p.add_argument("--gif-duration-ms", type=int, default=450)
    p.add_argument("--tag", default="co_real_delta_control_planview")
    return p.parse_args()


def pair_files(case_dir: Path, control_dir: Path, pattern: str) -> list[tuple[Path, Path, pd.Timestamp]]:
    case_files = sorted(case_dir.glob(pattern))
    if not case_files:
        raise SystemExit(f"No encontre archivos caso en {case_dir} con patron {pattern}")
    control_by_name = {p.name: p for p in control_dir.glob(pattern)}
    pairs = []
    missing = []
    for cf in case_files:
        ctl = control_by_name.get(cf.name)
        if ctl is None:
            missing.append(cf.name)
            continue
        with Dataset(cf) as nc:
            time = pd.Timestamp(parse_wrf_times(nc)[0])
        pairs.append((cf, ctl, time))
    if missing:
        raise SystemExit("Faltan wrfout control para: " + ", ".join(missing[:5]))
    return pairs


def read_delta_ppbv(case_path: Path, control_path: Path, varname: str) -> np.ndarray:
    with Dataset(case_path) as case, Dataset(control_path) as ctl:
        case_var = read_var(case, varname, time_index=0)
        ctl_var = read_var(ctl, varname, time_index=0)
    delta = (case_var - ctl_var) * 1000.0
    return np.where(np.isfinite(delta) & (delta > 0.0), delta, 0.0)


def plan_field(delta: np.ndarray, mode: str) -> np.ndarray:
    if mode == "surface":
        return delta[0, :, :]
    return np.nanmax(delta, axis=0)


def connected_component(mask: np.ndarray, seed_j: int, seed_i: int) -> np.ndarray:
    """Return the 4-neighbor component connected to the seed in a boolean mask."""
    out = np.zeros(mask.shape, dtype=bool)
    ny, nx = mask.shape
    if not (0 <= seed_j < ny and 0 <= seed_i < nx) or not mask[seed_j, seed_i]:
        return out

    stack = [(seed_j, seed_i)]
    out[seed_j, seed_i] = True
    while stack:
        j, i = stack.pop()
        for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
            if 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii] and not out[jj, ii]:
                out[jj, ii] = True
                stack.append((jj, ii))
    return out


def filtered_plan_field(
    plan: np.ndarray,
    *,
    threshold_ppbv: float,
    component_filter: str,
    j0: int,
    i0: int,
) -> tuple[np.ndarray, int]:
    visible = np.isfinite(plan) & (plan > threshold_ppbv)
    if component_filter == "source":
        keep = connected_component(visible, j0, i0)
    elif component_filter == "max" and np.any(visible):
        seed_j, seed_i = np.unravel_index(int(np.nanargmax(np.where(visible, plan, -np.inf))), plan.shape)
        keep = connected_component(visible, int(seed_j), int(seed_i))
    else:
        keep = visible
    return np.where(keep, plan, 0.0), int(np.count_nonzero(visible & ~keep))


def scan_diagnostics(
    pairs: list[tuple[Path, Path, pd.Timestamp]],
    *,
    varname: str,
    mode: str,
    j0: int,
    i0: int,
) -> tuple[pd.DataFrame, float]:
    rows = []
    plan_vmax = 0.0
    for case_path, control_path, time in pairs:
        delta = read_delta_ppbv(case_path, control_path, varname)
        plan = plan_field(delta, mode)
        plan_vmax = max(plan_vmax, float(np.nanmax(plan)))
        max_j, max_i = np.unravel_index(int(np.nanargmax(plan)), plan.shape)
        rows.append(
            {
                "time": time.strftime("%Y-%m-%d_%H:%M:%S"),
                "src_plan_delta_ppbv": float(plan[j0, i0]),
                "src_col_max_delta_ppbv": float(np.nanmax(delta[:, j0, i0])),
                "domain_plan_max_delta_ppbv": float(np.nanmax(plan)),
                "domain_3d_max_delta_ppbv": float(np.nanmax(delta)),
                "plan_max_j": int(max_j + 1),
                "plan_max_i": int(max_i + 1),
            }
        )
    return pd.DataFrame(rows), plan_vmax


def plot_frame(
    case_path: Path,
    control_path: Path,
    time: pd.Timestamp,
    outpath: Path,
    *,
    varname: str,
    mode: str,
    j0: int,
    i0: int,
    vmax: float,
    floor_ppbv: float,
    mask_threshold_ppbv: float,
    component_filter: str,
    metadata_text: str,
) -> None:
    delta = read_delta_ppbv(case_path, control_path, varname)
    plan = plan_field(delta, mode)
    plan_filtered, removed_pixels = filtered_plan_field(
        plan,
        threshold_ppbv=mask_threshold_ppbv,
        component_filter=component_filter,
        j0=j0,
        i0=i0,
    )
    zplot = np.ma.masked_less_equal(plan_filtered, floor_ppbv)

    with Dataset(case_path) as nc:
        lat = read_var(nc, "XLAT", time_index=0)
        lon = read_var(nc, "XLONG", time_index=0)
        hgt = read_var(nc, "HGT", time_index=0)

    src_lat = float(lat[j0, i0])
    src_lon = float(lon[j0, i0])
    max_j, max_i = np.unravel_index(int(np.nanargmax(plan)), plan.shape)

    fig = plt.figure(figsize=(9, 8), constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1)
    mesh = ax.pcolormesh(
        lon,
        lat,
        zplot,
        shading="auto",
        cmap="magma",
        norm=LogNorm(vmin=floor_ppbv, vmax=max(vmax, floor_ppbv * 10.0)),
    )
    topo = ax.contour(lon, lat, hgt, levels=12, colors="0.65", linewidths=0.45, alpha=0.75)
    ax.clabel(topo, inline=True, fontsize=6, fmt="%.0f")

    ax.axvline(src_lon, color="cyan", linestyle="--", linewidth=1.1, label=f"lon fuente {src_lon:.4f}")
    ax.axhline(src_lat, color="white", linestyle=":", linewidth=1.0, label=f"lat fuente {src_lat:.4f}")
    ax.scatter([src_lon], [src_lat], s=75, c="red", edgecolors="white", linewidths=1.0, zorder=5)
    ax.scatter(
        [float(lon[max_j, max_i])],
        [float(lat[max_j, max_i])],
        s=55,
        marker="x",
        c="lime",
        linewidths=1.8,
        zorder=5,
        label="maximo",
    )
    ax.text(
        src_lon,
        src_lat,
        f"  fuente i={i0 + 1} j={j0 + 1}\n  {src_lat:.4f}, {src_lon:.4f}",
        color="white",
        fontsize=8,
        ha="left",
        va="bottom",
        path_effects=[],
    )
    ax.text(
        0.012,
        0.285,
        metadata_text + f"\nFiltro visual: componente={component_filter}; umbral={mask_threshold_ppbv:g} ppbv; pixeles removidos={removed_pixels}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="0.08",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.35", "alpha": 0.86},
    )

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(
        f"Delta CO pulso - control | {time.strftime('%Y-%m-%d %H:%M')} | "
        f"{'maximo de columna' if mode == 'colmax' else 'superficie'}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Delta CO (ppbv)")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    add_figure_footer(fig, script_path=Path(__file__).resolve(), output_path=outpath)
    fig.subplots_adjust(left=0.09, right=0.88, top=0.92, bottom=0.15)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def build_metadata_text(
    args: argparse.Namespace,
    case_path: Path,
    control_path: Path,
    *,
    src_lat: float,
    src_lon: float,
    j0: int,
    i0: int,
) -> str:
    return "\n".join(
        [
            f"Experimento: delta_co = pulso - control | variable={args.var} | mapa={'max columna' if args.mode == 'colmax' else 'superficie'}",
            f"Liberacion: {args.release_type}; masa={args.pulse_mass_ton:g} ton CO; duracion={args.pulse_duration_hours:g} h",
            f"Inicio pulso={args.pulse_start}; intervalo wrfchemi={args.emissions_interval_min:g} min",
            f"Fuente: i={i0 + 1}, j={j0 + 1}; lat={src_lat:.4f}, lon={src_lon:.4f}",
            f"Referencia: meridiano lon={src_lon:.4f} y latitud fuente",
            f"WRF pulso: {case_path.name}",
            f"WRF control: {control_path.name}",
        ]
    )


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()
    control_dir = Path(args.control_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    frames_dir = outdir / f"{args.tag}_frames"
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_files(case_dir, control_dir, args.pattern)
    j0 = args.j - 1
    i0 = args.i - 1
    df, plan_vmax = scan_diagnostics(pairs, varname=args.var, mode=args.mode, j0=j0, i0=i0)
    color_vmax = args.vmax_ppbv if args.vmax_ppbv is not None else max(plan_vmax, args.floor_ppbv * 10.0)
    mask_threshold_ppbv = args.mask_threshold_ppbv if args.mask_threshold_ppbv is not None else args.floor_ppbv

    csv_path = outdir / f"{args.tag}_diagnostico_delta_co_planview.csv"
    df.to_csv(csv_path, index=False)

    frame_paths = []
    for n, (case_path, control_path, time) in enumerate(pairs):
        frame = frames_dir / f"{args.tag}_{n:03d}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        with Dataset(case_path) as nc:
            lat0 = read_var(nc, "XLAT", time_index=0)
            lon0 = read_var(nc, "XLONG", time_index=0)
        metadata_text = build_metadata_text(
            args,
            case_path,
            control_path,
            src_lat=float(lat0[j0, i0]),
            src_lon=float(lon0[j0, i0]),
            j0=j0,
            i0=i0,
        )
        plot_frame(
            case_path,
            control_path,
            time,
            frame,
            varname=args.var,
            mode=args.mode,
            j0=j0,
            i0=i0,
            vmax=color_vmax,
            floor_ppbv=args.floor_ppbv,
            mask_threshold_ppbv=mask_threshold_ppbv,
            component_filter=args.component_filter,
            metadata_text=metadata_text,
        )
        frame_paths.append(frame)

    gif_path = outdir / f"{args.tag}_delta_co_planview.gif"
    gif = build_gif_from_pngs(frame_paths, gif_path, duration_ms=args.gif_duration_ms)
    print(f"CSV: {csv_path}")
    print(f"Frames: {frames_dir}")
    if gif:
        print(f"GIF: {gif}")
    else:
        print("GIF: no generado")
    print(f"Maximo delta real en mapa: {plan_vmax:.6g} ppbv")
    print(f"Maximo usado para color: {color_vmax:.6g} ppbv")


if __name__ == "__main__":
    main()
