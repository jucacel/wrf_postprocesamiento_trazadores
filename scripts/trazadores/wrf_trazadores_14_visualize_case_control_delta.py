#!/usr/bin/env python3
"""Visualize the isolated CO pulse as case minus no-emission control."""

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

from wrf_trazadores_00_core import (  # noqa: E402
    add_figure_footer,
    build_gif_from_pngs,
    compute_height_agl_and_dz,
    parse_wrf_times,
    read_var,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grafica delta CO = caso con pulso - caso control sin emision local."
    )
    p.add_argument("--case-dir", required=True, help="Directorio con wrfout del caso con pulso.")
    p.add_argument("--control-dir", required=True, help="Directorio con wrfout del control.")
    p.add_argument("--outdir", required=True, help="Directorio de salida.")
    p.add_argument("--pattern", default="wrfout_*", help="Patron de wrfout dentro de cada directorio.")
    p.add_argument("--var", default="co", help="Variable de CO en wrfout.")
    p.add_argument("--i", type=int, default=56, help="Indice fuente WRF 1-based.")
    p.add_argument("--j", type=int, default=66, help="Indice fuente WRF 1-based.")
    p.add_argument("--axis", choices=["x", "y"], default="x", help="Corte vertical por x o y.")
    p.add_argument("--max-height-agl", type=float, default=3000.0, help="Altura AGL a mostrar.")
    p.add_argument("--floor-ppbv", type=float, default=1.0e-3, help="Piso positivo para escala log en ppbv.")
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
        help="none=todos los pixeles positivos; max=solo componente conectado al maximo del corte; source=solo componente conectado a la columna fuente.",
    )
    p.add_argument("--pulse-mass-ton", type=float, default=1.0, help="Masa total emitida de CO en toneladas.")
    p.add_argument("--pulse-duration-hours", type=float, default=1.0, help="Duracion de liberacion del pulso.")
    p.add_argument("--pulse-start", default="2024-09-01_00:00:00", help="Inicio de liberacion del pulso.")
    p.add_argument("--release-type", default="pulso no continuo", help="Tipo de liberacion.")
    p.add_argument("--emissions-interval-min", type=float, default=60.0, help="Intervalo de wrfchemi.")
    p.add_argument("--gif-duration-ms", type=int, default=450)
    p.add_argument("--tag", default="co_real_delta_control")
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


def get_line(data: np.ndarray, axis: str, j0: int, i0: int) -> np.ndarray:
    return data[:, j0, :] if axis == "x" else data[:, :, i0]


def get_line_2d(data: np.ndarray, axis: str, j0: int, i0: int) -> np.ndarray:
    return data[j0, :] if axis == "x" else data[:, i0]


def connected_component(mask: np.ndarray, seed_k: int, seed_x: int) -> np.ndarray:
    """Return the 4-neighbor component connected to the seed in a vertical transect."""
    out = np.zeros(mask.shape, dtype=bool)
    nz, nx = mask.shape
    if not (0 <= seed_k < nz and 0 <= seed_x < nx) or not mask[seed_k, seed_x]:
        return out

    stack = [(seed_k, seed_x)]
    out[seed_k, seed_x] = True
    while stack:
        k, x = stack.pop()
        for kk, xx in ((k - 1, x), (k + 1, x), (k, x - 1), (k, x + 1)):
            if 0 <= kk < nz and 0 <= xx < nx and mask[kk, xx] and not out[kk, xx]:
                out[kk, xx] = True
                stack.append((kk, xx))
    return out


def filtered_line_field(
    line: np.ndarray,
    *,
    threshold_ppbv: float,
    component_filter: str,
    source_x: int,
) -> tuple[np.ndarray, int]:
    visible = np.isfinite(line) & (line > threshold_ppbv)
    if component_filter == "source" and 0 <= source_x < line.shape[1] and np.any(visible[:, source_x]):
        seed_k = int(np.nanargmax(np.where(visible[:, source_x], line[:, source_x], -np.inf)))
        keep = connected_component(visible, seed_k, source_x)
    elif component_filter == "max" and np.any(visible):
        seed_k, seed_x = np.unravel_index(int(np.nanargmax(np.where(visible, line, -np.inf))), line.shape)
        keep = connected_component(visible, int(seed_k), int(seed_x))
    else:
        keep = visible
    return np.where(keep, line, 0.0), int(np.count_nonzero(visible & ~keep))


def scan_diagnostics(
    pairs: list[tuple[Path, Path, pd.Timestamp]],
    *,
    varname: str,
    axis: str,
    j0: int,
    i0: int,
) -> tuple[pd.DataFrame, float]:
    rows = []
    global_frame_max = 0.0
    for case_path, control_path, time in pairs:
        delta = read_delta_ppbv(case_path, control_path, varname)
        line = get_line(delta, axis, j0, i0)
        global_frame_max = max(global_frame_max, float(np.nanmax(line)))
        col = delta[:, j0, i0]
        local = delta[:, max(0, j0 - 2) : j0 + 3, max(0, i0 - 2) : i0 + 3]
        max_idx = np.unravel_index(int(np.nanargmax(delta)), delta.shape)
        rows.append(
            {
                "time": time.strftime("%Y-%m-%d_%H:%M:%S"),
                "src_col_max_delta_ppbv": float(np.nanmax(col)),
                "src_sfc_delta_ppbv": float(delta[0, j0, i0]),
                "local_5x5_col_max_delta_ppbv": float(np.nanmax(local)),
                "domain_max_delta_ppbv": float(np.nanmax(delta)),
                "domain_sum_positive_delta_ppbv": float(np.nansum(delta)),
                "max_k": int(max_idx[0] + 1),
                "max_j": int(max_idx[1] + 1),
                "max_i": int(max_idx[2] + 1),
            }
        )
    return pd.DataFrame(rows), global_frame_max


def plot_frame(
    case_path: Path,
    control_path: Path,
    time: pd.Timestamp,
    outpath: Path,
    *,
    varname: str,
    axis: str,
    j0: int,
    i0: int,
    vmax: float,
    floor_ppbv: float,
    mask_threshold_ppbv: float,
    component_filter: str,
    max_height_agl: float,
    metadata_text: str,
) -> None:
    delta = read_delta_ppbv(case_path, control_path, varname)
    with Dataset(case_path) as nc:
        lat = read_var(nc, "XLAT", time_index=0)
        lon = read_var(nc, "XLONG", time_index=0)
        hgt = read_var(nc, "HGT", time_index=0)
        pblh = read_var(nc, "PBLH", time_index=0)
        z_agl, _ = compute_height_agl_and_dz(nc, 0)

    line = get_line(delta, axis, j0, i0)
    zline = get_line(z_agl, axis, j0, i0)
    terrain = get_line_2d(hgt, axis, j0, i0)
    pblh_line = get_line_2d(pblh, axis, j0, i0)
    coord = get_line_2d(lon if axis == "x" else lat, axis, j0, i0)
    y = zline + terrain[np.newaxis, :]
    source_x = i0 if axis == "x" else j0
    line_filtered, removed_pixels = filtered_line_field(
        line,
        threshold_ppbv=mask_threshold_ppbv,
        component_filter=component_filter,
        source_x=source_x,
    )
    zplot = np.ma.masked_less_equal(line_filtered, floor_ppbv)

    fig = plt.figure(figsize=(12, 7), constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1)
    mesh = ax.pcolormesh(
        coord,
        y,
        zplot,
        shading="auto",
        cmap="magma",
        norm=LogNorm(vmin=floor_ppbv, vmax=max(vmax, floor_ppbv * 10.0)),
    )
    ax.fill_between(coord, terrain, np.nanmin(terrain) - 100, color="0.45", alpha=0.95)
    ax.plot(coord, terrain + pblh_line, color="cyan", linewidth=1.5, label="PBLH")
    source_coord = float(lon[j0, i0] if axis == "x" else lat[j0, i0])
    ax.axvline(source_coord, color="white", linestyle="--", linewidth=1.3, label="fuente")
    ax.set_ylim(float(np.nanmin(terrain)), float(np.nanmax(terrain)) + max_height_agl)
    ax.set_xlabel("Longitud" if axis == "x" else "Latitud")
    ax.set_ylabel("Altura MSL (m)")
    ax.set_title(
        f"Delta CO pulso - control | {time.strftime('%Y-%m-%d %H:%M')} | "
        f"corte {'W-E' if axis == 'x' else 'S-N'}"
    )
    ax.text(
        0.39,
        0.985,
        metadata_text + f"\nFiltro visual: componente={component_filter}; umbral={mask_threshold_ppbv:g} ppbv; pixeles removidos={removed_pixels}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="0.08",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.35", "alpha": 0.86},
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("Delta CO (ppbv)")
    ax.legend(loc="upper right", framealpha=0.85)

    inset = ax.inset_axes([0.035, 0.62, 0.24, 0.32])
    inset.contourf(lon, lat, hgt, levels=18, cmap="terrain")
    inset.plot(lon[j0, :], lat[j0, :], color="white", linewidth=1.0, alpha=0.9) if axis == "x" else inset.plot(lon[:, i0], lat[:, i0], color="white", linewidth=1.0, alpha=0.9)
    inset.scatter([lon[j0, i0]], [lat[j0, i0]], s=35, c="red", edgecolors="white", linewidths=0.7)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("corte/fuente", fontsize=9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    add_figure_footer(fig, script_path=Path(__file__).resolve(), output_path=outpath)
    fig.subplots_adjust(left=0.07, right=0.92, top=0.92, bottom=0.16)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def build_metadata_text(args: argparse.Namespace, case_path: Path, control_path: Path, j0: int, i0: int) -> str:
    return "\n".join(
        [
            f"Experimento: delta_co = pulso - control | variable={args.var}",
            f"Liberacion: {args.release_type}; masa={args.pulse_mass_ton:g} ton CO; duracion={args.pulse_duration_hours:g} h",
            f"Inicio pulso={args.pulse_start}; intervalo wrfchemi={args.emissions_interval_min:g} min",
            f"Fuente: i={i0 + 1}, j={j0 + 1}; corte={'longitud' if args.axis == 'x' else 'latitud'} de referencia",
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
    df, transect_vmax = scan_diagnostics(pairs, varname=args.var, axis=args.axis, j0=j0, i0=i0)
    domain_vmax = float(df["domain_max_delta_ppbv"].max()) if not df.empty else 0.0
    color_vmax = max(transect_vmax, args.floor_ppbv * 10.0)
    mask_threshold_ppbv = args.mask_threshold_ppbv if args.mask_threshold_ppbv is not None else args.floor_ppbv

    csv_path = outdir / f"{args.tag}_diagnostico_delta_co.csv"
    df.to_csv(csv_path, index=False)

    frame_paths = []
    for n, (case_path, control_path, time) in enumerate(pairs):
        frame = frames_dir / f"{args.tag}_{n:03d}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        metadata_text = build_metadata_text(args, case_path, control_path, j0, i0)
        plot_frame(
            case_path,
            control_path,
            time,
            frame,
            varname=args.var,
            axis=args.axis,
            j0=j0,
            i0=i0,
            vmax=color_vmax,
            floor_ppbv=args.floor_ppbv,
            mask_threshold_ppbv=mask_threshold_ppbv,
            component_filter=args.component_filter,
            max_height_agl=args.max_height_agl,
            metadata_text=metadata_text,
        )
        frame_paths.append(frame)

    gif_path = outdir / f"{args.tag}_delta_co.gif"
    gif = build_gif_from_pngs(frame_paths, gif_path, duration_ms=args.gif_duration_ms)
    print(f"CSV: {csv_path}")
    print(f"Frames: {frames_dir}")
    if gif:
        print(f"GIF: {gif}")
    else:
        print("GIF: no generado")
    print(f"Maximo delta real en corte: {transect_vmax:.6g} ppbv")
    print(f"Maximo delta real en dominio: {domain_vmax:.6g} ppbv")
    print(f"Maximo usado para color: {color_vmax:.6g} ppbv")
    if domain_vmax <= 0.0:
        print("Aviso: delta_co es cero en todo el dominio; el pulso no cambia co frente al control.")


if __name__ == "__main__":
    main()
