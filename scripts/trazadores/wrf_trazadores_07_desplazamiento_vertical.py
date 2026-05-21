"""
Visualizacion del desplazamiento vertical de trazadores WRF.

Productos:
- PNG tiempo-altura con la masa proxy del trazador integrada por niveles;
- CSV con diagnosticos de altura del penacho y fraccion sobre PBLH;
- PNG por tiempo de un corte vertical por la fuente/centro;
- GIF opcional del corte vertical.

El script esta pensado para corridas con N_SLV/S_SLV, tr17_* o trazadores 3D
equivalentes en archivos wrfout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from netCDF4 import Dataset

from wrf_trazadores_00_core import (
    add_common_cli_arguments,
    add_figure_footer,
    build_gif_from_pngs,
    build_region_mask,
    compute_height_agl_and_dz,
    get_latlon_2d,
    infer_domain_tag,
    make_time_range_tag,
    make_tracer_output_name,
    maybe_apply_common_gui_options,
    parse_wrf_times,
    read_tracer_step,
    read_var,
    resolve_common_inputs,
    safe_tag,
    save_rows_csv,
    split_tracer_name_mode,
    title_with_subtitle,
    validate_wrf_files,
)


DEFAULT_MAX_HEIGHT_M = 2500.0
DEFAULT_HEIGHT_BIN_M = 25.0
DEFAULT_FRAME_MIN_FRACTION = 1.0e-3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grafica como se desplaza verticalmente un trazador WRF: "
            "tiempo-altura, CSV diagnostico y GIF de corte vertical."
        )
    )
    add_common_cli_arguments(parser)
    parser.add_argument("--source-lat", type=float, default=None, help="Latitud de la fuente/pulso.")
    parser.add_argument("--source-lon", type=float, default=None, help="Longitud de la fuente/pulso.")
    parser.add_argument(
        "--source-label",
        default="Pulso",
        help="Etiqueta informativa de la fuente/pulso.",
    )
    parser.add_argument(
        "--i",
        type=int,
        default=None,
        help="Indice WRF 1-based oeste-este para el corte/columna. Ej.: 56.",
    )
    parser.add_argument(
        "--j",
        type=int,
        default=None,
        help="Indice WRF 1-based sur-norte para el corte/columna. Ej.: 66.",
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y"],
        default="x",
        help="Orientacion del corte: x fija j y recorre oeste-este; y fija i y recorre sur-norte.",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=DEFAULT_MAX_HEIGHT_M,
        help=(
            "Altura maxima AGL inicial a considerar en metros. "
            "Si usa --auto-max-height, el grafico se recorta luego al techo real del trazador."
        ),
    )
    parser.add_argument(
        "--auto-max-height",
        action="store_true",
        help="Recorta el eje AGL al ztop maximo del trazador segun --auto-height-frac.",
    )
    parser.add_argument(
        "--auto-height-frac",
        type=float,
        default=0.01,
        help=(
            "Fraccion del maximo instantaneo usada para definir el techo automatico. "
            "0.01 equivale al 1%% del maximo; 0.1 enfoca el nucleo fuerte."
        ),
    )
    parser.add_argument(
        "--height-padding",
        type=float,
        default=1.08,
        help="Factor de margen aplicado al techo automatico del eje AGL.",
    )
    parser.add_argument(
        "--height-bin",
        type=float,
        default=DEFAULT_HEIGHT_BIN_M,
        help="Espesor de bins verticales para el panel tiempo-altura, en metros.",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Salto temporal para frames del corte vertical.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Genera PNG por tiempo pero no arma GIF.",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=450,
        help="Duracion de cada frame del GIF.",
    )
    parser.add_argument(
        "--log-color",
        action="store_true",
        help="Usa escala logaritmica para el color del trazador.",
    )
    parser.add_argument(
        "--frame-min-fraction",
        type=float,
        default=DEFAULT_FRAME_MIN_FRACTION,
        help="Oculta en frames valores menores a esta fraccion del maximo global.",
    )
    parser.add_argument(
        "--threshold-fracs",
        type=float,
        nargs="+",
        default=[0.10, 0.01, 0.001],
        help="Fracciones del maximo instantaneo usadas para ztop diagnostico.",
    )
    parser.add_argument(
        "--x-coordinate",
        choices=["auto", "distance", "lon", "lat"],
        default="auto",
        help=(
            "Coordenada horizontal para los frames: auto usa longitud en cortes X "
            "y latitud en cortes Y."
        ),
    )
    parser.add_argument(
        "--frame-height-reference",
        choices=["msl", "agl"],
        default="msl",
        help="Referencia vertical de los frames. msl permite dibujar topografia.",
    )
    parser.add_argument(
        "--hide-terrain",
        action="store_true",
        help="No dibuja la topografia en los frames.",
    )
    parser.add_argument(
        "--hide-inset-map",
        action="store_true",
        help="No dibuja el mapa inserto de ubicacion del pulso en los frames.",
    )
    parser.add_argument(
        "--pulse-kind",
        choices=["auto", "inerte", "no-inerte"],
        default="auto",
        help="Tipo de trazador que se anotara en las figuras.",
    )
    parser.add_argument(
        "--pulse-duration-min",
        type=float,
        default=0.0,
        help="Duracion de la emision/pulso en minutos. 0 indica pulso inicial instantaneo.",
    )
    parser.add_argument(
        "--pulse-amount-label",
        default=None,
        help=(
            "Cantidad emitida a mostrar literalmente. Si se omite, se calcula "
            "la suma inicial proxy del trazador en el primer wrfout."
        ),
    )
    parser.add_argument(
        "--pulse-mass-label",
        default=None,
        help=(
            "Masa fisica emitida para anotar literalmente. Si se omite, se indica "
            "que la masa fisica no esta definida para trazadores adimensionales."
        ),
    )
    parser.add_argument(
        "--pulse-note",
        default=None,
        help="Nota corta adicional sobre el pulso para anotar en las figuras.",
    )
    parser.add_argument(
        "--pulse-place",
        default=None,
        help="Ciudad, corregimiento o municipio donde se origina el pulso.",
    )
    return parser


def nearest_grid_point(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lat: float,
    lon: float,
) -> tuple[int, int]:
    distance2 = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
    j_idx, i_idx = np.unravel_index(int(np.nanargmin(distance2)), lat2d.shape)
    return int(j_idx), int(i_idx)


def source_index_from_args_or_tracer(
    args: argparse.Namespace,
    wrf_files: list[Path],
    tracer_name: str,
    baseline_3d: np.ndarray | None,
) -> tuple[int, int, str]:
    with Dataset(wrf_files[0]) as nc:
        lat2d, lon2d = get_latlon_2d(nc)
        ny, nx = lat2d.shape

        if args.i is not None or args.j is not None:
            if args.i is None or args.j is None:
                raise SystemExit("Debe indicar --i y --j juntos.")
            i_idx = args.i - 1
            j_idx = args.j - 1
            if not (0 <= i_idx < nx and 0 <= j_idx < ny):
                raise SystemExit(
                    f"--i/--j fuera del dominio. Rango valido: i=1..{nx}, j=1..{ny}."
                )
            return j_idx, i_idx, f"i={args.i}, j={args.j}"

        if args.source_lat is not None or args.source_lon is not None:
            if args.source_lat is None or args.source_lon is None:
                raise SystemExit("Debe indicar --source-lat y --source-lon juntos.")
            j_idx, i_idx = nearest_grid_point(
                lat2d,
                lon2d,
                float(args.source_lat),
                float(args.source_lon),
            )
            return j_idx, i_idx, f"fuente cercana a lat/lon ({args.source_lat}, {args.source_lon})"

        tracer0 = read_tracer_step(nc, tracer_name, 0, baseline_3d=baseline_3d)
        column = np.nansum(tracer0, axis=0)
        if np.any(np.isfinite(column) & (column > 0.0)):
            j_idx, i_idx = np.unravel_index(int(np.nanargmax(column)), column.shape)
            return int(j_idx), int(i_idx), "maximo inicial del trazador"

        return ny // 2, nx // 2, "centro geometrico"


def build_time_edges(times: list[pd.Timestamp]) -> np.ndarray:
    x = mdates.date2num(pd.to_datetime(times).to_pydatetime())
    if len(x) == 1:
        return np.array([x[0] - 0.5 / 24.0, x[0] + 0.5 / 24.0])
    mids = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mids[0] - x[0])
    last = x[-1] + (x[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def infer_tracer_kind(nc: Dataset, tracer_name: str, requested_kind: str) -> str:
    if requested_kind == "inerte":
        return "inerte"
    if requested_kind == "no-inerte":
        return "no inerte"

    name_l = tracer_name.lower()
    if name_l in {"co", "co_t"}:
        chem_opt = getattr(nc, "CHEM_OPT", None)
        if chem_opt is not None:
            return f"CO real WRF-Chem (CHEM_OPT={chem_opt})"
        return "CO real WRF-Chem"

    if tracer_name.upper() in {"N_SLV", "S_SLV"} or name_l.startswith("tr17"):
        chem_opt = getattr(nc, "CHEM_OPT", None)
        if chem_opt is not None:
            return f"inerte (CHEM_OPT={chem_opt})"
        return "inerte"
    return "no clasificado"


def format_duration(duration_min: float) -> str:
    if duration_min <= 0.0:
        return "instantaneo en condicion inicial"
    if duration_min < 60.0:
        return f"{duration_min:g} min"
    return f"{duration_min / 60.0:g} h"


def build_pulse_summary(
    args: argparse.Namespace,
    wrf_files: list[Path],
    tracer_name: str,
    baseline_3d: np.ndarray | None,
    j_idx: int,
    i_idx: int,
) -> dict[str, object]:
    with Dataset(wrf_files[0]) as nc:
        times = parse_wrf_times(nc)
        time0 = pd.Timestamp(times[0]) if len(times) else pd.NaT
        lat2d, lon2d = get_latlon_2d(nc)
        hgt = read_var(nc, "HGT", time_index=0) if "HGT" in nc.variables else np.zeros_like(lat2d)
        tracer0 = read_tracer_step(nc, tracer_name, 0, baseline_3d=baseline_3d)
        active = np.isfinite(tracer0) & (tracer0 > 0.0)
        units = getattr(nc.variables.get(tracer_name), "units", "")

        amount = float(np.nansum(np.where(active, tracer0, 0.0)))
        max_value = float(np.nanmax(np.where(active, tracer0, np.nan))) if np.any(active) else 0.0
        active_cells_3d = int(np.sum(active))
        active_columns = int(np.sum(np.nansum(np.where(active, 1, 0), axis=0) > 0))
        active_layers = np.where(np.nansum(np.where(active, 1, 0), axis=(1, 2)) > 0)[0]
        first_layer = int(active_layers[0] + 1) if active_layers.size else None
        last_layer = int(active_layers[-1] + 1) if active_layers.size else None

        if args.pulse_amount_label:
            amount_label = args.pulse_amount_label
        else:
            unit_label = units or "unidades del trazador"
            amount_label = (
                f"carga proxy inicial={amount:.3g} {unit_label}; "
                f"max={max_value:.3g}; {active_columns} columnas / {active_cells_3d} celdas"
            )

        if args.pulse_mass_label:
            mass_label = args.pulse_mass_label
        elif tracer_name.lower() in {"co", "co_t"}:
            mass_label = "no inferida del wrfout; indiquela con --pulse-mass-label"
        elif units and units.lower() in {"dimensionless", "1", "unitless"}:
            mass_label = "no definida fisicamente; N_SLV es adimensional"
        else:
            mass_label = "no indicada"

        return {
            "lat": float(lat2d[j_idx, i_idx]),
            "lon": float(lon2d[j_idx, i_idx]),
            "i": int(i_idx + 1),
            "j": int(j_idx + 1),
            "hgt": float(hgt[j_idx, i_idx]),
            "time0": time0,
            "kind": infer_tracer_kind(nc, tracer_name, args.pulse_kind),
            "duration": format_duration(float(args.pulse_duration_min)),
            "amount_label": amount_label,
            "mass_label": mass_label,
            "units": units,
            "first_layer": first_layer,
            "last_layer": last_layer,
            "note": args.pulse_note,
            "place": args.pulse_place,
        }


def pulse_summary_text(summary: dict[str, object], *, compact: bool = False) -> str:
    prefix = f"Origen: {summary['place']}; " if summary.get("place") else "Origen: "
    where = (
        f"{prefix}lon={summary['lon']:.4f}, lat={summary['lat']:.4f} "
        f"(i={summary['i']}, j={summary['j']}), HGT={summary['hgt']:.0f} m"
    )
    if summary.get("first_layer") is not None:
        layer = f"k={summary['first_layer']}"
        if summary.get("last_layer") != summary.get("first_layer"):
            layer += f"-{summary['last_layer']}"
        where = f"{where}, {layer}"

    lines = [
        where,
        f"Tipo: trazador {summary['kind']}",
        f"Masa emitida: {summary['mass_label']}",
        f"Pulso: {summary['amount_label']}",
        f"Duracion: {summary['duration']}",
    ]
    if summary.get("note"):
        lines.append(f"Nota: {summary['note']}")

    if compact:
        return "\n".join(lines[:4])
    return "\n".join(lines)


def summarize_vertical_evolution(
    wrf_files: list[Path],
    tracer_name: str,
    baseline_3d: np.ndarray | None,
    region_mask: np.ndarray,
    height_edges: np.ndarray,
    threshold_fracs: list[float],
) -> tuple[list[dict], np.ndarray, float]:
    rows: list[dict] = []
    profiles: list[np.ndarray] = []
    global_max = 0.0

    for path in wrf_files:
        with Dataset(path) as nc:
            times = parse_wrf_times(nc)
            for time_index, time_value in enumerate(times):
                tracer = read_tracer_step(nc, tracer_name, time_index, baseline_3d=baseline_3d)
                z_agl, dz = compute_height_agl_and_dz(nc, time_index)
                pblh = read_var(nc, "PBLH", time_index=time_index)

                region3d = region_mask[np.newaxis, :, :]
                valid = (
                    region3d
                    & np.isfinite(tracer)
                    & np.isfinite(z_agl)
                    & np.isfinite(dz)
                    & (tracer > 0.0)
                    & (z_agl >= 0.0)
                )
                weights = np.where(valid, tracer * np.maximum(dz, 0.0), 0.0)

                profile, _ = np.histogram(
                    z_agl[valid],
                    bins=height_edges,
                    weights=weights[valid],
                )
                profiles.append(profile.astype(float))

                total = float(np.nansum(weights))
                above_mask = z_agl > pblh[np.newaxis, :, :]
                above = float(np.nansum(np.where(above_mask, weights, 0.0)))
                frac_above = above / total if total > 0 else np.nan
                max_total = float(np.nanmax(np.where(region3d, tracer, np.nan)))
                global_max = max(global_max, max_total if np.isfinite(max_total) else 0.0)

                row: dict[str, object] = {
                    "time": pd.Timestamp(time_value),
                    "file": path.name,
                    "total_mass_proxy": total,
                    "mass_proxy_above_pblh": above,
                    "fraction_above_pblh": frac_above,
                    "max_tracer": max_total,
                    "pblh_mean_m": float(np.nanmean(np.where(region_mask, pblh, np.nan))),
                    "pblh_max_m": float(np.nanmax(np.where(region_mask, pblh, np.nan))),
                }

                for frac in threshold_fracs:
                    key = f"ztop_{safe_tag(str(frac))}_maxfrac_m"
                    if max_total > 0 and np.isfinite(max_total):
                        active = valid & (tracer >= frac * max_total)
                        row[key] = float(np.nanmax(z_agl[active])) if np.any(active) else np.nan
                    else:
                        row[key] = np.nan

                rows.append(row)

    return rows, np.vstack(profiles), global_max


def plot_time_height(
    rows: list[dict],
    profiles: np.ndarray,
    height_edges: np.ndarray,
    tracer_display_name: str,
    outpath: Path,
    *,
    log_color: bool,
    subtitle: str,
    pulse_summary: dict[str, object],
) -> Path:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    times = list(pd.to_datetime(df["time"]))
    time_edges = build_time_edges(times)
    z_centers = 0.5 * (height_edges[:-1] + height_edges[1:])
    z_plot = profiles.T

    positive = z_plot[np.isfinite(z_plot) & (z_plot > 0.0)]
    if positive.size and log_color:
        norm = LogNorm(vmin=max(float(np.nanmin(positive)), 1.0e-12), vmax=float(np.nanmax(positive)))
        plot_data = np.ma.masked_less_equal(z_plot, 0.0)
        cbar_label = f"{tracer_display_name} * dz integrado por nivel (log)"
    else:
        norm = None
        plot_data = np.ma.masked_invalid(z_plot)
        cbar_label = f"{tracer_display_name} * dz integrado por nivel"

    fig, ax = plt.subplots(figsize=(12, 7))
    pcm = ax.pcolormesh(
        time_edges,
        height_edges,
        plot_data,
        shading="auto",
        cmap="magma",
        norm=norm,
    )
    fig.colorbar(pcm, ax=ax, label=cbar_label)
    ax.plot(times, df["pblh_mean_m"], color="cyan", linewidth=1.6, label="PBLH media")
    ax.plot(times, df["pblh_max_m"], color="white", linewidth=1.0, alpha=0.8, label="PBLH maxima")
    for col in [name for name in df.columns if name.startswith("ztop_")]:
        ax.plot(times, df[col], linewidth=1.0, alpha=0.75, label=col.replace("_maxfrac_m", ""))

    ax.set_ylabel("Altura AGL (m)")
    ax.set_xlabel("Tiempo")
    ax.set_ylim(float(height_edges[0]), float(height_edges[-1]))
    ax.set_title(title_with_subtitle(f"Desplazamiento vertical - {tracer_display_name}", subtitle))
    ax.text(
        0.012,
        0.982,
        pulse_summary_text(pulse_summary, compact=True),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "0.05", "edgecolor": "0.8", "alpha": 0.72},
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    add_figure_footer(fig, script_path=__file__, output_path=outpath)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return outpath


def ztop_column_name(frac: float) -> str:
    return f"ztop_{safe_tag(str(frac))}_maxfrac_m"


def resolve_effective_max_height(
    rows: list[dict],
    requested_max_height: float,
    *,
    auto_max_height: bool,
    auto_height_frac: float,
    height_padding: float,
    height_bin: float,
) -> float:
    if not auto_max_height:
        return float(requested_max_height)

    df = pd.DataFrame(rows)
    preferred = ztop_column_name(auto_height_frac)
    if preferred in df.columns:
        candidates = pd.to_numeric(df[preferred], errors="coerce")
    else:
        ztop_cols = [col for col in df.columns if col.startswith("ztop_")]
        candidates = (
            pd.to_numeric(df[ztop_cols].max(axis=1), errors="coerce")
            if ztop_cols
            else pd.Series(dtype=float)
        )

    candidates = candidates[np.isfinite(candidates)]
    candidates = candidates[candidates > 0.0]
    if candidates.empty:
        return float(requested_max_height)

    raw_top = float(candidates.max()) * max(float(height_padding), 1.0)
    if height_bin > 0:
        raw_top = np.ceil(raw_top / height_bin) * height_bin
    return max(float(height_bin), min(float(requested_max_height), float(raw_top)))


def crop_profiles_to_height(
    profiles: np.ndarray,
    height_edges: np.ndarray,
    max_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    keep_edges = height_edges <= max_height
    if np.count_nonzero(keep_edges) < 2:
        return profiles[:, :1], height_edges[:2]

    last_edge_idx = int(np.where(keep_edges)[0][-1])
    cropped_edges = height_edges[: last_edge_idx + 1]
    cropped_profiles = profiles[:, :last_edge_idx]
    return cropped_profiles, cropped_edges


def distance_axis_km(nc: Dataset, length: int, axis: str) -> np.ndarray:
    spacing = getattr(nc, "DX" if axis == "x" else "DY", None)
    try:
        spacing_m = float(spacing)
    except Exception:
        spacing_m = 1000.0
    return np.arange(length, dtype=float) * spacing_m / 1000.0


def tracer_plot_label(nc: Dataset, tracer_name: str, tracer_display_name: str) -> str:
    var = nc.variables.get(tracer_name)
    units = getattr(var, "units", "") if var is not None else ""
    desc = getattr(var, "description", "") if var is not None else ""

    if tracer_name.lower() == "co":
        base = "CO - monoxido de carbono"
    elif tracer_name.lower() == "co_t":
        base = "CO_T - trazador de CO"
    elif tracer_name.upper() == "N_SLV":
        base = "N_SLV - trazador inerte norte"
    elif tracer_name.upper() == "S_SLV":
        base = "S_SLV - trazador inerte sur"
    elif desc and desc.lower() != tracer_name.lower():
        base = f"{tracer_display_name} - {desc}"
    else:
        base = tracer_display_name

    if units:
        return f"{base} [{units}]"
    return base


def section_coordinates(
    nc: Dataset,
    axis: str,
    j_idx: int,
    i_idx: int,
    length: int,
    x_coordinate: str,
) -> tuple[np.ndarray, float, str]:
    lat2d, lon2d = get_latlon_2d(nc)
    mode = x_coordinate
    if mode == "auto":
        mode = "lon" if axis == "x" else "lat"

    if axis == "x":
        if mode == "lon":
            coord = lon2d[j_idx, :]
            marker = float(coord[i_idx])
            label = "Longitud (grados)"
        elif mode == "lat":
            coord = lat2d[j_idx, :]
            marker = float(coord[i_idx])
            label = "Latitud (grados)"
        else:
            coord = distance_axis_km(nc, length, "x")
            marker = float(coord[i_idx])
            label = "Distancia oeste-este (km)"
    else:
        if mode == "lat":
            coord = lat2d[:, i_idx]
            marker = float(coord[j_idx])
            label = "Latitud (grados)"
        elif mode == "lon":
            coord = lon2d[:, i_idx]
            marker = float(coord[j_idx])
            label = "Longitud (grados)"
        else:
            coord = distance_axis_km(nc, length, "y")
            marker = float(coord[j_idx])
            label = "Distancia sur-norte (km)"

    return np.asarray(coord, dtype=float), marker, label


def add_pulse_location_inset(
    ax,
    nc: Dataset,
    axis: str,
    j_idx: int,
    i_idx: int,
    pulse_summary: dict[str, object],
) -> None:
    lat2d, lon2d = get_latlon_2d(nc)
    hgt = read_var(nc, "HGT", time_index=0) if "HGT" in nc.variables else np.zeros_like(lat2d)

    inset = ax.inset_axes([0.705, 0.075, 0.275, 0.34])
    terrain = np.ma.masked_invalid(hgt)
    pcm = inset.pcolormesh(
        lon2d,
        lat2d,
        terrain,
        shading="auto",
        cmap="terrain",
        alpha=0.88,
    )

    if axis == "x":
        inset.plot(lon2d[j_idx, :], lat2d[j_idx, :], color="white", linewidth=1.2, label="Transecto")
    else:
        inset.plot(lon2d[:, i_idx], lat2d[:, i_idx], color="white", linewidth=1.2, label="Transecto")

    inset.scatter(
        [float(pulse_summary["lon"])],
        [float(pulse_summary["lat"])],
        s=42,
        marker="*",
        color="red",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label="Pulso",
    )
    if pulse_summary.get("place"):
        place = str(pulse_summary["place"])
        place = place.replace(", Valle del Cauca", "")
        place = place.replace(", ", "\n", 1)
        inset.text(
            0.035,
            0.955,
            place,
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "0.05", "edgecolor": "white", "alpha": 0.72},
        )
    inset.set_title("Ubicacion del pulso", fontsize=8, pad=3)
    inset.set_xlabel("Lon", fontsize=7)
    inset.set_ylabel("Lat", fontsize=7)
    inset.tick_params(labelsize=6, length=2)
    inset.grid(alpha=0.25, linewidth=0.4)
    inset.legend(loc="lower left", fontsize=6, framealpha=0.82)
    for spine in inset.spines.values():
        spine.set_edgecolor("0.15")
        spine.set_linewidth(0.8)

    cax = inset.inset_axes([0.50, 0.055, 0.43, 0.055])
    cbar = ax.figure.colorbar(pcm, cax=cax, orientation="horizontal")
    cbar.set_label("HGT (m)", fontsize=5, labelpad=-1)
    cbar.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    cax.set_facecolor((1, 1, 1, 0.72))


def plot_vertical_frame(
    nc: Dataset,
    time_index: int,
    time_value: pd.Timestamp,
    tracer_name: str,
    baseline_3d: np.ndarray | None,
    j_idx: int,
    i_idx: int,
    axis: str,
    max_height: float,
    global_max: float,
    frame_min_fraction: float,
    tracer_display_name: str,
    outpath: Path,
    *,
    log_color: bool,
    x_coordinate: str,
    frame_height_reference: str,
    show_terrain: bool,
    pulse_summary: dict[str, object],
    show_inset_map: bool,
) -> Path:
    import matplotlib.pyplot as plt

    tracer = read_tracer_step(nc, tracer_name, time_index, baseline_3d=baseline_3d)
    z_agl, _ = compute_height_agl_and_dz(nc, time_index)
    pblh = read_var(nc, "PBLH", time_index=time_index)
    hgt = read_var(nc, "HGT", time_index=time_index) if "HGT" in nc.variables else np.zeros_like(pblh)
    label = tracer_plot_label(nc, tracer_name, tracer_display_name)

    threshold = max(global_max * frame_min_fraction, 0.0)
    if axis == "x":
        field = tracer[:, j_idx, :]
        terrain = hgt[j_idx, :]
        zsec_agl = z_agl[:, j_idx, :]
        pblh_agl = pblh[j_idx, :]
        coord, marker_x, xlabel = section_coordinates(
            nc, axis, j_idx, i_idx, field.shape[1], x_coordinate
        )
        title_axis = f"transecto W-E por j={j_idx + 1}"
    else:
        field = tracer[:, :, i_idx]
        terrain = hgt[:, i_idx]
        zsec_agl = z_agl[:, :, i_idx]
        pblh_agl = pblh[:, i_idx]
        coord, marker_x, xlabel = section_coordinates(
            nc, axis, j_idx, i_idx, field.shape[1], x_coordinate
        )
        title_axis = f"transecto S-N por i={i_idx + 1}"

    if frame_height_reference == "msl":
        zsec = zsec_agl + terrain[np.newaxis, :]
        pblh_line = pblh_agl + terrain
        ylabel = "Altura MSL (m)"
        y_min = max(0.0, float(np.nanmin(terrain)) - 100.0)
        y_max = float(np.nanmax(terrain)) + max_height
    else:
        zsec = zsec_agl
        pblh_line = pblh_agl
        ylabel = "Altura AGL (m)"
        y_min = 0.0
        y_max = max_height

    field = np.where(field >= threshold, field, np.nan)
    x2d = np.broadcast_to(coord[np.newaxis, :], field.shape)

    positive = field[np.isfinite(field) & (field > 0.0)]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    if log_color:
        norm = LogNorm(
            vmin=max(threshold, 1.0e-12),
            vmax=max(global_max, max(threshold, 1.0e-12) * 10.0),
        )
    else:
        norm = Normalize(vmin=0.0, vmax=max(global_max, 1.0e-12))

    pcm = ax.pcolormesh(
        x2d,
        zsec,
        np.ma.masked_invalid(field),
        shading="auto",
        cmap="magma",
        norm=norm,
    )
    fig.colorbar(pcm, ax=ax, label=label)
    if frame_height_reference == "msl" and show_terrain:
        ax.fill_between(
            coord,
            y_min,
            terrain,
            color="0.35",
            alpha=0.85,
            linewidth=0,
            label="Topografia (HGT)",
        )
        ax.plot(coord, terrain, color="0.12", linewidth=0.8)
    ax.plot(coord, pblh_line, color="cyan", linewidth=1.5, label="PBLH")
    ax.axvline(marker_x, color="white", linestyle="--", linewidth=1.0, alpha=0.85, label="fuente/centro")
    ax.text(
        0.012,
        0.982,
        pulse_summary_text(pulse_summary, compact=True),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "0.05", "edgecolor": "0.8", "alpha": 0.74},
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        title_with_subtitle(
            f"{label} - {time_value:%Y-%m-%d %H:%M}",
            f"{title_axis}; terreno gris; linea cyan=PBLH",
        )
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    if show_inset_map:
        add_pulse_location_inset(ax, nc, axis, j_idx, i_idx, pulse_summary)
    add_figure_footer(fig, script_path=__file__, output_path=outpath)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def make_vertical_frames(
    wrf_files: list[Path],
    tracer_name: str,
    baseline_3d: np.ndarray | None,
    j_idx: int,
    i_idx: int,
    axis: str,
    max_height: float,
    global_max: float,
    frame_min_fraction: float,
    tracer_display_name: str,
    frames_dir: Path,
    *,
    time_stride: int,
    log_color: bool,
    x_coordinate: str,
    frame_height_reference: str,
    show_terrain: bool,
    pulse_summary: dict[str, object],
    show_inset_map: bool,
) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    global_step = 0
    for path in wrf_files:
        with Dataset(path) as nc:
            times = parse_wrf_times(nc)
            for time_index, time_value in enumerate(times):
                if global_step % max(time_stride, 1) != 0:
                    global_step += 1
                    continue
                outpath = frames_dir / f"frame_{global_step:04d}_{time_value:%Y%m%d_%H%M%S}.png"
                frame = plot_vertical_frame(
                    nc,
                    time_index,
                    pd.Timestamp(time_value),
                    tracer_name,
                    baseline_3d,
                    j_idx,
                    i_idx,
                    axis,
                    max_height,
                    global_max,
                    frame_min_fraction,
                    tracer_display_name,
                    outpath,
                    log_color=log_color,
                    x_coordinate=x_coordinate,
                    frame_height_reference=frame_height_reference,
                    show_terrain=show_terrain,
                    pulse_summary=pulse_summary,
                    show_inset_map=show_inset_map,
                )
                frames.append(frame)
                global_step += 1
    return frames


def main() -> None:
    args = build_parser().parse_args()
    args = maybe_apply_common_gui_options(
        args,
        title="Opciones - desplazamiento vertical",
        include_delta_from_initial=True,
    )
    wrf_files, outdir = resolve_common_inputs(
        args,
        default_subdir="wrf_trazadores_07_desplazamiento_vertical",
    )

    tracer_name_arg, delta_from_initial = split_tracer_name_mode(
        args.tracer_name,
        args.delta_from_initial,
    )
    tracer_name, tracer_units, all_times = validate_wrf_files(
        wrf_files,
        tracer_name=tracer_name_arg,
        required_vars=["PH", "PHB", "PBLH"],
    )
    tracer_display_name = make_tracer_output_name(tracer_name, delta_from_initial)
    baseline_3d = None
    if delta_from_initial:
        with Dataset(wrf_files[0]) as nc0:
            baseline_3d = read_var(nc0, tracer_name, time_index=0)

    with Dataset(wrf_files[0]) as nc0:
        lat2d, lon2d = get_latlon_2d(nc0)
        region_mask, region_label = build_region_mask(
            lat2d,
            lon2d,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            shapefile=None,
        )

    j_idx, i_idx, source_label = source_index_from_args_or_tracer(
        args,
        wrf_files,
        tracer_name,
        baseline_3d,
    )
    pulse_summary = build_pulse_summary(
        args,
        wrf_files,
        tracer_name,
        baseline_3d,
        j_idx,
        i_idx,
    )

    height_edges = np.arange(0.0, args.max_height + args.height_bin, args.height_bin)
    if height_edges.size < 2:
        raise SystemExit("--max-height y --height-bin generan menos de dos niveles.")

    rows, profiles, global_max = summarize_vertical_evolution(
        wrf_files,
        tracer_name,
        baseline_3d,
        region_mask,
        height_edges,
        args.threshold_fracs,
    )

    domain_tag = infer_domain_tag(wrf_files[0])
    time_tag = make_time_range_tag(all_times)
    case_tag = safe_tag(args.tag or f"{domain_tag}_{region_label}_{tracer_display_name}")

    csv_path = outdir / f"{case_tag}_desplazamiento_vertical_{time_tag}.csv"
    save_rows_csv(rows, csv_path)

    subtitle = (
        f"region={region_label}; corte={args.axis}; referencia={source_label}; "
        f"i={i_idx + 1}, j={j_idx + 1}"
    )
    effective_max_height = resolve_effective_max_height(
        rows,
        args.max_height,
        auto_max_height=args.auto_max_height,
        auto_height_frac=args.auto_height_frac,
        height_padding=args.height_padding,
        height_bin=args.height_bin,
    )
    plot_profiles, plot_height_edges = crop_profiles_to_height(
        profiles,
        height_edges,
        effective_max_height,
    )

    time_height_png = outdir / f"{case_tag}_tiempo_altura_{time_tag}.png"
    plot_time_height(
        rows,
        plot_profiles,
        plot_height_edges,
        tracer_display_name,
        time_height_png,
        log_color=args.log_color,
        subtitle=subtitle,
        pulse_summary=pulse_summary,
    )

    frames_dir = outdir / f"{case_tag}_frames_corte_{args.axis}_{time_tag}"
    frames = make_vertical_frames(
        wrf_files,
        tracer_name,
        baseline_3d,
        j_idx,
        i_idx,
        args.axis,
        effective_max_height,
        global_max,
        args.frame_min_fraction,
        tracer_display_name,
        frames_dir,
        time_stride=args.time_stride,
        log_color=args.log_color,
        x_coordinate=args.x_coordinate,
        frame_height_reference=args.frame_height_reference,
        show_terrain=not args.hide_terrain,
        pulse_summary=pulse_summary,
        show_inset_map=not args.hide_inset_map,
    )

    gif_path = None
    if not args.no_gif:
        gif_path = build_gif_from_pngs(
            frames,
            outdir / f"{case_tag}_corte_vertical_{args.axis}_{time_tag}.gif",
            duration_ms=args.gif_duration_ms,
        )

    print("Listo.")
    print(f"CSV: {csv_path}")
    print(f"Tiempo-altura: {time_height_png}")
    print(f"Frames: {frames_dir}")
    print(f"Altura maxima graficada AGL: {effective_max_height:.1f} m")
    print("Pulso:")
    print(pulse_summary_text(pulse_summary))
    if gif_path is not None:
        print(f"GIF: {gif_path}")
    elif not args.no_gif:
        print("GIF: no generado; revise que haya al menos dos frames y Pillow disponible.")


if __name__ == "__main__":
    main()
