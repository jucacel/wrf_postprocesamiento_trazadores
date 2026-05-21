"""
Mapas horarios del penacho del trazador a una altura AGL fija.

La figura sigue la ventana espacial usada en los ejercicios de ventilacion
de wrf_urbano para Cali, Yumbo, Palmira y Rozo:
- lat 3.35 a 3.65
- lon -76.55 a -76.25
- zoom_out 0.12
- figsize 10 x 8

Cada salida muestra:
- el trazador en un corte horizontal a una altura AGL;
- el shape del area de estudio;
- el punto de emision del pulso;
- el centroide de la pluma visible;
- una flecha desde la fuente al centroide;
- el punto de maximo de la pluma;
- un pie de figura con la ruta del script y de la imagen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wrf_trazadores_00_core import (
    WORKSPACE_ROOT,
    add_common_cli_arguments,
    add_figure_footer,
    build_gif_from_pngs,
    compute_height_agl_and_dz,
    format_source_marker,
    get_latlon_2d,
    haversine_km,
    infer_domain_tag,
    iter_wrf_steps,
    load_tracer_baseline,
    make_time_range_tag,
    make_tracer_output_name,
    maybe_apply_common_gui_options,
    plot_shapefile_boundary,
    plot_source_marker,
    read_tracer_step,
    region_nanmax,
    resolve_common_inputs,
    resolve_shapefile_path,
    resolve_source_marker,
    safe_tag,
    save_rows_csv,
    split_tracer_name_mode,
    title_with_subtitle,
    validate_wrf_files,
    weighted_centroid,
)

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception:  # pragma: no cover
    tk = None
    messagebox = None
    ttk = None


LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.55, -76.25
ZOOM_OUT = 0.12
FIGSIZE = (10, 8)
CITY_LABELS = {
    "Cali": (-76.516651, 3.443662),
    "Yumbo": (-76.490915, 3.571668),
    "Rozo": (-76.384813, 3.610740),
    "Palmira": (-76.294686, 3.527459),
}
DEFAULT_HEIGHT_AGL_M = 10.0
DEFAULT_VISUAL_FRAC = 0.005


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un PNG por hora del penacho del trazador en un plano "
            "horizontal a una altura AGL fija, usando la ventana espacial "
            "de los ejercicios de ventilacion de wrf_urbano."
        )
    )
    add_common_cli_arguments(parser)
    parser.add_argument(
        "--target-height-m",
        type=float,
        default=DEFAULT_HEIGHT_AGL_M,
        help=(
            "Altura objetivo AGL en metros. Si queda por debajo del primer "
            "nivel de masa, se usa el primer nivel disponible."
        ),
    )
    parser.add_argument(
        "--visual-frac",
        type=float,
        default=DEFAULT_VISUAL_FRAC,
        help=(
            "Fraccion del maximo local usada como umbral visual para resaltar "
            "el penacho. Ej.: 0.03 = 3 por ciento del maximo."
        ),
    )
    parser.add_argument(
        "--visual-min-abs",
        type=float,
        default=None,
        help="Umbral absoluto opcional para visualizar el penacho.",
    )
    parser.add_argument(
        "--source-lat",
        type=float,
        default=None,
        help="Latitud manual del pulso.",
    )
    parser.add_argument(
        "--source-lon",
        type=float,
        default=None,
        help="Longitud manual del pulso.",
    )
    parser.add_argument(
        "--source-label",
        default="Pulso",
        help="Etiqueta para el punto de emision.",
    )
    return parser


def get_default_cali_shape() -> Path | None:
    candidates = [
        WORKSPACE_ROOT
        / "Z_WRF_OUT"
        / "SHAPES_CALI_YUMBO_PALMIRA_WGS84"
        / "MGN_ZONA_URBANA_CONTINUA_CALI_PALMIRA_YUMBO_WGS84.shp",
        WORKSPACE_ROOT / "SHAPES" / "MGN_ZONA_URBANA_CONTINUA_CALI_PALMIRA_YUMBO.shp",
        WORKSPACE_ROOT / "SHAPES" / "MGN_ZONA_URBANA_CONTINUA_CALI_PALMIRA_YUMBO.gpkg",
        WORKSPACE_ROOT / "SHAPES" / "shape01" / "MGN_URB_ZONA_URBANA_CALI_YUMBO_PALMIRA.shp",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def should_open_local_gui(args: argparse.Namespace) -> bool:
    if getattr(args, "no_gui_options", False):
        return False
    return bool(getattr(args, "gui_options", False) or len(__import__("sys").argv) == 1)


def maybe_apply_local_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "shapefile", None) is None and not getattr(args, "select_shapefile", False):
        default_shape = get_default_cali_shape()
        if default_shape is not None:
            args.shapefile = str(default_shape)
    return args


def maybe_apply_penacho_map_gui(args: argparse.Namespace) -> argparse.Namespace:
    if not should_open_local_gui(args):
        return args
    if tk is None or ttk is None or messagebox is None:
        return args

    root = tk.Tk()
    root.title("Mapa horario del penacho - opciones locales")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")

    ttk.Label(
        frame,
        text=(
            "Ajuste la altura AGL y el umbral visual del penacho. "
            "La ventana espacial se mantiene fija como en wrf_urbano."
        ),
        wraplength=600,
    ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

    target_height_var = tk.StringVar(value=str(getattr(args, "target_height_m", DEFAULT_HEIGHT_AGL_M)))
    visual_frac_var = tk.StringVar(value=str(getattr(args, "visual_frac", DEFAULT_VISUAL_FRAC)))
    visual_min_abs_var = tk.StringVar(
        value="" if getattr(args, "visual_min_abs", None) is None else str(args.visual_min_abs)
    )

    ttk.Label(frame, text="Altura objetivo (m AGL)").grid(row=1, column=0, sticky="w")
    ttk.Entry(frame, textvariable=target_height_var, width=12).grid(
        row=1, column=1, sticky="w", padx=(8, 10)
    )
    ttk.Label(frame, text="Umbral relativo (0-1)").grid(row=1, column=2, sticky="w")
    ttk.Entry(frame, textvariable=visual_frac_var, width=12).grid(
        row=1, column=3, sticky="w", padx=(8, 0)
    )

    ttk.Label(frame, text="Umbral abs. opcional").grid(row=2, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=visual_min_abs_var, width=12).grid(
        row=2, column=1, sticky="w", padx=(8, 10), pady=(8, 0)
    )
    ttk.Label(
        frame,
        text="Si se deja vacio, solo se usa el umbral relativo sobre el maximo.",
    ).grid(row=2, column=2, columnspan=2, sticky="w", pady=(8, 0))

    result = {"ok": False}

    def on_ok() -> None:
        try:
            args.target_height_m = float(target_height_var.get().strip())
            args.visual_frac = float(visual_frac_var.get().strip())
            text = visual_min_abs_var.get().strip()
            args.visual_min_abs = None if text == "" else float(text)
            if args.target_height_m < 0.0:
                raise ValueError("La altura objetivo debe ser positiva.")
            if args.visual_frac < 0.0:
                raise ValueError("El umbral relativo no puede ser negativo.")
            result["ok"] = True
            root.destroy()
        except ValueError as exc:
            messagebox.showerror("Opciones invalidas", str(exc), parent=root)

    def on_cancel() -> None:
        root.destroy()

    buttons = ttk.Frame(frame)
    buttons.grid(row=3, column=0, columnspan=4, sticky="e", pady=(14, 0))
    ttk.Button(buttons, text="Cancelar", command=on_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(buttons, text="Continuar", command=on_ok).grid(row=0, column=1)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()

    if not result["ok"]:
        raise SystemExit("Configuracion cancelada por el usuario.")
    return args


def nearest_grid_index(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lat: float,
    lon: float,
) -> tuple[int, int]:
    cos_lat = np.cos(np.deg2rad(np.nanmean(lat2d)))
    dist2 = (lat2d - lat) ** 2 + ((lon2d - lon) * cos_lat) ** 2
    flat = int(np.nanargmin(dist2))
    return tuple(int(v) for v in np.unravel_index(flat, lat2d.shape))


def signed_displacements_km(
    source_lat: float,
    source_lon: float,
    target_lat: float,
    target_lon: float,
) -> tuple[float, float]:
    dx = haversine_km(source_lat, source_lon, source_lat, target_lon)
    dy = haversine_km(source_lat, source_lon, target_lat, source_lon)
    if np.isfinite(dx) and target_lon < source_lon:
        dx = -dx
    if np.isfinite(dy) and target_lat < source_lat:
        dy = -dy
    return dx, dy


def select_hourly_timestamps(times: list[pd.Timestamp]) -> set[pd.Timestamp]:
    time_index = pd.to_datetime(times)
    hourly: set[pd.Timestamp] = set()
    if len(time_index) == 0:
        return hourly

    grouped: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for ts in time_index:
        hour_key = pd.Timestamp(ts).floor("h")
        grouped.setdefault(hour_key, []).append(pd.Timestamp(ts))

    for _, values in grouped.items():
        values = sorted(values)
        best = min(values, key=lambda ts: abs((ts - ts.floor("h")).total_seconds()))
        hourly.add(pd.Timestamp(best))
    return hourly


def interpolate_slice_at_agl(
    tracer_3d: np.ndarray,
    z_agl_3d: np.ndarray,
    target_agl_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    nz, ny, nx = tracer_3d.shape
    out = np.full((ny, nx), np.nan, dtype=float)
    used_height = np.full((ny, nx), np.nan, dtype=float)

    tracer = np.asarray(tracer_3d, dtype=float)
    z_agl = np.asarray(z_agl_3d, dtype=float)

    for iy in range(ny):
        for ix in range(nx):
            zcol = z_agl[:, iy, ix]
            tcol = tracer[:, iy, ix]
            valid = np.isfinite(zcol) & np.isfinite(tcol)
            if np.count_nonzero(valid) == 0:
                continue
            zv = zcol[valid]
            tv = tcol[valid]
            if zv.size == 1:
                out[iy, ix] = tv[0]
                used_height[iy, ix] = zv[0]
                continue
            order = np.argsort(zv)
            zv = zv[order]
            tv = tv[order]
            if target_agl_m <= zv[0]:
                out[iy, ix] = tv[0]
                used_height[iy, ix] = zv[0]
            elif target_agl_m >= zv[-1]:
                out[iy, ix] = np.nan
                used_height[iy, ix] = np.nan
            else:
                out[iy, ix] = np.interp(target_agl_m, zv, tv)
                used_height[iy, ix] = target_agl_m
    return out, used_height


def compute_visual_threshold(
    field_2d: np.ndarray,
    visual_frac: float,
    visual_min_abs: float | None,
) -> float:
    positive = np.asarray(field_2d, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return float("nan")
    threshold = float(np.nanmax(positive)) * float(visual_frac)
    if visual_min_abs is not None and np.isfinite(visual_min_abs):
        threshold = max(threshold, float(visual_min_abs))
    return threshold


def compute_visual_threshold_adaptive(
    field_2d: np.ndarray,
    window_mask: np.ndarray,
    visual_frac: float,
    visual_min_abs: float | None,
) -> tuple[float, float]:
    masked = np.where(window_mask, field_2d, np.nan)
    fractions = [
        float(visual_frac),
        float(visual_frac) / 3.0,
        float(visual_frac) / 10.0,
        0.0,
    ]
    tried: set[float] = set()
    for frac in fractions:
        frac = max(frac, 0.0)
        key = round(frac, 12)
        if key in tried:
            continue
        tried.add(key)
        threshold = compute_visual_threshold(
            masked,
            visual_frac=frac,
            visual_min_abs=visual_min_abs,
        )
        if not np.isfinite(threshold):
            continue
        plume_mask = (
            window_mask
            & np.isfinite(field_2d)
            & (field_2d >= threshold)
            & (field_2d > 0.0)
        )
        if np.any(plume_mask):
            return threshold, frac
    return float("nan"), float("nan")


def get_window_mask(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    return (
        (lat2d >= LAT_MIN - ZOOM_OUT)
        & (lat2d <= LAT_MAX + ZOOM_OUT)
        & (lon2d >= LON_MIN - ZOOM_OUT)
        & (lon2d <= LON_MAX + ZOOM_OUT)
    )


def plot_city_labels(ax) -> None:
    for name, (lon, lat) in CITY_LABELS.items():
        ax.text(
            lon,
            lat,
            name,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2},
        )


def plot_penacho_map(
    *,
    field_2d: np.ndarray,
    field_visible_2d: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    outpath: Path,
    tracer_display_name: str,
    tracer_units: str,
    time_value: pd.Timestamp,
    target_height_m: float,
    mean_used_height_m: float,
    source_marker: tuple[float, float, str] | None,
    centroid_marker: tuple[float, float] | None,
    max_marker: tuple[float, float] | None,
    shapefile_path: Path | None,
    subtitle: str | None,
    threshold: float,
    used_visual_frac: float,
    source_value: float,
    max_value: float,
    centroid_distance_km: float,
    centroid_dx_km: float,
    centroid_dy_km: float,
    script_path: str | Path | None,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor("white")

    visible = np.ma.masked_invalid(field_visible_2d)
    if np.ma.count(visible) > 0:
        vmax = float(np.nanmax(field_visible_2d))
        vmin = threshold if np.isfinite(threshold) else 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 1.05 if vmin > 0.0 else 1.0
        pcm = ax.pcolormesh(
            lon2d,
            lat2d,
            visible,
            shading="auto",
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label(f"{tracer_display_name} ({tracer_units})")
    else:
        ax.text(
            0.5,
            0.52,
            "Sin penacho visible con el umbral actual",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="0.3",
        )

    plot_shapefile_boundary(ax, shapefile_path, linewidth=0.9)
    plot_city_labels(ax)

    legend_items = []
    if plot_source_marker(ax, source_marker):
        legend_items.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="yellow", markeredgecolor="black", markersize=12, linestyle="None", label=source_marker[2])
        )

    if centroid_marker is not None:
        ax.scatter(
            centroid_marker[1],
            centroid_marker[0],
            s=72,
            c="cyan",
            edgecolors="black",
            linewidths=0.8,
            marker="o",
            zorder=9,
        )
        ax.text(
            centroid_marker[1],
            centroid_marker[0],
            " centroide",
            fontsize=8.5,
            color="black",
            zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.1},
        )
        legend_items.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="cyan", markeredgecolor="black", markersize=8, linestyle="None", label="Centroide")
        )

    if max_marker is not None:
        ax.scatter(
            max_marker[1],
            max_marker[0],
            s=52,
            c="red",
            edgecolors="white",
            linewidths=0.8,
            marker="X",
            zorder=9,
        )
        ax.text(
            max_marker[1],
            max_marker[0],
            " max",
            fontsize=8.5,
            color="black",
            zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.1},
        )
        legend_items.append(
            Line2D([0], [0], marker="X", color="w", markerfacecolor="red", markeredgecolor="white", markersize=8, linestyle="None", label="Maximo")
        )

    if source_marker is not None and centroid_marker is not None:
        ax.annotate(
            "",
            xy=(centroid_marker[1], centroid_marker[0]),
            xytext=(source_marker[1], source_marker[0]),
            arrowprops={"arrowstyle": "->", "color": "deepskyblue", "lw": 1.8},
            zorder=8,
        )
        legend_items.append(
            Line2D([0, 1], [0, 0], color="deepskyblue", lw=1.8, label="Fuente a centroide")
        )

    if legend_items:
        ax.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_xlim(LON_MIN - ZOOM_OUT, LON_MAX + ZOOM_OUT)
    ax.set_ylim(LAT_MIN - ZOOM_OUT, LAT_MAX + ZOOM_OUT)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    diag_bits = [
        f"altura objetivo={target_height_m:.1f} m AGL",
        f"altura usada media={mean_used_height_m:.1f} m",
    ]
    if np.isfinite(threshold):
        diag_bits.append(f"pluma visible >= {threshold:.3e}")
    if np.isfinite(used_visual_frac):
        diag_bits.append(f"fraccion visual usada={used_visual_frac:.3g}")
    title = (
        f"Penacho horario del trazador a altura de ventilacion - "
        f"{time_value:%Y-%m-%d %H:%M:%S}"
    )
    subtitle_parts = [subtitle] if subtitle else []
    subtitle_parts.append(" | ".join(diag_bits))
    ax.set_title(title_with_subtitle(title, " | ".join(part for part in subtitle_parts if part)))

    text_lines = []
    if np.isfinite(source_value):
        text_lines.append(f"Valor en fuente: {source_value:.3e}")
    if np.isfinite(max_value):
        text_lines.append(f"Maximo visible: {max_value:.3e}")
    if np.isfinite(centroid_distance_km):
        text_lines.append(f"Dist. fuente-centroide: {centroid_distance_km:.2f} km")
    if np.isfinite(centroid_dx_km) and np.isfinite(centroid_dy_km):
        text_lines.append(f"dX={centroid_dx_km:.2f} km | dY={centroid_dy_km:.2f} km")
    if text_lines:
        ax.text(
            0.985,
            0.02,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.88, "pad": 3.5},
        )

    add_figure_footer(fig, script_path=script_path, output_path=outpath)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
    plt.savefig(outpath, dpi=170)
    plt.close(fig)
    return outpath


def main() -> None:
    args = build_parser().parse_args()
    args = maybe_apply_local_defaults(args)
    args = maybe_apply_common_gui_options(
        args,
        title="Opciones - mapas horarios del penacho",
        include_region_box=False,
    )
    args = maybe_apply_penacho_map_gui(args)

    wrf_files, outdir = resolve_common_inputs(
        args,
        default_subdir="wrf_trazadores_06_mapa_penacho_altura_ventilacion",
    )
    shapefile_path = resolve_shapefile_path(args)

    tracer_name_arg, delta_from_initial = split_tracer_name_mode(
        args.tracer_name,
        args.delta_from_initial,
    )
    tracer_name, tracer_units, all_times = validate_wrf_files(
        wrf_files,
        tracer_name=tracer_name_arg,
        required_vars=["PH", "PHB"],
    )
    tracer_display_name = make_tracer_output_name(tracer_name, delta_from_initial)
    baseline_3d = (
        load_tracer_baseline(wrf_files, tracer_name) if delta_from_initial else None
    )

    from netCDF4 import Dataset

    with Dataset(wrf_files[0]) as nc0:
        lat2d, lon2d = get_latlon_2d(nc0)

    domain_tag = infer_domain_tag(wrf_files[0])
    source_marker = resolve_source_marker(args, wrf_files, domain_tag=domain_tag)
    subtitle = format_source_marker(source_marker)

    source_iy = source_ix = None
    if source_marker is not None:
        source_iy, source_ix = nearest_grid_index(
            lat2d,
            lon2d,
            source_marker[0],
            source_marker[1],
        )

    hourly_times = select_hourly_timestamps(all_times)
    selected_times = sorted(hourly_times)
    if not selected_times:
        raise SystemExit("No se pudieron seleccionar tiempos horarios.")

    case_tag = safe_tag(args.tag or f"{domain_tag}_{tracer_display_name}_altura_vent")
    time_tag = make_time_range_tag(selected_times)
    rows: list[dict] = []
    png_outputs: list[Path] = []
    created = 0
    window_mask = get_window_mask(lat2d, lon2d)

    for _, nc, time_index, time_value in iter_wrf_steps(wrf_files):
        if pd.Timestamp(time_value) not in hourly_times:
            continue

        tracer = read_tracer_step(
            nc,
            tracer_name,
            time_index=time_index,
            baseline_3d=baseline_3d,
        )
        z_agl_3d, _ = compute_height_agl_and_dz(nc, time_index=time_index)
        slice_2d, used_height_2d = interpolate_slice_at_agl(
            tracer,
            z_agl_3d,
            float(args.target_height_m),
        )
        threshold, used_visual_frac = compute_visual_threshold_adaptive(
            slice_2d,
            window_mask,
            visual_frac=float(args.visual_frac),
            visual_min_abs=args.visual_min_abs,
        )

        if np.isfinite(threshold):
            analysis_field = np.where(np.isfinite(slice_2d) & (slice_2d >= threshold), slice_2d, 0.0)
            visible_field = np.where(np.isfinite(slice_2d) & (slice_2d >= threshold), slice_2d, np.nan)
        else:
            analysis_field = np.zeros_like(slice_2d, dtype=float)
            visible_field = np.full_like(slice_2d, np.nan, dtype=float)

        plume_mask = window_mask & np.isfinite(analysis_field) & (analysis_field > 0.0)
        centroid_lat, centroid_lon = weighted_centroid(
            analysis_field,
            lat2d,
            lon2d,
            plume_mask,
        )

        if np.any(plume_mask):
            idx_flat = int(np.nanargmax(np.where(plume_mask, analysis_field, np.nan)))
            max_iy, max_ix = np.unravel_index(idx_flat, analysis_field.shape)
            max_lat = float(lat2d[max_iy, max_ix])
            max_lon = float(lon2d[max_iy, max_ix])
            max_value = float(analysis_field[max_iy, max_ix])
        else:
            max_lat = max_lon = max_value = float("nan")

        centroid_distance_km = centroid_dx_km = centroid_dy_km = float("nan")
        if source_marker is not None and np.isfinite(centroid_lat) and np.isfinite(centroid_lon):
            centroid_distance_km = haversine_km(
                source_marker[0],
                source_marker[1],
                centroid_lat,
                centroid_lon,
            )
            centroid_dx_km, centroid_dy_km = signed_displacements_km(
                source_marker[0],
                source_marker[1],
                centroid_lat,
                centroid_lon,
            )

        source_value = float("nan")
        source_used_height_m = float("nan")
        if source_iy is not None and source_ix is not None:
            source_value = float(slice_2d[source_iy, source_ix])
            source_used_height_m = float(used_height_2d[source_iy, source_ix])

        mean_used_height_m = float(np.nanmean(used_height_2d[window_mask]))
        outpath = outdir / (
            f"{case_tag}_penacho_{pd.Timestamp(time_value):%Y-%m-%d_%H%M%S}.png"
        )
        plot_penacho_map(
            field_2d=slice_2d,
            field_visible_2d=visible_field,
            lat2d=lat2d,
            lon2d=lon2d,
            outpath=outpath,
            tracer_display_name=tracer_display_name,
            tracer_units=tracer_units,
            time_value=pd.Timestamp(time_value),
            target_height_m=float(args.target_height_m),
            mean_used_height_m=mean_used_height_m,
            source_marker=source_marker,
            centroid_marker=None if not np.isfinite(centroid_lat) else (centroid_lat, centroid_lon),
            max_marker=None if not np.isfinite(max_lat) else (max_lat, max_lon),
            shapefile_path=shapefile_path,
            subtitle=subtitle,
            threshold=threshold,
            used_visual_frac=used_visual_frac,
            source_value=source_value,
            max_value=max_value,
            centroid_distance_km=centroid_distance_km,
            centroid_dx_km=centroid_dx_km,
            centroid_dy_km=centroid_dy_km,
            script_path=Path(__file__).resolve(),
        )
        png_outputs.append(outpath)

        rows.append(
            {
                "time": pd.Timestamp(time_value),
                "png_path": str(outpath),
                "target_height_agl_m": float(args.target_height_m),
                "mean_used_height_m": mean_used_height_m,
                "source_used_height_m": source_used_height_m,
                "visual_threshold": threshold,
                "visual_frac_requested": float(args.visual_frac),
                "visual_frac_used": used_visual_frac,
                "slice_max_visible": max_value,
                "slice_max_domain": region_nanmax(slice_2d, window_mask),
                "source_value": source_value,
                "centroide_lat": centroid_lat,
                "centroide_lon": centroid_lon,
                "max_lat": max_lat,
                "max_lon": max_lon,
                "distancia_fuente_centroide_km": centroid_distance_km,
                "dx_fuente_centroide_km": centroid_dx_km,
                "dy_fuente_centroide_km": centroid_dy_km,
                "tracer_name": tracer_display_name,
                "tracer_units": tracer_units,
            }
        )
        created += 1

    if created == 0:
        raise SystemExit("No se generaron mapas horarios.")

    csv_path = outdir / f"{case_tag}_resumen_horario_{time_tag}.csv"
    save_rows_csv(rows, csv_path)
    gif_path = None
    if len(png_outputs) > 1:
        gif_path = build_gif_from_pngs(
            png_outputs,
            outdir / f"{case_tag}_penacho_animacion.gif",
            duration_ms=500,
        )

    print(f"Trazador usado: {tracer_display_name} (variable base: {tracer_name})")
    print(f"Altura objetivo: {float(args.target_height_m):.1f} m AGL")
    if shapefile_path is not None:
        print(f"Shape usado: {shapefile_path}")
    print(f"Mapas generados: {created}")
    if gif_path is not None:
        print(f"GIF animado: {gif_path}")
    print(f"Resumen CSV: {csv_path}")
    print(f"Carpeta de salida: {outdir}")


if __name__ == "__main__":
    main()
