"""
Corte vertical del trazador inerte a lo largo de un transecto lat/lon.

La figura sigue la estructura de un corte meteorologico clasico:
- eje Y de altura;
- eje X rotulado con latitud/longitud a lo largo del transecto;
- campo relleno del trazador, por defecto en autodeteccion;
- contornos opcionales de temperatura o theta;
- terreno en la base;
- mapa inserto con shapefile, punto del pulso y linea del corte;
- subtitulo con coordenadas del pulso y pie con rutas de trazabilidad.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from wrf_trazadores_00_core import (
    add_common_cli_arguments,
    add_figure_footer,
    build_gif_from_pngs,
    compute_height_agl_and_dz,
    first_tracer_candidate_with_signal,
    format_source_marker,
    get_latlon_2d,
    haversine_km,
    infer_domain_tag,
    load_tracer_baseline,
    make_tracer_output_name,
    maybe_apply_common_gui_options,
    parse_wrf_times,
    plot_shapefile_boundary,
    plot_source_marker,
    read_tracer_step,
    read_var,
    resolve_common_inputs,
    resolve_shapefile_path,
    resolve_source_marker,
    safe_tag,
    split_tracer_name_mode,
    title_with_subtitle,
    validate_wrf_files,
)


DEFAULT_NPOINTS = 180
DEFAULT_MAX_HEIGHT_M = 18000.0
DEFAULT_LOCAL_TRANSECT_SPAN_FRACTION = 0.18
R_D_OVER_CP = 287.0 / 1004.0


def summarize_tracer_signal(
    wrf_files: list[Path],
    tracer_name: str,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "has_signal": False,
        "max_value": 0.0,
        "positive_cells": 0,
        "suggested_tracer": None,
    }
    if not wrf_files:
        return summary

    with Dataset(wrf_files[0]) as nc:
        var = nc.variables[tracer_name]
        has_time = "Time" in var.dimensions
        time_count = len(nc.dimensions["Time"]) if has_time and "Time" in nc.dimensions else 1
        sample_indices = [0]
        if time_count > 1:
            sample_indices.extend([time_count // 2, time_count - 1])
        sample_indices = sorted(set(sample_indices))

        max_value = 0.0
        positive_cells = 0
        has_signal = False
        for time_index in sample_indices:
            field = np.asarray(
                read_var(nc, tracer_name, time_index=time_index if has_time else None),
                dtype=float,
            )
            positive = np.isfinite(field) & (field > 0.0)
            if np.any(positive):
                has_signal = True
                positive_cells += int(np.sum(positive))
            if np.isfinite(field).any():
                max_value = max(max_value, float(np.nanmax(field)))

        summary["has_signal"] = has_signal
        summary["positive_cells"] = positive_cells
        summary["max_value"] = max_value
        if not summary["has_signal"]:
            summary["suggested_tracer"] = first_tracer_candidate_with_signal(
                nc,
                exclude=[tracer_name],
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera cortes verticales lat/lon del trazador inerte."
    )
    add_common_cli_arguments(parser)
    parser.add_argument("--source-lat", type=float, default=None, help="Latitud del pulso.")
    parser.add_argument("--source-lon", type=float, default=None, help="Longitud del pulso.")
    parser.add_argument(
        "--source-label",
        default="Pulso",
        help="Etiqueta para el punto de emision del pulso.",
    )
    parser.add_argument("--start-lat", type=float, default=None, help="Latitud inicial del corte.")
    parser.add_argument("--start-lon", type=float, default=None, help="Longitud inicial del corte.")
    parser.add_argument("--end-lat", type=float, default=None, help="Latitud final del corte.")
    parser.add_argument("--end-lon", type=float, default=None, help="Longitud final del corte.")
    parser.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="Indice temporal global a graficar. Si se omite, usa 0.",
    )
    parser.add_argument(
        "--time",
        default=None,
        help="Tiempo a graficar; se usa el wrfout mas cercano. Ej.: 2018-07-10_00:10:00.",
    )
    parser.add_argument(
        "--all-times",
        action="store_true",
        help="Genera una figura por cada tiempo seleccionado por --time-stride.",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Salto temporal cuando se usa --all-times.",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=DEFAULT_MAX_HEIGHT_M,
        help="Altura maxima del eje Y en metros.",
    )
    parser.add_argument(
        "--height-mode",
        choices=["msl", "agl"],
        default="msl",
        help="msl usa altura sobre nivel medio del mar y dibuja terreno; agl usa altura sobre suelo.",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=DEFAULT_NPOINTS,
        help="Numero de puntos horizontales del transecto.",
    )
    parser.add_argument(
        "--contour-var",
        choices=["temperature", "theta", "none"],
        default="none",
        help="Variable de contorno: temperature en C, theta en K o none.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Mapa de color para el trazador.",
    )
    parser.add_argument(
        "--visual-filter",
        choices=["plume", "positive", "none"],
        default="plume",
        help=(
            "Filtro visual para el corte: plume oculta fondo con umbral, "
            "positive muestra toda anomalia positiva, none no filtra."
        ),
    )
    parser.add_argument(
        "--plume-percentile",
        type=float,
        default=90.0,
        help="Percentil de valores positivos usado como umbral visual de pluma.",
    )
    parser.add_argument(
        "--plume-min-fraction",
        type=float,
        default=0.03,
        help="Fraccion del maximo positivo usada como umbral visual adicional.",
    )
    parser.add_argument(
        "--plume-min-value",
        type=float,
        default=None,
        help="Umbral absoluto opcional para ocultar fondo del trazador.",
    )
    return parser


def has_complete_transect(args: argparse.Namespace) -> bool:
    return all(
        getattr(args, name, None) is not None
        for name in ("start_lat", "start_lon", "end_lat", "end_lon")
    )


def default_transect(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    source_marker: tuple[float, float, str] | None,
    orientation: str = "nw_se",
) -> tuple[float, float, float, float]:
    lat_min, lat_max, lon_min, lon_max = transect_bounds(lat2d, lon2d)

    if source_marker is not None:
        center_lat = float(np.clip(source_marker[0], lat_min, lat_max))
        center_lon = float(np.clip(source_marker[1], lon_min, lon_max))
    else:
        center_lat = 0.5 * (lat_min + lat_max)
        center_lon = 0.5 * (lon_min + lon_max)

    dlat = lat_max - lat_min
    dlon = lon_max - lon_min
    span = DEFAULT_LOCAL_TRANSECT_SPAN_FRACTION
    local_lat_half = max(0.05, 0.5 * dlat * span)
    local_lon_half = max(0.05, 0.5 * dlon * span)

    local_lat_min = max(lat_min, center_lat - local_lat_half)
    local_lat_max = min(lat_max, center_lat + local_lat_half)
    local_lon_min = max(lon_min, center_lon - local_lon_half)
    local_lon_max = min(lon_max, center_lon + local_lon_half)

    return oriented_transect_from_center(
        center_lat,
        center_lon,
        local_lat_min,
        local_lat_max,
        local_lon_min,
        local_lon_max,
        orientation=orientation,
    )


def transect_bounds(lat2d: np.ndarray, lon2d: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.nanpercentile(lat2d, 5.0)),
        float(np.nanpercentile(lat2d, 95.0)),
        float(np.nanpercentile(lon2d, 5.0)),
        float(np.nanpercentile(lon2d, 95.0)),
    )


def oriented_transect_from_center(
    center_lat: float,
    center_lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    orientation: str,
) -> tuple[float, float, float, float]:
    if orientation == "horizontal":
        return center_lat, lon_min, center_lat, lon_max
    if orientation == "vertical":
        return lat_min, center_lon, lat_max, center_lon

    dlat = lat_max - lat_min
    dlon = lon_max - lon_min
    if not np.isfinite(dlat) or dlat <= 0.0 or not np.isfinite(dlon) or dlon <= 0.0:
        return lat_min, lon_min, lat_max, lon_max

    slope = -dlat / dlon if orientation == "nw_se" else dlat / dlon
    intersections: list[tuple[float, float]] = []

    for lon in (lon_min, lon_max):
        lat = center_lat + slope * (lon - center_lon)
        if lat_min <= lat <= lat_max:
            intersections.append((lat, lon))

    for lat in (lat_min, lat_max):
        if slope == 0.0:
            continue
        lon = center_lon + (lat - center_lat) / slope
        if lon_min <= lon <= lon_max:
            intersections.append((lat, lon))

    unique: list[tuple[float, float]] = []
    for lat, lon in intersections:
        if not any(abs(lat - old_lat) < 1.0e-8 and abs(lon - old_lon) < 1.0e-8 for old_lat, old_lon in unique):
            unique.append((lat, lon))

    if len(unique) < 2:
        if orientation == "nw_se":
            unique = [(lat_max, lon_min), (lat_min, lon_max)]
        else:
            unique = [(lat_min, lon_min), (lat_max, lon_max)]

    p1, p2 = unique[0], unique[-1]
    if orientation == "nw_se" and p1[0] < p2[0]:
        p1, p2 = p2, p1
    if orientation == "sw_ne" and p1[0] > p2[0]:
        p1, p2 = p2, p1
    return p1[0], p1[1], p2[0], p2[1]


def maybe_apply_vertical_gui_options(
    args: argparse.Namespace,
    times: pd.DatetimeIndex,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    source_marker: tuple[float, float, str] | None,
) -> argparse.Namespace:
    should_open = (
        not getattr(args, "no_gui_options", False)
        and (len(sys.argv) == 1 or getattr(args, "gui_options", False) or not has_complete_transect(args))
    )
    if not should_open:
        return args

    return apply_vertical_gui_options(args, times, lat2d, lon2d, source_marker)


def apply_vertical_gui_options(
    args: argparse.Namespace,
    times: pd.DatetimeIndex,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    source_marker: tuple[float, float, str] | None,
) -> argparse.Namespace:
    options = select_vertical_options_gui(args, times, lat2d, lon2d, source_marker)
    if options is None:
        raise SystemExit("Configuracion del corte vertical cancelada por el usuario.")

    args.start_lat = options["start_lat"]
    args.start_lon = options["start_lon"]
    args.end_lat = options["end_lat"]
    args.end_lon = options["end_lon"]
    args.time_index = options["time_index"]
    args.all_times = options["all_times"]
    args.time_stride = options["time_stride"]
    args.max_height = options["max_height"]
    args.height_mode = options["height_mode"]
    args.npoints = options["npoints"]
    args.contour_var = options["contour_var"]
    args.cmap = options["cmap"]
    args.visual_filter = options["visual_filter"]
    args.plume_percentile = options["plume_percentile"]
    args.plume_min_fraction = options["plume_min_fraction"]
    args.plume_min_value = options["plume_min_value"]
    return args


def select_vertical_options_gui(
    args: argparse.Namespace,
    times: pd.DatetimeIndex,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    source_marker: tuple[float, float, str] | None,
) -> dict | None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        raise SystemExit("tkinter no esta disponible para configurar el corte vertical.") from exc

    start_lat, start_lon, end_lat, end_lon = (
        (args.start_lat, args.start_lon, args.end_lat, args.end_lon)
        if has_complete_transect(args)
        else default_transect(lat2d, lon2d, source_marker, orientation="nw_se")
    )
    lat_min, lat_max, lon_min, lon_max = transect_bounds(lat2d, lon2d)
    if source_marker is not None:
        center_lat = float(np.clip(source_marker[0], lat_min, lat_max))
        center_lon = float(np.clip(source_marker[1], lon_min, lon_max))
    else:
        center_lat = 0.5 * (lat_min + lat_max)
        center_lon = 0.5 * (lon_min + lon_max)

    root = tk.Tk()
    root.title("Opciones - corte vertical del trazador")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")
    row = 0

    ttk.Label(
        frame,
        text=(
            "Defina la linea del corte y los tiempos a graficar. "
            "El eje X se rotulara con lat/lon y el mapa inserto mostrara la linea."
        ),
        wraplength=720,
    ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 10))
    row += 1

    time_labels = [f"{idx}: {pd.Timestamp(value):%Y-%m-%d %H:%M:%S}" for idx, value in enumerate(times)]
    initial_index = args.time_index if args.time_index is not None else 0
    initial_index = int(np.clip(initial_index, 0, max(len(time_labels) - 1, 0)))
    mode_var = tk.StringVar(value="todos" if args.all_times else "seleccionado")
    time_var = tk.StringVar(value=time_labels[initial_index] if time_labels else "0")
    stride_var = tk.StringVar(value=str(max(1, int(args.time_stride or 1))))

    ttk.Label(frame, text="Tiempos").grid(row=row, column=0, sticky="w")
    ttk.Combobox(
        frame,
        textvariable=mode_var,
        values=["seleccionado", "todos"],
        width=16,
        state="readonly",
    ).grid(row=row, column=1, sticky="w", padx=(8, 4))
    ttk.Combobox(
        frame,
        textvariable=time_var,
        values=time_labels,
        width=28,
        state="readonly",
    ).grid(row=row, column=2, sticky="w", padx=(8, 4))
    ttk.Label(frame, text="salto").grid(row=row, column=3, sticky="w")
    ttk.Entry(frame, textvariable=stride_var, width=6).grid(row=row, column=4, sticky="w", padx=(4, 0))
    row += 1

    start_lat_var = tk.StringVar(value=f"{float(start_lat):.6f}")
    start_lon_var = tk.StringVar(value=f"{float(start_lon):.6f}")
    end_lat_var = tk.StringVar(value=f"{float(end_lat):.6f}")
    end_lon_var = tk.StringVar(value=f"{float(end_lon):.6f}")

    ttk.Label(frame, text="Inicio corte").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Label(frame, text="lat").grid(row=row, column=1, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=start_lat_var, width=12).grid(row=row, column=1, sticky="e", padx=(24, 8), pady=(8, 0))
    ttk.Label(frame, text="lon").grid(row=row, column=2, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=start_lon_var, width=12).grid(row=row, column=2, sticky="e", padx=(24, 8), pady=(8, 0))
    row += 1

    ttk.Label(frame, text="Fin corte").grid(row=row, column=0, sticky="w")
    ttk.Label(frame, text="lat").grid(row=row, column=1, sticky="w")
    ttk.Entry(frame, textvariable=end_lat_var, width=12).grid(row=row, column=1, sticky="e", padx=(24, 8))
    ttk.Label(frame, text="lon").grid(row=row, column=2, sticky="w")
    ttk.Entry(frame, textvariable=end_lon_var, width=12).grid(row=row, column=2, sticky="e", padx=(24, 8))
    row += 1

    def set_transect_orientation(orientation: str) -> None:
        slat, slon, elat, elon = oriented_transect_from_center(
            center_lat,
            center_lon,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            orientation,
        )
        start_lat_var.set(f"{slat:.6f}")
        start_lon_var.set(f"{slon:.6f}")
        end_lat_var.set(f"{elat:.6f}")
        end_lon_var.set(f"{elon:.6f}")

    ttk.Label(frame, text="Orientacion rapida").grid(row=row, column=0, sticky="w", pady=(6, 0))
    orient_frame = ttk.Frame(frame)
    orient_frame.grid(row=row, column=1, columnspan=4, sticky="w", padx=(8, 0), pady=(6, 0))
    ttk.Button(orient_frame, text="Diagonal NW-SE", command=lambda: set_transect_orientation("nw_se")).grid(row=0, column=0, padx=(0, 6))
    ttk.Button(orient_frame, text="Diagonal SW-NE", command=lambda: set_transect_orientation("sw_ne")).grid(row=0, column=1, padx=(0, 6))
    ttk.Button(orient_frame, text="Horizontal", command=lambda: set_transect_orientation("horizontal")).grid(row=0, column=2, padx=(0, 6))
    ttk.Button(orient_frame, text="Vertical", command=lambda: set_transect_orientation("vertical")).grid(row=0, column=3, padx=(0, 6))
    row += 1

    max_height_var = tk.StringVar(value=str(float(args.max_height or DEFAULT_MAX_HEIGHT_M)))
    npoints_var = tk.StringVar(value=str(int(args.npoints or DEFAULT_NPOINTS)))
    height_mode_var = tk.StringVar(value=args.height_mode or "msl")
    contour_var = tk.StringVar(value=args.contour_var or "none")
    cmap_var = tk.StringVar(value=args.cmap or "magma")
    visual_filter_var = tk.StringVar(value=args.visual_filter or "plume")
    plume_percentile_var = tk.StringVar(value=str(float(args.plume_percentile)))
    plume_fraction_var = tk.StringVar(value=str(float(args.plume_min_fraction)))
    plume_min_value_var = tk.StringVar(
        value="" if args.plume_min_value is None else str(float(args.plume_min_value))
    )

    ttk.Label(frame, text="Altura max (m)").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=max_height_var, width=12).grid(row=row, column=1, sticky="w", padx=(8, 4), pady=(8, 0))
    ttk.Label(frame, text="modo altura").grid(row=row, column=2, sticky="w", pady=(8, 0))
    ttk.Combobox(
        frame,
        textvariable=height_mode_var,
        values=["msl", "agl"],
        width=10,
        state="readonly",
    ).grid(row=row, column=2, sticky="e", padx=(70, 8), pady=(8, 0))
    row += 1

    ttk.Label(frame, text="Puntos transecto").grid(row=row, column=0, sticky="w")
    ttk.Entry(frame, textvariable=npoints_var, width=12).grid(row=row, column=1, sticky="w", padx=(8, 4))
    ttk.Label(frame, text="contornos").grid(row=row, column=2, sticky="w")
    ttk.Combobox(
        frame,
        textvariable=contour_var,
        values=["temperature", "theta", "none"],
        width=14,
        state="readonly",
    ).grid(row=row, column=2, sticky="e", padx=(70, 8))
    row += 1

    ttk.Label(frame, text="Colormap").grid(row=row, column=0, sticky="w")
    ttk.Combobox(
        frame,
        textvariable=cmap_var,
        values=["magma", "inferno", "plasma", "viridis", "turbo", "Reds", "YlOrRd"],
        width=14,
    ).grid(row=row, column=1, sticky="w", padx=(8, 4))
    row += 1

    ttk.Label(frame, text="Filtro visual").grid(row=row, column=0, sticky="w")
    ttk.Combobox(
        frame,
        textvariable=visual_filter_var,
        values=["plume", "positive", "none"],
        width=14,
        state="readonly",
    ).grid(row=row, column=1, sticky="w", padx=(8, 4))
    ttk.Label(frame, text="plume oculta fondo y resalta la pluma").grid(
        row=row,
        column=2,
        columnspan=3,
        sticky="w",
    )
    row += 1

    ttk.Label(frame, text="Umbral pluma").grid(row=row, column=0, sticky="w")
    plume_frame = ttk.Frame(frame)
    plume_frame.grid(row=row, column=1, columnspan=4, sticky="w", padx=(8, 0))
    for col, (label, var, width) in enumerate(
        [
            ("percentil", plume_percentile_var, 7),
            ("fraccion max", plume_fraction_var, 7),
            ("valor abs opc.", plume_min_value_var, 10),
        ]
    ):
        ttk.Label(plume_frame, text=label).grid(row=0, column=2 * col, sticky="w")
        ttk.Entry(plume_frame, textvariable=var, width=width).grid(row=0, column=2 * col + 1, sticky="w", padx=(3, 10))
    row += 1

    result: dict | None = None

    def parse_float(value: str, label: str) -> float:
        try:
            return float(value.strip())
        except ValueError:
            raise ValueError(f"Valor numerico invalido en {label}.")

    def parse_int(value: str, label: str, min_value: int = 1) -> int:
        try:
            parsed = int(value.strip())
        except ValueError:
            raise ValueError(f"Valor entero invalido en {label}.")
        if parsed < min_value:
            raise ValueError(f"{label} debe ser >= {min_value}.")
        return parsed

    def parse_optional_float(value: str, label: str) -> float | None:
        cleaned = value.strip()
        if not cleaned:
            return None
        return parse_float(cleaned, label)

    def on_ok() -> None:
        nonlocal result
        try:
            selected_time_index = 0
            if time_var.get():
                selected_time_index = int(time_var.get().split(":", 1)[0])
            result = {
                "start_lat": parse_float(start_lat_var.get(), "latitud inicial"),
                "start_lon": parse_float(start_lon_var.get(), "longitud inicial"),
                "end_lat": parse_float(end_lat_var.get(), "latitud final"),
                "end_lon": parse_float(end_lon_var.get(), "longitud final"),
                "time_index": selected_time_index,
                "all_times": mode_var.get() == "todos",
                "time_stride": parse_int(stride_var.get(), "salto temporal"),
                "max_height": parse_float(max_height_var.get(), "altura maxima"),
                "height_mode": height_mode_var.get(),
                "npoints": parse_int(npoints_var.get(), "puntos del transecto", min_value=2),
                "contour_var": contour_var.get(),
                "cmap": cmap_var.get().strip() or "magma",
                "visual_filter": visual_filter_var.get(),
                "plume_percentile": parse_float(plume_percentile_var.get(), "percentil de pluma"),
                "plume_min_fraction": parse_float(plume_fraction_var.get(), "fraccion del maximo"),
                "plume_min_value": parse_optional_float(plume_min_value_var.get(), "umbral absoluto"),
            }
            root.destroy()
        except ValueError as exc:
            messagebox.showerror("Opciones invalidas", str(exc), parent=root)

    def on_cancel() -> None:
        nonlocal result
        result = None
        root.destroy()

    buttons = ttk.Frame(frame)
    buttons.grid(row=row, column=0, columnspan=5, sticky="e", pady=(14, 0))
    ttk.Button(buttons, text="Cancelar", command=on_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(buttons, text="Continuar", command=on_ok).grid(row=0, column=1)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return result


def list_wrf_step_refs(wrf_files: list[Path]) -> list[tuple[Path, int, pd.Timestamp]]:
    refs: list[tuple[Path, int, pd.Timestamp]] = []
    for path in wrf_files:
        with Dataset(path) as nc:
            times = parse_wrf_times(nc)
        for time_index, time_value in enumerate(times):
            refs.append((path, time_index, pd.Timestamp(time_value)))
    return refs


def selected_step_indices(args: argparse.Namespace, step_refs: list[tuple[Path, int, pd.Timestamp]]) -> list[int]:
    if not step_refs:
        raise SystemExit("No hay pasos temporales disponibles.")

    if args.all_times:
        stride = max(1, int(args.time_stride or 1))
        return list(range(0, len(step_refs), stride))

    if args.time:
        requested = pd.to_datetime(args.time.replace("_", " "), errors="coerce")
        if pd.isna(requested):
            raise SystemExit(f"No se pudo interpretar --time: {args.time}")
        diffs = [abs((ref[2] - requested).total_seconds()) for ref in step_refs]
        return [int(np.argmin(diffs))]

    index = int(args.time_index or 0)
    if index < 0 or index >= len(step_refs):
        raise SystemExit(f"--time-index fuera de rango: {index}. Rango valido: 0-{len(step_refs) - 1}")
    return [index]


def build_transect_points(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(float(args.start_lat), float(args.end_lat), int(args.npoints))
    lon = np.linspace(float(args.start_lon), float(args.end_lon), int(args.npoints))
    distance = np.zeros_like(lat, dtype=float)
    for idx in range(1, len(lat)):
        distance[idx] = distance[idx - 1] + haversine_km(
            lat[idx - 1],
            lon[idx - 1],
            lat[idx],
            lon[idx],
        )
    return lat, lon, distance


def nearest_grid_indices(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    transect_lat: np.ndarray,
    transect_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    iy = np.empty(transect_lat.size, dtype=int)
    ix = np.empty(transect_lat.size, dtype=int)
    cos_lat = np.cos(np.deg2rad(np.nanmean(lat2d)))
    for idx, (lat, lon) in enumerate(zip(transect_lat, transect_lon, strict=True)):
        dist2 = (lat2d - lat) ** 2 + ((lon2d - lon) * cos_lat) ** 2
        flat = int(np.nanargmin(dist2))
        iy[idx], ix[idx] = np.unravel_index(flat, lat2d.shape)
    return iy, ix


def sample_3d(field_3d: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    return np.asarray(field_3d[:, iy, ix], dtype=float)


def sample_2d(field_2d: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    return np.asarray(field_2d[iy, ix], dtype=float)


def compute_temperature_c(nc: Dataset, time_index: int) -> np.ndarray | None:
    required = {"T", "P", "PB"}
    if not required.issubset(nc.variables):
        return None

    theta = read_var(nc, "T", time_index=time_index) + 300.0
    pressure = read_var(nc, "P", time_index=time_index) + read_var(nc, "PB", time_index=time_index)
    with np.errstate(invalid="ignore"):
        temperature_k = theta * np.power(pressure / 100000.0, R_D_OVER_CP)
    return temperature_k - 273.15


def compute_theta_k(nc: Dataset, time_index: int) -> np.ndarray | None:
    if "T" not in nc.variables:
        return None
    return read_var(nc, "T", time_index=time_index) + 300.0


def build_contour_field(
    nc: Dataset,
    contour_var: str,
    time_index: int,
) -> tuple[np.ndarray | None, str]:
    if contour_var == "none":
        return None, ""
    if contour_var == "theta":
        return compute_theta_k(nc, time_index), "theta (K)"
    temperature = compute_temperature_c(nc, time_index)
    return temperature, "Temperatura (C)"


def format_latlon_label(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0.0 else "S"
    ew = "E" if lon >= 0.0 else "W"
    return f"{abs(lat):.2f}{ns}\n{abs(lon):.2f}{ew}"


def nearest_distance_for_marker(
    marker: tuple[float, float, str] | None,
    transect_lat: np.ndarray,
    transect_lon: np.ndarray,
    distance_km: np.ndarray,
) -> float | None:
    if marker is None:
        return None
    lat, lon, _ = marker
    dist2 = (transect_lat - lat) ** 2 + (transect_lon - lon) ** 2
    return float(distance_km[int(np.nanargmin(dist2))])


def nearest_index_for_marker(
    marker: tuple[float, float, str] | None,
    transect_lat: np.ndarray,
    transect_lon: np.ndarray,
) -> int | None:
    if marker is None:
        return None
    lat, lon, _ = marker
    dist2 = (transect_lat - lat) ** 2 + (transect_lon - lon) ** 2
    return int(np.nanargmin(dist2))


def section_centerlines(
    field: np.ndarray,
    z_plot: np.ndarray,
    max_height: float,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    nz, nx = field.shape
    z_peak = np.full(nx, np.nan, dtype=float)
    z_centroid = np.full(nx, np.nan, dtype=float)

    for idx in range(nx):
        column = np.asarray(field[:, idx], dtype=float)
        zcol = np.asarray(z_plot[:, idx], dtype=float)
        valid = np.isfinite(column) & np.isfinite(zcol) & (zcol <= max_height) & (column > 0.0)
        if threshold is not None and np.isfinite(threshold) and threshold > 0.0:
            valid = valid & (column >= threshold)
        if not np.any(valid):
            continue
        weights = column[valid]
        heights = zcol[valid]
        z_peak[idx] = float(heights[int(np.argmax(weights))])
        total = float(np.sum(weights))
        if np.isfinite(total) and total > 0.0:
            z_centroid[idx] = float(np.sum(weights * heights) / total)

    return z_peak, z_centroid


def section_metrics(
    field: np.ndarray,
    z_msl: np.ndarray,
    z_agl: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    max_height: float,
    height_mode: str,
) -> dict:
    z_plot = z_msl if height_mode == "msl" else z_agl
    valid = np.isfinite(field) & np.isfinite(z_plot) & (z_plot <= max_height)
    weights = np.where(valid, field, 0.0)
    total = float(np.sum(weights))

    if total > 0.0:
        centroid_height_msl = float(np.sum(weights * z_msl) / total)
        centroid_height_agl = float(np.sum(weights * z_agl) / total)
        centroid_lat = float(np.sum(weights * grid_lat[np.newaxis, :]) / total)
        centroid_lon = float(np.sum(weights * grid_lon[np.newaxis, :]) / total)
    else:
        centroid_height_msl = float("nan")
        centroid_height_agl = float("nan")
        centroid_lat = float("nan")
        centroid_lon = float("nan")

    masked = np.where(valid, field, np.nan)
    if np.isfinite(masked).any():
        max_flat = int(np.nanargmax(masked))
        max_k, max_i = np.unravel_index(max_flat, masked.shape)
        max_value = float(masked[max_k, max_i])
        max_height_msl = float(z_msl[max_k, max_i])
        max_height_agl = float(z_agl[max_k, max_i])
        max_lat = float(grid_lat[max_i])
        max_lon = float(grid_lon[max_i])
    else:
        max_value = float("nan")
        max_height_msl = float("nan")
        max_height_agl = float("nan")
        max_lat = float("nan")
        max_lon = float("nan")

    return {
        "max_value_section": max_value,
        "max_height_msl_m": max_height_msl,
        "max_height_agl_m": max_height_agl,
        "max_lat": max_lat,
        "max_lon": max_lon,
        "centroid_height_msl_m": centroid_height_msl,
        "centroid_height_agl_m": centroid_height_agl,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "section_integral_proxy": total,
    }


def positive_color_limits(field: np.ndarray) -> tuple[float, float]:
    values = field[np.isfinite(field) & (field > 0.0)]
    if values.size == 0:
        return 0.0, 1.0
    vmax = float(np.nanpercentile(values, 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return 0.0, vmax


def plume_visual_threshold(field: np.ndarray, args: argparse.Namespace) -> float | None:
    values = field[np.isfinite(field) & (field > 0.0)]
    if values.size == 0:
        return None

    thresholds = []
    percentile = float(np.clip(args.plume_percentile, 0.0, 100.0))
    if percentile > 0.0:
        thresholds.append(float(np.nanpercentile(values, percentile)))

    fraction = max(0.0, float(args.plume_min_fraction))
    if fraction > 0.0:
        thresholds.append(float(np.nanmax(values) * fraction))

    if args.plume_min_value is not None:
        thresholds.append(float(args.plume_min_value))

    thresholds = [value for value in thresholds if np.isfinite(value) and value > 0.0]
    if not thresholds:
        return None
    return max(thresholds)


def adaptive_plume_visual_threshold(
    field: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float | None, float | None, float | None]:
    base_percentile = float(np.clip(args.plume_percentile, 0.0, 100.0))
    base_fraction = max(0.0, float(args.plume_min_fraction))
    trials = [
        (base_percentile, base_fraction),
        (max(0.0, base_percentile - 20.0), base_fraction / 3.0),
        (max(0.0, base_percentile - 40.0), base_fraction / 10.0),
        (0.0, 0.0),
    ]
    seen: set[tuple[float, float]] = set()
    for percentile, fraction in trials:
        key = (round(percentile, 6), round(fraction, 12))
        if key in seen:
            continue
        seen.add(key)
        trial_args = argparse.Namespace(**vars(args))
        trial_args.plume_percentile = percentile
        trial_args.plume_min_fraction = fraction
        threshold = plume_visual_threshold(field, trial_args)
        if threshold is None:
            positive = np.where(np.isfinite(field) & (field > 0.0), field, np.nan)
            if np.isfinite(positive).any():
                return None, percentile, fraction
            continue
        filtered = np.where(field >= threshold, field, np.nan)
        if np.isfinite(filtered).any():
            return threshold, percentile, fraction
    return None, None, None


def apply_visual_filter(
    field: np.ndarray,
    z_plot: np.ndarray,
    max_height: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float | None, str, float | None, float | None]:
    plot_field = np.where(z_plot <= max_height, field, np.nan)
    mode = args.visual_filter

    if mode == "none":
        return plot_field, None, "sin filtro visual", None, None

    if mode == "positive":
        filtered = np.where(plot_field > 0.0, plot_field, np.nan)
        return filtered, 0.0, "anomalia positiva", None, None

    threshold, used_percentile, used_fraction = adaptive_plume_visual_threshold(
        plot_field,
        args,
    )
    if threshold is None:
        positive = np.where(plot_field > 0.0, plot_field, np.nan)
        if np.isfinite(positive).any():
            return positive, None, "pluma: mostrando positivos", used_percentile, used_fraction
        return positive, None, "pluma: sin senal positiva", used_percentile, used_fraction

    filtered = np.where(plot_field >= threshold, plot_field, np.nan)
    if not np.isfinite(filtered).any():
        filtered = np.where(plot_field > 0.0, plot_field, np.nan)
        return filtered, None, "pluma: umbral sin datos, mostrando positivos", used_percentile, used_fraction

    return filtered, threshold, f"pluma >= {threshold:.3g}", used_percentile, used_fraction


def compute_effective_max_height(
    *,
    field: np.ndarray,
    plot_field: np.ndarray,
    z_plot: np.ndarray,
    z_agl: np.ndarray,
    terrain: np.ndarray,
    pblh_line: np.ndarray | None,
    requested_max_height: float,
) -> tuple[float, str | None]:
    requested = float(requested_max_height)
    terrain_max = float(np.nanmax(terrain)) if np.isfinite(terrain).any() else 0.0
    default_like = abs(requested - DEFAULT_MAX_HEIGHT_M) < 1.0e-6

    positive_mask = np.isfinite(plot_field)
    if not np.any(positive_mask):
        positive_mask = np.isfinite(field) & (field > 0.0)
    if not np.any(positive_mask):
        return requested, None

    plume_top_agl = float(np.nanmax(np.where(positive_mask, z_agl, np.nan)))
    plume_top_plot = float(np.nanmax(np.where(positive_mask, z_plot, np.nan)))
    if not np.isfinite(plume_top_agl) or plume_top_agl <= 0.0:
        return requested, None

    pblh_top = float(np.nanmax(pblh_line)) if pblh_line is not None and np.isfinite(pblh_line).any() else np.nan
    if np.isfinite(pblh_top):
        auto_max = max(
            terrain_max + max(1200.0, plume_top_agl * 2.2),
            plume_top_plot * 1.25,
            pblh_top * 1.15,
        )
    else:
        auto_max = max(
            terrain_max + max(1200.0, plume_top_agl * 2.2),
            plume_top_plot * 1.25,
        )

    auto_max = min(requested, max(1500.0, auto_max))
    if default_like and auto_max < 0.75 * requested:
        return auto_max, f"zoom vertical auto: 0-{auto_max:.0f} m"
    return requested, None


def plot_vertical_section(
    *,
    field: np.ndarray,
    contour_field: np.ndarray | None,
    contour_label: str,
    z_plot: np.ndarray,
    z_agl: np.ndarray,
    terrain: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    terrain2d: np.ndarray,
    transect_lat: np.ndarray,
    transect_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    distance_km: np.ndarray,
    pblh_line: np.ndarray | None,
    source_marker: tuple[float, float, str] | None,
    shapefile_path: Path | None,
    tracer_display_name: str,
    tracer_units: str,
    timestamp: pd.Timestamp,
    args: argparse.Namespace,
    outpath: Path,
    signal_summary: dict[str, object] | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    plot_field, visual_threshold, visual_label, used_percentile, used_fraction = apply_visual_filter(
        field,
        z_plot,
        float(args.max_height),
        args,
    )
    effective_max_height, zoom_label = compute_effective_max_height(
        field=field,
        plot_field=plot_field,
        z_plot=z_plot,
        z_agl=z_agl,
        terrain=terrain,
        pblh_line=pblh_line,
        requested_max_height=float(args.max_height),
    )
    plot_field = np.where(z_plot <= effective_max_height, plot_field, np.nan)
    vmin, vmax = positive_color_limits(plot_field)
    if visual_threshold is not None and np.isfinite(visual_threshold):
        vmin = float(visual_threshold)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin * 1.01
    x2d = np.tile(distance_km[np.newaxis, :], (field.shape[0], 1))

    fig, ax = plt.subplots(figsize=(11, 8.5))
    pcm = ax.pcolormesh(
        x2d,
        z_plot,
        np.ma.masked_invalid(plot_field),
        shading="auto",
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if contour_field is not None:
        contour_section = np.where(z_plot <= effective_max_height, contour_field, np.nan)
        values = contour_section[np.isfinite(contour_section)]
        if values.size > 0:
            levels = np.linspace(
                float(np.nanpercentile(values, 5.0)),
                float(np.nanpercentile(values, 95.0)),
                7,
            )
            if np.unique(np.round(levels, 6)).size > 1:
                cs = ax.contour(
                    x2d,
                    z_plot,
                    contour_section,
                    levels=levels,
                    colors="black",
                    linewidths=0.8,
                )
                ax.clabel(cs, fmt="%.1f", fontsize=8)

    if args.height_mode == "msl":
        ax.fill_between(
            distance_km,
            0.0,
            terrain,
            color="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )
        ax.plot(distance_km, terrain, color="black", linewidth=1.2, zorder=7)
    else:
        ax.axhline(0.0, color="black", linewidth=1.0)

    if pblh_line is not None:
        ax.plot(distance_km, pblh_line, color="limegreen", linestyle="--", linewidth=1.3, label="PBLH")

    marker_distance = nearest_distance_for_marker(source_marker, transect_lat, transect_lon, distance_km)
    if marker_distance is not None:
        ax.axvline(marker_distance, color="gold", linestyle=":", linewidth=1.6, label="Pulso")
    marker_index = nearest_index_for_marker(source_marker, transect_lat, transect_lon)
    if marker_index is not None:
        source_y = terrain[marker_index] if args.height_mode == "msl" else 0.0
        ax.scatter(
            [distance_km[marker_index]],
            [source_y],
            c="yellow",
            edgecolors="black",
            linewidths=0.8,
            s=120,
            marker="*",
            zorder=9,
        )
        ax.text(
            distance_km[marker_index],
            source_y,
            " Emision",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=10,
            va="bottom",
        )

    centerline_threshold = visual_threshold if visual_threshold is not None else None
    z_peak_line, z_centroid_line = section_centerlines(
        field,
        z_plot,
        float(effective_max_height),
        threshold=centerline_threshold,
    )
    if np.isfinite(z_peak_line).any():
        ax.plot(
            distance_km,
            z_peak_line,
            color="deepskyblue",
            linewidth=1.5,
            label="z_peak pluma",
            zorder=8,
        )
    if np.isfinite(z_centroid_line).any():
        ax.plot(
            distance_km,
            z_centroid_line,
            color="white",
            linewidth=1.2,
            linestyle="--",
            label="z_centroid pluma",
            zorder=8,
        )

    tick_count = min(6, len(distance_km))
    tick_idx = np.linspace(0, len(distance_km) - 1, tick_count, dtype=int)
    ax.set_xticks(distance_km[tick_idx])
    ax.set_xticklabels(
        [format_latlon_label(grid_lat[idx], grid_lon[idx]) for idx in tick_idx],
        rotation=45,
        ha="right",
    )
    ax.set_xlim(float(distance_km[0]), float(distance_km[-1]))
    ax.set_ylim(0.0, float(effective_max_height))
    ax.set_ylabel("Height (m MSL)" if args.height_mode == "msl" else "Height (m AGL)")
    ax.set_xlabel("Distancia sobre transecto (km) / Lat-Lon")
    if marker_distance is not None and np.isfinite(marker_distance):
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda x: x - marker_distance,
                lambda x: x + marker_distance,
            ),
        )
        secax.set_xlabel("Distancia desde pulso (km)")

    subtitle_parts = [
        f"Transecto: {args.start_lat:.5f},{args.start_lon:.5f} -> {args.end_lat:.5f},{args.end_lon:.5f}",
    ]
    source_text = format_source_marker(source_marker)
    if source_text:
        subtitle_parts.append(source_text)
    if contour_label:
        subtitle_parts.append(f"Contornos: {contour_label}")
    subtitle_parts.append(f"Visualizacion: {visual_label}")
    if used_percentile is not None or used_fraction is not None:
        subtitle_parts.append(
            "umbral adaptado: "
            f"percentil={used_percentile if used_percentile is not None else 0:.0f}, "
            f"frac={used_fraction if used_fraction is not None else 0:.3g}"
        )
    if zoom_label:
        subtitle_parts.append(zoom_label)
    ax.set_title(
        title_with_subtitle(
            f"Corte vertical {tracer_display_name} - {timestamp:%Y-%m-%d_%H:%M:%S}",
            " | ".join(subtitle_parts),
        )
    )
    if pblh_line is not None or marker_distance is not None or np.isfinite(z_peak_line).any() or np.isfinite(z_centroid_line).any():
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    has_visible_signal = bool(np.isfinite(plot_field).any())
    if not has_visible_signal:
        message = [f"Sin senal positiva visible para {tracer_display_name}"]
        if signal_summary is not None:
            max_value = float(signal_summary.get("max_value", 0.0) or 0.0)
            if max_value > 0.0:
                message.append(f"Maximo en archivo: {max_value:.3g}")
            suggested = signal_summary.get("suggested_tracer")
            if suggested:
                message.append(f"Sugerencia: probar {suggested}")
        ax.text(
            0.5,
            0.55,
            "\n".join(message),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="firebrick",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "firebrick", "alpha": 0.92},
            zorder=20,
        )

    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.16, fraction=0.07)
    unit_label = tracer_units if tracer_units and tracer_units.lower() != "unknown" else "unidades WRF"
    cbar.set_label(f"{tracer_display_name} ({unit_label})")
    if visual_threshold is not None:
        cbar.ax.text(
            1.0,
            -1.35,
            f"Fondo oculto: valores < {visual_threshold:.3g}",
            transform=cbar.ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.25",
        )

    inset = inset_axes(ax, width="34%", height="34%", loc="upper right", borderpad=0.7)
    inset.pcolormesh(lon2d, lat2d, terrain2d, shading="auto", cmap="terrain")
    inset.plot(transect_lon, transect_lat, color="blue", linewidth=1.4)
    inset.scatter([transect_lon[0], transect_lon[-1]], [transect_lat[0], transect_lat[-1]], c=["white", "yellow"], edgecolors="black", s=24, zorder=5)
    plot_shapefile_boundary(inset, shapefile_path, linewidth=0.8)
    plot_source_marker(inset, source_marker)
    lon_margin = max(0.03, 0.12 * abs(float(np.nanmax(transect_lon) - np.nanmin(transect_lon))))
    lat_margin = max(0.03, 0.12 * abs(float(np.nanmax(transect_lat) - np.nanmin(transect_lat))))
    inset.set_xlim(float(np.nanmin(transect_lon) - lon_margin), float(np.nanmax(transect_lon) + lon_margin))
    inset.set_ylim(float(np.nanmin(transect_lat) - lat_margin), float(np.nanmax(transect_lat) + lat_margin))
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Transecto", fontsize=8)

    add_figure_footer(
        fig,
        script_path=Path(__file__).resolve(),
        output_path=outpath,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.22)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return outpath


def process_step(
    *,
    path: Path,
    time_index: int,
    timestamp: pd.Timestamp,
    tracer_name: str,
    baseline_3d: np.ndarray | None,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    transect_lat: np.ndarray,
    transect_lon: np.ndarray,
    distance_km: np.ndarray,
    source_marker: tuple[float, float, str] | None,
    shapefile_path: Path | None,
    tracer_display_name: str,
    tracer_units: str,
    args: argparse.Namespace,
    outdir: Path,
    case_tag: str,
    signal_summary: dict[str, object] | None,
) -> tuple[Path, dict]:
    with Dataset(path) as nc:
        tracer = read_tracer_step(
            nc,
            tracer_name,
            time_index=time_index,
            baseline_3d=baseline_3d,
        )
        z_agl_3d, _ = compute_height_agl_and_dz(nc, time_index=time_index)
        terrain2d = read_var(nc, "HGT", time_index=time_index) if "HGT" in nc.variables else np.zeros_like(lat2d)
        z_msl_3d = z_agl_3d + terrain2d[np.newaxis, :, :]

        contour_3d, contour_label = build_contour_field(nc, args.contour_var, time_index)
        pblh_line = None
        if "PBLH" in nc.variables:
            pblh = sample_2d(read_var(nc, "PBLH", time_index=time_index), iy, ix)
            terrain_line = sample_2d(terrain2d, iy, ix)
            pblh_line = pblh + terrain_line if args.height_mode == "msl" else pblh

    section = sample_3d(tracer, iy, ix)
    z_agl_section = sample_3d(z_agl_3d, iy, ix)
    z_msl_section = sample_3d(z_msl_3d, iy, ix)
    z_plot = z_msl_section if args.height_mode == "msl" else z_agl_section
    terrain_line = sample_2d(terrain2d, iy, ix)
    contour_section = sample_3d(contour_3d, iy, ix) if contour_3d is not None else None
    grid_lat = sample_2d(lat2d, iy, ix)
    grid_lon = sample_2d(lon2d, iy, ix)

    timestamp_tag = f"{timestamp:%Y-%m-%d_%H%M%S}"
    outpath = outdir / f"{case_tag}_corte_vertical_{timestamp_tag}.png"
    plot_field_preview, _, _, _, _ = apply_visual_filter(
        section,
        z_plot,
        float(args.max_height),
        args,
    )
    effective_max_height, _ = compute_effective_max_height(
        field=section,
        plot_field=plot_field_preview,
        z_plot=z_plot,
        z_agl=z_agl_section,
        terrain=terrain_line,
        pblh_line=pblh_line,
        requested_max_height=float(args.max_height),
    )
    plot_vertical_section(
        field=section,
        contour_field=contour_section,
        contour_label=contour_label,
        z_plot=z_plot,
        z_agl=z_agl_section,
        terrain=terrain_line,
        lat2d=lat2d,
        lon2d=lon2d,
        terrain2d=terrain2d,
        transect_lat=transect_lat,
        transect_lon=transect_lon,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        distance_km=distance_km,
        pblh_line=pblh_line,
        source_marker=source_marker,
        shapefile_path=shapefile_path,
        tracer_display_name=tracer_display_name,
        tracer_units=tracer_units,
        timestamp=timestamp,
        args=args,
        outpath=outpath,
        signal_summary=signal_summary,
    )

    metrics = section_metrics(
        section,
        z_msl_section,
        z_agl_section,
        grid_lat,
        grid_lon,
        max_height=float(effective_max_height),
        height_mode=args.height_mode,
    )
    _, visual_threshold, visual_label, used_percentile, used_fraction = apply_visual_filter(
        section,
        z_plot,
        float(effective_max_height),
        args,
    )
    metrics.update(
        {
            "time": timestamp,
            "wrfout": str(path),
            "time_index_in_file": time_index,
            "figure": str(outpath),
            "start_lat": args.start_lat,
            "start_lon": args.start_lon,
            "end_lat": args.end_lat,
            "end_lon": args.end_lon,
            "visual_filter": args.visual_filter,
            "visual_threshold": visual_threshold,
            "visual_label": visual_label,
            "visual_percentile_used": used_percentile,
            "visual_fraction_used": used_fraction,
            "has_signal_in_file": bool(signal_summary.get("has_signal")) if signal_summary else None,
            "suggested_tracer": signal_summary.get("suggested_tracer") if signal_summary else None,
        }
    )
    return outpath, metrics


def main() -> None:
    args = build_parser().parse_args()
    args = maybe_apply_common_gui_options(
        args,
        title="Opciones - corte vertical del trazador",
        include_domain=True,
        include_region_box=False,
        include_input=True,
        allow_wrf_files=True,
    )

    wrf_files, outdir = resolve_common_inputs(
        args,
        default_subdir="wrf_trazadores_05_corte_vertical",
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
    signal_summary = summarize_tracer_signal(wrf_files, tracer_name)

    with Dataset(wrf_files[0]) as nc0:
        lat2d, lon2d = get_latlon_2d(nc0)

    domain_tag = infer_domain_tag(wrf_files[0])
    source_marker = resolve_source_marker(args, wrf_files, domain_tag=domain_tag)
    args = maybe_apply_vertical_gui_options(args, all_times, lat2d, lon2d, source_marker)

    if not has_complete_transect(args):
        args.start_lat, args.start_lon, args.end_lat, args.end_lon = default_transect(
            lat2d,
            lon2d,
            source_marker,
            orientation="nw_se",
        )

    transect_lat, transect_lon, distance_km = build_transect_points(args)
    iy, ix = nearest_grid_indices(lat2d, lon2d, transect_lat, transect_lon)

    step_refs = list_wrf_step_refs(wrf_files)
    indices = selected_step_indices(args, step_refs)
    case_tag = safe_tag(args.tag or f"{domain_tag}_{tracer_display_name}_corte_vertical")

    outputs = []
    rows = []
    for global_index in indices:
        path, time_index, timestamp = step_refs[global_index]
        outpath, metrics = process_step(
            path=path,
            time_index=time_index,
            timestamp=timestamp,
            tracer_name=tracer_name,
            baseline_3d=baseline_3d,
            lat2d=lat2d,
            lon2d=lon2d,
            iy=iy,
            ix=ix,
            transect_lat=transect_lat,
            transect_lon=transect_lon,
            distance_km=distance_km,
            source_marker=source_marker,
            shapefile_path=shapefile_path,
            tracer_display_name=tracer_display_name,
            tracer_units=tracer_units,
            args=args,
            outdir=outdir,
            case_tag=case_tag,
            signal_summary=signal_summary,
        )
        outputs.append(outpath)
        rows.append(metrics)

    csv_path = outdir / f"{case_tag}_resumen_desplazamiento_vertical.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    gif_path = None
    if len(outputs) > 1:
        gif_path = build_gif_from_pngs(
            outputs,
            outdir / f"{case_tag}_corte_vertical_animacion.gif",
            duration_ms=500,
        )

    print(f"Trazador usado: {tracer_display_name} (variable base: {tracer_name})")
    if not bool(signal_summary.get("has_signal")):
        suggested = signal_summary.get("suggested_tracer")
        print(
            "Aviso: la variable seleccionada no tiene senal positiva en el archivo."
            + (f" Pruebe con: {suggested}" if suggested else "")
        )
    print(
        "Transecto: "
        f"{args.start_lat:.6f},{args.start_lon:.6f} -> "
        f"{args.end_lat:.6f},{args.end_lon:.6f}"
    )
    print(f"Figuras generadas: {len(outputs)}")
    if outputs:
        print(f"Primera figura: {outputs[0]}")
        print(f"Ultima figura: {outputs[-1]}")
    if gif_path is not None:
        print(f"GIF animado: {gif_path}")
    print(f"CSV resumen vertical: {csv_path}")


if __name__ == "__main__":
    main()
