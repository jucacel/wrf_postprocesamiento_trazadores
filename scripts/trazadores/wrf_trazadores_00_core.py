"""
Nucleo comun para diagnosticos de trazadores en salidas WRF.

Este modulo centraliza:
- seleccion de archivos y carpeta de salida;
- validacion minima de wrfout;
- deteccion de la variable de trazador;
- parseo robusto de tiempos WRF;
- lectura segura de variables NetCDF;
- calculo de alturas AGL y espesores de capa;
- integracion vertical y fraccion sobre PBLH;
- mascaras espaciales simples por caja lat-lon;
- exportes comunes de CSV y mapas PNG.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from netCDF4 import Dataset

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover - tkinter puede no estar disponible
    tk = None
    filedialog = None
    messagebox = None
    ttk = None


GRAVITY = 9.81
WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_WRF_OUT_ROOT = WORKSPACE_ROOT / "Z_WRF_OUT"
DEFAULT_POSTPROC_DIRNAME = "POSTPROC_TRAZADORES"
INVALID_WRF_SUFFIXES = {".zip", ".gz", ".bz2", ".xz", ".7z"}

TRACER_PRIORITY = [
    "N_SLV",
    "S_SLV",
    "CO_T",
    "co_t",
    "tr17_1",
    "tr17_2",
    "tr17_3",
    "tr17_4",
    "tr17_5",
    "tr17_6",
    "tr17_7",
    "tr17_8",
    "co",
]

MET_4D_EXCLUSIONS = {
    "u",
    "v",
    "w",
    "t",
    "p",
    "pb",
    "ph",
    "phb",
    "qvapor",
    "qcloud",
    "qrain",
    "qice",
    "qsnow",
    "qgraup",
    "qnr",
    "qni",
    "qnc",
    "qng",
    "qns",
    "cldfra",
    "smoke",
    "mut",
    "mu",
    "mub",
}


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    add_gui_options_cli_arguments(parser)
    parser.add_argument(
        "wrf_files",
        nargs="*",
        help="Lista de archivos wrfout. Si se omite, se intenta seleccionar por GUI.",
    )
    parser.add_argument(
        "--outdir",
        help="Carpeta de salida. Si se omite, se usa POSTPROC_TRAZADORES en la corrida.",
    )
    parser.add_argument(
        "--select-outdir",
        action="store_true",
        help="Abre una ventana para seleccionar carpeta de salida.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Carpeta que contiene los wrfout de una corrida. "
            "Si se omite, se abre una ventana para seleccionar wrfout."
        ),
    )
    parser.add_argument(
        "--auto-run-dir",
        action="store_true",
        help=(
            "Si no se indican wrfout ni --run-dir, usa automaticamente la corrida "
            "mas reciente en Z_WRF_OUT en vez de abrir ventana."
        ),
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Filtra wrfout por dominio, por ejemplo d01, d02 o d03.",
    )
    parser.add_argument(
        "--tracer-name",
        default=None,
        help=(
            "Nombre explicito del trazador. Ej.: N_SLV, S_SLV, CO_T, tr17_1, "
            "co o delta_co."
        ),
    )
    parser.add_argument(
        "--delta-from-initial",
        action="store_true",
        help=(
            "Resta el primer tiempo del trazador y conserva solo la anomalia "
            "positiva. Util para co cuando no existe CO_T."
        ),
    )
    parser.add_argument("--lat-min", type=float, default=None, help="Latitud minima.")
    parser.add_argument("--lat-max", type=float, default=None, help="Latitud maxima.")
    parser.add_argument("--lon-min", type=float, default=None, help="Longitud minima.")
    parser.add_argument("--lon-max", type=float, default=None, help="Longitud maxima.")
    parser.add_argument(
        "--shapefile",
        default=None,
        help=(
            "Shapefile o GeoPackage opcional para mascara espacial. "
            "Se combina con la caja lat/lon si tambien se indica."
        ),
    )
    parser.add_argument(
        "--select-shapefile",
        action="store_true",
        help="Abre una ventana para seleccionar shapefile de mascara espacial.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Etiqueta opcional del caso para nombres de salida.",
    )


def add_gui_options_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gui-options",
        action="store_true",
        help="Abre una ventana para configurar opciones complementarias.",
    )
    parser.add_argument(
        "--no-gui-options",
        action="store_true",
        help="No abre la ventana de opciones aunque el script se ejecute sin argumentos.",
    )


def should_open_gui_options(args: argparse.Namespace) -> bool:
    if getattr(args, "no_gui_options", False):
        return False
    return bool(getattr(args, "gui_options", False) or len(sys.argv) == 1)


def maybe_apply_common_gui_options(
    args: argparse.Namespace,
    *,
    title: str = "Opciones WRF trazadores",
    include_domain: bool = True,
    include_region_box: bool = True,
    include_input: bool = True,
    allow_wrf_files: bool = True,
    include_delta_from_initial: bool = False,
    include_presence_threshold: bool = False,
) -> argparse.Namespace:
    if not should_open_gui_options(args):
        return args
    return apply_common_gui_options(
        args,
        title=title,
        include_domain=include_domain,
        include_region_box=include_region_box,
        include_input=include_input,
        allow_wrf_files=allow_wrf_files,
        include_delta_from_initial=include_delta_from_initial,
        include_presence_threshold=include_presence_threshold,
    )


def apply_common_gui_options(
    args: argparse.Namespace,
    *,
    title: str = "Opciones WRF trazadores",
    include_domain: bool = True,
    include_region_box: bool = True,
    include_input: bool = True,
    allow_wrf_files: bool = True,
    include_delta_from_initial: bool = False,
    include_presence_threshold: bool = False,
) -> argparse.Namespace:
    options = select_common_options_gui(
        args,
        title=title,
        include_domain=include_domain,
        include_region_box=include_region_box,
        include_input=include_input,
        allow_wrf_files=allow_wrf_files,
        include_delta_from_initial=include_delta_from_initial,
        include_presence_threshold=include_presence_threshold,
    )
    if options is None:
        raise SystemExit("Configuracion cancelada por el usuario.")

    input_mode = options.get("input_mode")
    if input_mode == "run_dir" and options.get("run_dir"):
        setattr(args, "run_dir", options["run_dir"])
        setattr(args, "auto_run_dir", False)
        if hasattr(args, "wrf_files"):
            args.wrf_files = []
    elif input_mode == "wrf_files" and options.get("wrf_files") and hasattr(args, "wrf_files"):
        args.wrf_files = options["wrf_files"]
        setattr(args, "run_dir", None)
        setattr(args, "auto_run_dir", False)
    elif input_mode == "latest":
        setattr(args, "auto_run_dir", True)
        setattr(args, "run_dir", None)
        if hasattr(args, "wrf_files"):
            args.wrf_files = []

    if include_domain:
        domain = options.get("domain")
        setattr(args, "domain", None if domain in (None, "auto") else domain)

    tracer_name = options.get("tracer_name")
    if tracer_name == "auto":
        setattr(args, "tracer_name", None)
        setattr(args, "delta_from_initial", False)
    elif tracer_name:
        setattr(args, "tracer_name", tracer_name)
        setattr(
            args,
            "delta_from_initial",
            bool(options.get("delta_from_initial", False)),
        )
    elif include_delta_from_initial:
        setattr(args, "delta_from_initial", bool(options.get("delta_from_initial", False)))

    shapefile_path = options.get("shapefile")
    use_shapefile = options.get("use_shapefile", False)
    if shapefile_path:
        setattr(args, "shapefile", shapefile_path)
        setattr(args, "select_shapefile", False)
    elif use_shapefile:
        setattr(args, "shapefile", None)
        setattr(args, "select_shapefile", True)
    else:
        setattr(args, "shapefile", None)
        setattr(args, "select_shapefile", False)

    outdir = options.get("outdir")
    use_outdir = options.get("use_outdir", False)
    if outdir:
        setattr(args, "outdir", outdir)
        setattr(args, "select_outdir", False)
    elif use_outdir:
        setattr(args, "outdir", None)
        setattr(args, "select_outdir", True)
    else:
        setattr(args, "outdir", None)
        setattr(args, "select_outdir", False)

    tag = options.get("tag")
    if tag:
        setattr(args, "tag", tag)

    if include_presence_threshold and hasattr(args, "presence_threshold"):
        threshold = options.get("presence_threshold")
        if threshold is not None:
            setattr(args, "presence_threshold", threshold)

    if include_region_box:
        box = options.get("region_box")
        if box is None:
            setattr(args, "lat_min", None)
            setattr(args, "lat_max", None)
            setattr(args, "lon_min", None)
            setattr(args, "lon_max", None)
        else:
            lat_min, lat_max, lon_min, lon_max = box
            setattr(args, "lat_min", lat_min)
            setattr(args, "lat_max", lat_max)
            setattr(args, "lon_min", lon_min)
            setattr(args, "lon_max", lon_max)

    source = options.get("source")
    if source is not None:
        source_lat, source_lon, source_label = source
        setattr(args, "source_lat", source_lat)
        setattr(args, "source_lon", source_lon)
        setattr(args, "source_label", source_label or "Pulso")

    return args


def get_default_run_dir() -> Path | None:
    return autodiscover_run_dir(DEFAULT_WRF_OUT_ROOT)


def get_default_browse_dir() -> Path:
    default_run_dir = get_default_run_dir()
    if default_run_dir is not None:
        return default_run_dir
    if DEFAULT_WRF_OUT_ROOT.exists():
        return DEFAULT_WRF_OUT_ROOT
    return WORKSPACE_ROOT


def select_common_options_gui(
    args: argparse.Namespace,
    *,
    title: str,
    include_domain: bool,
    include_region_box: bool,
    include_input: bool,
    allow_wrf_files: bool,
    include_delta_from_initial: bool,
    include_presence_threshold: bool,
) -> dict | None:
    if tk is None or filedialog is None or messagebox is None or ttk is None:
        raise SystemExit("tkinter no esta disponible para abrir ventanas de opciones.")

    root = tk.Tk()
    root.title(title)
    root.attributes("-topmost", True)
    root.resizable(False, False)

    selected_wrf_files = [
        str(Path(path).expanduser().resolve())
        for path in getattr(args, "wrf_files", []) or []
    ]
    default_run_dir = get_default_run_dir()
    current_run_dir = (
        str(Path(args.run_dir).expanduser().resolve())
        if getattr(args, "run_dir", None)
        else (str(default_run_dir) if default_run_dir is not None else "")
    )
    current_outdir = str(Path(args.outdir).expanduser().resolve()) if getattr(args, "outdir", None) else ""
    current_shapefile = str(Path(args.shapefile).expanduser().resolve()) if getattr(args, "shapefile", None) else ""

    row = 0
    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(
        frame,
        text=(
            "Configure las opciones. Si deja una ruta vacia y marca seleccion, "
            "se abrira otra ventana para escogerla."
        ),
        wraplength=720,
    ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 10))
    row += 1

    default_input_mode = "run_dir" if current_run_dir or not allow_wrf_files else "wrf_files"
    input_mode_var = tk.StringVar(value=default_input_mode)
    run_dir_var = tk.StringVar(value=current_run_dir)
    wrf_files_label_var = tk.StringVar(
        value=f"{len(selected_wrf_files)} archivo(s) seleccionado(s)"
        if selected_wrf_files
        else "sin archivos seleccionados"
    )

    if include_input:
        ttk.Label(frame, text="Entrada wrfout").grid(row=row, column=0, sticky="w")
        input_values = ["run_dir", "latest"]
        if allow_wrf_files:
            input_values.insert(1, "wrf_files")
        input_labels = {
            "run_dir": "Carpeta con wrfout",
            "wrf_files": "Archivos wrfout",
            "latest": "Corrida mas reciente en Z_WRF_OUT",
        }
        input_combo = ttk.Combobox(
            frame,
            textvariable=input_mode_var,
            values=input_values,
            width=18,
            state="readonly",
        )
        input_combo.grid(row=row, column=1, sticky="w", padx=(8, 4))
        ttk.Label(frame, text=" / ".join(input_labels[value] for value in input_values)).grid(
            row=row,
            column=2,
            columnspan=2,
            sticky="w",
        )
        row += 1

        ttk.Label(frame, text="Carpeta").grid(row=row, column=0, sticky="w")
        ttk.Entry(frame, textvariable=run_dir_var, width=62).grid(row=row, column=1, columnspan=2, sticky="we", padx=(8, 4))

        def browse_run_dir() -> None:
            directory = filedialog.askdirectory(
                parent=root,
                title="Seleccione carpeta de corrida con archivos wrfout",
                initialdir=str(get_default_browse_dir()),
            )
            if directory:
                run_dir_var.set(str(Path(directory).expanduser().resolve()))
                input_mode_var.set("run_dir")

        ttk.Button(frame, text="Buscar", command=browse_run_dir).grid(row=row, column=3, sticky="w")
        row += 1

        if allow_wrf_files:
            ttk.Label(frame, text="Archivos").grid(row=row, column=0, sticky="w")
            ttk.Label(frame, textvariable=wrf_files_label_var).grid(row=row, column=1, columnspan=2, sticky="w", padx=(8, 4))

            def browse_wrf_files() -> None:
                files = filedialog.askopenfilenames(
                    parent=root,
                    title="Seleccione archivos wrfout",
                    initialdir=str(get_default_browse_dir()),
                )
                if files:
                    selected_wrf_files.clear()
                    selected_wrf_files.extend(str(Path(path).expanduser().resolve()) for path in files)
                    wrf_files_label_var.set(f"{len(selected_wrf_files)} archivo(s) seleccionado(s)")
                    input_mode_var.set("wrf_files")

            ttk.Button(frame, text="Buscar", command=browse_wrf_files).grid(row=row, column=3, sticky="w")
            row += 1

    domain_var = tk.StringVar(value=normalize_domain(getattr(args, "domain", None)) or "auto")
    if include_domain:
        ttk.Label(frame, text="Dominio").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            frame,
            textvariable=domain_var,
            values=["auto", "d01", "d02", "d03", "d04"],
            width=18,
            state="readonly",
        ).grid(row=row, column=1, sticky="w", padx=(8, 4), pady=(8, 0))
        ttk.Label(frame, text="auto usa el dominio mas fino si hay varios").grid(row=row, column=2, columnspan=2, sticky="w", pady=(8, 0))
        row += 1

    tracer_initial = getattr(args, "tracer_name", None) or "auto"
    if getattr(args, "delta_from_initial", False) and tracer_initial == "co":
        tracer_initial = "delta_co"
    tracer_var = tk.StringVar(
        value=tracer_initial if tracer_initial in {"delta_co", "co", "CO_T", "N_SLV", "S_SLV", "auto"} else "custom"
    )
    tracer_custom_var = tk.StringVar(
        value="" if tracer_var.get() != "custom" else str(tracer_initial)
    )
    ttk.Label(frame, text="Trazador").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Combobox(
        frame,
        textvariable=tracer_var,
        values=["auto", "N_SLV", "S_SLV", "CO_T", "delta_co", "co", "custom"],
        width=18,
        state="readonly",
    ).grid(row=row, column=1, sticky="w", padx=(8, 4), pady=(8, 0))
    ttk.Entry(frame, textvariable=tracer_custom_var, width=24).grid(row=row, column=2, sticky="w", pady=(8, 0))
    ttk.Label(frame, text="custom solo si la variable tiene otro nombre").grid(row=row, column=3, sticky="w", pady=(8, 0))
    row += 1

    delta_from_initial_var = tk.BooleanVar(
        value=bool(getattr(args, "delta_from_initial", False))
    )
    if include_delta_from_initial:
        ttk.Checkbutton(
            frame,
            text="Restar primer tiempo (delta desde inicio)",
            variable=delta_from_initial_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(
            frame,
            text="Util para co; normalmente no hace falta para N_SLV/S_SLV",
        ).grid(row=row, column=3, sticky="w", pady=(8, 0))
        row += 1

    use_shape_var = tk.BooleanVar(value=bool(current_shapefile or getattr(args, "select_shapefile", False)))
    shapefile_var = tk.StringVar(value=current_shapefile)
    ttk.Checkbutton(frame, text="Usar shapefile / GeoPackage", variable=use_shape_var).grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=shapefile_var, width=62).grid(row=row, column=1, columnspan=2, sticky="we", padx=(8, 4), pady=(8, 0))

    def browse_shapefile() -> None:
        shp = filedialog.askopenfilename(
            parent=root,
            title="Seleccione shapefile o GeoPackage para mascara",
            initialdir=str(get_default_browse_dir()),
            filetypes=[
                ("Vector GIS", "*.shp *.gpkg"),
                ("Shapefile", "*.shp"),
                ("GeoPackage", "*.gpkg"),
                ("Todos", "*.*"),
            ],
        )
        if shp:
            shapefile_var.set(str(Path(shp).expanduser().resolve()))
            use_shape_var.set(True)

    ttk.Button(frame, text="Buscar", command=browse_shapefile).grid(row=row, column=3, sticky="w", pady=(8, 0))
    row += 1

    use_outdir_var = tk.BooleanVar(value=bool(current_outdir or getattr(args, "select_outdir", False)))
    outdir_var = tk.StringVar(value=current_outdir)
    ttk.Checkbutton(frame, text="Seleccionar carpeta de salida", variable=use_outdir_var).grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=outdir_var, width=62).grid(row=row, column=1, columnspan=2, sticky="we", padx=(8, 4), pady=(8, 0))

    def browse_outdir() -> None:
        directory = filedialog.askdirectory(
            parent=root,
            title="Seleccione carpeta de salida",
            initialdir=str(get_default_browse_dir()),
        )
        if directory:
            outdir_var.set(str(Path(directory).expanduser().resolve()))
            use_outdir_var.set(True)

    ttk.Button(frame, text="Buscar", command=browse_outdir).grid(row=row, column=3, sticky="w", pady=(8, 0))
    row += 1

    tag_var = tk.StringVar(value=getattr(args, "tag", None) or "")
    ttk.Label(frame, text="Etiqueta del caso").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=tag_var, width=32).grid(row=row, column=1, sticky="w", padx=(8, 4), pady=(8, 0))
    ttk.Label(frame, text="opcional, se usa en nombres de salida").grid(row=row, column=2, columnspan=2, sticky="w", pady=(8, 0))
    row += 1

    presence_threshold_var = tk.StringVar(
        value=str(getattr(args, "presence_threshold", 1.0e-12))
        if getattr(args, "presence_threshold", None) is not None
        else "1e-12"
    )
    if include_presence_threshold:
        ttk.Label(frame, text="Umbral de presencia").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frame, textvariable=presence_threshold_var, width=16).grid(
            row=row, column=1, sticky="w", padx=(8, 4), pady=(8, 0)
        )
        ttk.Label(
            frame,
            text="senal minima absoluta para declarar presencia",
        ).grid(row=row, column=2, columnspan=2, sticky="w", pady=(8, 0))
        row += 1

    lat_min_var = tk.StringVar(value="" if getattr(args, "lat_min", None) is None else str(args.lat_min))
    lat_max_var = tk.StringVar(value="" if getattr(args, "lat_max", None) is None else str(args.lat_max))
    lon_min_var = tk.StringVar(value="" if getattr(args, "lon_min", None) is None else str(args.lon_min))
    lon_max_var = tk.StringVar(value="" if getattr(args, "lon_max", None) is None else str(args.lon_max))
    use_box_var = tk.BooleanVar(
        value=all(
            getattr(args, name, None) is not None
            for name in ("lat_min", "lat_max", "lon_min", "lon_max")
        )
    )
    if include_region_box:
        ttk.Checkbutton(frame, text="Usar caja lat/lon", variable=use_box_var).grid(row=row, column=0, sticky="w", pady=(8, 0))
        box_frame = ttk.Frame(frame)
        box_frame.grid(row=row, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(8, 0))
        for col, (label, var) in enumerate(
            [
                ("lat min", lat_min_var),
                ("lat max", lat_max_var),
                ("lon min", lon_min_var),
                ("lon max", lon_max_var),
            ]
        ):
            ttk.Label(box_frame, text=label).grid(row=0, column=2 * col, sticky="w")
            ttk.Entry(box_frame, textvariable=var, width=10).grid(row=0, column=2 * col + 1, sticky="w", padx=(3, 8))
        row += 1

    source_manual_var = tk.BooleanVar(
        value=getattr(args, "source_lat", None) is not None
        and getattr(args, "source_lon", None) is not None
    )
    source_lat_var = tk.StringVar(value="" if getattr(args, "source_lat", None) is None else str(args.source_lat))
    source_lon_var = tk.StringVar(value="" if getattr(args, "source_lon", None) is None else str(args.source_lon))
    source_label_var = tk.StringVar(value=getattr(args, "source_label", None) or "Pulso")
    ttk.Checkbutton(
        frame,
        text="Indicar pulso manualmente",
        variable=source_manual_var,
    ).grid(row=row, column=0, sticky="w", pady=(8, 0))
    source_frame = ttk.Frame(frame)
    source_frame.grid(row=row, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(8, 0))
    for col, (label, var, width) in enumerate(
        [
            ("lat", source_lat_var, 11),
            ("lon", source_lon_var, 11),
            ("etiqueta", source_label_var, 14),
        ]
    ):
        ttk.Label(source_frame, text=label).grid(row=0, column=2 * col, sticky="w")
        ttk.Entry(source_frame, textvariable=var, width=width).grid(row=0, column=2 * col + 1, sticky="w", padx=(3, 8))
    row += 1

    result: dict | None = None

    def parse_float(var: tk.StringVar, name: str) -> float:
        try:
            return float(var.get().strip())
        except ValueError:
            raise ValueError(f"Valor numerico invalido en {name}.")

    def on_ok() -> None:
        nonlocal result
        try:
            tracer_value = tracer_var.get()
            if tracer_value == "custom":
                tracer_value = tracer_custom_var.get().strip()
                if not tracer_value:
                    raise ValueError("Debe escribir el nombre del trazador custom.")

            region_box = None
            if include_region_box and use_box_var.get():
                region_box = (
                    parse_float(lat_min_var, "lat min"),
                    parse_float(lat_max_var, "lat max"),
                    parse_float(lon_min_var, "lon min"),
                    parse_float(lon_max_var, "lon max"),
                )

            source = None
            if source_manual_var.get():
                source = (
                    parse_float(source_lat_var, "latitud del pulso"),
                    parse_float(source_lon_var, "longitud del pulso"),
                    source_label_var.get().strip() or "Pulso",
                )

            presence_threshold = None
            if include_presence_threshold:
                presence_threshold = parse_float(
                    presence_threshold_var,
                    "umbral de presencia",
                )

            wrf_files = selected_wrf_files if allow_wrf_files else []
            result = {
                "input_mode": input_mode_var.get() if include_input else None,
                "run_dir": run_dir_var.get().strip(),
                "wrf_files": wrf_files,
                "domain": domain_var.get(),
                "tracer_name": tracer_value,
                "use_shapefile": use_shape_var.get(),
                "shapefile": shapefile_var.get().strip(),
                "use_outdir": use_outdir_var.get(),
                "outdir": outdir_var.get().strip(),
                "tag": tag_var.get().strip(),
                "delta_from_initial": delta_from_initial_var.get(),
                "presence_threshold": presence_threshold,
                "region_box": region_box,
                "source": source,
            }
            root.destroy()
        except ValueError as exc:
            messagebox.showerror("Opciones invalidas", str(exc), parent=root)

    def on_cancel() -> None:
        nonlocal result
        result = None
        root.destroy()

    buttons = ttk.Frame(frame)
    buttons.grid(row=row, column=0, columnspan=4, sticky="e", pady=(14, 0))
    ttk.Button(buttons, text="Cancelar", command=on_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(buttons, text="Continuar", command=on_ok).grid(row=0, column=1)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return result


def resolve_common_inputs(
    args: argparse.Namespace,
    default_subdir: str,
) -> tuple[list[Path], Path]:
    source_dir: Path | None = None

    if args.wrf_files:
        wrf_files = sorted(Path(path).expanduser().resolve() for path in args.wrf_files)
        source_dir = infer_source_dir_from_files(wrf_files)
    else:
        if args.run_dir:
            source_dir = Path(args.run_dir).expanduser().resolve()
            wrf_files = list_wrf_files_in_dir(source_dir, domain=args.domain)
        else:
            source_dir = None
            if getattr(args, "auto_run_dir", False):
                source_dir = autodiscover_run_dir(DEFAULT_WRF_OUT_ROOT)
            if source_dir is not None:
                wrf_files = list_wrf_files_in_dir(source_dir, domain=args.domain)
            else:
                wrf_files = select_wrf_files_gui()
                source_dir = infer_source_dir_from_files(wrf_files)

    wrf_files = select_wrf_files_for_single_domain(wrf_files, domain=args.domain)

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    elif getattr(args, "select_outdir", False):
        outdir = select_output_dir_gui() / default_subdir
    else:
        outdir = default_output_dir(source_dir, default_subdir)

    outdir.mkdir(parents=True, exist_ok=True)
    return wrf_files, outdir


def infer_source_dir_from_files(wrf_files: Sequence[Path]) -> Path | None:
    if not wrf_files:
        return None
    common = Path(Path(wrf_files[0]).parent)
    for path in wrf_files[1:]:
        while not str(path).startswith(str(common)):
            if common.parent == common:
                return Path(wrf_files[0]).parent
            common = common.parent
    return common


def default_output_dir(source_dir: Path | None, default_subdir: str) -> Path:
    if source_dir is not None:
        return source_dir / DEFAULT_POSTPROC_DIRNAME / default_subdir
    return select_output_dir_gui() / default_subdir


def extract_domain_token(filename: str) -> str | None:
    match = re.search(r"(d\d{2})", filename.lower())
    if match:
        return match.group(1)
    return None


def normalize_domain(domain: str | None) -> str | None:
    if domain is None:
        return None
    token = domain.strip().lower()
    if re.fullmatch(r"\d{2}", token):
        token = f"d{token}"
    elif re.fullmatch(r"\d", token):
        token = f"d0{token}"
    return token


def select_wrf_files_for_single_domain(
    wrf_files: Sequence[Path],
    domain: str | None = None,
) -> list[Path]:
    files = sorted(Path(path) for path in wrf_files)
    if not files:
        return []

    if domain:
        domain_token = normalize_domain(domain)
        filtered = [path for path in files if domain_token in path.name.lower()]
        if not filtered:
            raise SystemExit(
                f"No se encontraron archivos wrfout para el dominio {domain_token}."
            )
        return filtered

    domains_present = sorted(
        {
            token
            for token in (extract_domain_token(path.name) for path in files)
            if token is not None
        }
    )
    if len(domains_present) <= 1:
        return files

    # Los diagnosticos 02-04 operan sobre una sola malla. Si el usuario
    # selecciona d01/d02/d03 juntos, usamos el dominio mas fino por defecto.
    domain_token = domains_present[-1]
    return [path for path in files if domain_token in path.name.lower()]


def list_wrf_files_in_dir(run_dir: Path, domain: str | None = None) -> list[Path]:
    if not run_dir.exists():
        raise SystemExit(f"No existe la carpeta de corrida: {run_dir}")
    if not run_dir.is_dir():
        raise SystemExit(f"La ruta no es un directorio: {run_dir}")

    wrf_files = [
        path
        for path in sorted(run_dir.iterdir())
        if path.is_file()
        and path.name.startswith("wrfout")
        and path.suffix.lower() not in INVALID_WRF_SUFFIXES
    ]
    wrf_files = select_wrf_files_for_single_domain(wrf_files, domain=domain)

    if not wrf_files:
        domain_msg = f" para el dominio {domain}" if domain else ""
        raise SystemExit(
            f"No se encontraron archivos wrfout{domain_msg} en {run_dir}"
        )
    return wrf_files


def discover_run_directories(root_dir: Path) -> list[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []

    run_dirs = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        has_wrfout = any(
            path.is_file()
            and path.name.startswith("wrfout")
            and path.suffix.lower() not in INVALID_WRF_SUFFIXES
            for path in child.iterdir()
        )
        if has_wrfout:
            run_dirs.append(child)
    return run_dirs


def autodiscover_run_dir(root_dir: Path) -> Path | None:
    run_dirs = discover_run_directories(root_dir)
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs[0]


def select_run_dir_gui() -> Path:
    if tk is None or filedialog is None:
        raise SystemExit(
            "No se recibio carpeta de corrida y tkinter no esta disponible."
        )

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    directory = filedialog.askdirectory(
        title="Seleccione carpeta de corrida con archivos wrfout",
        initialdir=str(get_default_browse_dir()),
    )
    root.destroy()

    if not directory:
        raise SystemExit("No se selecciono carpeta de corrida.")
    return Path(directory).expanduser().resolve()


def select_wrf_files_gui() -> list[Path]:
    if tk is None or filedialog is None:
        raise SystemExit(
            "No se recibieron archivos wrfout y tkinter no esta disponible."
        )

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    files = filedialog.askopenfilenames(
        title="Seleccione archivos wrfout",
        initialdir=str(get_default_browse_dir()),
    )
    root.destroy()

    paths = [Path(path).expanduser().resolve() for path in files]
    paths = [
        path
        for path in paths
        if path.name.startswith("wrfout")
        and path.suffix.lower() not in INVALID_WRF_SUFFIXES
    ]
    if not paths:
        raise SystemExit("No se seleccionaron archivos wrfout validos.")
    return sorted(paths)


def select_shapefile_gui() -> Path | None:
    if tk is None or filedialog is None:
        raise SystemExit(
            "Se solicito shapefile por ventana, pero tkinter no esta disponible."
        )

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    shp = filedialog.askopenfilename(
        title="Seleccione shapefile o GeoPackage para mascara",
        initialdir=str(get_default_browse_dir()),
        filetypes=[
            ("Vector GIS", "*.shp *.gpkg"),
            ("Shapefile", "*.shp"),
            ("GeoPackage", "*.gpkg"),
            ("Todos", "*.*"),
        ],
    )
    root.destroy()
    return Path(shp).expanduser().resolve() if shp else None


def resolve_shapefile_path(args: argparse.Namespace) -> Path | None:
    shp_arg = getattr(args, "shapefile", None)
    if shp_arg:
        shp = Path(shp_arg).expanduser().resolve()
        if not shp.exists():
            raise SystemExit(f"No existe el shapefile indicado: {shp}")
        return shp

    if getattr(args, "select_shapefile", False):
        return select_shapefile_gui()

    return None


def find_pulse_source_marker(
    wrf_files: Sequence[Path],
    domain_tag: str | None = None,
    label: str = "Pulso",
) -> tuple[float, float, str] | None:
    source_dir = infer_source_dir_from_files(wrf_files)
    if source_dir is None:
        return None

    domain = normalize_domain(domain_tag) if domain_tag else None
    patterns = []
    if domain:
        patterns.extend(
            [
                f"emissions*{domain}*.txt",
                f"*emissions*{domain}*.txt",
            ]
        )
    patterns.extend(["emissions*.txt", "*emissions*.txt"])

    candidates: list[Path] = []
    for pattern in patterns:
        for path in sorted(source_dir.glob(pattern)):
            if path.is_file() and path not in candidates:
                candidates.append(path)

    best: tuple[float, float, float] | None = None
    for path in candidates:
        if domain and domain not in path.name.lower():
            continue
        source = read_emission_source_from_text(path)
        if source is None:
            continue
        lat, lon, emission = source
        if best is None or emission > best[2]:
            best = (lat, lon, emission)

    if best is None:
        return None
    return best[0], best[1], label


def read_emission_source_from_text(path: Path) -> tuple[float, float, float] | None:
    best: tuple[float, float, float] | None = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 4:
                    continue
                try:
                    lon = float(parts[1])
                    lat = float(parts[2])
                    emission = float(parts[3])
                except ValueError:
                    continue
                if not np.isfinite(emission) or emission <= 0.0:
                    continue
                if best is None or emission > best[2]:
                    best = (lat, lon, emission)
    except OSError:
        return None
    return best


def resolve_source_marker(
    args: argparse.Namespace,
    wrf_files: Sequence[Path],
    domain_tag: str | None = None,
) -> tuple[float, float, str] | None:
    label = getattr(args, "source_label", None) or "Pulso"
    source_lat = getattr(args, "source_lat", None)
    source_lon = getattr(args, "source_lon", None)
    if source_lat is not None and source_lon is not None:
        return float(source_lat), float(source_lon), label
    if source_lat is not None or source_lon is not None:
        raise SystemExit("Debe indicar --source-lat y --source-lon juntos.")
    marker = find_pulse_source_marker(wrf_files, domain_tag=domain_tag, label=label)
    if marker is not None:
        return marker
    return infer_source_marker_from_tracer(wrf_files, label=label)


def infer_source_marker_from_tracer(
    wrf_files: Sequence[Path],
    label: str = "Pulso",
) -> tuple[float, float, str] | None:
    if not wrf_files:
        return None

    try:
        with Dataset(wrf_files[0]) as nc:
            tracer_name = detect_tracer_name(nc)
            lat2d, lon2d = get_latlon_2d(nc)
            tracer = read_var(nc, tracer_name, time_index=0)
    except Exception:
        return None

    tracer = np.asarray(tracer, dtype=float)
    if tracer.ndim != 3:
        return None

    positive = np.where(np.isfinite(tracer) & (tracer > 0.0), tracer, 0.0)
    if not np.any(positive > 0.0):
        return None

    surface = positive[0, :, :]
    if np.any(surface > 0.0):
        iy, ix = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    else:
        column = np.nansum(positive, axis=0)
        if not np.any(np.isfinite(column) & (column > 0.0)):
            return None
        iy, ix = np.unravel_index(int(np.nanargmax(column)), column.shape)

    return float(lat2d[iy, ix]), float(lon2d[iy, ix]), label


def select_output_dir_gui() -> Path:
    if tk is None or filedialog is None:
        raise SystemExit(
            "No se recibio carpeta de salida y tkinter no esta disponible."
        )

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    outdir = filedialog.askdirectory(
        title="Seleccione carpeta de salida",
        initialdir=str(get_default_browse_dir()),
    )
    root.destroy()

    if not outdir:
        raise SystemExit("No se selecciono carpeta de salida.")
    return Path(outdir).expanduser().resolve()


def safe_tag(text: str | None) -> str:
    if text is None:
        return "sin_tag"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "sin_tag"


def infer_domain_tag(path: Path) -> str:
    match = re.search(r"(d\d{2})", path.name)
    if match:
        return match.group(1)
    return "dominio"


def decode_times_row(row: np.ndarray) -> str:
    chars = []
    for value in row:
        if isinstance(value, (bytes, np.bytes_)):
            chars.append(value.decode("utf-8"))
        else:
            chars.append(str(value))
    return "".join(chars)


def parse_wrf_times(nc: Dataset) -> pd.DatetimeIndex:
    if "Times" in nc.variables:
        raw = nc.variables["Times"][:]
        if raw.ndim == 2:
            times_str = [decode_times_row(row) for row in raw]
        else:
            times_str = []
            for value in raw:
                if isinstance(value, (bytes, np.bytes_)):
                    times_str.append(value.decode("utf-8"))
                elif hasattr(value, "tobytes"):
                    times_str.append(value.tobytes().decode("utf-8"))
                else:
                    times_str.append(str(value))

        times = pd.to_datetime(times_str, format="%Y-%m-%d_%H:%M:%S", errors="coerce")
        if times.isna().any():
            times = pd.to_datetime(times_str, errors="coerce")
        if times.isna().any():
            raise ValueError("No se pudieron decodificar todos los tiempos WRF.")
        return pd.DatetimeIndex(times)

    if "XTIME" in nc.variables:
        xtime = np.asarray(nc.variables["XTIME"][:], dtype=float)
        t0 = pd.Timestamp("1970-01-01")
        return pd.DatetimeIndex(t0 + pd.to_timedelta(xtime, unit="m"))

    raise ValueError("No se encontro variable 'Times' ni 'XTIME'.")


def infer_dt_seconds(times: Sequence[pd.Timestamp]) -> float:
    time_index = pd.to_datetime(list(times))
    time_index = time_index[~pd.isna(time_index)]
    if len(time_index) < 2:
        raise ValueError("No se puede inferir el paso temporal con menos de dos tiempos.")

    diffs = np.diff(time_index.values.astype("datetime64[s]").astype("int64"))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("No se pudo inferir un paso temporal positivo.")
    return float(np.median(diffs))


def _candidate_score(name: str) -> tuple[int, int, str]:
    lname = name.lower()
    priority_index = len(TRACER_PRIORITY) + 1
    for idx, item in enumerate(TRACER_PRIORITY):
        if lname == item.lower():
            priority_index = idx
            break

    hint_score = 99
    if lname.startswith("tr"):
        hint_score = 0
    if "tracer" in lname:
        hint_score = min(hint_score, 1)
    if lname.endswith("_t"):
        hint_score = min(hint_score, 2)
    if lname in {"n_slv", "s_slv", "co_t", "co"}:
        hint_score = min(hint_score, 0)

    return priority_index, hint_score, lname


def list_tracer_candidates(nc: Dataset) -> list[str]:
    candidates = []
    for name, var in nc.variables.items():
        if "Time" not in var.dimensions or len(var.dimensions) != 4:
            continue
        lname = name.lower()
        if lname in MET_4D_EXCLUSIONS:
            continue
        if (
            lname.startswith("tr")
            or "tracer" in lname
            or lname.endswith("_t")
            or lname in {"n_slv", "s_slv", "co_t", "co"}
        ):
            candidates.append(name)

    candidates.sort(key=_candidate_score)
    return candidates


def split_tracer_name_mode(
    requested: str | None,
    delta_from_initial: bool = False,
) -> tuple[str | None, bool]:
    if requested is None:
        return None, delta_from_initial

    tracer_name = requested.strip()
    if tracer_name.lower().startswith("delta_"):
        tracer_name = tracer_name[6:]
        delta_from_initial = True
    return tracer_name or None, delta_from_initial


def make_tracer_output_name(tracer_name: str, delta_from_initial: bool = False) -> str:
    return f"delta_{tracer_name}" if delta_from_initial else tracer_name


def _resolve_variable_name_case_insensitive(nc: Dataset, requested: str) -> str:
    if requested in nc.variables:
        return requested

    requested_lower = requested.lower()
    matches = [name for name in nc.variables if name.lower() == requested_lower]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"El nombre solicitado '{requested}' coincide con varias variables: "
            f"{', '.join(matches)}"
        )
    raise ValueError(
        f"El trazador solicitado '{requested}' no existe en el archivo."
    )


def detect_tracer_name(nc: Dataset, requested: str | None = None) -> str:
    if requested:
        return _resolve_variable_name_case_insensitive(nc, requested)

    candidates = list_tracer_candidates(nc)
    if not candidates:
        raise ValueError(
            "No se detecto una variable de trazador. Use --tracer-name "
            "para indicarla explicitamente."
        )
    for name in candidates:
        if tracer_has_positive_signal(nc, name, time_index=0):
            return name
    return candidates[0]


def get_latlon_2d(nc: Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "XLAT" not in nc.variables or "XLONG" not in nc.variables:
        raise ValueError("El archivo no contiene XLAT y XLONG.")

    lat = read_var(nc, "XLAT", time_index=0)
    lon = read_var(nc, "XLONG", time_index=0)
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("XLAT/XLONG no tienen forma 2D esperada.")
    return lat, lon


def _slice_for_time(var, time_index: int | None):
    if time_index is None or "Time" not in var.dimensions:
        return var[:]

    index = [slice(None)] * var.ndim
    time_axis = var.dimensions.index("Time")
    index[time_axis] = time_index
    return var[tuple(index)]


def read_var(nc: Dataset, name: str, time_index: int | None = None) -> np.ndarray:
    if name not in nc.variables:
        raise ValueError(f"La variable requerida '{name}' no existe en el archivo.")

    var = nc.variables[name]
    data = _slice_for_time(var, time_index)
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data, np.nan)
    arr = np.asarray(data, dtype=float)

    fill_value = getattr(var, "_FillValue", None)
    if fill_value is None:
        fill_value = getattr(var, "missing_value", None)
    if fill_value is not None:
        arr = np.where(np.isclose(arr, float(fill_value)), np.nan, arr)
    return arr


def tracer_has_positive_signal(
    nc: Dataset,
    name: str,
    *,
    time_index: int | None = 0,
) -> bool:
    try:
        data = read_var(nc, name, time_index=time_index)
    except Exception:
        return False
    data = np.asarray(data, dtype=float)
    return bool(np.any(np.isfinite(data) & (data > 0.0)))


def first_tracer_candidate_with_signal(
    nc: Dataset,
    *,
    exclude: Iterable[str] | None = None,
) -> str | None:
    excluded = {item.lower() for item in (exclude or [])}
    for name in list_tracer_candidates(nc):
        if name.lower() in excluded:
            continue
        if tracer_has_positive_signal(nc, name, time_index=0):
            return name
    return None


def sanitize_tracer(tracer_3d: np.ndarray) -> np.ndarray:
    tracer = np.asarray(tracer_3d, dtype=float)
    tracer = np.where(np.isfinite(tracer) & (tracer > 0.0), tracer, 0.0)
    return tracer


def load_tracer_baseline(
    wrf_files: Sequence[Path],
    tracer_name: str,
) -> np.ndarray:
    if not wrf_files:
        raise ValueError("No se recibieron wrfout para construir la linea base.")

    with Dataset(wrf_files[0]) as nc0:
        resolved_name = detect_tracer_name(nc0, requested=tracer_name)
        baseline = read_var(nc0, resolved_name, time_index=0)
    return np.asarray(baseline, dtype=float)


def read_tracer_step(
    nc: Dataset,
    tracer_name: str,
    time_index: int,
    baseline_3d: np.ndarray | None = None,
) -> np.ndarray:
    tracer = read_var(nc, tracer_name, time_index=time_index)
    if baseline_3d is not None:
        tracer = tracer - baseline_3d
    return sanitize_tracer(tracer)


def get_grid_cell_area_m2(nc: Dataset) -> float | None:
    dx = getattr(nc, "DX", None)
    dy = getattr(nc, "DY", None)
    if dx is None or dy is None:
        return None
    try:
        dx = float(dx)
        dy = float(dy)
    except Exception:
        return None
    if not np.isfinite(dx) or not np.isfinite(dy):
        return None
    return dx * dy


def compute_height_agl_and_dz(
    nc: Dataset,
    time_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    ph = read_var(nc, "PH", time_index=time_index)
    phb = read_var(nc, "PHB", time_index=time_index)
    z_stag = (ph + phb) / GRAVITY
    dz = z_stag[1:, :, :] - z_stag[:-1, :, :]
    z_mid = 0.5 * (z_stag[1:, :, :] + z_stag[:-1, :, :])

    if "HGT" in nc.variables:
        hgt = read_var(nc, "HGT", time_index=time_index)
        z_agl = z_mid - hgt[np.newaxis, :, :]
    else:
        z_agl = z_mid - z_stag[0:1, :, :]

    return z_agl, dz


def integrate_column(tracer_3d: np.ndarray, dz_3d: np.ndarray) -> np.ndarray:
    weights = sanitize_tracer(tracer_3d) * np.where(np.isfinite(dz_3d), dz_3d, 0.0)
    return np.sum(weights, axis=0)


def fraction_above_pblh(
    tracer_3d: np.ndarray,
    z_agl_3d: np.ndarray,
    dz_3d: np.ndarray,
    pblh_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = sanitize_tracer(tracer_3d) * np.where(np.isfinite(dz_3d), dz_3d, 0.0)
    total = np.sum(weights, axis=0)
    above_mask = z_agl_3d > pblh_2d[np.newaxis, :, :]
    above = np.sum(np.where(above_mask, weights, 0.0), axis=0)
    below = np.maximum(total - above, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.divide(
            above,
            total,
            out=np.full_like(total, np.nan, dtype=float),
            where=total > 0.0,
        )
    return fraction, above, below, total


def build_region_mask(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    shapefile: str | Path | None = None,
) -> tuple[np.ndarray, str]:
    values = [lat_min, lat_max, lon_min, lon_max]
    has_box = not all(value is None for value in values)

    if has_box and any(value is None for value in values):
        raise ValueError(
            "Para una region por caja debe indicar lat-min, lat-max, lon-min y lon-max."
        )

    if has_box:
        mask = (
            (lat2d >= lat_min)
            & (lat2d <= lat_max)
            & (lon2d >= lon_min)
            & (lon2d <= lon_max)
        )
        label_parts = [
            f"box_lat{lat_min:.3f}_{lat_max:.3f}_"
            f"lon{lon_min:.3f}_{lon_max:.3f}"
        ]
    else:
        mask = np.ones_like(lat2d, dtype=bool)
        label_parts = []

    if shapefile is not None:
        shp = Path(shapefile).expanduser().resolve()
        mask = mask & build_shapefile_mask(lat2d, lon2d, shp)
        label_parts.append(f"shape_{shp.stem}")

    if not np.any(mask):
        raise ValueError("La mascara espacial seleccionada no contiene celdas.")

    label = "_".join(label_parts) if label_parts else "dominio_completo"
    return mask, safe_tag(label)


def build_shapefile_mask(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    shapefile: Path,
) -> np.ndarray:
    try:
        import geopandas as gpd
    except Exception as exc:
        raise RuntimeError(
            "Se requiere geopandas para usar mascara por shapefile."
        ) from exc

    gdf = gpd.read_file(shapefile).to_crs(epsg=4326)
    if gdf.empty:
        raise ValueError(f"El shapefile no tiene geometria: {shapefile}")

    geom = gdf.geometry.union_all()
    lon = np.asarray(lon2d, dtype=float)
    lat = np.asarray(lat2d, dtype=float)

    try:
        from shapely import contains_xy

        mask = contains_xy(geom, lon, lat)
    except Exception:
        try:
            from shapely import vectorized

            mask = vectorized.contains(geom, lon, lat)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo construir mascara desde shapefile: {shapefile}"
            ) from exc

    return np.asarray(mask, dtype=bool)


def _masked_values(field_2d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    field = np.asarray(field_2d, dtype=float)
    return field[np.isfinite(field) & mask_2d]


def region_nanmean(field_2d: np.ndarray, mask_2d: np.ndarray) -> float:
    values = _masked_values(field_2d, mask_2d)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def region_nanmax(field_2d: np.ndarray, mask_2d: np.ndarray) -> float:
    values = _masked_values(field_2d, mask_2d)
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def region_nansum(field_2d: np.ndarray, mask_2d: np.ndarray) -> float:
    values = _masked_values(field_2d, mask_2d)
    if values.size == 0:
        return float("nan")
    return float(np.sum(values))


def region_nanpercentile(field_2d: np.ndarray, mask_2d: np.ndarray, q: float) -> float:
    values = _masked_values(field_2d, mask_2d)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def weighted_centroid(
    field_2d: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    mask_2d: np.ndarray,
) -> tuple[float, float]:
    weights = np.where(mask_2d, np.asarray(field_2d, dtype=float), np.nan)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, np.nan)
    total = np.nansum(weights)
    if not np.isfinite(total) or total <= 0.0:
        return float("nan"), float("nan")

    centroid_lat = float(np.nansum(lat2d * weights) / total)
    centroid_lon = float(np.nansum(lon2d * weights) / total)
    return centroid_lat, centroid_lon


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return float("nan")

    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def make_time_range_tag(times: Sequence[pd.Timestamp]) -> str:
    time_index = pd.to_datetime(list(times))
    time_index = time_index[~pd.isna(time_index)]
    if len(time_index) == 0:
        return "sin_tiempo"
    time_index = time_index.sort_values()
    return (
        f"{time_index[0]:%Y-%m-%d_%H%M}_"
        f"{time_index[-1]:%Y-%m-%d_%H%M}"
    )


def save_rows_csv(rows: Sequence[dict], outpath: Path) -> Path:
    df = pd.DataFrame(rows)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
    df.to_csv(outpath, index=False)
    return outpath


def build_gif_from_pngs(
    png_paths: Sequence[str | Path],
    outpath: str | Path,
    *,
    duration_ms: int = 500,
) -> Path | None:
    frames = [Path(path).expanduser().resolve() for path in png_paths]
    frames = [path for path in frames if path.exists() and path.suffix.lower() == ".png"]
    if len(frames) < 2:
        return None

    try:
        from PIL import Image
    except Exception as exc:
        print(f"Aviso: no se pudo generar GIF porque Pillow no esta disponible ({exc}).")
        return None

    outpath = Path(outpath).expanduser().resolve()
    images = []
    try:
        for frame in frames:
            with Image.open(frame) as img:
                images.append(img.convert("RGBA"))
        images[0].save(
            outpath,
            save_all=True,
            append_images=images[1:],
            duration=int(duration_ms),
            loop=0,
        )
    except Exception as exc:
        print(f"Aviso: no se pudo generar GIF {outpath.name} ({exc}).")
        return None
    finally:
        for image in images:
            try:
                image.close()
            except Exception:
                pass
    return outpath


def plot_shapefile_boundary(
    ax,
    shapefile: str | Path | None,
    *,
    edgecolor: str = "black",
    linewidth: float = 1.0,
    label: str = "Area de estudio",
) -> bool:
    if shapefile is None:
        return False

    try:
        import geopandas as gpd

        gdf = gpd.read_file(Path(shapefile).expanduser().resolve()).to_crs(epsg=4326)
        if gdf.empty:
            return False
        gdf.boundary.plot(
            ax=ax,
            color=edgecolor,
            linewidth=linewidth,
            label=label,
            zorder=5,
        )
        return True
    except Exception as exc:
        print(f"Aviso: no se pudo dibujar shapefile ({exc}).")
        return False


def plot_source_marker(
    ax,
    marker: tuple[float, float, str] | None,
) -> bool:
    if marker is None or not np.isfinite(marker[0]) or not np.isfinite(marker[1]):
        return False

    lat, lon, label = marker
    label = label or "Pulso"
    ax.scatter(
        lon,
        lat,
        c="black",
        s=220,
        marker="o",
        alpha=0.22,
        linewidths=0.0,
        zorder=6,
    )
    ax.scatter(
        lon,
        lat,
        c="yellow",
        edgecolors="black",
        linewidths=0.8,
        s=120,
        marker="*",
        zorder=7.5,
        label=f"{label} de emision",
    )
    ax.scatter(
        lon,
        lat,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        s=260,
        marker="o",
        zorder=7.2,
    )
    ax.text(
        lon,
        lat,
        f" {label}",
        fontsize=9,
        fontweight="bold",
        color="black",
        zorder=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
    )
    return True


def format_source_marker(marker: tuple[float, float, str] | None) -> str | None:
    if marker is None or not np.isfinite(marker[0]) or not np.isfinite(marker[1]):
        return None

    lat, lon, label = marker
    label = label or "Pulso"
    return f"{label} de emision: lat={lat:.6f}, lon={lon:.6f}"


def format_source_markers_by_domain(
    markers_by_domain: dict[str, tuple[float, float, str] | None],
) -> str | None:
    parts = []
    for domain, marker in sorted(markers_by_domain.items()):
        if marker is None or not np.isfinite(marker[0]) or not np.isfinite(marker[1]):
            continue
        lat, lon, label = marker
        label = label or "Pulso"
        parts.append(f"{domain} {label}: lat={lat:.6f}, lon={lon:.6f}")

    if not parts:
        return None
    return "Pulsos de emision: " + " | ".join(parts)


def title_with_subtitle(title: str, subtitle: str | None, width: int = 105) -> str:
    if not subtitle:
        return title
    return f"{title}\n{textwrap.fill(subtitle, width=width)}"


def _format_footer_path(label: str, path: str | Path, width: int = 150) -> str:
    resolved = Path(path).expanduser().resolve()
    line = f"{label}: {resolved}"
    return textwrap.fill(line, width=width, subsequent_indent=" " * (len(label) + 2))


def add_figure_footer(
    fig,
    *,
    script_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> bool:
    lines = []
    if script_path is not None:
        lines.append(_format_footer_path("Script", script_path))
    if output_path is not None:
        lines.append(_format_footer_path("Figura", output_path))

    if not lines:
        return False

    fig.text(
        0.01,
        0.018,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="0.25",
    )
    return True


def save_map_png(
    field_2d: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    title: str,
    cbar_label: str,
    outpath: Path,
    cmap: str = "viridis",
    marker: tuple[float, float, str] | None = None,
    shapefile: str | Path | None = None,
    subtitle: str | None = None,
    script_path: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    masked_field = np.ma.masked_invalid(field_2d)
    fig, ax = plt.subplots(figsize=(9, 7))
    pcm = ax.pcolormesh(
        lon2d,
        lat2d,
        masked_field,
        shading="auto",
        cmap=cmap,
    )
    fig.colorbar(pcm, ax=ax, label=cbar_label)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(title_with_subtitle(title, subtitle))

    has_shape = plot_shapefile_boundary(ax, shapefile)
    has_marker = plot_source_marker(ax, marker)
    if has_shape or has_marker:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    has_footer = add_figure_footer(fig, script_path=script_path, output_path=outpath)
    if has_footer:
        fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    else:
        fig.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath


def iter_wrf_steps(
    wrf_files: Sequence[Path],
) -> Iterator[tuple[Path, Dataset, int, pd.Timestamp]]:
    for path in wrf_files:
        nc = Dataset(path)
        try:
            times = parse_wrf_times(nc)
            for time_index, time_value in enumerate(times):
                yield path, nc, time_index, pd.Timestamp(time_value)
        finally:
            nc.close()


def validate_wrf_files(
    wrf_files: Sequence[Path],
    tracer_name: str | None = None,
    required_vars: Iterable[str] | None = None,
) -> tuple[str, str, list[pd.Timestamp]]:
    if not wrf_files:
        raise ValueError("No se recibieron archivos wrfout.")

    required = set(required_vars or [])
    all_times: list[pd.Timestamp] = []
    lat_shape = None
    lon_shape = None
    resolved_tracer = tracer_name
    tracer_units = "unknown"

    for path in wrf_files:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        with Dataset(path) as nc:
            if "Time" not in nc.dimensions or len(nc.dimensions["Time"]) == 0:
                raise ValueError(f"El archivo no tiene dimension Time valida: {path}")

            lat2d, lon2d = get_latlon_2d(nc)
            if lat_shape is None:
                lat_shape = lat2d.shape
                lon_shape = lon2d.shape
            elif lat_shape != lat2d.shape or lon_shape != lon2d.shape:
                raise ValueError(
                    "Los wrfout no comparten la misma malla horizontal: "
                    f"{path.name}"
                )

            resolved_tracer = detect_tracer_name(nc, requested=resolved_tracer)
            if tracer_units == "unknown":
                tracer_units = getattr(
                    nc.variables[resolved_tracer],
                    "units",
                    "unknown",
                )

            missing = [
                name
                for name in (required | {"XLAT", "XLONG", resolved_tracer})
                if name not in nc.variables
            ]
            if missing:
                raise ValueError(
                    f"Faltan variables requeridas en {path.name}: {', '.join(missing)}"
                )

            times = parse_wrf_times(nc)
            if len(times) != len(nc.dimensions["Time"]):
                raise ValueError(
                    f"La dimension Time no coincide con Times/XTIME en {path.name}"
                )
            all_times.extend(pd.Timestamp(value) for value in times)

    return resolved_tracer, tracer_units, all_times
