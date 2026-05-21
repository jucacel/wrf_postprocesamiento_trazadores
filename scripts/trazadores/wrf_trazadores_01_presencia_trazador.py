"""
Verifica la presencia de trazadores en salidas WRF y genera una figura simple.

Objetivo:
- revisar una carpeta de corrida con uno o varios `wrfout`;
- detectar variables candidatas de trazador por dominio;
- identificar el instante de mayor senal superficial en la ventana de estudio;
- exportar un panel PNG que evidencie la presencia del trazador.

Ventana de estudio por defecto, heredada de `wrf_urbano`:
- latitud: 3.35 a 3.65
- longitud: -76.55 a -76.25
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from wrf_trazadores_00_core import (
    DEFAULT_WRF_OUT_ROOT,
    INVALID_WRF_SUFFIXES,
    add_figure_footer,
    add_gui_options_cli_arguments,
    autodiscover_run_dir,
    build_region_mask,
    default_output_dir,
    extract_domain_token,
    format_source_markers_by_domain,
    get_latlon_2d,
    list_tracer_candidates,
    load_tracer_baseline,
    make_tracer_output_name,
    maybe_apply_common_gui_options,
    parse_wrf_times,
    plot_shapefile_boundary,
    plot_source_marker,
    read_tracer_step,
    resolve_shapefile_path,
    resolve_source_marker,
    safe_tag,
    select_run_dir_gui,
    split_tracer_name_mode,
    title_with_subtitle,
)


LAT_MIN_DEFAULT = 3.35
LAT_MAX_DEFAULT = 3.65
LON_MIN_DEFAULT = -76.55
LON_MAX_DEFAULT = -76.25
ZOOM_OUT = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revisa si los wrfout tienen trazadores y genera una figura simple."
    )
    add_gui_options_cli_arguments(parser)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Carpeta que contiene los wrfout. Si se omite, se abre una ventana.",
    )
    parser.add_argument(
        "--auto-run-dir",
        action="store_true",
        help=(
            "Si no se indica --run-dir, usa automaticamente la corrida mas reciente "
            "en Z_WRF_OUT en vez de abrir ventana."
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
            "Nombre explicito del trazador a revisar. "
            "Puede usar N_SLV, S_SLV, CO_T, co o delta_co."
        ),
    )
    parser.add_argument(
        "--delta-from-initial",
        action="store_true",
        help=(
            "Resta el primer tiempo y conserva la anomalia positiva del trazador. "
            "Recomendado para co; no suele ser necesario para N_SLV/S_SLV."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Carpeta de salida. Si se omite, se crea POSTPROC_TRAZADORES en la corrida.",
    )
    parser.add_argument(
        "--select-outdir",
        action="store_true",
        help="Abre una ventana para seleccionar carpeta de salida.",
    )
    parser.add_argument("--lat-min", type=float, default=LAT_MIN_DEFAULT)
    parser.add_argument("--lat-max", type=float, default=LAT_MAX_DEFAULT)
    parser.add_argument("--lon-min", type=float, default=LON_MIN_DEFAULT)
    parser.add_argument("--lon-max", type=float, default=LON_MAX_DEFAULT)
    parser.add_argument(
        "--shapefile",
        default=None,
        help="Shapefile o GeoPackage opcional para mascara espacial.",
    )
    parser.add_argument(
        "--select-shapefile",
        action="store_true",
        help="Abre una ventana para seleccionar shapefile de mascara espacial.",
    )
    parser.add_argument("--source-lat", type=float, default=None, help="Latitud del pulso.")
    parser.add_argument("--source-lon", type=float, default=None, help="Longitud del pulso.")
    parser.add_argument(
        "--source-label",
        default="Pulso",
        help="Etiqueta para el punto de emision del pulso.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Etiqueta opcional para nombres de salida.",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=1.0e-12,
        help="Umbral minimo de senal absoluta para declarar presencia.",
    )
    return parser


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = (
            autodiscover_run_dir(DEFAULT_WRF_OUT_ROOT)
            if args.auto_run_dir
            else select_run_dir_gui()
        )
        if run_dir is None:
            raise SystemExit(
                "No se pudo autodetectar una corrida en Z_WRF_OUT. Use --run-dir."
            )
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"La carpeta de corrida no existe o no es valida: {run_dir}")
    return run_dir


def list_all_wrfout(run_dir: Path) -> list[Path]:
    files = [
        path
        for path in sorted(run_dir.iterdir())
        if path.is_file()
        and path.name.startswith("wrfout")
        and path.suffix.lower() not in INVALID_WRF_SUFFIXES
    ]
    if not files:
        raise SystemExit(f"No se encontraron archivos wrfout en {run_dir}")
    return files


def group_files_by_domain(wrf_files: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in wrf_files:
        domain = extract_domain_token(path.name) or "sin_dominio"
        grouped[domain].append(path)
    return dict(sorted(grouped.items()))


def build_mask(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    args: argparse.Namespace,
    shapefile_path: Path | None,
) -> tuple[np.ndarray, str]:
    try:
        mask, label = build_region_mask(
            lat2d,
            lon2d,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            shapefile=shapefile_path,
        )
    except ValueError as exc:
        if "no contiene celdas" not in str(exc).lower():
            raise
        # En dominios gruesos el poligono urbano puede no contener centros de celda.
        # Se conserva el shapefile como referencia visual y se usa la ventana base.
        mask, label = build_region_mask(
            lat2d,
            lon2d,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            shapefile=None,
        )
        if shapefile_path is not None:
            label = f"{label}_shape_solo_referencia"
    if np.any(mask):
        return mask, label
    return np.ones_like(lat2d, dtype=bool), "dominio_completo_sin_interseccion"


def choose_active_candidate(
    nc: Dataset,
    candidates: list[str],
    time_count: int,
    mask_2d: np.ndarray,
    threshold: float,
    baseline_map: dict[str, np.ndarray],
    delta_from_initial: bool,
) -> tuple[str | None, dict[str, float]]:
    best_name = None
    best_signal = -np.inf
    diagnostics: dict[str, float] = {}

    for candidate in candidates:
        candidate_best = 0.0
        baseline_3d = baseline_map.get(candidate) if delta_from_initial else None
        for time_index in range(time_count):
            tracer = read_tracer_step(
                nc,
                candidate,
                time_index=time_index,
                baseline_3d=baseline_3d,
            )
            surface = tracer[0, :, :]
            if np.any(mask_2d):
                surface = np.where(mask_2d, surface, np.nan)
            signal = float(np.nanmax(np.abs(surface))) if np.isfinite(surface).any() else 0.0
            candidate_best = max(candidate_best, signal)
        diagnostics[candidate] = candidate_best
        if candidate_best > best_signal:
            best_signal = candidate_best
            best_name = candidate

    if best_signal <= threshold:
        return None, diagnostics
    return best_name, diagnostics


def inspect_domain(
    domain: str,
    files: list[Path],
    args: argparse.Namespace,
    shapefile_path: Path | None,
) -> dict:
    best_record = None
    lat2d_ref = None
    lon2d_ref = None
    chosen_tracer, delta_from_initial = split_tracer_name_mode(
        args.tracer_name,
        args.delta_from_initial,
    )
    tracer_units = "unknown"
    candidate_summary: dict[str, float] = {}
    mask_label = "sin_mascara"
    baseline_map: dict[str, np.ndarray] = {}
    source_marker = resolve_source_marker(args, files, domain_tag=domain)

    for path in files:
        with Dataset(path) as nc:
            lat2d, lon2d = get_latlon_2d(nc)
            lat2d_ref = lat2d
            lon2d_ref = lon2d
            mask_2d, mask_label = build_mask(lat2d, lon2d, args, shapefile_path)
            time_values = parse_wrf_times(nc)
            time_count = len(time_values)

            if chosen_tracer is None:
                candidates = list_tracer_candidates(nc)
                if delta_from_initial:
                    for candidate in candidates:
                        baseline_map.setdefault(
                            candidate,
                            load_tracer_baseline(files, candidate),
                        )
                active_candidate, diagnostics = choose_active_candidate(
                    nc,
                    candidates,
                    time_count,
                    mask_2d,
                    args.presence_threshold,
                    baseline_map,
                    delta_from_initial,
                )
                for name, value in diagnostics.items():
                    candidate_summary[name] = max(candidate_summary.get(name, 0.0), value)
                tracer_name = active_candidate
            else:
                tracer_name = chosen_tracer if chosen_tracer in nc.variables else None
                if tracer_name is None:
                    lower_map = {name.lower(): name for name in nc.variables}
                    tracer_name = lower_map.get(chosen_tracer.lower())

            if tracer_name is None:
                continue

            if delta_from_initial and tracer_name not in baseline_map:
                baseline_map[tracer_name] = load_tracer_baseline(files, tracer_name)
            tracer_units = getattr(nc.variables[tracer_name], "units", "unknown")
            tracer_display_name = make_tracer_output_name(
                tracer_name,
                delta_from_initial,
            )
            for time_index, time_value in enumerate(time_values):
                tracer = read_tracer_step(
                    nc,
                    tracer_name,
                    time_index=time_index,
                    baseline_3d=baseline_map.get(tracer_name)
                    if delta_from_initial
                    else None,
                )
                surface = tracer[0, :, :]
                region_surface = np.where(mask_2d, surface, np.nan)
                if np.isfinite(region_surface).any():
                    peak_abs = float(np.nanmax(np.abs(region_surface)))
                    peak_value = float(np.nanmax(region_surface))
                else:
                    peak_abs = 0.0
                    peak_value = float("nan")
                candidate_summary[tracer_name] = max(
                    candidate_summary.get(tracer_name, 0.0),
                    peak_abs,
                )
                if peak_abs <= args.presence_threshold:
                    continue

                peak_flat = int(np.nanargmax(np.abs(region_surface)))
                iy, ix = np.unravel_index(peak_flat, region_surface.shape)
                record = {
                    "domain": domain,
                    "file": path,
                    "time": pd.Timestamp(time_value),
                    "tracer_name": tracer_name,
                    "tracer_display_name": tracer_display_name,
                    "tracer_units": tracer_units,
                    "surface_map": surface.copy(),
                    "lat2d": lat2d,
                    "lon2d": lon2d,
                    "mask_2d": mask_2d,
                    "mask_label": mask_label,
                    "source_marker": source_marker,
                    "peak_abs": peak_abs,
                    "peak_value": peak_value,
                    "peak_lat": float(lat2d[iy, ix]),
                    "peak_lon": float(lon2d[iy, ix]),
                }
                if best_record is None or peak_abs > best_record["peak_abs"]:
                    best_record = record

    if best_record is None:
        return {
            "domain": domain,
            "status": "sin_trazador",
            "tracer_name": chosen_tracer or "no_detectado",
            "tracer_display_name": make_tracer_output_name(
                chosen_tracer or "no_detectado",
                delta_from_initial,
            )
            if chosen_tracer
            else "no_detectado",
            "tracer_units": tracer_units,
            "candidate_summary": candidate_summary,
            "lat2d": lat2d_ref,
            "lon2d": lon2d_ref,
            "mask_label": mask_label,
            "source_marker": source_marker,
            "delta_from_initial": delta_from_initial,
        }

    best_record["status"] = "trazador_detectado"
    best_record["candidate_summary"] = candidate_summary
    best_record["delta_from_initial"] = delta_from_initial
    return best_record


def build_summary_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        row = {
            "domain": result["domain"],
            "status": result["status"],
            "tracer_name": result.get("tracer_name"),
            "tracer_display_name": result.get("tracer_display_name"),
            "tracer_units": result.get("tracer_units"),
            "mask_label": result.get("mask_label"),
        }
        if result["status"] == "trazador_detectado":
            row.update(
                {
                    "time": result["time"],
                    "file": str(result["file"]),
                    "peak_abs_surface": result["peak_abs"],
                    "peak_value_surface": result["peak_value"],
                    "peak_lat": result["peak_lat"],
                    "peak_lon": result["peak_lon"],
                }
            )
        else:
            max_candidate = 0.0
            if result["candidate_summary"]:
                max_candidate = max(result["candidate_summary"].values())
            row.update(
                {
                    "time": pd.NaT,
                    "file": None,
                    "peak_abs_surface": max_candidate,
                    "peak_value_surface": np.nan,
                    "peak_lat": np.nan,
                    "peak_lon": np.nan,
                }
            )
        rows.append(row)
    return rows


def plot_domain_panels(
    results: list[dict],
    outpath: Path,
    args: argparse.Namespace,
    case_label: str,
    shapefile_path: Path | None,
) -> Path:
    import matplotlib.pyplot as plt

    ncols = min(3, max(1, len(results)))
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, result in zip(axes, results):
        domain = result["domain"]
        if result["status"] != "trazador_detectado":
            ax.set_title(f"{domain} - sin trazador detectado")
            ax.text(
                0.5,
                0.55,
                "No se detecto senal\nde trazador en superficie",
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.25,
                f"Ventana: {result.get('mask_label', 'NA')}",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        lat2d = result["lat2d"]
        lon2d = result["lon2d"]
        surface_map = result["surface_map"]
        mask_2d = result["mask_2d"]
        plot_field = np.where(mask_2d, surface_map, np.nan)
        valid_values = plot_field[np.isfinite(plot_field)]
        if valid_values.size == 0:
            plot_field = surface_map
            valid_values = plot_field[np.isfinite(plot_field)]

        vmin = float(np.nanmin(valid_values))
        vmax = float(np.nanmax(valid_values))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin < 0.0 < vmax:
            vabs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vabs, vabs
            cmap = "RdBu_r"
        else:
            cmap = "magma"

        pcm = ax.pcolormesh(
            lon2d,
            lat2d,
            np.ma.masked_invalid(plot_field),
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plot_shapefile_boundary(ax, shapefile_path)
        plot_source_marker(ax, result.get("source_marker"))
        fig.colorbar(
            pcm,
            ax=ax,
            shrink=0.88,
            label=f"{result['tracer_display_name']} ({result['tracer_units']})",
        )
        ax.scatter(result["peak_lon"], result["peak_lat"], c="cyan", s=36, marker="x")
        ax.text(
            result["peak_lon"],
            result["peak_lat"],
            f" max {result['peak_abs']:.3g}",
            fontsize=8,
            color="cyan",
        )
        ax.set_xlim(args.lon_min - ZOOM_OUT, args.lon_max + ZOOM_OUT)
        ax.set_ylim(args.lat_min - ZOOM_OUT, args.lat_max + ZOOM_OUT)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_title(
            f"{domain} | {result['tracer_display_name']} | {result['time']:%Y-%m-%d %H:%M}"
        )
        if shapefile_path is not None or result.get("source_marker") is not None:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    for ax in axes[len(results):]:
        ax.set_axis_off()

    pulse_subtitle = format_source_markers_by_domain(
        {result["domain"]: result.get("source_marker") for result in results}
    )
    fig.suptitle(
        title_with_subtitle(
            f"Presencia de trazador en superficie - {case_label}",
            pulse_subtitle,
        ),
        fontsize=14,
        y=0.98,
    )
    add_figure_footer(
        fig,
        script_path=Path(__file__).resolve(),
        output_path=outpath,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.89, bottom=0.16, wspace=0.22, hspace=0.30)
    plt.savefig(outpath, dpi=170)
    plt.close(fig)
    return outpath


def write_candidate_report(results: list[dict], outpath: Path) -> Path:
    lines = []
    for result in results:
        lines.append(f"[{result['domain']}] {result['status']}")
        candidate_summary = result.get("candidate_summary", {})
        is_delta_mode = bool(result.get("delta_from_initial", False))
        if not candidate_summary:
            lines.append("  candidatos: ninguno")
            continue
        for name, value in sorted(candidate_summary.items(), key=lambda item: item[1], reverse=True):
            lines.append(
                f"  {make_tracer_output_name(name, is_delta_mode)}: "
                f"max_abs_superficie={value:.6g}"
            )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outpath


def main() -> None:
    args = build_parser().parse_args()
    args = maybe_apply_common_gui_options(
        args,
        title="Opciones - presencia de trazador",
        include_domain=True,
        include_region_box=True,
        include_input=True,
        allow_wrf_files=False,
        include_delta_from_initial=True,
        include_presence_threshold=True,
    )
    run_dir = resolve_run_dir(args)
    shapefile_path = resolve_shapefile_path(args)
    wrf_files = list_all_wrfout(run_dir)
    grouped = group_files_by_domain(wrf_files)
    if args.domain:
        grouped = {
            domain: files
            for domain, files in grouped.items()
            if domain.lower() == args.domain.lower()
        }
        if not grouped:
            raise SystemExit(
                f"No se encontraron archivos wrfout para el dominio {args.domain} en {run_dir}"
            )

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    elif args.select_outdir:
        from wrf_trazadores_00_core import select_output_dir_gui

        outdir = select_output_dir_gui() / "wrf_trazadores_01_presencia_trazador"
    else:
        outdir = default_output_dir(run_dir, "wrf_trazadores_01_presencia_trazador")
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for domain, files in grouped.items():
        results.append(inspect_domain(domain, files, args, shapefile_path))
    results.sort(key=lambda item: item["domain"])

    case_label = safe_tag(args.tag or run_dir.name)
    csv_path = outdir / f"{case_label}_resumen_presencia_trazador.csv"
    txt_path = outdir / f"{case_label}_candidatos_trazador.txt"
    png_path = outdir / f"{case_label}_panel_presencia_trazador.png"

    summary_rows = build_summary_rows(results)
    df = pd.DataFrame(summary_rows)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df.to_csv(csv_path, index=False)
    write_candidate_report(results, txt_path)
    plot_domain_panels(results, png_path, args, run_dir.name, shapefile_path)

    print(f"Carpeta revisada: {run_dir}")
    print(f"Dominios encontrados: {', '.join(grouped.keys())}")
    print(f"CSV resumen: {csv_path}")
    print(f"TXT candidatos: {txt_path}")
    print(f"PNG panel: {png_path}")

    detected = [result for result in results if result["status"] == "trazador_detectado"]
    if detected:
        for result in detected:
            print(
                f"{result['domain']}: {result['tracer_name']} "
                f"({result['tracer_display_name']}) "
                f"detectado en {result['time']:%Y-%m-%d %H:%M} "
                f"(max_abs_superficie={result['peak_abs']:.6g})"
            )
    else:
        print("No se detectaron trazadores activos en los wrfout revisados.")


if __name__ == "__main__":
    main()
