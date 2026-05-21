"""
Mapa y serie temporal de columna integrada del trazador.

Entradas:
- uno o varios archivos wrfout con un trazador 3D;
- PH y PHB para calcular espesores verticales.

Salidas:
- CSV con serie temporal regional de columna integrada;
- PNG con serie temporal;
- PNG con mapa de columna media;
- PNG con mapa del instante de maxima carga regional.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wrf_trazadores_00_core import (
    add_figure_footer,
    add_common_cli_arguments,
    build_region_mask,
    format_source_marker,
    get_grid_cell_area_m2,
    get_latlon_2d,
    haversine_km,
    infer_domain_tag,
    integrate_column,
    iter_wrf_steps,
    load_tracer_baseline,
    make_tracer_output_name,
    make_time_range_tag,
    maybe_apply_common_gui_options,
    read_tracer_step,
    region_nanmax,
    region_nanmean,
    region_nanpercentile,
    region_nansum,
    resolve_common_inputs,
    resolve_shapefile_path,
    resolve_source_marker,
    safe_tag,
    save_map_png,
    save_rows_csv,
    split_tracer_name_mode,
    title_with_subtitle,
    validate_wrf_files,
    weighted_centroid,
    compute_height_agl_and_dz,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula columna integrada del trazador en salidas WRF."
    )
    add_common_cli_arguments(parser)
    parser.add_argument(
        "--source-lat",
        type=float,
        default=None,
        help="Latitud opcional de la fuente para distancia al centroide.",
    )
    parser.add_argument(
        "--source-lon",
        type=float,
        default=None,
        help="Longitud opcional de la fuente para distancia al centroide.",
    )
    parser.add_argument(
        "--source-label",
        default="Pulso",
        help="Etiqueta para el punto de emision del pulso.",
    )
    return parser


def build_column_label(tracer_display_name: str, tracer_units: str) -> str:
    units = tracer_units or "unknown"
    if units.lower() == "unknown":
        return f"{tracer_display_name} integrado verticalmente"
    return f"{tracer_display_name} integrado verticalmente ({units} * m)"


def plot_timeseries(
    df: pd.DataFrame,
    tracer_display_name: str,
    outpath: Path,
    subtitle: str | None = None,
    script_path: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(df["time"], df["columna_media_region"], marker="o", label="Media")
    axes[0].plot(df["time"], df["columna_p95_region"], marker="s", label="P95")
    axes[0].plot(df["time"], df["columna_max_region"], marker="^", label="Max")
    axes[0].set_ylabel("Columna integrada")
    axes[0].set_title(
        title_with_subtitle(
            f"Serie regional de columna total - {tracer_display_name}",
            subtitle,
        )
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        df["time"],
        df["carga_integrada_region"],
        color="tab:red",
        marker="o",
        label="Carga integrada regional",
    )
    if "distancia_fuente_centroide_km" in df.columns:
        ax2 = axes[1].twinx()
        ax2.plot(
            df["time"],
            df["distancia_fuente_centroide_km"],
            color="tab:green",
            marker="s",
            label="Distancia fuente-centroide",
        )
        ax2.set_ylabel("Distancia (km)")
        lines = axes[1].lines + ax2.lines
        labels = [line.get_label() for line in lines]
        axes[1].legend(lines, labels, loc="upper left")
    else:
        axes[1].legend(loc="upper left")

    axes[1].set_ylabel("Carga integrada")
    axes[1].set_xlabel("Tiempo")
    axes[1].grid(alpha=0.3)

    add_figure_footer(fig, script_path=script_path, output_path=outpath)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath


def main() -> None:
    args = build_parser().parse_args()
    args = maybe_apply_common_gui_options(
        args,
        title="Opciones - columna total del trazador",
    )
    wrf_files, outdir = resolve_common_inputs(
        args,
        default_subdir="wrf_trazadores_02_columna_total",
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
        region_mask, region_label = build_region_mask(
            lat2d,
            lon2d,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            shapefile=shapefile_path,
        )
        cell_area_m2 = get_grid_cell_area_m2(nc0)

    domain_tag = infer_domain_tag(wrf_files[0])
    source_marker = resolve_source_marker(args, wrf_files, domain_tag=domain_tag)
    pulse_subtitle = format_source_marker(source_marker)
    time_tag = make_time_range_tag(all_times)
    case_tag = safe_tag(args.tag or f"{domain_tag}_{region_label}_{tracer_display_name}")
    column_label = build_column_label(tracer_display_name, tracer_units)

    sum_map = np.zeros_like(lat2d, dtype=float)
    count_map = np.zeros_like(lat2d, dtype=float)
    peak_map = None
    peak_metric = -np.inf
    peak_time = None
    rows = []

    for _, nc, time_index, time_value in iter_wrf_steps(wrf_files):
        tracer = read_tracer_step(
            nc,
            tracer_name,
            time_index=time_index,
            baseline_3d=baseline_3d,
        )
        _, dz = compute_height_agl_and_dz(nc, time_index=time_index)
        column_2d = integrate_column(tracer, dz)

        column_mean = region_nanmean(column_2d, region_mask)
        column_max = region_nanmax(column_2d, region_mask)
        column_p95 = region_nanpercentile(column_2d, region_mask, 95.0)
        column_sum = region_nansum(column_2d, region_mask)

        if np.isfinite(column_sum) and cell_area_m2 is not None:
            load_proxy = column_sum * cell_area_m2
        else:
            load_proxy = column_sum

        centroid_lat, centroid_lon = weighted_centroid(
            column_2d,
            lat2d,
            lon2d,
            region_mask,
        )

        row = {
            "time": time_value,
            "columna_media_region": column_mean,
            "columna_p95_region": column_p95,
            "columna_max_region": column_max,
            "columna_suma_region": column_sum,
            "carga_integrada_region": load_proxy,
            "centroide_lat": centroid_lat,
            "centroide_lon": centroid_lon,
        }
        if source_marker is not None:
            row["distancia_fuente_centroide_km"] = haversine_km(
                source_marker[0],
                source_marker[1],
                centroid_lat,
                centroid_lon,
            )
        rows.append(row)

        valid = np.isfinite(column_2d)
        sum_map += np.where(valid, column_2d, 0.0)
        count_map += valid.astype(float)

        if np.isfinite(load_proxy) and load_proxy > peak_metric:
            peak_metric = load_proxy
            peak_map = column_2d.copy()
            peak_time = pd.Timestamp(time_value)

    if not rows:
        raise SystemExit("No se generaron pasos temporales para la columna total.")

    mean_map = np.where(count_map > 0.0, sum_map / count_map, np.nan)
    df = pd.DataFrame(rows).sort_values("time")

    csv_path = outdir / f"{case_tag}_serie_columna_{time_tag}.csv"
    serie_png = outdir / f"{case_tag}_serie_columna_{time_tag}.png"
    mean_png = outdir / f"{case_tag}_mapa_columna_media_{time_tag}.png"

    save_rows_csv(rows, csv_path)
    plot_timeseries(
        df,
        tracer_display_name,
        serie_png,
        subtitle=pulse_subtitle,
        script_path=Path(__file__).resolve(),
    )
    save_map_png(
        mean_map,
        lat2d,
        lon2d,
        title=f"Columna media del trazador - {tracer_display_name}",
        cbar_label=column_label,
        outpath=mean_png,
        cmap="magma",
        marker=source_marker,
        shapefile=shapefile_path,
        subtitle=pulse_subtitle,
        script_path=Path(__file__).resolve(),
    )

    if peak_map is not None and peak_time is not None:
        peak_png = outdir / f"{case_tag}_mapa_columna_pico_{peak_time:%Y-%m-%d_%H%M}.png"
        save_map_png(
            peak_map,
            lat2d,
            lon2d,
            title=f"Columna al pico regional - {tracer_display_name} - {peak_time:%Y-%m-%d %H:%M}",
            cbar_label=column_label,
            outpath=peak_png,
            cmap="inferno",
            marker=source_marker,
            shapefile=shapefile_path,
            subtitle=pulse_subtitle,
            script_path=Path(__file__).resolve(),
        )

    print(f"Trazador usado: {tracer_display_name} (variable base: {tracer_name})")
    print(f"Archivos procesados: {len(wrf_files)}")
    print(f"Region: {region_label}")
    print(f"CSV serie: {csv_path}")
    print(f"Serie PNG: {serie_png}")
    print(f"Mapa medio: {mean_png}")
    if peak_time is not None:
        print(f"Pico regional: {peak_time:%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
