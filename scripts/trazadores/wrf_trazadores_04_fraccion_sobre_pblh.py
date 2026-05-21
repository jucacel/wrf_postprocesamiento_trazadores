"""
Fraccion de columna del trazador por encima y por debajo de PBLH.

Entradas:
- uno o varios wrfout con trazador 3D;
- PH y PHB para alturas AGL;
- PBLH para separar masa sobre y bajo la capa limite.

Salidas:
- CSV por tiempo con fracciones e integrales regionales;
- PNG con series temporales;
- PNG con mapa medio de fraccion sobre PBLH;
- TXT corto con resumen del evento.
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
    compute_height_agl_and_dz,
    format_source_marker,
    fraction_above_pblh,
    get_latlon_2d,
    infer_domain_tag,
    iter_wrf_steps,
    load_tracer_baseline,
    make_tracer_output_name,
    make_time_range_tag,
    maybe_apply_common_gui_options,
    read_var,
    read_tracer_step,
    region_nanmean,
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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula la fraccion del trazador por encima de PBLH."
    )
    add_common_cli_arguments(parser)
    parser.add_argument("--source-lat", type=float, default=None, help="Latitud del pulso.")
    parser.add_argument("--source-lon", type=float, default=None, help="Longitud del pulso.")
    parser.add_argument(
        "--source-label",
        default="Pulso",
        help="Etiqueta para el punto de emision del pulso.",
    )
    return parser


def plot_fraction_series(
    df: pd.DataFrame,
    tracer_display_name: str,
    outpath: Path,
    subtitle: str | None = None,
    script_path: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(df["time"], df["fraccion_sobre_pblh_region"], marker="o", label="Sobre PBLH")
    axes[0].plot(df["time"], df["fraccion_bajo_pblh_region"], marker="s", label="Bajo PBLH")
    axes[0].set_ylabel("Fraccion")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title(
        title_with_subtitle(
            f"Destino vertical del trazador - {tracer_display_name}",
            subtitle,
        )
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        df["time"],
        df["integral_sobre_pblh_region"],
        marker="o",
        color="tab:red",
        label="Integral sobre PBLH",
    )
    axes[1].plot(
        df["time"],
        df["integral_bajo_pblh_region"],
        marker="s",
        color="tab:blue",
        label="Integral bajo PBLH",
    )
    axes[1].set_ylabel("Integral")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        df["time"],
        df["pblh_media_region_m"],
        marker="o",
        color="tab:green",
        label="PBLH media regional",
    )
    axes[2].set_ylabel("PBLH (m)")
    axes[2].set_xlabel("Tiempo")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    add_figure_footer(fig, script_path=script_path, output_path=outpath)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath


def write_summary(df: pd.DataFrame, tracer_display_name: str, outpath: Path) -> Path:
    peak_row = None
    frac_series = df["fraccion_sobre_pblh_region"]
    if frac_series.notna().any():
        peak_idx = frac_series.idxmax()
        peak_row = df.loc[peak_idx]

    lines = [
        f"Trazador: {tracer_display_name}",
        f"Inicio: {pd.Timestamp(df['time'].min()):%Y-%m-%d %H:%M}",
        f"Fin: {pd.Timestamp(df['time'].max()):%Y-%m-%d %H:%M}",
        (
            "Fraccion media sobre PBLH: "
            f"{df['fraccion_sobre_pblh_region'].mean():.4f}"
        ),
        (
            "Fraccion maxima sobre PBLH: "
            + (
                f"{peak_row['fraccion_sobre_pblh_region']:.4f} "
                f"en {pd.Timestamp(peak_row['time']):%Y-%m-%d %H:%M}"
                if peak_row is not None
                else "sin valor valido (todo NaN)"
            )
        ),
        (
            "PBLH media regional: "
            f"{df['pblh_media_region_m'].mean():.2f} m"
        ),
    ]

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outpath


def main() -> None:
    args = build_parser().parse_args()
    args = maybe_apply_common_gui_options(
        args,
        title="Opciones - fraccion sobre PBLH",
    )
    wrf_files, outdir = resolve_common_inputs(
        args,
        default_subdir="wrf_trazadores_04_fraccion_sobre_pblh",
    )
    shapefile_path = resolve_shapefile_path(args)
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
    baseline_3d = (
        load_tracer_baseline(wrf_files, tracer_name) if delta_from_initial else None
    )

    from netCDF4 import Dataset

    with Dataset(wrf_files[0]) as nc0:
        lat2d, lon2d = get_latlon_2d(nc0)
        try:
            region_mask, region_label = build_region_mask(
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
            print(
                "Aviso: la mascara espacial seleccionada no intersecta este dominio. "
                "Se usara dominio completo."
            )
            region_mask = np.ones_like(lat2d, dtype=bool)
            region_label = "dominio_completo_fallback"

    domain_tag = infer_domain_tag(wrf_files[0])
    source_marker = resolve_source_marker(args, wrf_files, domain_tag=domain_tag)
    pulse_subtitle = format_source_marker(source_marker)
    time_tag = make_time_range_tag(all_times)
    case_tag = safe_tag(args.tag or f"{domain_tag}_{region_label}_{tracer_display_name}")

    sum_fraction_map = np.zeros_like(lat2d, dtype=float)
    count_fraction_map = np.zeros_like(lat2d, dtype=float)
    peak_fraction_map = None
    peak_fraction_value = -np.inf
    peak_fraction_time = None
    rows = []

    for _, nc, time_index, time_value in iter_wrf_steps(wrf_files):
        tracer = read_tracer_step(
            nc,
            tracer_name,
            time_index=time_index,
            baseline_3d=baseline_3d,
        )
        z_agl, dz = compute_height_agl_and_dz(nc, time_index=time_index)
        pblh = read_var(nc, "PBLH", time_index=time_index)

        frac_2d, above_2d, below_2d, total_2d = fraction_above_pblh(
            tracer,
            z_agl,
            dz,
            pblh,
        )

        above_region = region_nansum(above_2d, region_mask)
        below_region = region_nansum(below_2d, region_mask)
        total_region = region_nansum(total_2d, region_mask)
        pblh_mean = region_nanmean(pblh, region_mask)

        if np.isfinite(total_region) and total_region > 0.0:
            fraction_above_region = above_region / total_region
            fraction_below_region = below_region / total_region
        else:
            fraction_above_region = float("nan")
            fraction_below_region = float("nan")

        rows.append(
            {
                "time": time_value,
                "tracer_name": tracer_display_name,
                "tracer_units": tracer_units,
                "integral_total_region": total_region,
                "integral_sobre_pblh_region": above_region,
                "integral_bajo_pblh_region": below_region,
                "fraccion_sobre_pblh_region": fraction_above_region,
                "fraccion_bajo_pblh_region": fraction_below_region,
                "pblh_media_region_m": pblh_mean,
            }
        )

        valid = np.isfinite(frac_2d)
        sum_fraction_map += np.where(valid, frac_2d, 0.0)
        count_fraction_map += valid.astype(float)

        if np.isfinite(fraction_above_region) and fraction_above_region > peak_fraction_value:
            peak_fraction_value = fraction_above_region
            peak_fraction_map = frac_2d.copy()
            peak_fraction_time = pd.Timestamp(time_value)

    if not rows:
        raise SystemExit("No se generaron pasos temporales para fraccion sobre PBLH.")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_fraction_map = np.divide(
            sum_fraction_map,
            count_fraction_map,
            out=np.full_like(sum_fraction_map, np.nan, dtype=float),
            where=count_fraction_map > 0.0,
        )
    df = pd.DataFrame(rows).sort_values("time")

    csv_path = outdir / f"{case_tag}_fraccion_pblh_{time_tag}.csv"
    png_series = outdir / f"{case_tag}_serie_fraccion_pblh_{time_tag}.png"
    png_mean_map = outdir / f"{case_tag}_mapa_fraccion_media_pblh_{time_tag}.png"
    txt_summary = outdir / f"{case_tag}_resumen_fraccion_pblh_{time_tag}.txt"

    save_rows_csv(rows, csv_path)
    plot_fraction_series(
        df,
        tracer_display_name,
        png_series,
        subtitle=pulse_subtitle,
        script_path=Path(__file__).resolve(),
    )
    save_map_png(
        mean_fraction_map,
        lat2d,
        lon2d,
        title=f"Fraccion media de {tracer_display_name} sobre PBLH",
        cbar_label="Fraccion sobre PBLH",
        outpath=png_mean_map,
        cmap="viridis",
        marker=source_marker,
        shapefile=shapefile_path,
        subtitle=pulse_subtitle,
        script_path=Path(__file__).resolve(),
    )
    write_summary(df, tracer_display_name, txt_summary)

    if peak_fraction_map is not None and peak_fraction_time is not None:
        png_peak_map = outdir / (
            f"{case_tag}_mapa_fraccion_pico_{peak_fraction_time:%Y-%m-%d_%H%M}.png"
        )
        save_map_png(
            peak_fraction_map,
            lat2d,
            lon2d,
            title=(
                f"Fraccion sobre PBLH al pico regional - "
                f"{peak_fraction_time:%Y-%m-%d %H:%M}"
            ),
            cbar_label="Fraccion sobre PBLH",
            outpath=png_peak_map,
            cmap="plasma",
            marker=source_marker,
            shapefile=shapefile_path,
            subtitle=pulse_subtitle,
            script_path=Path(__file__).resolve(),
        )

    print(f"Trazador usado: {tracer_display_name} (variable base: {tracer_name})")
    print(f"Region: {region_label}")
    print(f"CSV serie: {csv_path}")
    print(f"Serie PNG: {png_series}")
    print(f"Mapa medio: {png_mean_map}")
    print(f"Resumen TXT: {txt_summary}")
    if peak_fraction_time is not None:
        print(
            "Maxima fraccion regional sobre PBLH: "
            f"{peak_fraction_value:.4f} en {peak_fraction_time:%Y-%m-%d %H:%M}"
        )


if __name__ == "__main__":
    main()
