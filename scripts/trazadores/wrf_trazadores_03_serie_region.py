"""
Serie temporal regional para diagnostico de trazadores en wrfout.

Entradas:
- uno o varios wrfout con trazador 3D;
- PH y PHB para la columna integrada.

Salidas:
- CSV con estadisticos regionales por tiempo;
- PNG con panel de series;
- resumen de desplazamiento del centroide si se indica una fuente.
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
    plot_shapefile_boundary,
    plot_source_marker,
    read_tracer_step,
    region_nanmax,
    region_nanmean,
    region_nanpercentile,
    region_nansum,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera series temporales regionales del trazador."
    )
    add_common_cli_arguments(parser)
    parser.add_argument(
        "--source-lat",
        type=float,
        default=None,
        help="Latitud de la fuente para medir distancia al centroide.",
    )
    parser.add_argument(
        "--source-lon",
        type=float,
        default=None,
        help="Longitud de la fuente para medir distancia al centroide.",
    )
    parser.add_argument(
        "--source-label",
        default="Pulso",
        help="Etiqueta para el punto de emision del pulso.",
    )
    return parser


def plot_series_panel(
    df: pd.DataFrame,
    tracer_display_name: str,
    outpath: Path,
    subtitle: str | None = None,
    script_path: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    has_source = {
        "distancia_fuente_centroide_km",
        "superficie_fuente",
        "columna_fuente",
        "dx_fuente_centroide_km",
        "dy_fuente_centroide_km",
    }.intersection(df.columns)
    nrows = 5 if has_source else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.0 * nrows + 1), sharex=True)

    axes[0].plot(df["time"], df["superficie_media_region"], marker="o", label="Superficie media")
    axes[0].plot(df["time"], df["superficie_max_region"], marker="^", label="Superficie max")
    axes[0].set_ylabel("Nivel bajo")
    axes[0].set_title(
        title_with_subtitle(
            f"Serie regional del trazador - {tracer_display_name}",
            subtitle,
        )
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["time"], df["columna_media_region"], marker="o", label="Columna media")
    axes[1].plot(df["time"], df["columna_p95_region"], marker="s", label="Columna P95")
    axes[1].plot(df["time"], df["columna_max_region"], marker="^", label="Columna max")
    axes[1].set_ylabel("Columna integrada")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    load_ax = axes[2]
    load_ax.plot(
        df["time"],
        df["carga_integrada_region"],
        color="tab:red",
        marker="o",
        label="Carga integrada",
    )
    load_ax.set_ylabel("Carga integrada")
    load_ax.legend(loc="upper left")
    load_ax.grid(alpha=0.3)
    load_ax.set_title("Carga integrada regional")

    if "superficie_fuente" in df.columns or "columna_fuente" in df.columns:
        disp_ax = axes[3]
        if "distancia_fuente_centroide_km" in df.columns:
            disp_ax.plot(
                df["time"],
                df["distancia_fuente_centroide_km"],
                color="tab:green",
                marker="s",
                label="Distancia fuente-centroide",
            )
        if "dx_fuente_centroide_km" in df.columns:
            disp_ax.plot(
                df["time"],
                df["dx_fuente_centroide_km"],
                color="tab:blue",
                marker=">",
                label="Desplazamiento E-O",
            )
        if "dy_fuente_centroide_km" in df.columns:
            disp_ax.plot(
                df["time"],
                df["dy_fuente_centroide_km"],
                color="tab:purple",
                marker="^",
                label="Desplazamiento N-S",
            )
        disp_ax.set_ylabel("Desplazamiento (km)")
        disp_ax.legend(loc="upper left")
        disp_ax.grid(alpha=0.3)
        disp_ax.set_title("Desplazamiento horizontal del centroide desde la fuente")

        source_ax = axes[4]
        if "superficie_fuente" in df.columns:
            source_ax.plot(
                df["time"],
                df["superficie_fuente"],
                color="tab:orange",
                marker="o",
                label="Superficie en fuente",
            )
        if "columna_fuente" in df.columns:
            ax_source2 = source_ax.twinx()
            ax_source2.plot(
                df["time"],
                df["columna_fuente"],
                color="tab:brown",
                marker="s",
                label="Columna en fuente",
            )
            ax_source2.set_ylabel("Columna en fuente")
            lines = source_ax.lines + ax_source2.lines
            labels = [line.get_label() for line in lines]
            source_ax.legend(lines, labels, loc="upper left")
        else:
            source_ax.legend(loc="upper left")
        source_ax.set_ylabel("Superficie en fuente")
        source_ax.grid(alpha=0.3)
        source_ax.set_title("Salida del trazador desde el punto de emision")

    axes[-1].set_xlabel("Tiempo")
    add_figure_footer(fig, script_path=script_path, output_path=outpath)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath


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


def plot_planview_track(
    df: pd.DataFrame,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    outpath: Path,
    tracer_display_name: str,
    source_marker: tuple[float, float, str] | None,
    shapefile_path: Path | None,
    subtitle: str | None = None,
    script_path: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 7.5))

    ax.set_facecolor("white")
    plot_shapefile_boundary(ax, shapefile_path)
    plot_source_marker(ax, source_marker)

    if source_marker is not None:
        sx, sy, _ = source_marker[1], source_marker[0], source_marker[2]
        for _, row in df.iterrows():
            if np.isfinite(row["centroide_lat"]) and np.isfinite(row["centroide_lon"]):
                ax.plot(
                    [sx, row["centroide_lon"]],
                    [sy, row["centroide_lat"]],
                    color="0.85",
                    linewidth=0.7,
                    zorder=2,
                )

    valid = np.isfinite(df["centroide_lat"]) & np.isfinite(df["centroide_lon"])
    track = df.loc[valid].copy()
    if not track.empty:
        ax.plot(
            track["centroide_lon"],
            track["centroide_lat"],
            color="tab:blue",
            linewidth=1.3,
            zorder=3,
            label="Trayectoria del centroide",
        )
        sc = ax.scatter(
            track["centroide_lon"],
            track["centroide_lat"],
            c=np.arange(len(track)),
            cmap="viridis",
            s=50,
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Orden temporal")
        first = track.iloc[0]
        last = track.iloc[-1]
        ax.text(first["centroide_lon"], first["centroide_lat"], " inicio", fontsize=8, color="tab:blue")
        ax.text(last["centroide_lon"], last["centroide_lat"], " fin", fontsize=8, color="tab:blue")

    extent_lon = list(track["centroide_lon"]) if not track.empty else []
    extent_lat = list(track["centroide_lat"]) if not track.empty else []
    if source_marker is not None:
        extent_lon.append(source_marker[1])
        extent_lat.append(source_marker[0])
    if not extent_lon:
        finite_lon = lon2d[np.isfinite(lon2d)]
        finite_lat = lat2d[np.isfinite(lat2d)]
        lon_min = float(np.nanpercentile(finite_lon, 2.0))
        lon_max = float(np.nanpercentile(finite_lon, 98.0))
        lat_min = float(np.nanpercentile(finite_lat, 2.0))
        lat_max = float(np.nanpercentile(finite_lat, 98.0))
    else:
        lon_min = float(np.nanmin(extent_lon))
        lon_max = float(np.nanmax(extent_lon))
        lat_min = float(np.nanmin(extent_lat))
        lat_max = float(np.nanmax(extent_lat))
    lon_pad = max(0.015, 0.35 * max(lon_max - lon_min, 0.01))
    lat_pad = max(0.015, 0.35 * max(lat_max - lat_min, 0.01))
    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(
        title_with_subtitle(
            f"Desplazamiento horizontal desde la fuente - {tracer_display_name}",
            subtitle,
        )
    )
    ax.grid(alpha=0.2)
    if track.empty:
        ax.text(0.5, 0.5, "Sin centroide valido", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    add_figure_footer(fig, script_path=script_path, output_path=outpath)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(outpath, dpi=160)
    plt.close(fig)
    return outpath


def main() -> None:
    args = build_parser().parse_args()
    args = maybe_apply_common_gui_options(
        args,
        title="Opciones - serie regional del trazador",
    )
    wrf_files, outdir = resolve_common_inputs(
        args,
        default_subdir="wrf_trazadores_03_serie_region",
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
        cell_area = get_grid_cell_area_m2(nc0)

    domain_tag = infer_domain_tag(wrf_files[0])
    source_marker = resolve_source_marker(args, wrf_files, domain_tag=domain_tag)
    source_iy = None
    source_ix = None
    if source_marker is not None:
        source_iy, source_ix = nearest_grid_index(
            lat2d,
            lon2d,
            source_marker[0],
            source_marker[1],
        )
    pulse_subtitle = format_source_marker(source_marker)
    time_tag = make_time_range_tag(all_times)
    case_tag = safe_tag(args.tag or f"{domain_tag}_{region_label}_{tracer_display_name}")

    rows = []
    for _, nc, time_index, time_value in iter_wrf_steps(wrf_files):
        tracer = read_tracer_step(
            nc,
            tracer_name,
            time_index=time_index,
            baseline_3d=baseline_3d,
        )
        surface_2d = tracer[0, :, :]
        _, dz = compute_height_agl_and_dz(nc, time_index=time_index)
        column_2d = integrate_column(tracer, dz)

        surface_mean = region_nanmean(surface_2d, region_mask)
        surface_max = region_nanmax(surface_2d, region_mask)
        column_mean = region_nanmean(column_2d, region_mask)
        column_max = region_nanmax(column_2d, region_mask)
        column_p95 = region_nanpercentile(column_2d, region_mask, 95.0)
        column_sum = region_nansum(column_2d, region_mask)
        load_proxy = column_sum * cell_area if cell_area is not None and pd.notna(column_sum) else column_sum
        centroid_lat, centroid_lon = weighted_centroid(
            column_2d,
            lat2d,
            lon2d,
            region_mask,
        )

        row = {
            "time": time_value,
            "tracer_name": tracer_display_name,
            "tracer_units": tracer_units,
            "superficie_media_region": surface_mean,
            "superficie_max_region": surface_max,
            "columna_media_region": column_mean,
            "columna_p95_region": column_p95,
            "columna_max_region": column_max,
            "columna_suma_region": column_sum,
            "carga_integrada_region": load_proxy,
            "centroide_lat": centroid_lat,
            "centroide_lon": centroid_lon,
        }
        if source_marker is not None:
            row["superficie_fuente"] = float(surface_2d[source_iy, source_ix])
            row["columna_fuente"] = float(column_2d[source_iy, source_ix])
            row["distancia_fuente_centroide_km"] = haversine_km(
                source_marker[0],
                source_marker[1],
                centroid_lat,
                centroid_lon,
            )
            dx_km, dy_km = signed_displacements_km(
                source_marker[0],
                source_marker[1],
                centroid_lat,
                centroid_lon,
            )
            row["dx_fuente_centroide_km"] = dx_km
            row["dy_fuente_centroide_km"] = dy_km
        rows.append(row)

    if not rows:
        raise SystemExit("No se generaron pasos temporales para la serie regional.")

    df = pd.DataFrame(rows).sort_values("time")
    csv_path = outdir / f"{case_tag}_serie_region_{time_tag}.csv"
    png_path = outdir / f"{case_tag}_serie_region_{time_tag}.png"
    map_path = outdir / f"{case_tag}_trayectoria_superior_{time_tag}.png"

    save_rows_csv(rows, csv_path)
    plot_series_panel(
        df,
        tracer_display_name,
        png_path,
        subtitle=pulse_subtitle,
        script_path=Path(__file__).resolve(),
    )
    plot_planview_track(
        df,
        lat2d,
        lon2d,
        map_path,
        tracer_display_name,
        source_marker,
        shapefile_path,
        subtitle=pulse_subtitle,
        script_path=Path(__file__).resolve(),
    )

    peak_row = None
    peak_series = df["carga_integrada_region"]
    if peak_series.notna().any():
        peak_idx = peak_series.idxmax()
        peak_row = df.loc[peak_idx]

    print(f"Trazador usado: {tracer_display_name} (variable base: {tracer_name})")
    print(f"Region: {region_label}")
    print(f"CSV serie: {csv_path}")
    print(f"Panel de series: {png_path}")
    print(f"Mapa trayectoria: {map_path}")
    if peak_row is not None:
        print(
            "Pico de carga integrada regional: "
            f"{pd.Timestamp(peak_row['time']):%Y-%m-%d %H:%M}"
        )
    else:
        print("Pico de carga integrada regional: sin valor valido (todo NaN)")
    if "distancia_fuente_centroide_km" in df.columns:
        print(
            "Distancia maxima fuente-centroide: "
            f"{df['distancia_fuente_centroide_km'].max():.2f} km"
        )


if __name__ == "__main__":
    main()
