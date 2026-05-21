import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tkinter as tk
from datetime import datetime
from collections import deque
from tkinter import filedialog, simpledialog, messagebox
from pathlib import Path

# ======================================================
# PARAMETROS POR DEFECTO (Allwine y Whiteman, 1994)
# ======================================================
DEFAULT_TAU_H = 24.0
DEFAULT_THRESHOLDS_KM = {
    "S_c": 130.0,   # km, estancamiento
    "R_c": 0.6,     # adimensional, recirculacion
    "S_cv": 250.0,  # km, ventilacion
    "R_cv": 0.2     # adimensional, ventilacion
}
DEBUG = True
BASE_TAU_H = 24.0
SCALE_S_WITH_TAU = True
ADAPTIVE_CTI = True
OUTPUT_BOTH_CTI = True
ADAPTIVE_METHOD = "percentiles"
ADAPTIVE_PCTS = {"S_c": 25, "S_cv": 75, "R_c": 75, "R_cv": 25}
ADAPTIVE_MIN_SAMPLES = 30
ADAPTIVE_R_EPS = 0.05
ADAPTIVE_S_EPS_KM = 5.0
FALLBACK_MASK_NEAREST_CELLS = 4
FALLBACK_WARN_DISTANCE_KM = 50.0
CITY_COORDS = {
    "Cali": (3.443662, -76.516651),
    "Yumbo": (3.571668, -76.490915),
    "Rozo": (3.610740, -76.384813),
    "Palmira": (3.527459, -76.294686),
    "Jamundi": (3.264000, -76.540000),
}
CITY_PATTERNS = {
    "Cali": "CALI",
    "Yumbo": "YUMBO",
    "Rozo": "ROZO",
    "Palmira": "PALMIRA",
    "Jamundi": "JAMUNDI|JAMUNDÍ",
}
CITY_CODES = {
    "Cali": "76001",
    "Yumbo": "76892",
    "Palmira": "76520",
    "Jamundi": "76364",
    "Rozo": "76520",  # Rozo es corregimiento de Palmira
}


def dprint(msg):
    if DEBUG:
        print(msg)


def thresholds_for_tau(tau_h):
    factor = tau_h / BASE_TAU_H if BASE_TAU_H > 0 else 1.0
    th = dict(DEFAULT_THRESHOLDS_KM)
    if SCALE_S_WITH_TAU:
        th["S_c"] = DEFAULT_THRESHOLDS_KM["S_c"] * factor
        th["S_cv"] = DEFAULT_THRESHOLDS_KM["S_cv"] * factor
    return th


def format_cti_note(thresholds_km, label=None):
    if thresholds_km is None:
        return ""
    prefix = f"{label}: " if label else "CTI: "
    return (
        f"{prefix}S_c={thresholds_km['S_c']:.2f} km, "
        f"R_c={thresholds_km['R_c']:.2f}, "
        f"S_cv={thresholds_km['S_cv']:.2f} km, "
        f"R_cv={thresholds_km['R_cv']:.2f}"
    )


def build_title(base, loc_label=None):
    if loc_label:
        return f"{base} | {loc_label}"
    return base


def adaptive_thresholds_from_series(S_km, R, base_thresholds, label=""):
    if not ADAPTIVE_CTI:
        return base_thresholds, False
    s = np.asarray(S_km, dtype=float).ravel()
    r = np.asarray(R, dtype=float).ravel()
    mask = np.isfinite(s) & np.isfinite(r)
    n = int(np.sum(mask))
    if n < ADAPTIVE_MIN_SAMPLES:
        print(
            f"CTI adaptativo: muestras insuficientes (N={n}). "
            "Se usan umbrales fijos."
        )
        return base_thresholds, False

    s = s[mask]
    r = np.clip(r[mask], 0.0, 1.0)
    try:
        S_c = float(np.nanpercentile(s, ADAPTIVE_PCTS["S_c"]))
        S_cv = float(np.nanpercentile(s, ADAPTIVE_PCTS["S_cv"]))
        R_c = float(np.nanpercentile(r, ADAPTIVE_PCTS["R_c"]))
        R_cv = float(np.nanpercentile(r, ADAPTIVE_PCTS["R_cv"]))
    except Exception:
        print("CTI adaptativo: no se pudieron calcular percentiles. Se usan umbrales fijos.")
        return base_thresholds, False

    if not np.isfinite(S_c) or not np.isfinite(S_cv) or not np.isfinite(R_c) or not np.isfinite(R_cv):
        print("CTI adaptativo: percentiles invalidos. Se usan umbrales fijos.")
        return base_thresholds, False

    if S_cv <= S_c:
        S_cv = S_c + ADAPTIVE_S_EPS_KM
    if R_c <= R_cv:
        mid = 0.5 * (R_c + R_cv)
        R_cv = max(0.0, min(mid - ADAPTIVE_R_EPS / 2.0, 1.0))
        R_c = min(1.0, max(mid + ADAPTIVE_R_EPS / 2.0, 0.0))
        if R_c <= R_cv:
            R_cv = max(0.0, R_cv - ADAPTIVE_R_EPS)
            R_c = min(1.0, R_c + ADAPTIVE_R_EPS)

    th = dict(base_thresholds)
    th["S_c"] = S_c
    th["S_cv"] = S_cv
    th["R_c"] = R_c
    th["R_cv"] = R_cv

    label_txt = f" ({label})" if label else ""
    print(
        "CTI adaptativo"
        f"{label_txt}: S_c={S_c:.2f} km, S_cv={S_cv:.2f} km, "
        f"R_c={R_c:.2f}, R_cv={R_cv:.2f} "
        f"(P{ADAPTIVE_PCTS['S_c']}/P{ADAPTIVE_PCTS['S_cv']} S; "
        f"P{ADAPTIVE_PCTS['R_cv']}/P{ADAPTIVE_PCTS['R_c']} R, N={n})"
    )
    return th, True


def adaptive_thresholds_from_df(df, base_thresholds, label=""):
    if df is None or df.empty:
        return base_thresholds, False
    if "S_km" in df.columns:
        s_km = df["S_km"].to_numpy()
    elif "S_m" in df.columns:
        s_km = df["S_m"].to_numpy() / 1000.0
    else:
        return base_thresholds, False
    if "R" not in df.columns:
        return base_thresholds, False
    return adaptive_thresholds_from_series(s_km, df["R"].to_numpy(), base_thresholds, label=label)


def build_cti_variants(df, base_thresholds, label_prefix=""):
    variants = []
    variants.append(
        {
            "tag": "fixed",
            "label": "CTI Allwine (1994/1999)",
            "thresholds": base_thresholds,
        }
    )
    adaptive_used = False
    adaptive_thresholds = None
    if ADAPTIVE_CTI:
        adaptive_thresholds, adaptive_used = adaptive_thresholds_from_df(
            df, base_thresholds, label=label_prefix
        )
    if adaptive_used and OUTPUT_BOTH_CTI:
        variants.append(
            {
                "tag": "adaptive",
                "label": "CTI adaptativo",
                "thresholds": adaptive_thresholds,
            }
        )
    elif adaptive_used:
        variants = [
            {
                "tag": "adaptive",
                "label": "CTI adaptativo",
                "thresholds": adaptive_thresholds,
            }
        ]
    return variants


def _mask_fill_values(arr, attrs):
    out = arr.astype(float, copy=True)
    fill = attrs.get("_FillValue", None)
    if fill is None:
        fill = attrs.get("missing_value", None)
    if fill is not None:
        out = np.where(np.isclose(out, fill), np.nan, out)
    # Filtro adicional por valores físicamente imposibles
    out = np.where(np.abs(out) > 200.0, np.nan, out)
    return out


def run_tag():
    return datetime.now().strftime("%Y%m%d%H%M")


def make_point_tag(nombre, lat, lon):
    if nombre and nombre != "Manual":
        return nombre.lower()
    lat_s = f"{lat:.4f}".replace("-", "m").replace(".", "p")
    lon_s = f"{lon:.4f}".replace("-", "m").replace(".", "p")
    return f"lat{lat_s}_lon{lon_s}"

# ======================================================
# INTERFAZ BASICA
# ======================================================
def seleccionar_archivo_wrf():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    archivos = filedialog.askopenfilenames(title="Seleccione archivos wrfout")
    root.destroy()
    if not archivos:
        raise SystemExit("No se selecciono archivo wrfout.")
    dprint(f"Archivos seleccionados: {len(archivos)}")
    for a in archivos:
        dprint(f"  - {a}")
    return [Path(a) for a in archivos]


def seleccionar_shapefile():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    shp = filedialog.askopenfilename(
        title="Seleccione shapefile de municipios/ciudades",
        filetypes=[("Shapefile", "*.shp"), ("Todos", "*.*")]
    )
    root.destroy()
    if shp == "":
        raise SystemExit("No se selecciono shapefile.")
    return Path(shp)


def cargar_wrf_archivos(archivos):
    ds_list = []
    for f in archivos:
        dprint(f"Abrir: {f}")
        ds = xr.open_dataset(f, decode_times=False, mask_and_scale=False)
        times = parse_wrf_times(ds)
        if "Time" in ds.sizes and len(times) == ds.sizes["Time"]:
            ds = ds.assign_coords(Time=("Time", times))
        ds_list.append(ds)

    if len(ds_list) == 1:
        return ds_list[0], ds_list

    ds = xr.concat(
        ds_list,
        dim="Time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    )
    if "Time" in ds.coords:
        ds = ds.sortby("Time")
    dprint(f"Concatenados {len(ds_list)} archivos.")
    return ds, ds_list


def obtener_horas_disponibles(archivos):
    tiempos = []
    for f in archivos:
        ds = xr.open_dataset(f, decode_times=False, mask_and_scale=False)
        try:
            t = parse_wrf_times(ds)
        finally:
            ds.close()
        if t is None:
            continue
        t = pd.to_datetime(t, errors="coerce")
        t = t[~pd.isna(t)]
        if len(t) > 0:
            tiempos.append(pd.DatetimeIndex(t))

    if not tiempos:
        return 0.0, None, 0

    all_times = tiempos[0]
    for t in tiempos[1:]:
        all_times = all_times.union(t)
    all_times = all_times.sort_values()

    if len(all_times) < 2:
        return 0.0, None, len(all_times)

    dt_seconds = infer_dt_seconds(all_times)
    nsteps = len(all_times)
    max_hours = (nsteps * dt_seconds) / 3600.0
    return max_hours, dt_seconds, nsteps


def procesar_mapa_streaming(archivos, tau_h, dt_seconds, thresholds_km):
    ds0 = xr.open_dataset(archivos[0], decode_times=False, mask_and_scale=False)
    time_dim, ydim, xdim = _extraer_dims_u10(ds0)
    lat2d, lon2d = _get_latlon_2d(ds0)
    shape2d = ds0["U10"].isel({time_dim: 0}).values.shape
    ds0.close()

    window = int(round((tau_h * 3600.0) / dt_seconds))
    if window < 1:
        raise SystemExit("Ventana invalida para tau.")

    if window <= 1:
        print(
            "Aviso: tau equivale a un solo paso temporal. "
            "La recirculación no se puede caracterizar bien con ventana=1."
        )

    # Acumuladores globales
    count_valid = np.zeros(shape2d, dtype=np.int32)
    sum_S = np.zeros(shape2d, dtype=np.float64)
    sum_R = np.zeros(shape2d, dtype=np.float64)
    count_stag = np.zeros(shape2d, dtype=np.int32)
    count_recirc = np.zeros(shape2d, dtype=np.int32)
    count_vent = np.zeros(shape2d, dtype=np.int32)

    # Sumas en la ventana
    sum_speed = np.zeros(shape2d, dtype=np.float64)
    sum_u = np.zeros(shape2d, dtype=np.float64)
    sum_v = np.zeros(shape2d, dtype=np.float64)
    valid_count = np.zeros(shape2d, dtype=np.int16)

    buf_speed = deque(maxlen=window)
    buf_u = deque(maxlen=window)
    buf_v = deque(maxlen=window)
    buf_valid = deque(maxlen=window)

    S_c = thresholds_km["S_c"] * 1000.0
    R_c = thresholds_km["R_c"]
    S_cv = thresholds_km["S_cv"] * 1000.0
    R_cv = thresholds_km["R_cv"]

    total_steps = 0
    for f in archivos:
        ds = xr.open_dataset(f, decode_times=False, mask_and_scale=False)
        ntime = int(ds.sizes.get(time_dim, ds["U10"].shape[0]))
        for i in range(ntime):
            u = ds["U10"].isel({time_dim: i}).values
            v = ds["V10"].isel({time_dim: i}).values
            u = _mask_fill_values(u, ds["U10"].attrs)
            v = _mask_fill_values(v, ds["V10"].attrs)
            valid = np.isfinite(u) & np.isfinite(v)
            u0 = np.where(valid, u, 0.0)
            v0 = np.where(valid, v, 0.0)
            speed0 = np.where(valid, np.sqrt(u0 * u0 + v0 * v0), 0.0)
            valid_i = valid.astype(np.int8)

            if len(buf_speed) == window:
                old_speed = buf_speed.popleft()
                old_u = buf_u.popleft()
                old_v = buf_v.popleft()
                old_valid = buf_valid.popleft()
                sum_speed -= old_speed
                sum_u -= old_u
                sum_v -= old_v
                valid_count -= old_valid

            buf_speed.append(speed0)
            buf_u.append(u0)
            buf_v.append(v0)
            buf_valid.append(valid_i)
            sum_speed += speed0
            sum_u += u0
            sum_v += v0
            valid_count += valid_i

            if len(buf_speed) == window:
                full = valid_count == window
                if np.any(full):
                    S = sum_speed * dt_seconds
                    X = sum_v * dt_seconds
                    Y = sum_u * dt_seconds
                    L = np.sqrt(X ** 2 + Y ** 2)
                    denom = np.where(S > 0, S, 1.0)
                    numer = np.where(S > 0, L, 0.0)
                    R = 1.0 - (numer / denom)
                    R = np.clip(R, 0.0, 1.0)
                    mask = full & np.isfinite(S) & np.isfinite(R)
                    count_valid += mask.astype(np.int32)
                    sum_S += np.where(mask, S, 0.0)
                    sum_R += np.where(mask, R, 0.0)
                    count_stag += (mask & (S <= S_c)).astype(np.int32)
                    count_recirc += (mask & (R >= R_c)).astype(np.int32)
                    count_vent += (mask & (R <= R_cv) & (S >= S_cv)).astype(np.int32)

            total_steps += 1
            if DEBUG and total_steps % 50 == 0:
                dprint(f"Progreso mapa: {total_steps} pasos")

        ds.close()

    with np.errstate(invalid="ignore", divide="ignore"):
        S_mean_km = np.where(count_valid > 0, (sum_S / count_valid) / 1000.0, np.nan)
        R_mean = np.where(count_valid > 0, (sum_R / count_valid), np.nan)
        pct_stag = np.where(count_valid > 0, 100.0 * count_stag / count_valid, np.nan)
        pct_recirc = np.where(count_valid > 0, 100.0 * count_recirc / count_valid, np.nan)
        pct_vent = np.where(count_valid > 0, 100.0 * count_vent / count_valid, np.nan)

    ds_out = xr.Dataset(
        {
            "S_mean_km": ((ydim, xdim), S_mean_km),
            "R_mean": ((ydim, xdim), R_mean),
            "pct_stagnation": ((ydim, xdim), pct_stag),
            "pct_recirculation": ((ydim, xdim), pct_recirc),
            "pct_ventilation": ((ydim, xdim), pct_vent),
        }
    )
    ds_out = ds_out.assign_attrs({"tau_h": tau_h, "dt_seconds": dt_seconds})
    return ds_out, lat2d, lon2d


def procesar_ciudad_promedio(archivos, tau_h, thresholds_km, outdir, run_id, mask, ciudad):
    ds0 = xr.open_dataset(archivos[0], decode_times=False, mask_and_scale=False)
    time_dim, ydim, xdim = _extraer_dims_u10(ds0)
    ds0.close()

    dfs = []
    for f in archivos:
        ds = xr.open_dataset(f, decode_times=False, mask_and_scale=False)
        times = parse_wrf_times(ds)
        ntime = int(ds.sizes.get(time_dim, ds["U10"].shape[0]))
        for i in range(ntime):
            u = ds["U10"].isel({time_dim: i}).values
            v = ds["V10"].isel({time_dim: i}).values
            u = _mask_fill_values(u, ds["U10"].attrs)
            v = _mask_fill_values(v, ds["V10"].attrs)
            valid = mask & np.isfinite(u) & np.isfinite(v)
            if not np.any(valid):
                u_mean = np.nan
                v_mean = np.nan
            else:
                u_mean = float(np.nanmean(u[valid]))
                v_mean = float(np.nanmean(v[valid]))
            dfs.append({"Time": times[i], "U10": u_mean, "V10": v_mean})
        ds.close()

    df_all = pd.DataFrame(dfs)
    df_all = df_all.dropna(subset=["Time", "U10", "V10"]).sort_values("Time")
    df_all = df_all.drop_duplicates(subset=["Time"]).reset_index(drop=True)
    dprint(f"Serie ciudad promedio: {len(df_all)} registros")
    if df_all.empty:
        raise SystemExit("No hay datos validos para ciudad promedio.")

    dt_seconds = infer_dt_seconds(df_all["Time"])
    print(f"Paso temporal estimado (ciudad promedio): {dt_seconds:.1f} s")

    u_da = xr.DataArray(df_all["U10"].to_numpy(), dims=("Time",), coords={"Time": df_all["Time"].to_numpy()})
    v_da = xr.DataArray(df_all["V10"].to_numpy(), dims=("Time",), coords={"Time": df_all["Time"].to_numpy()})

    S, L, R, theta = compute_aw_indices(u_da, v_da, dt_seconds, tau_h, time_dim="Time")
    df = xr.Dataset({"S_m": S, "L_m": L, "R": R, "theta_deg": theta}).to_dataframe().reset_index()
    df["S_km"] = df["S_m"] / 1000.0
    df["L_km"] = df["L_m"] / 1000.0
    df = df.dropna(subset=["S_m", "R"])
    thresholds_used = thresholds_km
    adaptive_used = False
    if ADAPTIVE_CTI:
        thresholds_used, adaptive_used = adaptive_thresholds_from_df(
            df, thresholds_km, label=f"Ciudad promedio {ciudad}"
        )

    variants = build_cti_variants(df, thresholds_km, label_prefix=f"Ciudad promedio {ciudad}")

    for var in variants:
        th = var["thresholds"]
        label = var["label"]
        tag_cti = f"{ciudad.lower()}_avg_{run_id}_{var['tag']}"

        df_cti = df.copy()
        df_cti["stagnation"] = df_cti["S_m"] <= th["S_c"] * 1000.0
        df_cti["recirculation"] = df_cti["R"] >= th["R_c"]
        df_cti["ventilation"] = (df_cti["R"] <= th["R_cv"]) & (
            df_cti["S_m"] >= th["S_cv"] * 1000.0
        )

        out_csv = outdir / f"aw_ciudad_promedio_{tag_cti}_tau{int(tau_h)}h.csv"
        df_cti.to_csv(out_csv, index=False)
        print(f"CSV ciudad promedio guardado: {out_csv}")
        print_event_summary(df_cti, f"Ciudad promedio ({ciudad}) - {label}")
        if not df_cti.empty:
            pct_low_r = (df_cti["R"] <= th["R_cv"]).mean() * 100.0
            pct_high_s = (df_cti["S_m"] >= th["S_cv"] * 1000.0).mean() * 100.0
            pct_both = (df_cti["ventilation"]).mean() * 100.0
            print(
                f"{label} | Condiciones ventilación | %R<=R_cv: {pct_low_r:.1f} "
                f"%S>=S_cv: {pct_high_s:.1f} %Ambas: {pct_both:.1f}"
            )

        plot_point_timeseries(df_cti, th, outdir, tau_h, tag_cti, cti_label=label, loc_label=ciudad)
        plot_sr_scatter(df_cti, th, outdir, tau_h, tag_cti, cti_label=label, loc_label=ciudad)
        plot_sr_isopleth(df_cti, th, outdir, tau_h, tag_cti, cti_label=label, loc_label=ciudad)
        plot_event_timeline(df_cti, th, outdir, tau_h, tag_cti, cti_label=label, loc_label=ciudad)
        plot_daily_percentages(
            df_cti,
            outdir,
            tau_h,
            tag_cti,
            thresholds_km=th,
            cti_label=label,
            print_label=label,
            loc_label=ciudad,
        )
        plot_daily_counts(df_cti, outdir, tau_h, tag_cti, thresholds_km=th, cti_label=label, loc_label=ciudad)

    return df, variants


def procesar_ciudad_incidencia(
    archivos,
    tau_h,
    dt_seconds,
    cti_variants,
    mask,
    outdir,
    run_id,
    ciudad,
):
    ds0 = xr.open_dataset(archivos[0], decode_times=False, mask_and_scale=False)
    time_dim, ydim, xdim = _extraer_dims_u10(ds0)
    ds0.close()

    window = int(round((tau_h * 3600.0) / dt_seconds))
    if window < 1:
        raise SystemExit("Ventana invalida para tau.")

    buf_speed = deque(maxlen=window)
    buf_u = deque(maxlen=window)
    buf_v = deque(maxlen=window)
    buf_valid = deque(maxlen=window)

    sum_speed = np.zeros(mask.shape, dtype=np.float64)
    sum_u = np.zeros(mask.shape, dtype=np.float64)
    sum_v = np.zeros(mask.shape, dtype=np.float64)
    valid_count = np.zeros(mask.shape, dtype=np.int16)

    if not cti_variants:
        raise SystemExit("No hay CTI disponibles para incidencia.")

    meta = []
    for var in cti_variants:
        th = var["thresholds"]
        meta.append(
            {
                "tag": var["tag"],
                "label": var["label"],
                "thresholds": th,
                "S_c": th["S_c"] * 1000.0,
                "R_c": th["R_c"],
                "S_cv": th["S_cv"] * 1000.0,
                "R_cv": th["R_cv"],
            }
        )

    series_by_tag = {m["tag"]: [] for m in meta}

    for f in archivos:
        ds = xr.open_dataset(f, decode_times=False, mask_and_scale=False)
        times = parse_wrf_times(ds)
        ntime = int(ds.sizes.get(time_dim, ds["U10"].shape[0]))
        for i in range(ntime):
            u = ds["U10"].isel({time_dim: i}).values
            v = ds["V10"].isel({time_dim: i}).values
            u = _mask_fill_values(u, ds["U10"].attrs)
            v = _mask_fill_values(v, ds["V10"].attrs)
            valid = mask & np.isfinite(u) & np.isfinite(v)
            u0 = np.where(valid, u, 0.0)
            v0 = np.where(valid, v, 0.0)
            speed0 = np.where(valid, np.sqrt(u0 * u0 + v0 * v0), 0.0)
            valid_i = valid.astype(np.int8)

            if len(buf_speed) == window:
                sum_speed -= buf_speed.popleft()
                sum_u -= buf_u.popleft()
                sum_v -= buf_v.popleft()
                valid_count -= buf_valid.popleft()

            buf_speed.append(speed0)
            buf_u.append(u0)
            buf_v.append(v0)
            buf_valid.append(valid_i)
            sum_speed += speed0
            sum_u += u0
            sum_v += v0
            valid_count += valid_i

            if len(buf_speed) == window:
                full = valid_count == window
                denom_count = np.sum(full)
                if denom_count > 0:
                    S = sum_speed * dt_seconds
                    X = sum_v * dt_seconds
                    Y = sum_u * dt_seconds
                    L = np.sqrt(X ** 2 + Y ** 2)
                    denom_s = np.where(S > 0, S, 1.0)
                    numer = np.where(S > 0, L, 0.0)
                    R = 1.0 - (numer / denom_s)
                    R = np.clip(R, 0.0, 1.0)
                    for m in meta:
                        frac_stag = float(np.sum((S <= m["S_c"]) & full) / denom_count)
                        frac_rec = float(np.sum((R >= m["R_c"]) & full) / denom_count)
                        frac_vent = float(
                            np.sum((R <= m["R_cv"]) & (S >= m["S_cv"]) & full) / denom_count
                        )
                        series_by_tag[m["tag"]].append(
                            {
                                "Time": times[i],
                                "stagnation": frac_stag,
                                "recirculation": frac_rec,
                                "ventilation": frac_vent,
                            }
                        )
                else:
                    for m in meta:
                        series_by_tag[m["tag"]].append(
                            {
                                "Time": times[i],
                                "stagnation": np.nan,
                                "recirculation": np.nan,
                                "ventilation": np.nan,
                            }
                        )
        ds.close()

    outputs = {}
    for m in meta:
        series = series_by_tag[m["tag"]]
        df = pd.DataFrame(series)
        df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
        if df.empty:
            print(f"Sin datos para incidencia ({m['label']}).")
            continue

        tag = f"{ciudad.lower()}_inc_{run_id}_{m['tag']}"
        out_csv = outdir / f"aw_ciudad_incidencia_{tag}_tau{int(tau_h)}h.csv"
        df.to_csv(out_csv, index=False)
        print(f"CSV ciudad incidencia guardado: {out_csv}")
        print_event_summary(df, f"Ciudad incidencia ({ciudad}) - {m['label']}")

        # Graficas de porcentajes diarios (area)
        df_pct = df.copy()
        df_pct["stagnation"] = df_pct["stagnation"] * 100.0
        df_pct["recirculation"] = df_pct["recirculation"] * 100.0
        df_pct["ventilation"] = df_pct["ventilation"] * 100.0
        plot_daily_percentages(
            df_pct,
            outdir,
            tau_h,
            tag,
            thresholds_km=m["thresholds"],
            cti_label=m["label"],
            print_label=m["label"],
            loc_label=ciudad,
        )
        plot_daily_counts(
            df,
            outdir,
            tau_h,
            tag,
            thresholds_km=m["thresholds"],
            cti_label=m["label"],
            loc_label=ciudad,
        )
        outputs[m["tag"]] = df
    return outputs


def _extraer_dims_u10(ds):
    u_dims = ds["U10"].dims
    if "Time" in u_dims:
        time_dim = "Time"
    else:
        time_dim = u_dims[0]
    if len(u_dims) < 3:
        raise ValueError("U10 no tiene dimensiones espaciales esperadas.")
    ydim = u_dims[-2]
    xdim = u_dims[-1]
    return time_dim, ydim, xdim


def procesar_punto(archivos, tau_h, outdir, run_id, thresholds_km):
    ds0 = xr.open_dataset(archivos[0], decode_times=False, mask_and_scale=False)
    if "XLAT" not in ds0.variables or "XLONG" not in ds0.variables:
        ds0.close()
        raise ValueError("Se requieren XLAT y XLONG para modo punto.")

    time_dim, ydim, xdim = _extraer_dims_u10(ds0)
    lat2d = ds0["XLAT"]
    lon2d = ds0["XLONG"]
    if time_dim in lat2d.dims:
        lat2d = lat2d.isel({time_dim: 0})
    if time_dim in lon2d.dims:
        lon2d = lon2d.isel({time_dim: 0})
    lat2d = lat2d.values
    lon2d = lon2d.values

    lat_pt, lon_pt, nombre_pt = seleccionar_punto()
    point_tag = make_point_tag(nombre_pt, lat_pt, lon_pt)
    tag = f"{point_tag}_{run_id}"
    print(f"Punto seleccionado: {nombre_pt} (lat={lat_pt:.5f}, lon={lon_pt:.5f})")
    iy, ix = nearest_ij(lat2d, lon2d, lat_pt, lon_pt)
    valid_mask = np.isfinite(ds0["U10"].isel({time_dim: 0}).values) & np.isfinite(
        ds0["V10"].isel({time_dim: 0}).values
    )
    if not valid_mask[iy, ix]:
        alt = nearest_valid_ij(lat2d, lon2d, valid_mask, lat_pt, lon_pt)
        if alt is not None:
            iy, ix = alt
            print(
                "Aviso: punto sin datos validos. "
                f"Se usa celda valida mas cercana (iy={iy}, ix={ix})."
            )
        else:
            print("Aviso: no hay celdas validas cercanas. Resultados pueden ser vacios.")
    dprint(f"Celda usada: iy={iy}, ix={ix} | lat={lat2d[iy, ix]:.5f}, lon={lon2d[iy, ix]:.5f}")
    ds0.close()

    dfs = []
    for f in archivos:
        ds = xr.open_dataset(f, decode_times=False, mask_and_scale=False)
        times = parse_wrf_times(ds)
        if len(times) != ds.sizes.get(time_dim, len(times)):
            dprint(f"Aviso: Times no coincide en {f}")
        u = ds["U10"].isel({ydim: iy, xdim: ix}).values
        v = ds["V10"].isel({ydim: iy, xdim: ix}).values
        u = _mask_fill_values(u, ds["U10"].attrs)
        v = _mask_fill_values(v, ds["V10"].attrs)
        if DEBUG:
            dprint(
                f"{Path(f).name}: n={len(times)} "
                f"U10 fin={np.isfinite(u).sum()}/{len(u)} "
                f"V10 fin={np.isfinite(v).sum()}/{len(v)}"
            )
        df = pd.DataFrame({"Time": times, "U10": u, "V10": v})
        dfs.append(df)
        ds.close()

    if not dfs:
        raise SystemExit("No se pudieron extraer series del punto.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["Time"])
    df_all = df_all.dropna(subset=["U10", "V10"])
    df_all = df_all.sort_values("Time").drop_duplicates(subset=["Time"]).reset_index(drop=True)
    dprint(f"Serie total: {len(df_all)} registros")

    dt_seconds = infer_dt_seconds(df_all["Time"])
    print(f"Paso temporal estimado: {dt_seconds:.1f} s ({dt_seconds/60.0:.2f} min)")

    df_idx = df_all.set_index("Time").sort_index()
    freq = f"{int(round(dt_seconds))}S"
    full_time = pd.date_range(df_idx.index.min(), df_idx.index.max(), freq=freq)
    if len(full_time) != len(df_idx):
        dprint("Reindexando a grilla temporal regular.")
        df_idx = df_idx.reindex(full_time)
    if df_idx["U10"].isna().any() or df_idx["V10"].isna().any():
        dprint("Interpolando valores faltantes en U10/V10.")
        df_idx["U10"] = df_idx["U10"].interpolate(method="time", limit_direction="both")
        df_idx["V10"] = df_idx["V10"].interpolate(method="time", limit_direction="both")
    df_all = df_idx.reset_index().rename(columns={"index": "Time"})

    u_da = xr.DataArray(
        df_all["U10"].to_numpy(),
        dims=("Time",),
        coords={"Time": df_all["Time"].to_numpy()},
    )
    v_da = xr.DataArray(
        df_all["V10"].to_numpy(),
        dims=("Time",),
        coords={"Time": df_all["Time"].to_numpy()},
    )

    ntime = int(u_da.sizes.get("Time", u_da.shape[0]))
    window = int(round((tau_h * 3600.0) / dt_seconds))
    dprint(f"Timesteps: {ntime} | Ventana requerida: {window}")
    if window <= 1:
        print(
            "Aviso: tau equivale a un solo paso temporal. "
            "La recirculación no se puede caracterizar bien con ventana=1."
        )
    if ntime < window:
        raise SystemExit(
            f"No hay suficientes timesteps ({ntime}) para tau={tau_h} h. "
            "Use un tau menor o seleccione/concatene mas archivos."
        )

    S, L, R, theta = compute_aw_indices(u_da, v_da, dt_seconds, tau_h, time_dim="Time")

    df = xr.Dataset(
        {"S_m": S, "L_m": L, "R": R, "theta_deg": theta}
    ).to_dataframe().reset_index()
    df["S_km"] = df["S_m"] / 1000.0
    df["L_km"] = df["L_m"] / 1000.0
    df = df.dropna(subset=["S_m", "R"])
    dprint(f"Filas validas: {len(df)}")
    if df.empty:
        print("Aviso: no hay datos validos en el punto. Revise lat/lon o tau.")
    if DEBUG and not df.empty:
        dprint(f"S_m min/max: {df['S_m'].min():.1f} / {df['S_m'].max():.1f}")
        dprint(f"R min/max: {df['R'].min():.3f} / {df['R'].max():.3f}")

    variants = build_cti_variants(df, thresholds_km, label_prefix=f"Punto {nombre_pt}")
    loc_label = nombre_pt if nombre_pt != "Manual" else f"Manual ({lat_pt:.3f}, {lon_pt:.3f})"

    for var in variants:
        th = var["thresholds"]
        label = var["label"]
        tag_cti = f"{tag}_{var['tag']}"

        df_cti = df.copy()
        df_cti["stagnation"] = df_cti["S_m"] <= th["S_c"] * 1000.0
        df_cti["recirculation"] = df_cti["R"] >= th["R_c"]
        df_cti["ventilation"] = (df_cti["R"] <= th["R_cv"]) & (
            df_cti["S_m"] >= th["S_cv"] * 1000.0
        )

        out_csv = outdir / f"aw_indices_punto_{tag_cti}_tau{int(tau_h)}h.csv"
        df_cti.to_csv(out_csv, index=False)
        print(f"CSV guardado: {out_csv}")
        print(
            "Columnas CSV: Time, S_m, L_m, R, theta_deg, S_km, L_km, "
            "stagnation, recirculation, ventilation"
        )

        pct_s = 100.0 * df_cti["stagnation"].mean()
        pct_r = 100.0 * df_cti["recirculation"].mean()
        pct_v = 100.0 * df_cti["ventilation"].mean()
        print(f"{label} | Estancamiento: {pct_s:.1f}%")
        print(f"{label} | Recirculacion: {pct_r:.1f}%")
        print(f"{label} | Ventilacion  : {pct_v:.1f}%")

        plot_point_timeseries(df_cti, th, outdir, tau_h, tag_cti, cti_label=label, loc_label=loc_label)
        plot_sr_scatter(df_cti, th, outdir, tau_h, tag_cti, cti_label=label, loc_label=loc_label)
        plot_sr_isopleth(df_cti, th, outdir, tau_h, tag_cti, cti_label=label, loc_label=loc_label)
        plot_event_timeline(df_cti, th, outdir, tau_h, tag_cti, cti_label=label, loc_label=loc_label)
        plot_daily_percentages(
            df_cti,
            outdir,
            tau_h,
            tag_cti,
            thresholds_km=th,
            cti_label=label,
            print_label=label,
            loc_label=loc_label,
        )


def seleccionar_modo():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    modo = simpledialog.askstring(
        "Modo de salida",
        "Escriba 'punto', 'ciudad' o 'mapa' (default: mapa):"
    )
    root.destroy()
    if not modo:
        return "mapa"
    modo = modo.strip().lower()
    if modo not in {"punto", "ciudad", "mapa"}:
        messagebox.showwarning("Modo invalido", "Se usara el modo 'mapa'.")
        return "mapa"
    return modo


def solicitar_float(titulo, mensaje, default):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    valor = simpledialog.askfloat(titulo, mensaje, initialvalue=default)
    root.destroy()
    if valor is None:
        return default
    return float(valor)


def solicitar_latlon():
    lat = solicitar_float("Latitud", "Latitud (grados decimales):", 3.45)
    lon = solicitar_float("Longitud", "Longitud (grados decimales):", -76.5)
    return lat, lon


def seleccionar_tau_disponible(opciones, max_horas):
    root = tk.Tk()
    root.title("Seleccionar tau")
    root.attributes("-topmost", True)

    tk.Label(
        root,
        text=f"Horas disponibles aprox.: {max_horas:.1f} h\nSeleccione tau (en horas):",
    ).pack(padx=10, pady=5)

    lb = tk.Listbox(
        root,
        selectmode=tk.SINGLE,
        width=12,
        height=min(12, len(opciones)),
        exportselection=False,
    )
    for v in opciones:
        lb.insert(tk.END, f"{v} h")
    lb.pack(padx=10, pady=5)

    seleccion = {"valor": None}

    def aceptar():
        idxs = lb.curselection()
        if not idxs:
            messagebox.showwarning("Sin seleccion", "Seleccione un valor de tau.")
            return
        seleccion["valor"] = float(opciones[idxs[0]])
        root.destroy()

    btn = tk.Button(root, text="Aceptar", command=aceptar)
    btn.pack(pady=(0, 10))
    root.mainloop()

    if seleccion["valor"] is None:
        raise SystemExit("No se selecciono tau.")
    return seleccion["valor"]


def seleccionar_punto():
    opciones = ["Manual (lat/lon)"] + list(CITY_COORDS.keys())
    root = tk.Tk()
    root.title("Seleccionar punto")
    root.attributes("-topmost", True)

    tk.Label(root, text="Seleccione un punto:").pack(padx=10, pady=5)
    lb = tk.Listbox(
        root,
        selectmode=tk.SINGLE,
        width=28,
        height=len(opciones),
        exportselection=False,
    )
    for o in opciones:
        lb.insert(tk.END, o)
    lb.pack(padx=10, pady=5)

    seleccion = {"valor": None}

    def aceptar():
        idxs = lb.curselection()
        if not idxs:
            messagebox.showwarning("Sin seleccion", "Seleccione un punto.")
            return
        seleccion["valor"] = opciones[idxs[0]]
        root.destroy()

    btn = tk.Button(root, text="Aceptar", command=aceptar)
    btn.pack(pady=(0, 10))
    root.mainloop()

    if seleccion["valor"] is None:
        raise SystemExit("No se selecciono un punto.")

    if seleccion["valor"].startswith("Manual"):
        lat, lon = solicitar_latlon()
        return lat, lon, "Manual"

    nombre = seleccion["valor"]
    lat, lon = CITY_COORDS[nombre]
    return lat, lon, nombre


def seleccionar_ciudad():
    opciones = list(CITY_COORDS.keys())
    root = tk.Tk()
    root.title("Seleccionar ciudad")
    root.attributes("-topmost", True)
    tk.Label(root, text="Seleccione una ciudad/municipio:").pack(padx=10, pady=5)
    lb = tk.Listbox(
        root,
        selectmode=tk.SINGLE,
        width=22,
        height=len(opciones),
        exportselection=False,
    )
    for o in opciones:
        lb.insert(tk.END, o)
    lb.pack(padx=10, pady=5)

    seleccion = {"valor": None}

    def aceptar():
        idxs = lb.curselection()
        if not idxs:
            messagebox.showwarning("Sin seleccion", "Seleccione una ciudad.")
            return
        seleccion["valor"] = opciones[idxs[0]]
        root.destroy()

    btn = tk.Button(root, text="Aceptar", command=aceptar)
    btn.pack(pady=(0, 10))
    root.mainloop()

    if seleccion["valor"] is None:
        raise SystemExit("No se selecciono ciudad.")
    return seleccion["valor"]


def _score_columna(gdf, col):
    score = 0
    name = col.upper()
    if any(k in name for k in ["NOMBRE", "NAME", "MPNOMBRE", "NOM_MPIO", "NOM_MUN", "MUNICIP"]):
        score += 5
    try:
        vals = gdf[col].astype(str).str.upper()
    except Exception:
        return score
    if vals.str.contains("CALI|YUMBO|PALM|ROZO|JAMUND", regex=True, na=False).any():
        score += 10
    codes = vals.str.replace(r"\D", "", regex=True)
    if any((codes == c).any() for c in CITY_CODES.values()):
        score += 3
    return score


def seleccionar_columna_shp(gdf):
    columnas = list(gdf.columns)
    scores = {c: _score_columna(gdf, c) for c in columnas}
    mejor = max(scores, key=scores.get)
    return mejor


def solicitar_si_no(titulo, mensaje):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    resp = messagebox.askyesno(titulo, mensaje)
    root.destroy()
    return bool(resp)


# ======================================================
# TIEMPOS WRF
# ======================================================
def parse_wrf_times(ds):
    if "Times" in ds.variables:
        times_raw = ds["Times"].values
        times_str = []
        if times_raw.ndim == 2:
            for row in times_raw:
                chars = []
                for c in row:
                    if isinstance(c, (bytes, np.bytes_)):
                        chars.append(c.decode("utf-8"))
                    else:
                        chars.append(str(c))
                times_str.append("".join(chars))
        else:
            for t in times_raw:
                if isinstance(t, (bytes, np.bytes_)):
                    times_str.append(t.decode("utf-8"))
                elif hasattr(t, "tobytes"):
                    times_str.append(t.tobytes().decode("utf-8"))
                else:
                    times_str.append(str(t))
        times = pd.to_datetime(
            times_str, format="%Y-%m-%d_%H:%M:%S", errors="coerce"
        )
        if times.isna().any():
            times = pd.to_datetime(times_str, errors="coerce")
        return times

    if "XTIME" in ds.variables:
        xtime = ds["XTIME"].values.astype(float)
        t0 = pd.Timestamp("1970-01-01")
        return t0 + pd.to_timedelta(xtime, unit="m")

    raise ValueError("No se encontro variable 'Times' ni 'XTIME'.")


def infer_dt_seconds(times):
    times = pd.to_datetime(times)
    times = times[~pd.isna(times)]
    tvals = times.values.astype("datetime64[s]").astype("int64")
    diffs = np.diff(tvals)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("No se pudo inferir el paso temporal.")
    return float(np.median(diffs))


# ======================================================
# INDICES ALLWINE-WHITEMAN
# ======================================================
def compute_aw_indices(u, v, dt_seconds, tau_hours, time_dim="Time"):
    tau_seconds = tau_hours * 3600.0
    window_float = tau_seconds / dt_seconds
    window = int(round(window_float))
    if window < 1:
        raise ValueError("tau_hours demasiado pequeno para el paso temporal.")
    if not np.isclose(window_float, window, rtol=0, atol=1e-6):
        mismatch = abs(tau_seconds - window * dt_seconds)
        tol = max(0.02 * tau_seconds, 120.0)  # 2% o 120 s
        if mismatch > tol:
            raise ValueError(
                f"tau_hours ({tau_hours}) no es multiplo del paso temporal."
            )
        dprint(
            "Aviso: tau no es multiplo exacto de dt. "
            f"Se ajusta dt a {tau_seconds / window:.2f} s (error {mismatch:.1f} s)."
        )
        dt_seconds = tau_seconds / window

    speed = np.sqrt(u ** 2 + v ** 2)

    sum_speed = (
        speed.rolling({time_dim: window}, min_periods=window)
        .sum()
        .shift({time_dim: -(window - 1)})
    )
    sum_v = (
        v.rolling({time_dim: window}, min_periods=window)
        .sum()
        .shift({time_dim: -(window - 1)})
    )
    sum_u = (
        u.rolling({time_dim: window}, min_periods=window)
        .sum()
        .shift({time_dim: -(window - 1)})
    )

    S = sum_speed * dt_seconds
    X = sum_v * dt_seconds
    Y = sum_u * dt_seconds
    L = np.sqrt(X ** 2 + Y ** 2)

    # evitar divisiones por cero sin generar warnings
    denom = xr.where(S > 0, S, 1.0)
    numer = xr.where(S > 0, L, 0.0)
    R = 1.0 - (numer / denom)
    R = R.clip(min=0.0, max=1.0)
    theta = (np.degrees(np.arctan2(Y, X)) + 360.0) % 360.0

    return S, L, R, theta


def classify_aw(S, R, thresholds_km, time_dim="Time"):
    S_c = thresholds_km["S_c"] * 1000.0
    R_c = thresholds_km["R_c"]
    S_cv = thresholds_km["S_cv"] * 1000.0
    R_cv = thresholds_km["R_cv"]

    valid = np.isfinite(S) & np.isfinite(R)

    stagnation = (S <= S_c) & valid
    recirculation = (R >= R_c) & valid
    ventilation = (R <= R_cv) & (S >= S_cv) & valid

    count_valid = valid.sum(dim=time_dim)
    pct_stag = xr.where(count_valid > 0, 100.0 * stagnation.sum(dim=time_dim) / count_valid, np.nan)
    pct_recirc = xr.where(count_valid > 0, 100.0 * recirculation.sum(dim=time_dim) / count_valid, np.nan)
    pct_vent = xr.where(count_valid > 0, 100.0 * ventilation.sum(dim=time_dim) / count_valid, np.nan)

    return stagnation, recirculation, ventilation, pct_stag, pct_recirc, pct_vent


def nearest_ij(lat2d, lon2d, lat, lon):
    dist2 = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
    idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return idx


def nearest_valid_ij(lat2d, lon2d, valid_mask, lat, lon):
    if valid_mask is None or not np.any(valid_mask):
        return None
    dist2 = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
    dist2 = np.where(valid_mask, dist2, np.inf)
    if not np.isfinite(dist2).any():
        return None
    return np.unravel_index(np.nanargmin(dist2), dist2.shape)


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def mask_nearest_cells(lat2d, lon2d, lat0, lon0, k):
    dist2 = (lat2d - lat0) ** 2 + (lon2d - lon0) ** 2
    flat_idx = np.argsort(dist2, axis=None)[:k]
    mask = np.zeros(lat2d.shape, dtype=bool)
    iy, ix = np.unravel_index(flat_idx, lat2d.shape)
    mask[iy, ix] = True
    # distancia al más cercano
    iy0, ix0 = iy[0], ix[0]
    dist_km = float(haversine_km(lon0, lat0, lon2d[iy0, ix0], lat2d[iy0, ix0]))
    return mask, dist_km


def fallback_mask_ciudad(lat2d, lon2d, ciudad):
    if ciudad not in CITY_COORDS:
        return None
    lat0, lon0 = CITY_COORDS[ciudad]
    k = max(1, int(FALLBACK_MASK_NEAREST_CELLS))
    mask, dist_km = mask_nearest_cells(lat2d, lon2d, lat0, lon0, k)
    print(
        f"Aviso: máscara ciudad vacía. "
        f"Se usan {k} celdas más cercanas al centro de {ciudad} "
        f"(distancia ~{dist_km:.1f} km)."
    )
    if dist_km > FALLBACK_WARN_DISTANCE_KM:
        print(
            "Aviso: el centro de la ciudad está lejos del dominio WRF. "
            "Revise dominio o shapefile."
        )
    return mask


def find_time_column(df):
    if "Time" in df.columns:
        return "Time"
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None


def plot_point_timeseries(df, thresholds_km, outdir, tau_h, tag, cti_label=None, loc_label=None):
    dfp = df.dropna(subset=["S_km", "R"])
    if dfp.empty:
        print("No hay datos validos para graficar.")
        return

    time_col = find_time_column(dfp)
    if time_col is None:
        print("No se encontro columna Time para graficar.")
        return

    s_c = thresholds_km["S_c"]
    r_c = thresholds_km["R_c"]
    s_cv = thresholds_km["S_cv"]
    r_cv = thresholds_km["R_cv"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(dfp[time_col], dfp["S_km"], color="tab:blue", linewidth=1.2)
    axes[0].axhline(s_c, color="tab:red", linestyle="--", label=f"S_c={s_c} km")
    axes[0].axhline(s_cv, color="tab:green", linestyle="--", label=f"S_cv={s_cv} km")
    axes[0].set_ylabel("S (km)")
    axes[0].set_title(build_title(f"Serie temporal S y R (tau={tau_h} h)", loc_label))
    axes[0].legend(loc="upper right")

    axes[1].plot(dfp[time_col], dfp["R"], color="tab:purple", linewidth=1.2)
    axes[1].axhline(r_c, color="tab:red", linestyle="--", label=f"R_c={r_c}")
    axes[1].axhline(r_cv, color="tab:green", linestyle="--", label=f"R_cv={r_cv}")
    axes[1].set_ylabel("R")
    axes[1].set_xlabel("Tiempo")
    axes[1].legend(loc="upper right")

    fig.text(
        0.5,
        0.03,
        "Siglas: S=wind run (km), R=factor de recirculación; "
        "S_c/R_c umbrales de estancamiento/recirculación; "
        "S_cv/R_cv umbrales de ventilación.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    cti_note = format_cti_note(thresholds_km, cti_label)
    if cti_note:
        fig.text(0.5, 0.01, cti_note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out_png = outdir / f"aw_timeseries_{tag}_tau{int(tau_h)}h.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Grafica guardada: {out_png}")


def plot_sr_scatter(df, thresholds_km, outdir, tau_h, tag, cti_label=None, loc_label=None):
    dfp = df.dropna(subset=["S_km", "R"])
    if dfp.empty:
        print("No hay datos validos para grafica S-R.")
        return

    s_c = thresholds_km["S_c"]
    r_c = thresholds_km["R_c"]
    s_cv = thresholds_km["S_cv"]
    r_cv = thresholds_km["R_cv"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(dfp["S_km"], dfp["R"], s=10, alpha=0.5, color="tab:blue", edgecolors="none")
    ax.axvline(s_c, color="tab:red", linestyle="--", label=f"S_c={s_c} km")
    ax.axvline(s_cv, color="tab:green", linestyle="--", label=f"S_cv={s_cv} km")
    ax.axhline(r_c, color="tab:red", linestyle=":", label=f"R_c={r_c}")
    ax.axhline(r_cv, color="tab:green", linestyle=":", label=f"R_cv={r_cv}")
    ax.set_xlabel("S (km)")
    ax.set_ylabel("R")
    ax.set_title(build_title(f"Diagrama S-R (tau={tau_h} h)", loc_label))
    ax.text(
        0.02,
        0.02,
        "S: wind run (km) | R: factor de recirculación",
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, linewidth=0),
    )
    fig.text(
        0.5,
        0.03,
        "Siglas: S_c umbral de estancamiento, R_c umbral de recirculación, "
        "S_cv/R_cv umbrales de ventilación.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    cti_note = format_cti_note(thresholds_km, cti_label)
    if cti_note:
        fig.text(0.5, 0.01, cti_note, ha="center", va="bottom", fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out_png = outdir / f"aw_scatter_SR_{tag}_tau{int(tau_h)}h.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Grafica guardada: {out_png}")


def plot_sr_isopleth(df, thresholds_km, outdir, tau_h, tag, cti_label=None, loc_label=None):
    dfp = df.dropna(subset=["S_km", "R"])
    if dfp.empty:
        print("No hay datos validos para isopletas S-R.")
        return

    s = dfp["S_km"].to_numpy()
    r = dfp["R"].to_numpy()
    mask = np.isfinite(s) & np.isfinite(r)
    s = s[mask]
    r = r[mask]
    if s.size < 10:
        print("Muy pocos datos para isopletas S-R.")
        return

    s_step = 25.0
    s_max = np.nanmax(s)
    s_max_bin = max(s_step, np.ceil(s_max / s_step) * s_step)
    s_bins = np.arange(0.0, s_max_bin + s_step, s_step)

    r_step = 0.05
    r_bins = np.arange(0.0, 1.0 + r_step, r_step)

    hist, s_edges, r_edges = np.histogram2d(s, r, bins=[s_bins, r_bins])
    total = hist.sum()
    if total == 0:
        print("Histograma vacio para isopletas S-R.")
        return

    joint_pct = (hist / total) * 100.0
    cum_r = np.cumsum(joint_pct, axis=1)
    cum_sr = np.cumsum(cum_r[::-1, :], axis=0)[::-1, :]

    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    Sg, Rg = np.meshgrid(s_centers, r_centers, indexing="ij")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    levels = np.arange(5, 105, 5)
    cf = ax.contourf(Sg, Rg, cum_sr, levels=levels, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Porcentaje acumulado (%)")

    s_c = thresholds_km["S_c"]
    r_c = thresholds_km["R_c"]
    s_cv = thresholds_km["S_cv"]
    r_cv = thresholds_km["R_cv"]
    ax.axvline(s_c, color="tab:red", linestyle="--", linewidth=1.0)
    ax.axvline(s_cv, color="tab:green", linestyle="--", linewidth=1.0)
    ax.axhline(r_c, color="tab:red", linestyle=":", linewidth=1.0)
    ax.axhline(r_cv, color="tab:green", linestyle=":", linewidth=1.0)

    ax.set_xlabel("S (km)")
    ax.set_ylabel("R")
    ax.set_title(build_title(f"Isopletas S-R (tau={tau_h} h)", loc_label))
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2)

    cti_note = format_cti_note(thresholds_km, cti_label)
    if cti_note:
        fig.text(0.5, 0.01, cti_note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_png = outdir / f"aw_isopleth_SR_{tag}_tau{int(tau_h)}h.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Grafica guardada: {out_png}")


def plot_event_timeline(df, thresholds_km, outdir, tau_h, tag, cti_label=None, loc_label=None):
    dfp = df.dropna(subset=["S_km", "R"])
    if dfp.empty:
        print("No hay datos validos para linea de eventos.")
        return

    time_col = find_time_column(dfp)
    if time_col is None:
        print("No se encontro columna Time para linea de eventos.")
        return

    fig, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True)
    ax_s = axes[0]
    ax_r = axes[1]
    ax_ev = axes[2:]

    s_c = thresholds_km["S_c"]
    s_cv = thresholds_km["S_cv"]
    r_c = thresholds_km["R_c"]
    r_cv = thresholds_km["R_cv"]

    ax_s.plot(dfp[time_col], dfp["S_km"], color="tab:blue", linewidth=1.0)
    ax_s.axhline(s_c, color="tab:red", linestyle="--", linewidth=0.8)
    ax_s.axhline(s_cv, color="tab:green", linestyle="--", linewidth=0.8)
    ax_s.set_ylabel("S (km)")
    ax_s.grid(True, axis="x", alpha=0.2)
    # Escala robusta para S: evita que outliers aplasten el detalle
    s_vals = dfp["S_km"].to_numpy()
    s_vals = s_vals[np.isfinite(s_vals)]
    if s_vals.size > 0:
        p = np.nanpercentile(s_vals, 98)
        upper = max(p, s_c * 1.05, s_cv * 1.05)
        if upper <= 0:
            upper = 1.0
        ax_s.set_ylim(0, upper)

    ax_r.plot(dfp[time_col], dfp["R"], color="tab:purple", linewidth=1.0)
    ax_r.axhline(r_c, color="tab:red", linestyle="--", linewidth=0.8)
    ax_r.axhline(r_cv, color="tab:green", linestyle="--", linewidth=0.8)
    ax_r.set_ylabel("R (adim)")
    ax_r.grid(True, axis="x", alpha=0.2)

    eventos = [
        ("stagnation", "Estancamiento", "tab:red", "S ≤ S_c (km)"),
        ("recirculation", "Recirculación", "tab:purple", "R ≥ R_c (adim)"),
        ("ventilation", "Ventilación", "tab:green", "R ≤ R_cv (adim) y S ≥ S_cv (km)"),
    ]

    for ax, (col, label, color, cond) in zip(ax_ev, eventos):
        mask = dfp[col].astype(bool).to_numpy()
        t = dfp[time_col].to_numpy()
        if mask.any():
            ax.step(t, mask.astype(int), where="post", color=color, linewidth=1.0)
            ax.fill_between(t, 0, mask.astype(int), step="post", color=color, alpha=0.25)
        else:
            ax.text(0.5, 0.5, "Sin eventos", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, axis="x", alpha=0.2)

    ax_ev[-1].set_xlabel("Tiempo")
    fig.suptitle(build_title(f"Línea de eventos (tau={tau_h} h)", loc_label))

    cond_text = (
        f"Estancamiento: {eventos[0][3]}   |   "
        f"Recirculación: {eventos[1][3]}   |   "
        f"Ventilación: {eventos[2][3]}"
    )
    umbral_text = (
        "Líneas de umbral: rojo discontínuo = S_c (panel S) / R_c (panel R); "
        "verde discontínuo = S_cv (panel S) / R_cv (panel R)."
    )
    cti_note = format_cti_note(thresholds_km, cti_label)
    if cti_note:
        fig.text(0.5, 0.058, cti_note, ha="center", va="bottom", fontsize=8)
    fig.text(0.5, 0.035, cond_text, ha="center", va="bottom", fontsize=9)
    fig.text(0.5, 0.012, umbral_text, ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0.11, 1, 0.95])

    out_png = outdir / f"aw_eventos_{tag}_tau{int(tau_h)}h.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Grafica guardada: {out_png}")


def plot_daily_percentages(
    df,
    outdir,
    tau_h,
    tag,
    thresholds_km=None,
    cti_label=None,
    print_label=None,
    loc_label=None,
):
    cols = ["stagnation", "recirculation", "ventilation"]
    if not all(c in df.columns for c in cols):
        print("No hay columnas de eventos para porcentajes diarios.")
        return
    dfp = df.copy()
    if "S_m" in dfp.columns and "R" in dfp.columns:
        dfp = dfp.dropna(subset=["S_m", "R"])
    dfp = dfp.dropna(subset=cols)
    if dfp.empty:
        print("No hay datos validos para porcentajes diarios.")
        return

    time_col = find_time_column(dfp)
    if time_col is None:
        print("No se encontro columna Time para porcentajes diarios.")
        return

    dfp[time_col] = pd.to_datetime(dfp[time_col], errors="coerce")
    dfp = dfp.dropna(subset=[time_col])
    if dfp.empty:
        print("No hay fechas validas para porcentajes diarios.")
        return

    dfp = dfp.set_index(time_col)
    daily = dfp[cols].resample("D").mean()
    counts = dfp[["stagnation"]].resample("D").count().rename(columns={"stagnation": "n"})

    full_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_days)
    counts = counts.reindex(full_days).fillna(0)

    if daily.dropna(how="all").empty:
        print("No hay datos diarios para graficar.")
        return
    if daily.max().max() <= 1.0:
        daily = daily * 100.0
    daily = daily.clip(lower=0.0, upper=100.0)
    daily = daily.fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(daily.index))
    w = 0.25
    ax.bar(x - w, daily["stagnation"], width=w, label="Estancamiento", color="tab:red", alpha=0.8)
    ax.bar(x, daily["recirculation"], width=w, label="Recirculación", color="tab:purple", alpha=0.8)
    ax.bar(x + w, daily["ventilation"], width=w, label="Ventilación", color="tab:green", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in full_days], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_xlabel("Día")
    ax.set_title(build_title(f"Porcentaje diario de eventos (tau={tau_h} h)", loc_label))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    # Marcar días sin datos
    missing = counts["n"] == 0
    if missing.any():
        missing_days = [d.strftime("%Y-%m-%d") for d in full_days[missing.values]]
        print(f"Días sin datos (porcentaje diario): {', '.join(missing_days)}")
        for i, is_missing in enumerate(missing.values):
            if is_missing:
                ax.axvspan(i - 0.5, i + 0.5, color="0.9", zorder=0)

    # Imprimir número de ventanas válidas por día
    if print_label:
        print(f"Ventanas válidas por día (porcentaje diario) [{print_label}]:")
    else:
        print("Ventanas válidas por día (porcentaje diario):")
    for d in full_days:
        n = int(counts.loc[d, "n"]) if d in counts.index else 0
        if n == 0:
            print(f"  {d.strftime('%Y-%m-%d')}: N=0 (sin datos)")
        else:
            vals = daily.loc[d]
            print(
                f"  {d.strftime('%Y-%m-%d')}: N={n} | "
                f"%Est={vals['stagnation']:.1f} "
                f"%Rec={vals['recirculation']:.1f} "
                f"%Vent={vals['ventilation']:.1f}"
            )

    cti_note = format_cti_note(thresholds_km, cti_label)
    if cti_note:
        fig.text(0.5, 0.01, cti_note, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        fig.tight_layout()
    out_png = outdir / f"aw_porcentajes_diarios_{tag}_tau{int(tau_h)}h.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Grafica guardada: {out_png}")


def plot_daily_counts(df, outdir, tau_h, tag, thresholds_km=None, cti_label=None, loc_label=None):
    cols = ["stagnation", "recirculation", "ventilation"]
    if not all(c in df.columns for c in cols):
        print("No hay columnas de eventos para conteos diarios.")
        return
    dfp = df.copy()
    time_col = find_time_column(dfp)
    if time_col is None:
        print("No se encontro columna Time para conteos diarios.")
        return
    dfp[time_col] = pd.to_datetime(dfp[time_col], errors="coerce")
    dfp = dfp.dropna(subset=[time_col])
    if dfp.empty:
        print("No hay datos validos para conteos diarios.")
        return

    dfp = dfp.set_index(time_col)
    vals = dfp[cols].copy()

    # Si los valores están en 0-100, convertir a fracción para decidir binario
    if vals.max().max() > 1.0:
        vals = vals / 100.0
    # Si no son estrictamente 0/1, binarizar por presencia (>0)
    uniques = set(np.unique(np.nan_to_num(vals.values)))
    is_binary = uniques.issubset({0.0, 1.0})
    if not is_binary:
        vals = (vals > 0.0).astype(int)

    counts = vals.resample("D").sum()
    full_days = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    counts = counts.reindex(full_days).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(full_days))
    w = 0.25
    ax.bar(x - w, counts["stagnation"], width=w, label="Estancamiento", color="tab:red", alpha=0.8)
    ax.bar(x, counts["recirculation"], width=w, label="Recirculación", color="tab:purple", alpha=0.8)
    ax.bar(x + w, counts["ventilation"], width=w, label="Ventilación", color="tab:green", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in full_days], rotation=45, ha="right")
    ax.set_ylabel("Número de eventos (ventanas)")
    ax.set_xlabel("Día")
    ax.set_title(build_title(f"Conteo diario de eventos (tau={tau_h} h)", loc_label))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    cti_note = format_cti_note(thresholds_km, cti_label)
    if cti_note:
        fig.text(0.5, 0.01, cti_note, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        fig.tight_layout()
    out_png = outdir / f"aw_conteo_diario_{tag}_tau{int(tau_h)}h.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Grafica guardada: {out_png}")


def print_event_summary(df, label):
    cols = ["stagnation", "recirculation", "ventilation"]
    if not all(c in df.columns for c in cols):
        return
    n = len(df)
    if n == 0:
        print(f"{label}: sin datos.")
        return
    vals = df[cols].copy()
    if vals.max().max() > 1.0:
        vals = vals / 100.0
    pct = vals.mean() * 100.0
    print(
        f"{label}: N={n} | %Est={pct['stagnation']:.1f} "
        f"%Rec={pct['recirculation']:.1f} %Vent={pct['ventilation']:.1f}"
    )


def _get_latlon_2d(ds):
    if "XLAT" in ds.variables and "XLONG" in ds.variables:
        lat = ds["XLAT"]
        lon = ds["XLONG"]
        if "Time" in lat.dims:
            lat = lat.isel(Time=0)
        if "Time" in lon.dims:
            lon = lon.isel(Time=0)
        return lat.values, lon.values
    return None, None


def detectar_columnas_texto(gdf):
    cols = []
    for c in gdf.columns:
        if gdf[c].dtype == object or str(gdf[c].dtype).startswith("string"):
            cols.append(c)
    return cols


def filtrar_ciudad_en_gdf(gdf, ciudad):
    cols = detectar_columnas_texto(gdf)
    if not cols:
        return None, None, None

    def _buscar_por_pat(ciudad_pat):
        best = (0, None, None)
        for c in cols:
            vals = gdf[c].astype(str).str.upper()
            mask = vals.str.contains(ciudad_pat, regex=True, na=False)
            n = int(mask.sum())
            if n > best[0]:
                best = (n, c, mask)
        return best

    def _buscar_por_code(code):
        if not code:
            return (0, None, None)
        best = (0, None, None)
        for c in cols:
            vals = gdf[c].astype(str)
            codes = vals.str.replace(r"\D", "", regex=True)
            mask = (codes == code) | (codes == code[-3:])
            n = int(mask.sum())
            if n > best[0]:
                best = (n, c, mask)
        return best

    pattern = CITY_PATTERNS.get(ciudad, ciudad.upper())
    n_pat, col_pat, mask_pat = _buscar_por_pat(pattern)
    if n_pat > 0:
        return col_pat, mask_pat, "nombre"

    code = CITY_CODES.get(ciudad)
    n_code, col_code, mask_code = _buscar_por_code(code)
    if n_code > 0:
        return col_code, mask_code, "codigo"

    # Alias: Rozo -> Palmira si no hay coincidencias
    if ciudad.lower() == "rozo":
        alias = "Palmira"
        pattern = CITY_PATTERNS.get(alias, alias.upper())
        n_pat, col_pat, mask_pat = _buscar_por_pat(pattern)
        if n_pat > 0:
            print("Aviso: Rozo no está en el shapefile. Se usa Palmira como aproximación.")
            return col_pat, mask_pat, "alias_nombre"
        code = CITY_CODES.get(alias)
        n_code, col_code, mask_code = _buscar_por_code(code)
        if n_code > 0:
            print("Aviso: Rozo no está en el shapefile. Se usa Palmira como aproximación.")
            return col_code, mask_code, "alias_codigo"

    return None, None, None


def cargar_geometria_ciudad(shp_path, ciudad):
    import geopandas as gpd

    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    if gdf.empty:
        raise SystemExit("El shapefile no tiene geometria.")
    col, mask, modo = filtrar_ciudad_en_gdf(gdf, ciudad)
    if col is None or mask is None or not mask.any():
        print(f"Aviso: no se encontro '{ciudad}' en el shapefile seleccionado.")
        return gdf, None
    if DEBUG:
        print(f"Shapefile: columna '{col}' usada por {modo} para '{ciudad}'.")
    geom = gdf.loc[mask].geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.loc[mask].unary_union
    return gdf, geom


def construir_mascara_ciudad(lat2d, lon2d, geom):
    try:
        from shapely import contains_xy
        mask = contains_xy(geom, lon2d, lat2d)
    except Exception:
        from shapely import vectorized
        mask = vectorized.contains(geom, lon2d, lat2d)
    return mask


def plot_map_field(lat, lon, data2d, title, cbar_label, out_png, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(8, 6))
    if lat is not None and lon is not None:
        cf = ax.pcolormesh(lon, lat, data2d, shading="auto", cmap=cmap)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
    else:
        cf = ax.imshow(data2d, origin="lower", cmap=cmap)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Mapa guardado: {out_png}")


def plot_map_outputs(ds_out, lat, lon, outdir, tau_h, run_id):
    fields = [
        ("S_mean_km", "S medio (km)", "S medio (km)", "viridis"),
        ("R_mean", "R medio (adim)", "R medio (adim)", "magma"),
        ("pct_stagnation", "Estancamiento (%)", "Porcentaje (%)", "Reds"),
        ("pct_recirculation", "Recirculación (%)", "Porcentaje (%)", "Purples"),
        ("pct_ventilation", "Ventilación (%)", "Porcentaje (%)", "Greens"),
    ]
    for var, title, cbar, cmap in fields:
        if var not in ds_out:
            continue
        data = ds_out[var].values
        out_png = outdir / f"aw_map_{var}_{run_id}_tau{int(tau_h)}h.png"
        plot_map_field(lat, lon, data, title, cbar, out_png, cmap=cmap)


def print_map_summary(ds_out):
    vars_ = ["S_mean_km", "R_mean", "pct_stagnation", "pct_recirculation", "pct_ventilation"]
    for v in vars_:
        if v not in ds_out:
            continue
        arr = ds_out[v].values
        if not np.isfinite(arr).any():
            print(f"{v}: sin datos validos.")
            continue
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        mean = float(np.nanmean(arr))
        frac_pos = float(np.mean(np.nan_to_num(arr) > 0.0) * 100.0)
        print(f"{v}: min={mn:.3f} max={mx:.3f} mean={mean:.3f} | >0%: {frac_pos:.1f}%")


# ======================================================
# MAIN
# ======================================================
def main():
    print("Allwine y Whiteman (1994) - Estancamiento, Recirculacion, Ventilacion")
    print("=" * 72)
    print(
        "Modo 'punto' : serie temporal en una celda (CSV + graficas).\n"
        "Modo 'ciudad': promedio y incidencia en poligono urbano (CSV + graficas).\n"
        "Modo 'mapa'  : campos espaciales y porcentajes (NetCDF)."
    )

    archivos = seleccionar_archivo_wrf()
    modo = seleccionar_modo()

    max_horas, dt_seconds, _ = obtener_horas_disponibles(archivos)
    if max_horas <= 0:
        raise SystemExit("No se pudo estimar horas disponibles a partir de los archivos.")
    if max_horas < 12:
        opciones = [round(max_horas, 1)]
    else:
        max_tau = int(np.floor(max_horas / 12.0)) * 12
        opciones = list(range(12, max_tau + 1, 12))
    tau_h = seleccionar_tau_disponible(opciones, max_horas)

    print(f"Archivos: {len(archivos)}")
    print(f"Primero: {archivos[0]}")
    print(f"Modo: {modo}")
    print(f"Tau: {tau_h} h")
    thresholds_km = thresholds_for_tau(tau_h)
    scale_note = ""
    if SCALE_S_WITH_TAU and not np.isclose(tau_h, BASE_TAU_H):
        scale_note = f" (S escalado x{tau_h/BASE_TAU_H:.2f})"
    print(
        "CTI (km/adim): "
        f"S_c={thresholds_km['S_c']:.2f}, "
        f"R_c={thresholds_km['R_c']}, "
        f"S_cv={thresholds_km['S_cv']:.2f}, "
        f"R_cv={thresholds_km['R_cv']}"
        f"{scale_note}"
    )
    if ADAPTIVE_CTI:
        print(
            "CTI adaptativo: activado "
            f"(percentiles S: P{ADAPTIVE_PCTS['S_c']}/P{ADAPTIVE_PCTS['S_cv']}, "
            f"R: P{ADAPTIVE_PCTS['R_cv']}/P{ADAPTIVE_PCTS['R_c']})."
        )
        if OUTPUT_BOTH_CTI:
            print("CTI: se generarán salidas para CTI fijo y CTI adaptativo.")
        if modo == "mapa":
            print("CTI adaptativo: en modo mapa se mantiene CTI fijo por rendimiento.")
    print(
        "Siglas: S=wind run (km), L=distancia neta (km), R=factor de recirculación (adim), "
        "θ=dirección de transporte (°).\n"
        "CTI: S_c=umbral de estancamiento, R_c=umbral de recirculación, "
        "S_cv=umbral de ventilación, R_cv=umbral de ventilación."
    )

    outdir = Path("outputs") / "allwine_whiteman"
    outdir.mkdir(parents=True, exist_ok=True)

    run_id = run_tag()
    if modo == "punto":
        procesar_punto(archivos, tau_h, outdir, run_id, thresholds_km)
        return

    if modo == "ciudad":
        shp = seleccionar_shapefile()
        ciudad = seleccionar_ciudad()
        gdf, geom = cargar_geometria_ciudad(shp, ciudad)
        ds0 = xr.open_dataset(archivos[0], decode_times=False, mask_and_scale=False)
        lat2d, lon2d = _get_latlon_2d(ds0)
        ds0.close()
        if lat2d is None or lon2d is None:
            raise SystemExit("No se encontraron XLAT/XLONG para construir mascara.")
        mask = None
        if geom is not None:
            mask = construir_mascara_ciudad(lat2d, lon2d, geom)
        if mask is None or not np.any(mask):
            mask = fallback_mask_ciudad(lat2d, lon2d, ciudad)
        if mask is None or not np.any(mask):
            raise SystemExit("Mascara ciudad vacia. Revise shapefile/ciudad.")

        print(f"Ciudad seleccionada: {ciudad} | Celdas en mascara: {int(mask.sum())}")
        _, cti_variants = procesar_ciudad_promedio(
            archivos, tau_h, thresholds_km, outdir, run_id, mask, ciudad
        )
        procesar_ciudad_incidencia(
            archivos, tau_h, dt_seconds, cti_variants, mask, outdir, run_id, ciudad
        )
        return

    print(f"Paso temporal estimado: {dt_seconds:.1f} s ({dt_seconds/60.0:.2f} min)")

    ds_out, lat2d, lon2d = procesar_mapa_streaming(archivos, tau_h, dt_seconds, thresholds_km)

    print_map_summary(ds_out)

    out_nc = outdir / f"aw_indices_map_tau{int(tau_h)}h_{run_id}.nc"
    ds_out.to_netcdf(out_nc)
    print(f"NetCDF guardado: {out_nc}")
    print("Variables NetCDF: S_mean_km, R_mean, pct_stagnation, pct_recirculation, pct_ventilation")

    generar_mapas = solicitar_si_no(
        "Mapas PNG",
        "¿Desea generar mapas PNG de S_mean, R_mean y porcentajes?"
    )
    if generar_mapas:
        plot_map_outputs(ds_out, lat2d, lon2d, outdir, tau_h, run_id)


if __name__ == "__main__":
    main()
