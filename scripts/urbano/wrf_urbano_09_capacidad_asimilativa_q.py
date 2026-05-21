import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
import re

# ======================================================
# CONFIGURACION ESPACIAL Y CONSTANTES
# ======================================================
LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.55, -76.25

PM10_LIMITE_UG_M3 = 75.0  # Resolucion 2254 de 2017 (24 horas)
KG_POR_UG = 1e-9
SEG_DIA = 86400.0
FACTOR_KG_DIA = KG_POR_UG * SEG_DIA
# Sugerencias preliminares de concentración de fondo B (ug/m3).
# Ajustar con datos locales de monitoreo.
B_SUG_BAJO = 25.0
B_SUG_ALTO = 70.0
B_DEFAULT = B_SUG_ALTO

# ======================================================
# INTERFAZ INTERACTIVA
# ======================================================
def seleccionar_archivos_wrf():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    archivos = filedialog.askopenfilenames(title="Seleccione archivos wrfout")
    root.destroy()
    if not archivos:
        raise SystemExit("No se seleccionaron archivos wrfout.")
    return sorted(archivos)


def seleccionar_shapefile():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    shp = filedialog.askopenfilename(
        title="Seleccione shapefile urbano",
        filetypes=[("Shapefile", "*.shp")],
    )
    root.destroy()
    if shp == "":
        raise SystemExit("No se seleccionó shapefile.")
    return shp


def seleccionar_directorio_salida():
    root = tk.Tk()
    root.withdraw()
    out = filedialog.askdirectory(title="Seleccione carpeta de salida")
    root.destroy()
    if out == "":
        raise SystemExit("No se seleccionó carpeta de salida.")
    return Path(out)


def seleccionar_dias_disponibles(fechas):
    root = tk.Tk()
    root.title("Seleccionar días")
    root.attributes("-topmost", True)

    tk.Label(root, text="Seleccione uno o más días:").pack(padx=10, pady=5)
    lb = tk.Listbox(
        root,
        selectmode=tk.MULTIPLE,
        width=20,
        height=min(15, len(fechas)),
        exportselection=False,
    )
    for f in fechas:
        lb.insert(tk.END, f)
    lb.pack(padx=10, pady=5)

    seleccion = {"dias": []}

    def seleccionar_todo():
        lb.select_set(0, tk.END)

    def aceptar():
        indices = lb.curselection()
        if not indices:
            messagebox.showwarning("Sin selección", "Seleccione al menos un día.")
            return
        seleccion["dias"] = [fechas[i] for i in indices]
        root.destroy()

    botones = tk.Frame(root)
    botones.pack(pady=(0, 10))
    tk.Button(botones, text="Seleccionar todo", command=seleccionar_todo).pack(
        side=tk.LEFT, padx=5
    )
    tk.Button(botones, text="Aceptar", command=aceptar).pack(side=tk.LEFT, padx=5)

    root.mainloop()

    if not seleccion["dias"]:
        raise SystemExit("No se seleccionaron días.")
    return seleccion["dias"]


def solicitar_concentracion_fondo():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    valor = simpledialog.askfloat(
        "Concentración de fondo",
        "Ingrese concentración de fondo B (ug/m3).\n"
        f"Sugerencias preliminares: bajo {B_SUG_BAJO:.1f}, alto {B_SUG_ALTO:.1f}.\n"
        "Use 0 si no aplica:",
        minvalue=0.0,
        initialvalue=B_DEFAULT,
    )
    root.destroy()
    if valor is None:
        raise SystemExit("No se ingresó la concentración de fondo.")
    return float(valor)


# ======================================================
# UTILIDADES
# ======================================================
def extraer_fecha_archivo(nombre):
    match = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", nombre)
    if not match:
        return None
    return match.group(0).replace("_", "-")


def _haversine(lon1, lat1, lon2, lat2):
    r = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def obtener_longitud_caracteristica(ds):
    for key in ("DX", "dx"):
        if key in ds.attrs:
            dx = float(ds.attrs[key])
            dy = float(ds.attrs.get("DY", ds.attrs.get("dy", dx)))
            return float(np.sqrt(dx * dy))
        if key in ds:
            try:
                dx = float(ds[key].values)
                dy = float(ds["DY"].values) if "DY" in ds else dx
                return float(np.sqrt(dx * dy))
            except Exception:
                pass

    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    i = lat.shape[0] // 2
    j = lon.shape[1] // 2
    if j + 1 < lon.shape[1]:
        dx = _haversine(lon[i, j], lat[i, j], lon[i, j + 1], lat[i, j + 1])
    else:
        dx = _haversine(lon[i, j - 1], lat[i, j - 1], lon[i, j], lat[i, j])
    if i + 1 < lat.shape[0]:
        dy = _haversine(lon[i, j], lat[i, j], lon[i + 1, j], lat[i + 1, j])
    else:
        dy = _haversine(lon[i - 1, j], lat[i - 1, j], lon[i, j], lat[i, j])
    return float(np.sqrt(dx * dy))


def cargar_geometria(shp):
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    if gdf.empty:
        raise SystemExit("El shapefile no tiene geometría.")
    return gdf.unary_union


def mascara_region(ds, geom=None):
    lat = ds["XLAT"].isel(Time=0)
    lon = ds["XLONG"].isel(Time=0)
    mask_box = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)

    if geom is None:
        return mask_box

    try:
        from shapely import vectorized

        mask_shape = vectorized.contains(geom, lon.values, lat.values)
        mask_shape = xr.DataArray(mask_shape, coords=lat.coords, dims=lat.dims)
        return mask_box & mask_shape
    except Exception as exc:
        print(f"Aviso: no se pudo aplicar máscara del shapefile ({exc}). Se usa caja.")
        return mask_box


# ======================================================
# CALCULOS
# ======================================================
def calcular_vc(u10, v10, pblh):
    ws = np.sqrt(u10**2 + v10**2)
    return ws * pblh


def calcular_q(vc, L, c_limite, c_fondo, factor_tiempo):
    delta_c = c_limite - c_fondo
    return delta_c * vc / L * factor_tiempo


def calcular_vc_medio(ds, mask):
    u10 = ds["U10"]
    v10 = ds["V10"]
    pblh = ds["PBLH"]

    vc_full = calcular_vc(u10, v10, pblh)
    vc_mean = vc_full.where(mask).mean(dim=("Time", "south_north", "west_east"))
    ntime = int(vc_full.sizes.get("Time", vc_full.shape[0]))
    return float(vc_mean), ntime


def calcular_precipitacion_diaria(ds, mask):
    if "RAINC" not in ds or "RAINNC" not in ds:
        return np.nan
    rain = ds["RAINC"] + ds["RAINNC"]
    rain_dia = (rain.isel(Time=-1) - rain.isel(Time=0)).where(mask).mean(
        dim=("south_north", "west_east")
    )
    return float(rain_dia)


# ======================================================
# GRAFICAS
# ======================================================
def grafica_serie_temporal(fechas, q, vc, precip, outpath, c_fondo):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(fechas))
    width = 0.35

    ax1.plot(
        x,
        q,
        color="tab:red",
        marker="o",
        linewidth=2,
        markerfacecolor="white",
        markeredgecolor="tab:red",
        label="q (kg/m²·día)",
        zorder=3,
    )
    ax1.set_ylabel("q (kg/m²·día)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.bar(
        x - width / 2,
        vc,
        width=width,
        color="tab:blue",
        alpha=0.45,
        label="VC (m²/s)",
        zorder=1,
    )
    ax2.set_ylabel("VC (m²/s)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.bar(
        x + width / 2,
        precip,
        width=width,
        color="tab:green",
        alpha=0.35,
        label="Precipitación (mm)",
        zorder=1,
    )
    ax3.set_ylabel("Precipitación (mm)", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")

    ax2.patch.set_visible(False)
    ax3.patch.set_visible(False)
    ax1.set_zorder(3)
    ax1.patch.set_visible(False)

    ax1.set_xlabel("Fecha")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fechas, rotation=45, ha="right")

    lines_labels = [ax.get_legend_handles_labels() for ax in (ax1, ax2)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines_labels3 = ax3.get_legend_handles_labels()
    lines += lines_labels3[0]
    labels += lines_labels3[1]
    ax1.legend(
        lines,
        labels,
        loc="lower right",
        bbox_to_anchor=(1.35, -0.26),
        frameon=True,
    )

    plt.title(f"Serie temporal diaria - q, VC y Precipitación | B={c_fondo:.1f} ug/m3")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def grafica_dispersion(fechas, q, vc, precip, outpath, c_fondo):
    q = np.asarray(q)
    vc = np.asarray(vc)
    precip = np.asarray(precip)

    valid_color = np.isfinite(q) & np.isfinite(vc) & np.isfinite(precip)
    valid_basic = np.isfinite(q) & np.isfinite(vc) & ~np.isfinite(precip)

    fig, ax = plt.subplots(figsize=(7, 6))

    if valid_color.any():
        sc = ax.scatter(
            vc[valid_color],
            q[valid_color],
            c=precip[valid_color],
            cmap="viridis",
            s=70,
            edgecolor="black",
            alpha=0.85,
            label="Con precipitación",
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Precipitación diaria (mm)")

    if valid_basic.any():
        ax.scatter(
            vc[valid_basic],
            q[valid_basic],
            color="gray",
            s=70,
            edgecolor="black",
            alpha=0.85,
            label="Sin precipitación",
        )

    for i, fecha in enumerate(fechas):
        if np.isfinite(vc[i]) and np.isfinite(q[i]):
            ax.text(vc[i], q[i], fecha, fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("VC (m²/s)")
    ax.set_ylabel("q (kg/m²·día)")
    ax.set_title(f"Dispersión q - VC con precipitación | B={c_fondo:.1f} ug/m3")

    if valid_basic.any() or valid_color.any():
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ======================================================
# MAIN
# ======================================================
def main():
    archivos = [Path(a) for a in seleccionar_archivos_wrf()]
    shp = seleccionar_shapefile()
    outdir = seleccionar_directorio_salida()
    outdir.mkdir(parents=True, exist_ok=True)

    c_fondo = solicitar_concentracion_fondo()

    fechas = []
    archivos_por_fecha = {}
    for p in archivos:
        fecha = extraer_fecha_archivo(p.name)
        if fecha:
            archivos_por_fecha.setdefault(fecha, []).append(p)
            if fecha not in fechas:
                fechas.append(fecha)

    if not fechas:
        raise SystemExit("No se encontraron archivos wrfout con fecha válida.")

    fechas = sorted(fechas)
    dias_sel = seleccionar_dias_disponibles(fechas)

    try:
        geom = cargar_geometria(shp)
    except Exception as exc:
        print(f"Aviso: no se pudo leer shapefile ({exc}). Se usará solo caja.")
        geom = None

    resultados = []

    print("\nRESUMEN DIARIO (días seleccionados)")
    print("-" * 90)
    print(f"{'Fecha':<12}{'VC(m²/s)':>12}{'q(kg/m²·día)':>16}{'Prec(mm)':>12}")
    print("-" * 90)

    for fecha in dias_sel:
        archivos_dia = archivos_por_fecha.get(fecha, [])
        if not archivos_dia:
            continue

        vc_sum = 0.0
        L_sum = 0.0
        ntime_sum = 0
        precip_sum = 0.0
        precip_ok = False

        for f in archivos_dia:
            ds = xr.open_dataset(f, decode_times=False)
            mask = mascara_region(ds, geom).astype(bool)

            L = obtener_longitud_caracteristica(ds)
            vc_mean, ntime = calcular_vc_medio(ds, mask)

            if np.isfinite(vc_mean):
                vc_sum += vc_mean * ntime
                L_sum += L * ntime
                ntime_sum += ntime

            p_dia = calcular_precipitacion_diaria(ds, mask)
            if np.isfinite(p_dia):
                precip_sum += p_dia
                precip_ok = True

            ds.close()

        if ntime_sum > 0:
            vc_dia = vc_sum / ntime_sum
            L_dia = L_sum / ntime_sum
            q_dia = calcular_q(vc_dia, L_dia, PM10_LIMITE_UG_M3, c_fondo, FACTOR_KG_DIA)
        else:
            vc_dia = np.nan
            q_dia = np.nan

        precip_dia = precip_sum if precip_ok else np.nan

        resultados.append((fecha, vc_dia, q_dia, precip_dia))
        print(f"{fecha:<12}{vc_dia:12.2f}{q_dia:16.4f}{precip_dia:12.2f}")

    if not resultados:
        raise SystemExit("No hay datos para los días seleccionados.")

    fechas_sel = [r[0] for r in resultados]
    vc_sel = [r[1] for r in resultados]
    q_sel = [r[2] for r in resultados]
    precip_sel = [r[3] for r in resultados]

    graficas_dir = outdir / "graficas_q"
    graficas_dir.mkdir(parents=True, exist_ok=True)

    grafica_serie_temporal(
        fechas_sel,
        q_sel,
        vc_sel,
        precip_sel,
        graficas_dir / "serie_q_vc_precip_diaria.png",
        c_fondo,
    )

    grafica_dispersion(
        fechas_sel,
        q_sel,
        vc_sel,
        precip_sel,
        graficas_dir / "dispersion_q_vc_precip.png",
        c_fondo,
    )

    print("\nGráficas generadas en:")
    print(f"  {graficas_dir}")


if __name__ == "__main__":
    main()
