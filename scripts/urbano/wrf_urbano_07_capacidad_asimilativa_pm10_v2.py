import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
from scipy.interpolate import griddata
import re

# ======================================================
# CONFIGURACION
# ======================================================
LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.55, -76.25
ZOOM_OUT = 0.12
EMAX_DEFAULT_MAX = 5000
EMAX_LEVELS = 20

PM10_LIMITE_UG_M3 = 75.0  # Resolucion 2254 de 2017 (24 horas)
KG_POR_UG = 1e-9
SEG_DIA = 86400.0
SEG_HORA = 3600.0
FACTOR_KG_DIA = KG_POR_UG * SEG_DIA
FACTOR_KG_HORA = KG_POR_UG * SEG_HORA

# ======================================================
# INTERFAZ INTERACTIVA
# ======================================================

def seleccionar_archivos_wrf():
    archivos = []
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        archivo = filedialog.askopenfilename(title="Seleccione un archivo wrfout")
        root.destroy()

        if archivo == "":
            break
        archivos.append(archivo)

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        continuar = messagebox.askyesno("Agregar otro", "¿Desea agregar otro archivo wrfout?")
        root.destroy()
        if not continuar:
            break

    if not archivos:
        raise SystemExit("No se seleccionaron archivos wrfout.")
    return sorted(archivos)


def seleccionar_fecha_disponible(fechas):
    opciones = "\\n".join([f"{i}. {f}" for i, f in enumerate(fechas, 1)])
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = simpledialog.askinteger(
            "Seleccione un día",
            "Días disponibles:\\n" + opciones + "\\n\\nIngrese el número:",
            minvalue=1,
            maxvalue=len(fechas)
        )
        root.destroy()
        if sel is None:
            raise SystemExit("No se seleccionó un día.")
        return fechas[sel - 1]


def seleccionar_modo_escala():
    opciones = "\n".join([
        "1. Manual",
        "2. Percentil 95 (selección)",
        "3. Tope fijo por temporada (seco/húmedo)"
    ])
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = simpledialog.askinteger(
            "Modo de escala",
            "Seleccione el modo de escala:\n" + opciones + "\n\nIngrese el número:",
            minvalue=1,
            maxvalue=3
        )
        root.destroy()
        if sel is None:
            raise SystemExit("No se seleccionó modo de escala.")
        if sel == 1:
            return "manual"
        if sel == 2:
            return "p95"
        if sel == 3:
            return "temporada"


def seleccionar_temporada():
    opciones = "\n".join([
        "1. Seco",
        "2. Húmedo"
    ])
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = simpledialog.askinteger(
            "Temporada",
            "Seleccione la temporada:\n" + opciones + "\n\nIngrese el número:",
            minvalue=1,
            maxvalue=2
        )
        root.destroy()
        if sel is None:
            raise SystemExit("No se seleccionó temporada.")
        return "seco" if sel == 1 else "humedo"


def seleccionar_shapefile():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    shp = filedialog.askopenfilename(
        title="Seleccione shapefile urbano",
        filetypes=[("Shapefile", "*.shp")]
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


def seleccionar_emax_maximo(etiqueta, valor_sugerido, texto_sugerencia):
    if valor_sugerido is None or not np.isfinite(valor_sugerido):
        valor_sugerido = EMAX_DEFAULT_MAX
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    emax_max = simpledialog.askfloat(
        "Tope máximo de Emax",
        f"{texto_sugerencia}: {valor_sugerido:.3g}\n"
        f"Ingrese el tope máximo para {etiqueta}:",
        minvalue=0.0,
        initialvalue=valor_sugerido
    )
    root.destroy()
    if emax_max is None:
        raise SystemExit("No se seleccionó tope máximo de Emax.")
    return float(emax_max)


# ======================================================
# UTILIDADES
# ======================================================

def extraer_fecha_archivo(nombre):
    match = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", nombre)
    if not match:
        return None
    return match.group(0).replace("_", "-")


def detectar_columna_nombre(gdf):
    for c in gdf.columns:
        if gdf[c].dtype == object or str(gdf[c].dtype).startswith("string"):
            if gdf[c].astype(str).str.upper().str.contains(
                "CALI|SILO|YUMBO|PALM|ROZO", regex=True, na=False
            ).any():
                return c
    raise RuntimeError("No se detectó columna de nombres.")


def haversine(lon1, lat1, lon2, lat2):
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
    # Prioriza DX/DY del WRF si existen
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

    # Fallback con lat/lon
    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    i = lat.shape[0] // 2
    j = lon.shape[1] // 2
    if j + 1 < lon.shape[1]:
        dx = haversine(lon[i, j], lat[i, j], lon[i, j + 1], lat[i, j + 1])
    else:
        dx = haversine(lon[i, j - 1], lat[i, j - 1], lon[i, j], lat[i, j])
    if i + 1 < lat.shape[0]:
        dy = haversine(lon[i, j], lat[i, j], lon[i + 1, j], lat[i + 1, j])
    else:
        dy = haversine(lon[i - 1, j], lat[i - 1, j], lon[i, j], lat[i, j])
    return float(np.sqrt(dx * dy))


def obtener_resolucion_malla(ds):
    # Prioriza DX/DY del WRF si existen
    for key in ("DX", "dx"):
        if key in ds.attrs:
            dx = float(ds.attrs[key])
            dy = float(ds.attrs.get("DY", ds.attrs.get("dy", dx)))
            return dx, dy
        if key in ds:
            try:
                dx = float(ds[key].values)
                dy = float(ds["DY"].values) if "DY" in ds else dx
                return dx, dy
            except Exception:
                pass

    # Fallback con lat/lon
    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    i = lat.shape[0] // 2
    j = lon.shape[1] // 2
    if j + 1 < lon.shape[1]:
        dx = haversine(lon[i, j], lat[i, j], lon[i, j + 1], lat[i, j + 1])
    else:
        dx = haversine(lon[i, j - 1], lat[i, j - 1], lon[i, j], lat[i, j])
    if i + 1 < lat.shape[0]:
        dy = haversine(lon[i, j], lat[i, j], lon[i + 1, j], lat[i + 1, j])
    else:
        dy = haversine(lon[i - 1, j], lat[i - 1, j], lon[i, j], lat[i, j])
    return float(dx), float(dy)


def formatear_resolucion(dx, dy):
    def fmt(v):
        if v is None or not np.isfinite(v):
            return "NA"
        if v >= 1000:
            return f"{v / 1000:.1f} km"
        return f"{v:.0f} m"
    return f"Resolución: {fmt(dx)} × {fmt(dy)}"


def formatear_hora(xtime_min):
    horas_total = (xtime_min / 60.0) % 24
    hora = int(np.floor(horas_total + 1e-6))
    minutos = int(np.round((horas_total - hora) * 60))
    if minutos == 60:
        hora = (hora + 1) % 24
        minutos = 0
    return hora, minutos


# ======================================================
# CALCULOS
# ======================================================

def calcular_vc(u10, v10, pblh):
    ws = np.sqrt(u10 ** 2 + v10 ** 2)
    return ws * pblh


def calcular_emax(vc, L, factor_tiempo):
    # VC (m2/s), L (m) => Emax (kg/tiempo)
    return PM10_LIMITE_UG_M3 * vc * L * factor_tiempo


def calcular_emax_percentil_95(archivos):
    p95_h = None
    p95_d = None
    for f in archivos:
        ds = xr.open_dataset(f, decode_times=False)
        L = obtener_longitud_caracteristica(ds)
        u10 = ds["U10"]
        v10 = ds["V10"]
        pblh = ds["PBLH"]
        vc = calcular_vc(u10, v10, pblh)

        emax_h = calcular_emax(vc, L, FACTOR_KG_HORA)
        emax_h_vals = np.asarray(emax_h.values)
        emax_h_vals = emax_h_vals[np.isfinite(emax_h_vals)]
        if emax_h_vals.size:
            emax_h_val = float(np.nanpercentile(emax_h_vals, 95))
            p95_h = emax_h_val if p95_h is None else max(p95_h, emax_h_val)

        emax_d = calcular_emax(vc.mean(dim="Time"), L, FACTOR_KG_DIA)
        emax_d_vals = np.asarray(emax_d.values)
        emax_d_vals = emax_d_vals[np.isfinite(emax_d_vals)]
        if emax_d_vals.size:
            emax_d_val = float(np.nanpercentile(emax_d_vals, 95))
            p95_d = emax_d_val if p95_d is None else max(p95_d, emax_d_val)

        ds.close()

    if p95_h is None or not np.isfinite(p95_h):
        p95_h = EMAX_DEFAULT_MAX
    if p95_d is None or not np.isfinite(p95_d):
        p95_d = EMAX_DEFAULT_MAX

    return p95_h, p95_d


# ======================================================
# MAPA
# ======================================================

def mapa_emax_viento(emax, u10, v10, ds, shp, titulo, outpath, cbar_label, emax_max, resolucion_texto):
    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values

    gdf = gpd.read_file(shp).to_crs(epsg=4326)

    data = np.asarray(emax.values)
    valid = data[np.isfinite(data)]
    if valid.size:
        q1, q2 = np.nanpercentile(valid, [33, 66])
    else:
        q1 = q2 = np.nan

    # ---- Grilla de flechas
    nx, ny = 66, 66
    lon_g = np.linspace(LON_MIN - ZOOM_OUT, LON_MAX + ZOOM_OUT, nx)
    lat_g = np.linspace(LAT_MIN - ZOOM_OUT, LAT_MAX + ZOOM_OUT, ny)
    lon_q, lat_q = np.meshgrid(lon_g, lat_g)

    u_i = griddata(
        (lon.flatten(), lat.flatten()),
        u10.values.flatten(),
        (lon_q, lat_q),
        method="linear"
    )

    v_i = griddata(
        (lon.flatten(), lat.flatten()),
        v10.values.flatten(),
        (lon_q, lat_q),
        method="linear"
    )

    mag = np.sqrt(u_i ** 2 + v_i ** 2)
    mag = np.nan_to_num(mag, nan=0.0)
    u_n = np.nan_to_num(u_i, nan=0.0) / (mag + 1e-6)
    v_n = np.nan_to_num(v_i, nan=0.0) / (mag + 1e-6)

    # ---- Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    levels = np.linspace(0, emax_max, EMAX_LEVELS + 1)
    cf = ax.contourf(
        lon, lat, emax.values,
        levels=levels,
        cmap="coolwarm_r",
        extend="max"
    )
    ax.contour(lon, lat, emax.values, colors="k", linewidths=0.4, levels=levels)
    plt.colorbar(cf, ax=ax, label=cbar_label)

    if valid.size:
        etiqueta_cat = (
            f"Categorias ({cbar_label}):\n"
            f"Baja < {q1:.2g}\n"
            f"Media {q1:.2g} - {q2:.2g}\n"
            f"Alta > {q2:.2g}"
        )
        fig.text(
            0.02, 0.02, etiqueta_cat,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray")
        )

    if resolucion_texto:
        fig.text(
            0.98, 0.02, resolucion_texto,
            fontsize=9,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray")
        )

    ax.quiver(
        lon_q[::3, ::3],
        lat_q[::3, ::3],
        u_n[::3, ::3],
        v_n[::3, ::3],
        scale=28,
        width=0.0028,
        color="black",
        zorder=6
    )

    # ---- Shapes
    gdf.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.8,
        zorder=7
    )

    # ---- Rótulos (posiciones fijas)
    etiquetas = {
        "Cali": (-76.516651, 3.443662),
        "Yumbo": (-76.490915, 3.571668),
        "Rozo": (-76.384813, 3.610740),
        "Palmira": (-76.294686, 3.527459)
    }

    for nombre, (x, y) in etiquetas.items():
        ax.text(
            x, y, nombre,
            fontsize=10,
            fontweight="bold",
            ha="center",
            zorder=8
        )

    ax.set_xlim(LON_MIN - ZOOM_OUT, LON_MAX + ZOOM_OUT)
    ax.set_ylim(LAT_MIN - ZOOM_OUT, LAT_MAX + ZOOM_OUT)

    ax.set_title(titulo)

    bottom_margin = 0.12 if (valid.size or resolucion_texto) else 0.0
    plt.tight_layout(rect=(0, bottom_margin, 1, 1))
    plt.savefig(outpath, dpi=150)
    plt.close()


# ======================================================
# MAIN
# ======================================================

def main():
    archivos = [Path(a) for a in seleccionar_archivos_wrf()]

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
    fecha_sel = seleccionar_fecha_disponible(fechas)

    shp = seleccionar_shapefile()
    outdir = seleccionar_directorio_salida()
    modo_escala = seleccionar_modo_escala()

    if modo_escala in ("p95", "temporada"):
        print("Calculando percentil 95 de Emax para sugerencias de escala...")
        p95_h_sug, p95_d_sug = calcular_emax_percentil_95(archivos_por_fecha[fecha_sel])

    scale_label = ""
    if modo_escala == "manual":
        emax_max_h = seleccionar_emax_maximo(
            "Emax (kg/h)",
            EMAX_DEFAULT_MAX,
            "Sugerencia"
        )
        emax_max_d = seleccionar_emax_maximo(
            "Emax (kg/día)",
            EMAX_DEFAULT_MAX,
            "Sugerencia"
        )
        scale_label = "Escala manual"
    elif modo_escala == "p95":
        emax_max_h = seleccionar_emax_maximo(
            "Emax (kg/h)",
            p95_h_sug,
            "Percentil 95 observado"
        )
        emax_max_d = seleccionar_emax_maximo(
            "Emax (kg/día)",
            p95_d_sug,
            "Percentil 95 observado"
        )
        scale_label = "Escala P95"
    else:
        temporada = seleccionar_temporada()
        emax_max_h = seleccionar_emax_maximo(
            f"Emax (kg/h) - {temporada}",
            p95_h_sug,
            f"P95 sugerido ({temporada})"
        )
        emax_max_d = seleccionar_emax_maximo(
            f"Emax (kg/día) - {temporada}",
            p95_d_sug,
            f"P95 sugerido ({temporada})"
        )
        scale_label = f"Escala {temporada}"

    outdir.mkdir(parents=True, exist_ok=True)

    for f in archivos_por_fecha[fecha_sel]:
        ds = xr.open_dataset(f, decode_times=False)
        etiqueta = Path(f).name.replace("wrfout_", "")

        L = obtener_longitud_caracteristica(ds)
        dx, dy = obtener_resolucion_malla(ds)
        resolucion_texto = formatear_resolucion(dx, dy)

        u10 = ds["U10"]
        v10 = ds["V10"]
        pblh = ds["PBLH"]

        vc = calcular_vc(u10, v10, pblh)

        # ---- Mapas horarios (Emax en kg/h)
        xtimes = ds["XTIME"].values
        for idx in range(len(xtimes)):
            hora, minutos = formatear_hora(xtimes[idx])
            emax_h = calcular_emax(vc.isel(Time=idx), L, FACTOR_KG_HORA)
            titulo = (
                f"Emax PM10 (kg/h) | {fecha_sel} {hora:02d}:{minutos:02d}"
                f" | Tope {emax_max_h:.3g} | {scale_label}"
            )
            outpath = outdir / f"Emax_PM10_h{hora:02d}{minutos:02d}_{fecha_sel}_{etiqueta}.png"
            mapa_emax_viento(
                emax_h,
                u10.isel(Time=idx),
                v10.isel(Time=idx),
                ds,
                shp,
                titulo,
                outpath,
                "Emax PM10 (kg/h)",
                emax_max_h,
                resolucion_texto
            )

        # ---- Mapa diario (Emax en kg/día)
        vc_mean = vc.mean(dim="Time")
        u10_mean = u10.mean(dim="Time")
        v10_mean = v10.mean(dim="Time")
        emax_d = calcular_emax(vc_mean, L, FACTOR_KG_DIA)
        titulo_d = (
            f"Emax PM10 diario (kg/día) | {fecha_sel} | Tope {emax_max_d:.3g}"
            f" | {scale_label}"
        )
        outpath_d = outdir / f"Emax_PM10_diario_{fecha_sel}_{etiqueta}.png"
        mapa_emax_viento(
            emax_d,
            u10_mean,
            v10_mean,
            ds,
            shp,
            titulo_d,
            outpath_d,
            "Emax PM10 (kg/día)",
            emax_max_d,
            resolucion_texto
        )

        ds.close()

    print(f"✔ Mapas Emax PM10 generados para {fecha_sel}.")


if __name__ == "__main__":
    main()
