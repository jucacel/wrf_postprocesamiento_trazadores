import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
import re

try:
    from scipy.interpolate import griddata, Rbf
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# ======================================================
# CONFIGURACION ESPACIAL Y CONSTANTES
# ======================================================
LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.55, -76.25
ZOOM_OUT = 0.12

PM10_LIMITE_UG_M3 = 75.0  # Resolucion 2254 de 2017 (24 horas)
KG_POR_UG = 1e-9
SEG_DIA = 86400.0
FACTOR_KG_DIA = KG_POR_UG * SEG_DIA
Q_DEFAULT_MAX = 5000
Q_LEVELS = 20
# Sugerencias preliminares de concentración de fondo B (ug/m3).
# Ajustar con datos locales de monitoreo.
B_SUG_BAJO = 25.0
B_SUG_ALTO = 70.0
B_DEFAULT = B_SUG_ALTO
Q_EN_KM2 = True  # True: kg/km²·día, False: kg/m²·día
MODO_MASCARA = "auto"  # "auto", "caja", "shape", "ninguna"
MIN_CELDAS_SHAPE = 20
DEBUG = True
ANOTAR_VALORES_CIUDAD = True
IMPRIMIR_VALORES_CIUDAD = True
INTERPOLAR_MAPA = True
INTERP_NX = 200
INTERP_NY = 200
INTERP_METODO = "idw"  # "idw", "linear", "nearest", "rbf"
INTERP_RELLENAR_NAN = True  # Solo aplica a linear/nearest
IDW_POTENCIA = 2.0
RBF_FUNC = "multiquadric"
RBF_SMOOTH = 0.0
BUFFER_KM_SHAPE = 0  # Expande el shape en km para incluir más celdas

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


def seleccionar_modo_escala():
    opciones = "\n".join([
        "1. Manual",
        "2. Percentil 95 (selección)"
    ])
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = simpledialog.askinteger(
            "Modo de escala",
            "Seleccione el modo de escala:\n" + opciones + "\n\nIngrese el número:",
            minvalue=1,
            maxvalue=2
        )
        root.destroy()
        if sel is None:
            raise SystemExit("No se seleccionó modo de escala.")
        if sel == 1:
            return "manual"
        if sel == 2:
            return "p95"


def seleccionar_q_maximo(etiqueta, valor_sugerido, texto_sugerencia):
    if valor_sugerido is None or not np.isfinite(valor_sugerido):
        valor_sugerido = Q_DEFAULT_MAX
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    q_max = simpledialog.askfloat(
        "Tope máximo de q",
        f"{texto_sugerencia}: {valor_sugerido:.3g}\n"
        f"Ingrese el tope máximo para {etiqueta}:",
        minvalue=0.0,
        initialvalue=valor_sugerido
    )
    root.destroy()
    if q_max is None:
        raise SystemExit("No se seleccionó tope máximo de q.")
    return float(q_max)


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


def obtener_resolucion_malla(ds):
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
    return float(dx), float(dy)


def formatear_resolucion(dx, dy):
    def fmt(v):
        if v is None or not np.isfinite(v):
            return "NA"
        if v >= 1000:
            return f"{v / 1000:.1f} km"
        return f"{v:.0f} m"
    return f"Resolución: {fmt(dx)} × {fmt(dy)}"


def cargar_geometria(shp):
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    if gdf.empty:
        raise SystemExit("El shapefile no tiene geometría.")
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union

    if BUFFER_KM_SHAPE > 0:
        try:
            gdf_proj = gdf.to_crs(epsg=32618)  # UTM zona 18N (Cali)
            geom_buf = gdf_proj.buffer(BUFFER_KM_SHAPE * 1000.0).unary_union
            geom = (
                gpd.GeoSeries([geom_buf], crs=32618)
                .to_crs(epsg=4326)
                .iloc[0]
            )
            print(f"Buffer aplicado al shape: {BUFFER_KM_SHAPE} km")
        except Exception as exc:
            print(f"Aviso: no se pudo aplicar buffer al shape ({exc}).")

    return gdf, geom


def aplicar_mascara(q_map, lat, lon, geom=None, debug=False):
    mask_box = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
    if debug:
        total = mask_box.size
        n_true = int(mask_box.sum().values) if hasattr(mask_box, "values") else int(mask_box.sum())
        print(f"Debug | Mascara caja: {n_true}/{total} celdas dentro del rango.")

    if MODO_MASCARA == "ninguna":
        if debug:
            print("Debug | Modo máscara: ninguna (sin recorte).")
        return q_map

    if MODO_MASCARA == "caja":
        if debug:
            print("Debug | Modo máscara: caja.")
        return q_map.where(mask_box)

    if hasattr(mask_box, "values") and not np.any(mask_box.values):
        if debug:
            print("Debug | Mascara caja vacía. Se usa dominio completo.")
        q_mask = q_map
    else:
        q_mask = q_map.where(mask_box)

    if geom is None or MODO_MASCARA == "caja":
        return q_mask

    try:
        try:
            from shapely import contains_xy

            mask_shape = contains_xy(geom, lon.values, lat.values)
        except Exception:
            from shapely import vectorized

            mask_shape = vectorized.contains(geom, lon.values, lat.values)

        if debug:
            n_shape = int(mask_shape.sum())
            print(f"Debug | Mascara shape: {n_shape}/{mask_shape.size} celdas dentro.")
        if MODO_MASCARA == "auto" and n_shape < MIN_CELDAS_SHAPE:
            print(
                f"Aviso: máscara shape con pocas celdas ({n_shape}). "
                f"Se usa caja (min={MIN_CELDAS_SHAPE})."
            )
            return q_mask
        mask_shape = xr.DataArray(mask_shape, coords=lat.coords, dims=lat.dims)
        q_mask_shape = q_mask.where(mask_shape)
        if not np.isfinite(q_mask_shape.values).any():
            print("Aviso: máscara del shapefile vacía. Se usa caja.")
            return q_mask
        return q_mask_shape
    except Exception as exc:
        print(f"Aviso: no se pudo aplicar máscara del shapefile ({exc}). Se usa caja.")
        return q_mask


# ======================================================
# CALCULOS
# ======================================================
def calcular_vc(u10, v10, pblh):
    ws = np.sqrt(u10**2 + v10**2)
    return ws * pblh


def calcular_q(vc, L, c_limite, c_fondo, factor_tiempo):
    delta_c = c_limite - c_fondo
    return delta_c * vc / L * factor_tiempo


def ajustar_unidades_q(q_map):
    if Q_EN_KM2:
        return q_map * 1e6  # m² -> km²
    return q_map


def unidad_q():
    return "kg/km²·día" if Q_EN_KM2 else "kg/m²·día"


def etiqueta_q():
    return f"q ({unidad_q()})"


def vc_medio_diario_2d(ds):
    u10 = ds["U10"]
    v10 = ds["V10"]
    pblh = ds["PBLH"]
    vc_full = calcular_vc(u10, v10, pblh)
    vc_mean = vc_full.mean(dim="Time")
    ntime = int(vc_full.sizes.get("Time", vc_full.shape[0]))
    return vc_mean, ntime


# ======================================================
# MAPA
# ======================================================
def _idw_interpolar(lon_vals, lat_vals, q_vals, lon_q, lat_q, power):
    vals = q_vals.astype(float)
    mask = np.isfinite(vals)
    x = lon_vals[mask]
    y = lat_vals[mask]
    v = vals[mask]
    if v.size == 0:
        return np.full_like(lon_q, np.nan, dtype=float)
    dx = lon_q[..., None] - x
    dy = lat_q[..., None] - y
    dist2 = dx * dx + dy * dy
    dist2 = np.where(dist2 == 0, 1e-12, dist2)
    w = 1.0 / np.power(dist2, power / 2.0)
    return np.sum(w * v, axis=-1) / np.sum(w, axis=-1)


def interpolar_q(lon_vals, lat_vals, q_vals, nx, ny, metodo, bounds=None):
    if bounds is None:
        lon_min, lon_max = float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))
        lat_min, lat_max = float(np.nanmin(lat_vals)), float(np.nanmax(lat_vals))
    else:
        lon_min, lon_max, lat_min, lat_max = bounds
    lon_g = np.linspace(lon_min, lon_max, nx)
    lat_g = np.linspace(lat_min, lat_max, ny)
    lon_q, lat_q = np.meshgrid(lon_g, lat_g)
    if metodo == "idw":
        q_i = _idw_interpolar(lon_vals, lat_vals, q_vals, lon_q, lat_q, IDW_POTENCIA)
        return lon_q, lat_q, q_i
    if metodo == "rbf":
        if not _SCIPY_OK:
            return lon_q, lat_q, None
        rbf = Rbf(lon_vals, lat_vals, q_vals, function=RBF_FUNC, smooth=RBF_SMOOTH)
        q_i = rbf(lon_q, lat_q)
        return lon_q, lat_q, q_i

    if not _SCIPY_OK:
        return lon_q, lat_q, None
    points = np.column_stack([lon_vals, lat_vals])
    q_i = griddata(points, q_vals, (lon_q, lat_q), method=metodo)
    if INTERP_RELLENAR_NAN and metodo != "nearest":
        q_nn = griddata(points, q_vals, (lon_q, lat_q), method="nearest")
        q_i = np.where(np.isnan(q_i), q_nn, q_i)
    return lon_q, lat_q, q_i


def mapa_q(
    q_2d,
    lat,
    lon,
    gdf,
    fecha,
    outpath,
    q_max,
    resolucion_texto,
    c_fondo,
    escala_label,
    debug=False,
    valores_ciudad=None,
):
    lat_vals = lat.values if hasattr(lat, "values") else lat
    lon_vals = lon.values if hasattr(lon, "values") else lon

    q_vals = q_2d.values
    finite = np.isfinite(q_vals)
    n_finite = int(finite.sum())

    if debug:
        if n_finite > 0:
            print(
                f"Debug | q_map min={np.nanmin(q_vals):.4e} "
                f"max={np.nanmax(q_vals):.4e} "
                f"fin={n_finite}/{finite.size}"
            )
        else:
            print("Debug | q_map sin valores finitos.")

    fig, ax = plt.subplots(figsize=(10, 8))

    plotted = False

    cmap = "RdYlBu"
    if q_max is None or not np.isfinite(q_max) or q_max <= 0:
        q_max = Q_DEFAULT_MAX

    levels = np.linspace(0, q_max, Q_LEVELS + 1)

    if n_finite > 0:
        q1, q2 = np.nanpercentile(q_vals[finite], [33, 66])
    else:
        q1 = q2 = np.nan

    if INTERPOLAR_MAPA and n_finite >= 4 and (INTERP_METODO == "idw" or _SCIPY_OK):
        if MODO_MASCARA == "ninguna":
            bounds = (
                float(np.nanmin(lon_vals)),
                float(np.nanmax(lon_vals)),
                float(np.nanmin(lat_vals)),
                float(np.nanmax(lat_vals)),
            )
        else:
            bounds = (
                LON_MIN - ZOOM_OUT,
                LON_MAX + ZOOM_OUT,
                LAT_MIN - ZOOM_OUT,
                LAT_MAX + ZOOM_OUT,
            )
        lon_q, lat_q, q_i = interpolar_q(
            lon_vals[finite],
            lat_vals[finite],
            q_vals[finite],
            INTERP_NX,
            INTERP_NY,
            INTERP_METODO,
            bounds=bounds,
        )
        if q_i is not None:
            cf = ax.contourf(
                lon_q,
                lat_q,
                q_i,
                levels=levels,
                cmap=cmap,
                extend="max",
            )
            plt.colorbar(cf, ax=ax, label=etiqueta_q())
            plotted = True
        else:
            print("Aviso: no se pudo interpolar, se usa grilla original.")

    if not plotted and n_finite < 30:
        vmin = 0.0
        vmax = q_max if q_max > 0 else 1.0
        sc = ax.scatter(
            lon_vals[finite],
            lat_vals[finite],
            c=q_vals[finite],
            s=180,
            marker="s",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="k",
            linewidth=0.5,
            zorder=4,
        )
        plt.colorbar(sc, ax=ax, label=etiqueta_q())
        plotted = True

    if not plotted:
        cf = ax.contourf(
            lon_vals,
            lat_vals,
            q_vals,
            levels=levels,
            cmap=cmap,
            extend="max",
        )
        ax.contour(lon_vals, lat_vals, q_vals, colors="k", linewidths=0.4, levels=levels)
        plt.colorbar(cf, ax=ax, label=etiqueta_q())

    gdf.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.8,
        zorder=7,
    )

    etiquetas = {
        "Cali": (-76.516651, 3.443662),
        "Yumbo": (-76.490915, 3.571668),
        "Rozo": (-76.384813, 3.610740),
        "Palmira": (-76.294686, 3.527459),
    }

    for nombre, (x, y) in etiquetas.items():
        ax.text(
            x,
            y,
            nombre,
            fontsize=10,
            fontweight="bold",
            ha="center",
            zorder=8,
        )
        if ANOTAR_VALORES_CIUDAD and valores_ciudad and nombre in valores_ciudad:
            val = valores_ciudad.get(nombre)
            if np.isfinite(val):
                ax.text(
                    x,
                    y - 0.015,
                    f"{val:.1f}",
                    fontsize=9,
                    ha="center",
                    color="black",
                    zorder=9,
                )

    ax.set_xlim(LON_MIN - ZOOM_OUT, LON_MAX + ZOOM_OUT)
    ax.set_ylim(LAT_MIN - ZOOM_OUT, LAT_MAX + ZOOM_OUT)

    ax.set_title(
        f"q PM10 diario ({unidad_q()}) | {fecha} | B {c_fondo:.1f} ug/m3"
        f" | Tope {q_max:.3g} | {escala_label}"
    )

    if n_finite > 0:
        etiqueta_cat = (
            f"Categorías ({etiqueta_q()}):\n"
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

    bottom_margin = 0.12 if (n_finite > 0 or resolucion_texto) else 0.0
    plt.tight_layout(rect=(0, bottom_margin, 1, 1))
    plt.savefig(outpath, dpi=150)
    plt.close()


# ======================================================
# MAIN
# ======================================================
def valor_en_punto(q_map, lat, lon, x, y):
    lat_vals = lat.values if hasattr(lat, "values") else lat
    lon_vals = lon.values if hasattr(lon, "values") else lon
    dist2 = (lon_vals - x) ** 2 + (lat_vals - y) ** 2
    idx = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return float(q_map.values[idx])


def main():
    debug = DEBUG
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
        gdf, geom = cargar_geometria(shp)
    except Exception as exc:
        raise SystemExit(f"No se pudo leer shapefile: {exc}") from exc

    mapas_dir = outdir / "mapas_q"
    mapas_dir.mkdir(parents=True, exist_ok=True)

    if Q_EN_KM2:
        print("Unidades de q: kg/km²·día")
    else:
        print("Unidades de q: kg/m²·día")
    print(f"Modo de máscara: {MODO_MASCARA}")
    if INTERPOLAR_MAPA:
        relleno = "sí" if INTERP_RELLENAR_NAN else "no"
        print(f"Interpolación: {INTERP_METODO} | Relleno NaN: {relleno}")
    if INTERPOLAR_MAPA and INTERP_METODO in ("linear", "nearest", "rbf") and not _SCIPY_OK:
        print("Aviso: SciPy no disponible; no se interpolará el mapa.")

    resumen_lines = []
    resumen_lines.append("RESUMEN DIARIO - q (capacidad asimilativa por área)")
    resumen_lines.append(f"Unidades: {unidad_q()}")
    resumen_lines.append(f"Modo de máscara: {MODO_MASCARA}")
    resumen_lines.append(
        f"Interpolación: {INTERP_METODO} | Relleno NaN: {'sí' if INTERP_RELLENAR_NAN else 'no'}"
    )
    resumen_lines.append("")

    resolucion_texto = ""
    try:
        ds0 = xr.open_dataset(archivos[0], decode_times=False)
        dx, dy = obtener_resolucion_malla(ds0)
        ds0.close()
        resolucion_texto = formatear_resolucion(dx, dy)
    except Exception:
        resolucion_texto = ""

    resultados = []
    p95_vals = []

    for fecha in dias_sel:
        archivos_dia = archivos_por_fecha.get(fecha, [])
        if not archivos_dia:
            continue

        vc_sum = None
        ntime_sum = 0
        L_sum = 0.0
        lat_ref = None
        lon_ref = None

        for f in archivos_dia:
            ds = xr.open_dataset(f, decode_times=False)
            if lat_ref is None:
                lat_ref = ds["XLAT"].isel(Time=0).load()
                lon_ref = ds["XLONG"].isel(Time=0).load()

            L = obtener_longitud_caracteristica(ds)
            vc_mean_2d, ntime = vc_medio_diario_2d(ds)
            vc_mean_2d = vc_mean_2d.load()

            if vc_sum is None:
                vc_sum = vc_mean_2d * ntime
            else:
                vc_sum = vc_sum + vc_mean_2d * ntime
            L_sum += L * ntime
            ntime_sum += ntime
            ds.close()

        if ntime_sum == 0 or vc_sum is None or lat_ref is None or lon_ref is None:
            print(f"Aviso: sin datos válidos para {fecha}.")
            continue

        vc_mean_dia = vc_sum / ntime_sum
        L_dia = L_sum / ntime_sum

        q_map = calcular_q(
            vc_mean_dia, L_dia, PM10_LIMITE_UG_M3, c_fondo, FACTOR_KG_DIA
        ).load()
        q_map = ajustar_unidades_q(q_map)

        if debug:
            lat_min = float(np.nanmin(lat_ref.values))
            lat_max = float(np.nanmax(lat_ref.values))
            lon_min = float(np.nanmin(lon_ref.values))
            lon_max = float(np.nanmax(lon_ref.values))
            print(f"Debug | Dominio lat: {lat_min:.4f} a {lat_max:.4f}")
            print(f"Debug | Dominio lon: {lon_min:.4f} a {lon_max:.4f}")
            print(f"Debug | L_dia: {L_dia:.3f} m | C_fondo: {c_fondo:.2f} ug/m3")
            vc_vals = vc_mean_dia.values
            if np.isfinite(vc_vals).any():
                print(
                    f"Debug | VC mean min={np.nanmin(vc_vals):.3e} "
                    f"max={np.nanmax(vc_vals):.3e} "
                    f"fin={np.isfinite(vc_vals).sum()}/{vc_vals.size}"
                )
            else:
                print("Debug | VC mean sin valores finitos.")

        q_map = aplicar_mascara(q_map, lat_ref, lon_ref, geom, debug=debug)

        etiquetas = {
            "Cali": (-76.516651, 3.443662),
            "Yumbo": (-76.490915, 3.571668),
            "Palmira": (-76.294686, 3.527459),
        }
        valores_ciudad = {}
        if ANOTAR_VALORES_CIUDAD or IMPRIMIR_VALORES_CIUDAD:
            if IMPRIMIR_VALORES_CIUDAD:
                print(f"\nValores q (aprox.) - {fecha}")
            resumen_lines.append(f"Fecha: {fecha}")
            for nombre, (x, y) in etiquetas.items():
                try:
                    q_val = valor_en_punto(q_map, lat_ref, lon_ref, x, y)
                except Exception:
                    q_val = np.nan
                valores_ciudad[nombre] = q_val
                if IMPRIMIR_VALORES_CIUDAD:
                    if np.isfinite(q_val):
                        print(f"  {nombre:<8}: {q_val:.2f} {unidad_q()}")
                    else:
                        print(f"  {nombre:<8}: NaN")
                if np.isfinite(q_val):
                    resumen_lines.append(f"  {nombre:<8}: {q_val:.2f} {unidad_q()}")
                else:
                    resumen_lines.append(f"  {nombre:<8}: NaN")
            resumen_lines.append("")

        if np.isfinite(q_map.values).any():
            p95_vals.append(float(np.nanpercentile(q_map.values, 95)))

        resultados.append(
            {
                "fecha": fecha,
                "q_map": q_map,
                "lat": lat_ref,
                "lon": lon_ref,
                "valores_ciudad": valores_ciudad if valores_ciudad else None,
            }
        )

    if not resultados:
        raise SystemExit("No hay datos válidos para los días seleccionados.")

    p95_sug = max(p95_vals) if p95_vals else Q_DEFAULT_MAX

    modo_escala = seleccionar_modo_escala()
    if modo_escala == "manual":
        q_max = seleccionar_q_maximo(etiqueta_q(), Q_DEFAULT_MAX, "Sugerencia")
        escala_label = "Escala manual"
    else:
        q_max = seleccionar_q_maximo(etiqueta_q(), p95_sug, "Percentil 95 observado")
        escala_label = "Escala P95"

    resumen_lines.insert(1, f"Tope q: {q_max:.3g} ({unidad_q()}) | Modo escala: {modo_escala}")

    for item in resultados:
        outpath = mapas_dir / f"q_PM10_diario_{item['fecha']}.png"
        mapa_q(
            item["q_map"],
            item["lat"],
            item["lon"],
            gdf,
            item["fecha"],
            outpath,
            q_max,
            resolucion_texto,
            c_fondo,
            escala_label,
            debug=debug,
            valores_ciudad=item["valores_ciudad"],
        )

    print("\nMapas de q generados en:")
    print(f"  {mapas_dir}")

    resumen_path = mapas_dir / "resumen_q_diario.txt"
    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("\n".join(resumen_lines))
    print(f"Resumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()
