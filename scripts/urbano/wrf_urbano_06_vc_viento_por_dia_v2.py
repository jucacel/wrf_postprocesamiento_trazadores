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
# DOMINIO ESPACIAL
# ======================================================
LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.55, -76.25
ZOOM_OUT = 0.12
VC_DEFAULT_MAX = 5000
VC_LEVELS = 20

# ======================================================
# INTERFAZ INTERACTIVA (PATRÓN CORRECTO)
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
    opciones = "\n".join([f"{i}. {f}" for i, f in enumerate(fechas, 1)])
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = simpledialog.askinteger(
            "Seleccione un día",
            "Días disponibles:\n" + opciones + "\n\nIngrese el número:",
            minvalue=1,
            maxvalue=len(fechas)
        )
        root.destroy()
        if sel is None:
            raise SystemExit("No se seleccionó un día.")
        return fechas[sel - 1]

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

def seleccionar_vc_maximo(valor_defecto=VC_DEFAULT_MAX):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    vc_max = simpledialog.askfloat(
        "Tope máximo de VC",
        "Ingrese el valor máximo de VC (m² s⁻¹):",
        minvalue=0.0,
        initialvalue=valor_defecto
    )
    root.destroy()
    if vc_max is None:
        raise SystemExit("No se seleccionó tope máximo de VC.")
    return float(vc_max)

def extraer_fecha_archivo(nombre):
    match = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", nombre)
    if not match:
        return None
    return match.group(0).replace("_", "-")


# ======================================================
# UTILIDADES SHAPE
# ======================================================
def detectar_columna_nombre(gdf):
    for c in gdf.columns:
        # Acepta columnas de texto como object o string y tolera nulos.
        if gdf[c].dtype == object or str(gdf[c].dtype).startswith("string"):
            if gdf[c].astype(str).str.upper().str.contains(
                "CALI|SILO|YUMBO|PALM|ROZO", regex=True, na=False
            ).any():
                return c
    raise RuntimeError("No se detectó columna de nombres.")

# ======================================================
# EXTRAER VC Y VIENTO POR ÍNDICE
# ======================================================
def extraer_vc_viento_idx(ds, idx):
    u10 = ds["U10"].isel(Time=idx)
    v10 = ds["V10"].isel(Time=idx)
    pblh = ds["PBLH"].isel(Time=idx)

    ws = np.sqrt(u10**2 + v10**2)
    vc = ws * pblh

    return vc, u10, v10

def formatear_hora(xtime_min):
    horas_total = (xtime_min / 60.0) % 24
    hora = int(np.floor(horas_total + 1e-6))
    minutos = int(np.round((horas_total - hora) * 60))
    if minutos == 60:
        hora = (hora + 1) % 24
        minutos = 0
    return hora, minutos

# ======================================================
# MAPA VC + FLECHAS
# ======================================================
def mapa_vc_viento(vc, u10, v10, ds, shp, hora, minutos, outdir, etiqueta, vc_max, fecha):

    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values

    gdf = gpd.read_file(shp).to_crs(epsg=4326)

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

    mag = np.sqrt(u_i**2 + v_i**2)
    # Evita NaNs fuera del convex hull para no romper el quiver.
    mag = np.nan_to_num(mag, nan=0.0)
    u_n = np.nan_to_num(u_i, nan=0.0) / (mag + 1e-6)
    v_n = np.nan_to_num(v_i, nan=0.0) / (mag + 1e-6)

    # ---- Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    levels = np.linspace(0, vc_max, VC_LEVELS + 1)
    cf = ax.contourf(
        lon, lat, vc.values,
        levels=levels,
        cmap="coolwarm_r",
        extend="max"
    )
    ax.contour(lon, lat, vc.values, colors="k", linewidths=0.4, levels=levels)
    plt.colorbar(cf, ax=ax, label="Coeficiente de ventilación (m²/s)")

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

    # ---- Shapes (sin clip)
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

    ax.set_title(f"VC + viento | {fecha} | Hora {hora:02d}:{minutos:02d}")

    plt.tight_layout()
    plt.savefig(outdir / f"VC_viento_h{hora:02d}{minutos:02d}_{etiqueta}.png", dpi=150)
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
    vc_max = seleccionar_vc_maximo()

    outdir.mkdir(parents=True, exist_ok=True)

    for f in archivos_por_fecha[fecha_sel]:
        ds = xr.open_dataset(f, decode_times=False)
        etiqueta = Path(f).name.replace("wrfout_", "")
        xtimes = ds["XTIME"].values
        for idx in range(len(xtimes)):
            hora, minutos = formatear_hora(xtimes[idx])
            vc, u10, v10 = extraer_vc_viento_idx(ds, idx)
            mapa_vc_viento(vc, u10, v10, ds, shp, hora, minutos, outdir, etiqueta, vc_max, fecha_sel)
        ds.close()

    print(f"✔ Mapas generados correctamente para {fecha_sel}.")

if __name__ == "__main__":
    main()
