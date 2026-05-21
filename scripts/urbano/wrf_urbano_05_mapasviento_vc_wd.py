import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path
from scipy.interpolate import griddata

# ======================================================
# DOMINIO ESPACIAL
# ======================================================
LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.55, -76.25
ZOOM_OUT = 0.12

# ======================================================
# INTERFAZ INTERACTIVA (PATRÓN CORRECTO)
# ======================================================
def seleccionar_archivos_wrf():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    archivos = filedialog.askopenfilenames(
        title="Seleccione archivos wrfout"
    )
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

def seleccionar_hora():
    root = tk.Tk()
    root.withdraw()
    hora = simpledialog.askinteger(
        "Hora de análisis",
        "Seleccione la hora (0–23)",
        minvalue=0,
        maxvalue=23
    )
    root.destroy()
    if hora is None:
        raise SystemExit("No se seleccionó hora.")
    return hora

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
# EXTRAER VC Y VIENTO POR HORA
# ======================================================
def extraer_vc_viento(ds, hora):
    horas = (ds["XTIME"].values / 60.0) % 24
    idx = int(np.argmin(np.abs(horas - hora)))

    u10 = ds["U10"].isel(Time=idx)
    v10 = ds["V10"].isel(Time=idx)
    pblh = ds["PBLH"].isel(Time=idx)

    ws = np.sqrt(u10**2 + v10**2)
    vc = ws * pblh

    return vc, u10, v10

# ======================================================
# MAPA VC + FLECHAS
# ======================================================
def mapa_vc_viento(vc, u10, v10, ds, shp, hora, outdir, etiqueta):

    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values

    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    col = detectar_columna_nombre(gdf)

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

    cf = ax.contourf(
        lon, lat, vc.values,
        levels=20,
        cmap="coolwarm",
        extend="both"
    )
    ax.contour(lon, lat, vc.values, colors="k", linewidths=0.4)
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

    ax.set_title(f"VC + viento | Hora {hora:02d}:00")

    plt.tight_layout()
    plt.savefig(outdir / f"VC_viento_h{hora:02d}_{etiqueta}.png", dpi=150)
    plt.close()

# ======================================================
# MAIN
# ======================================================
def main():
    archivos = seleccionar_archivos_wrf()
    shp = seleccionar_shapefile()
    outdir = seleccionar_directorio_salida()
    hora = seleccionar_hora()

    outdir.mkdir(parents=True, exist_ok=True)

    for f in archivos:
        ds = xr.open_dataset(f, decode_times=False)
        etiqueta = Path(f).name.replace("wrfout_", "")
        vc, u10, v10 = extraer_vc_viento(ds, hora)
        mapa_vc_viento(vc, u10, v10, ds, shp, hora, outdir, etiqueta)
        ds.close()

    print("✔ Mapas generados correctamente.")

if __name__ == "__main__":
    main()
