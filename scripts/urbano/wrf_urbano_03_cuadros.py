import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import geopandas as gpd

# ======================================================
# CONFIGURACIÓN ESPACIAL
# ======================================================
LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.546012, -76.25
ZOOM_OUT = 0.1

# Coordenadas manuales de rótulos
CALI_LAT, CALI_LON = 3.4516, -76.5320
ROZO_LAT, ROZO_LON = 3.621598, -76.390036

# Reemplazos semánticos del shape
REEMPLAZOS_NOMBRES = {
    "EL CHOCHO": "Yumbo",
    "BOLO ITALIA 2": "Palmira"
}

# ======================================================
# CLASIFICACIÓN DE PRECIPITACIÓN
# ======================================================
def clasificar_precipitacion(p_mm):
    if p_mm < 1:
        return "Seco"
    elif p_mm < 5:
        return "Lluvia débil"
    elif p_mm < 20:
        return "Lluvia moderada"
    else:
        return "Lluvia fuerte"

# ======================================================
# DETECCIÓN DE COLUMNA DE NOMBRE EN SHAPE
# ======================================================
def detectar_columna_nombre(gdf):
    if "zu_cnmbre" in gdf.columns:
        return "zu_cnmbre"
    raise ValueError(
        f"No se encontró columna de nombre. Columnas disponibles: {list(gdf.columns)}"
    )

# ======================================================
# INTERFAZ GRÁFICA
# ======================================================
def seleccionar_archivos_wrf():
    root = tk.Tk(); root.withdraw()
    files = filedialog.askopenfilenames(title="Seleccionar archivos wrfout")
    root.destroy()
    return sorted([f for f in files if Path(f).name.startswith("wrfout")])

def seleccionar_shapefile():
    root = tk.Tk(); root.withdraw()
    shp = filedialog.askopenfilename(
        title="Seleccionar shapefile urbano",
        filetypes=[("Shapefile", "*.shp")]
    )
    root.destroy()
    return shp

def seleccionar_directorio_salida():
    root = tk.Tk(); root.withdraw()
    out = filedialog.askdirectory(title="Seleccionar carpeta de salida")
    root.destroy()
    return Path(out)

def seleccionar_rango_fechas(fechas):
    root = tk.Tk()
    root.title("Seleccionar rango de días")

    tk.Label(root, text="Día inicial").grid(row=0, column=0)
    tk.Label(root, text="Día final").grid(row=0, column=1)

    lb_ini = tk.Listbox(root, exportselection=False)
    lb_fin = tk.Listbox(root, exportselection=False)

    for f in fechas:
        lb_ini.insert(tk.END, f)
        lb_fin.insert(tk.END, f)

    lb_ini.grid(row=1, column=0)
    lb_fin.grid(row=1, column=1)

    seleccion = {}

    def aceptar():
        if not lb_ini.curselection() or not lb_fin.curselection():
            root.destroy()
            raise SystemExit("No se seleccionó rango de fechas.")
        i0 = lb_ini.curselection()[0]
        i1 = lb_fin.curselection()[0]
        if i1 < i0:
            root.destroy()
            raise SystemExit("Rango inválido.")
        seleccion["i0"], seleccion["i1"] = i0, i1
        root.destroy()

    tk.Button(root, text="Aceptar", command=aceptar).grid(
        row=2, column=0, columnspan=2
    )
    root.mainloop()
    return seleccion["i0"], seleccion["i1"]

# ======================================================
# MÁSCARA ESPACIAL
# ======================================================
def mascara_region(ds):
    lat = ds["XLAT"].isel(Time=0)
    lon = ds["XLONG"].isel(Time=0)
    return (
        (lat >= LAT_MIN) & (lat <= LAT_MAX) &
        (lon >= LON_MIN) & (lon <= LON_MAX)
    )

# ======================================================
# CÁLCULOS METEOROLÓGICOS
# ======================================================
def calcular_precipitacion_diaria(ds, mask):
    rain = ds["RAINC"] + ds["RAINNC"]
    return float(
        (rain.isel(Time=-1) - rain.isel(Time=0))
        .where(mask)
        .mean()
    )

def calcular_vc(ds, mask):
    u10 = ds["U10"]
    v10 = ds["V10"]
    pblh = ds["PBLH"]

    ws = np.sqrt(u10**2 + v10**2)
    vc_full = ws * pblh
    vc_map = vc_full.mean(dim="Time")

    vc_mean = float(vc_full.where(mask).mean())
    pblh_mean = float(pblh.where(mask).mean())

    return vc_mean, pblh_mean, vc_map

# ======================================================
# MAPA DE VC
# ======================================================
def mapa_vc(vc_2d, ds, shp, fecha, mapas_dir):
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    nombre_col = detectar_columna_nombre(gdf)

    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        vc_2d.values,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="auto"
    )

    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)

    # Yumbo y Palmira desde el shape
    for _, row in gdf.iterrows():
        nombre_raw = row[nombre_col]
        if nombre_raw in REEMPLAZOS_NOMBRES:
            nombre = REEMPLAZOS_NOMBRES[nombre_raw]
            p = row.geometry.representative_point()
            ax.text(
                p.x, p.y, nombre,
                fontsize=9, fontweight="bold",
                ha="center", va="center"
            )

    # Cali (manual)
    ax.text(
        CALI_LON, CALI_LAT, "Cali",
        fontsize=9, fontweight="bold",
        ha="center", va="center"
    )

    # Rozo (manual)
    ax.text(
        ROZO_LON, ROZO_LAT, "Rozo",
        fontsize=9, fontweight="bold",
        ha="center", va="center"
    )

    ax.set_xlim(LON_MIN - ZOOM_OUT, LON_MAX + ZOOM_OUT)
    ax.set_ylim(LAT_MIN - ZOOM_OUT, LAT_MAX + ZOOM_OUT)

    fig.colorbar(im, ax=ax, label="VC (m² s⁻¹)")
    ax.set_title(f"VC promedio diario – {fecha}")

    plt.tight_layout()
    plt.savefig(mapas_dir / f"VC_{fecha}.png", dpi=150)
    plt.close()

# ======================================================
# GRÁFICA TEMPORAL
# ======================================================
def grafica_temporal(fechas, prec, vc, pblh, outdir):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(fechas, prec,color="gray")
    ax1.set_ylabel("Precipitación diaria (mm)")

    ax2 = ax1.twinx()
    ax2.plot(fechas, vc, color="red", marker="o", linewidth=2, label="VC (m² s⁻¹)")
    ax2.plot(fechas, pblh, color="green", marker="s", linewidth=2, label="PBLH (m)")
    ax2.set_ylabel("VC / PBLH")
    ax2.legend()

    plt.title("Serie temporal diaria – Precipitación, VC y PBLH")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outdir / "serie_temporal.png", dpi=150)
    plt.close()

# ======================================================
# MAIN
# ======================================================
def main():
    archivos = seleccionar_archivos_wrf()
    shp = seleccionar_shapefile()
    outdir = seleccionar_directorio_salida()

    mapas_dir = outdir / "mapas_vc"
    graficas_dir = outdir / "graficas"
    mapas_dir.mkdir(parents=True, exist_ok=True)
    graficas_dir.mkdir(parents=True, exist_ok=True)

    fechas = [Path(a).name.split("_")[2] for a in archivos]
    i0, i1 = seleccionar_rango_fechas(fechas)

    archivos = archivos[i0:i1+1]
    fechas = fechas[i0:i1+1]

    prec, vc, pblh = [], [], []

    print("\nDESCRIPCIÓN DE VARIABLES")
    print("Prec(mm) : Precipitación acumulada diaria (RAINC + RAINNC)")
    print("PBLH(m)  : Altura media diaria de la capa límite planetaria")
    print("VC       : Coeficiente de ventilación (m² s⁻¹)\n")

    print("RESUMEN DIARIO")
    print("-"*95)
    print(f"{'Fecha':<12}{'Prec(mm)':>10}{'Nivel':>18}{'PBLH(m)':>12}{'VC':>10}")
    print("-"*95)

    for f, fecha in zip(archivos, fechas):
        ds = xr.open_dataset(f, decode_times=False)
        mask = mascara_region(ds)

        p = calcular_precipitacion_diaria(ds, mask)
        vc_m, pbl_m, vc_2d = calcular_vc(ds, mask)

        prec.append(p)
        vc.append(vc_m)
        pblh.append(pbl_m)

        print(
            f"{fecha:<12}{p:10.2f}{clasificar_precipitacion(p):>18}"
            f"{pbl_m:12.1f}{vc_m:10.1f}"
        )

        mapa_vc(vc_2d, ds, shp, fecha, mapas_dir)
        ds.close()

    grafica_temporal(fechas, prec, vc, pblh, graficas_dir)

    print("\nResultados guardados en:")
    print(f"  Carpeta base : {outdir}")
    print(f"  Mapas VC     : {mapas_dir}")
    print(f"  Gráficas     : {graficas_dir}")

# ======================================================
if __name__ == "__main__":
    main()
