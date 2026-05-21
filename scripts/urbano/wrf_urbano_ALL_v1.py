#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wrf_urbano_ALL_v1.py  –  Script unificado independiente de análisis WRF
Versión 1.0 | 2026-03-08

Integra 11 herramientas de post-procesamiento de archivos wrfout:
  1.  Clasificación de precipitación diaria (R0-R3)
  2.  Mapas de VC por régimen + extremos PBLH
  3a. Mapa VC imshow + gráfica serie temporal
  3b. Mapa VC contornos rellenos + gráfica serie temporal
  4.  Ciclo horario promedio: PBLH, viento, VC, precipitación
  5.  Mapa VC + flechas de viento para una hora elegida
  6.  Mapa VC + flechas por cada timestep de un día elegido
  7.  Emax PM10 horario y diario (mapas + escalas)
  8.  Emax PM10 serie temporal + dispersión
  9.  q PM10 (capacidad asimilativa por área) serie + dispersión
  10. Mapa espacial de q por celda con interpolación
  11. Índices Allwine-Whiteman (estancamiento/recirculación/ventilación)

No depende de ningún otro script externo.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 0 – IMPORTACIONES
# ─────────────────────────────────────────────────────────────────────────────
import math
import os
import re
import threading
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, scrolledtext

try:
    import xarray as xr
    XR_OK = True
except ImportError:
    XR_OK = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import geopandas as gpd
    GEO_OK = True
except ImportError:
    GEO_OK = False

try:
    from scipy.interpolate import griddata
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    PD_OK = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 – CONSTANTES GLOBALES COMPARTIDAS
# ─────────────────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.55, -76.25
ZOOM_OUT = 0.12

# Constantes PM10 (Resolución 2254 de 2017)
PM10_LIMITE_UG_M3 = 75.0
KG_POR_UG  = 1e-9
SEG_DIA    = 86400.0
SEG_HORA   = 3600.0
FACTOR_KG_DIA  = KG_POR_UG * SEG_DIA
FACTOR_KG_HORA = KG_POR_UG * SEG_HORA

# Etiquetas de ciudades (posiciones fijas)
CITY_LABELS = {
    "Cali":    (-76.516651, 3.443662),
    "Yumbo":   (-76.490915, 3.571668),
    "Rozo":    (-76.384813, 3.610740),
    "Palmira": (-76.294686, 3.527459),
}

# Parámetros Allwine-Whiteman (1994)
AW_DEFAULT_TAU_H = 24.0
AW_DEFAULT_THRESHOLDS = {"S_c": 130.0, "R_c": 0.6, "S_cv": 250.0, "R_cv": 0.2}

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 – FUNCIONES AUXILIARES COMPARTIDAS
# ─────────────────────────────────────────────────────────────────────────────

def _check_deps(*flags_names):
    """Verifica dependencias y muestra error si falta alguna."""
    missing = [n for f, n in flags_names if not f]
    if missing:
        messagebox.showerror("Dependencias faltantes",
                             "Se requieren:\n  " + "\n  ".join(missing))
        return False
    return True


# ── Selectores GUI ────────────────────────────────────────────────────────────

def gui_seleccionar_archivos_wrf(multiples=True) -> List[Path]:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    if multiples:
        archivos = filedialog.askopenfilenames(title="Seleccione archivos wrfout")
    else:
        archivos = [filedialog.askopenfilename(title="Seleccione archivo wrfout")]
    root.destroy()
    archivos = [Path(a) for a in archivos if a and Path(a).name.startswith("wrfout")]
    if not archivos:
        messagebox.showwarning("Sin archivos", "No se seleccionaron archivos wrfout válidos.")
    return sorted(archivos)


def gui_seleccionar_shapefile() -> Optional[Path]:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    shp = filedialog.askopenfilename(
        title="Seleccione shapefile",
        filetypes=[("Shapefile", "*.shp"), ("GeoPackage", "*.gpkg"), ("Todos", "*.*")]
    )
    root.destroy()
    return Path(shp) if shp else None


def gui_seleccionar_directorio() -> Optional[Path]:
    root = tk.Tk(); root.withdraw()
    out = filedialog.askdirectory(title="Seleccione carpeta de salida")
    root.destroy()
    return Path(out) if out else None


def gui_seleccionar_hora(titulo="Hora de análisis") -> Optional[int]:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    hora = simpledialog.askinteger(titulo, "Seleccione la hora (0–23)", minvalue=0, maxvalue=23)
    root.destroy()
    return hora


def gui_seleccionar_float(titulo, mensaje, valor_default=1000.0, minval=0.0) -> Optional[float]:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    val = simpledialog.askfloat(titulo, mensaje, initialvalue=valor_default, minvalue=minval)
    root.destroy()
    return val


def gui_seleccionar_dias(fechas: List[str]) -> List[str]:
    """Ventana con Listbox de selección múltiple de fechas."""
    root = tk.Tk(); root.title("Seleccionar días"); root.attributes("-topmost", True)
    tk.Label(root, text="Seleccione uno o más días:").pack(padx=10, pady=5)
    lb = tk.Listbox(root, selectmode=tk.MULTIPLE, width=20,
                    height=min(15, len(fechas)), exportselection=False)
    for f in fechas:
        lb.insert(tk.END, f)
    lb.pack(padx=10, pady=5)
    seleccion = {"dias": []}

    def todo(): lb.select_set(0, tk.END)
    def aceptar():
        idx = lb.curselection()
        if not idx:
            messagebox.showwarning("Sin selección", "Seleccione al menos un día.")
            return
        seleccion["dias"] = [fechas[i] for i in idx]
        root.destroy()

    bf = tk.Frame(root); bf.pack(pady=(0, 10))
    tk.Button(bf, text="Seleccionar todo", command=todo).pack(side=tk.LEFT, padx=5)
    tk.Button(bf, text="Aceptar", command=aceptar).pack(side=tk.LEFT, padx=5)
    root.mainloop()
    return seleccion["dias"]


def gui_seleccionar_rango_fechas(fechas: List[str]) -> Tuple[int, int]:
    root = tk.Tk(); root.title("Rango de días")
    tk.Label(root, text="Día inicial").grid(row=0, column=0)
    tk.Label(root, text="Día final").grid(row=0, column=1)
    lb_ini = tk.Listbox(root, exportselection=False)
    lb_fin = tk.Listbox(root, exportselection=False)
    for f in fechas:
        lb_ini.insert(tk.END, f); lb_fin.insert(tk.END, f)
    lb_ini.grid(row=1, column=0); lb_fin.grid(row=1, column=1)
    sel = {}

    def aceptar():
        if not lb_ini.curselection() or not lb_fin.curselection():
            messagebox.showwarning("Selección", "Seleccione ambos extremos."); return
        i0 = lb_ini.curselection()[0]; i1 = lb_fin.curselection()[0]
        if i1 < i0:
            messagebox.showwarning("Rango", "El día final debe ser ≥ día inicial."); return
        sel["i0"], sel["i1"] = i0, i1; root.destroy()

    tk.Button(root, text="Aceptar", command=aceptar).grid(row=2, column=0, columnspan=2)
    root.mainloop()
    return sel.get("i0", 0), sel.get("i1", len(fechas)-1)


# ── Utilidades WRF ─────────────────────────────────────────────────────────────

def extraer_fecha_wrf(nombre: str) -> str:
    """Extrae fecha YYYY-MM-DD desde nombre de archivo wrfout."""
    m = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", nombre)
    if m:
        return m.group(0).replace("_", "-")
    return nombre.replace("wrfout_d01_", "").replace("_00_00_00", "")[:10]


def mascara_region(ds) -> object:
    lat = ds["XLAT"].isel(Time=0)
    lon = ds["XLONG"].isel(Time=0)
    return (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)


def calcular_vc_2d(ds, tiempo_idx=None):
    u = ds["U10"] if tiempo_idx is None else ds["U10"].isel(Time=tiempo_idx)
    v = ds["V10"] if tiempo_idx is None else ds["V10"].isel(Time=tiempo_idx)
    p = ds["PBLH"] if tiempo_idx is None else ds["PBLH"].isel(Time=tiempo_idx)
    ws = np.sqrt(u**2 + v**2)
    return ws * p


def calcular_precipitacion_diaria(ds, mask=None):
    rain = ds["RAINC"] + ds["RAINNC"]
    r = rain.isel(Time=-1) - rain.isel(Time=0)
    if mask is not None:
        r = r.where(mask)
    return float(r.mean())


def clasificar_precipitacion(p_mm: float) -> str:
    if p_mm <= 1.0:   return "R0_seco"
    if p_mm <= 5.0:   return "R1_lluvia_debil"
    if p_mm <= 20.0:  return "R2_lluvia_moderada"
    return "R3_lluvia_fuerte"


def haversine(lon1, lat1, lon2, lat2) -> float:
    R = 6371000.0
    lo1, la1, lo2, la2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlo, dla = lo2-lo1, la2-la1
    a = np.sin(dla/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlo/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def obtener_longitud_caracteristica(ds) -> float:
    for key in ("DX", "dx"):
        if key in ds.attrs:
            dx = float(ds.attrs[key])
            dy = float(ds.attrs.get("DY", ds.attrs.get("dy", dx)))
            return float(np.sqrt(dx * dy))
    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    i, j = lat.shape[0]//2, lat.shape[1]//2
    dx = haversine(lon[i,j], lat[i,j], lon[i,j+1], lat[i,j+1]) if j+1 < lon.shape[1] else haversine(lon[i,j-1], lat[i,j-1], lon[i,j], lat[i,j])
    dy = haversine(lon[i,j], lat[i,j], lon[i+1,j], lat[i+1,j]) if i+1 < lat.shape[0] else haversine(lon[i-1,j], lat[i-1,j], lon[i,j], lat[i,j])
    return float(np.sqrt(dx*dy))


def formatear_hora_xtime(xtime_min) -> Tuple[int, int]:
    total = (float(xtime_min) / 60.0) % 24
    h = int(np.floor(total + 1e-6))
    m = int(np.round((total - h) * 60))
    if m == 60: h = (h+1)%24; m = 0
    return h, m


def detectar_columna_nombre(gdf) -> str:
    for c in gdf.columns:
        if gdf[c].dtype == object or str(gdf[c].dtype).startswith("string"):
            if gdf[c].astype(str).str.upper().str.contains(
                    "CALI|SILO|YUMBO|PALM|ROZO", regex=True, na=False).any():
                return c
    if "zu_cnmbre" in gdf.columns:
        return "zu_cnmbre"
    raise RuntimeError("No se detectó columna de nombres en el shapefile.")


def agregar_labels_ciudades(ax):
    for nombre, (x, y) in CITY_LABELS.items():
        ax.text(x, y, nombre, fontsize=9, fontweight="bold", ha="center", zorder=8)


def quiver_normalizado(ax, ds, u2d, v2d, nx=66, ny=66):
    """Agrega flechas de dirección de viento normalizadas sobre el eje ax."""
    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    lon_g = np.linspace(LON_MIN-ZOOM_OUT, LON_MAX+ZOOM_OUT, nx)
    lat_g = np.linspace(LAT_MIN-ZOOM_OUT, LAT_MAX+ZOOM_OUT, ny)
    lon_q, lat_q = np.meshgrid(lon_g, lat_g)
    pts = (lon.flatten(), lat.flatten())
    u_i = griddata(pts, np.asarray(u2d).flatten(), (lon_q, lat_q), method="linear")
    v_i = griddata(pts, np.asarray(v2d).flatten(), (lon_q, lat_q), method="linear")
    mag = np.sqrt(u_i**2 + v_i**2); mag = np.nan_to_num(mag, nan=0.0)
    un = np.nan_to_num(u_i, nan=0.0) / (mag+1e-6)
    vn = np.nan_to_num(v_i, nan=0.0) / (mag+1e-6)
    ax.quiver(lon_q[::3,::3], lat_q[::3,::3], un[::3,::3], vn[::3,::3],
              scale=28, width=0.0028, color="black", zorder=6)


def promedio_horario_lista(horas_list, data_list):
    """Promedia valores por hora UTC [0-23] sobre varios días."""
    horas_unicas = np.arange(24)
    promedio = np.zeros(24)
    for h in horas_unicas:
        vals = []
        for horas, data in zip(horas_list, data_list):
            vals.extend(data[horas.astype(int) == h])
        promedio[h] = np.mean(vals) if vals else np.nan
    return horas_unicas, promedio


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 1 – CLASIFICACIÓN DE PRECIPITACIÓN DIARIA
# ═════════════════════════════════════════════════════════════════════════════

def run_01_clasificacion_prec():
    if not _check_deps((XR_OK, "xarray")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return

    lines = [
        "CLASIFICACIÓN DIARIA POR REGÍMENES DE PRECIPITACIÓN (WRF)",
        "Región tropical – Valle geográfico del río Cauca",
        "=" * 70,
        f"{'Fecha':<12} {'P(mm)':>10} {'Régimen':>30}",
        "-" * 60,
    ]
    for arch in archivos:
        fecha = extraer_fecha_wrf(arch.name)
        ds = xr.open_dataset(arch, decode_times=False, mask_and_scale=False)
        p = calcular_precipitacion_diaria(ds)
        reg = clasificar_precipitacion(p)
        lines.append(f"{fecha:<12} {p:10.2f} {reg:>30}")
        ds.close()

    lines += ["", "Regímenes:", "  R0_seco (P≤1mm) | R1_débil (1<P≤5) | R2_moderada (5<P≤20) | R3_fuerte (P>20)"]

    # Mostrar en ventana de texto
    win = tk.Toplevel()
    win.title("Precipitación – Régimen diario")
    win.attributes("-topmost", True)
    st = scrolledtext.ScrolledText(win, width=75, height=22, font=("Courier", 11))
    st.pack(padx=10, pady=10)
    st.insert("1.0", "\n".join(lines))
    st.configure(state="disabled")
    tk.Button(win, text="Cerrar", command=win.destroy).pack(pady=6)


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 2 – MAPAS VC POR RÉGIMEN + EXTREMOS PBLH
# ═════════════════════════════════════════════════════════════════════════════

def run_02_vc_por_regimen():
    if not _check_deps((XR_OK, "xarray"), (MPL_OK, "matplotlib")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return
    outdir.mkdir(parents=True, exist_ok=True)

    vc_diarios = {"promedio": [], "max": [], "min": [], "mediodia": []}
    regimenes, fechas, p_mm_list = [], [], []

    for arch in archivos:
        fecha = extraer_fecha_wrf(arch.name)
        fechas.append(fecha)
        ds = xr.open_dataset(arch, decode_times=False, mask_and_scale=False)

        p = calcular_precipitacion_diaria(ds)
        p_mm_list.append(p)
        regimenes.append(clasificar_precipitacion(p))

        u10 = ds["U10"].load(); v10 = ds["V10"].load(); pblh = ds["PBLH"].load()
        ws = np.sqrt(u10**2 + v10**2)
        vc_prom = (ws * pblh).mean(dim="Time", keep_attrs=True)

        pblh_mean = pblh.mean(dim=("south_north", "west_east"))
        idx_max = int(pblh_mean.argmax(dim="Time"))
        idx_min = int(pblh_mean.argmin(dim="Time"))
        t_med = len(ds["Time"]) // 2

        vc_diarios["promedio"].append(vc_prom)
        vc_diarios["max"].append((ws.isel(Time=idx_max) * pblh.isel(Time=idx_max)))
        vc_diarios["min"].append((ws.isel(Time=idx_min) * pblh.isel(Time=idx_min)))
        vc_diarios["mediodia"].append((ws.isel(Time=t_med) * pblh.isel(Time=t_med)))
        ds.close()

    regimenes_unicos = ["R0_seco", "R1_lluvia_debil", "R2_lluvia_moderada", "R3_lluvia_fuerte"]
    tipos_vc = ["promedio", "max", "min", "mediodia"]
    conteo = 0
    for r in regimenes_unicos:
        idx_r = [i for i, reg in enumerate(regimenes) if reg == r]
        if not idx_r: continue
        for tipo in tipos_vc:
            campos = [vc_diarios[tipo][i] for i in idx_r]
            vc_stack = xr.concat(campos, dim="dia")
            vc_media = vc_stack.mean("dia")
            fig, ax = plt.subplots(figsize=(8, 6))
            vc_media.plot(ax=ax, cmap="viridis")
            ax.set_title(f"VC {tipo} promedio – {r}")
            plt.tight_layout()
            outpath = outdir / f"VC_{tipo}_{r}.png"
            plt.savefig(outpath, dpi=150)
            plt.close()
            conteo += 1

    messagebox.showinfo("Listo", f"{conteo} mapas guardados en:\n{outdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 3a – MAPA VC IMSHOW + SERIE TEMPORAL (03_cuadros)
# ═════════════════════════════════════════════════════════════════════════════

REEMPLAZOS_SHAPE = {"EL CHOCHO": "Yumbo", "BOLO ITALIA 2": "Palmira"}
CALI_LON, CALI_LAT = -76.5320, 3.4516
ROZO_LON, ROZO_LAT = -76.390036, 3.621598


def _mapa_vc_imshow(vc_2d, ds, gdf, fecha, mapas_dir):
    nombre_col = detectar_columna_nombre(gdf)
    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vc_2d.values, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)
    for _, row in gdf.iterrows():
        nombre_raw = row[nombre_col]
        if nombre_raw in REEMPLAZOS_SHAPE:
            p = row.geometry.representative_point()
            ax.text(p.x, p.y, REEMPLAZOS_SHAPE[nombre_raw], fontsize=9, fontweight="bold", ha="center", va="center")
    ax.text(CALI_LON, CALI_LAT, "Cali", fontsize=9, fontweight="bold", ha="center", va="center")
    ax.text(ROZO_LON, ROZO_LAT, "Rozo", fontsize=9, fontweight="bold", ha="center", va="center")
    ax.set_xlim(LON_MIN-0.1, LON_MAX+0.1); ax.set_ylim(LAT_MIN-0.1, LAT_MAX+0.1)
    fig.colorbar(im, ax=ax, label="VC (m² s⁻¹)")
    ax.set_title(f"VC promedio diario – {fecha}")
    plt.tight_layout()
    plt.savefig(mapas_dir / f"VC_{fecha}.png", dpi=150); plt.close()


def _grafica_serie_pblh_vc(fechas, prec, vc, pblh, outdir, nombre="serie_temporal.png"):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(fechas, prec, color="gray"); ax1.set_ylabel("Precipitación diaria (mm)")
    ax2 = ax1.twinx()
    ax2.plot(fechas, vc,   color="red",   marker="o", linewidth=2, label="VC (m² s⁻¹)")
    ax2.plot(fechas, pblh, color="green", marker="s", linewidth=2, label="PBLH (m)")
    ax2.set_ylabel("VC / PBLH"); ax2.legend()
    plt.title("Serie temporal diaria – Precipitación, VC y PBLH")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(outdir / nombre, dpi=150); plt.close()


def run_03a_vc_imshow():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(GEO_OK,"geopandas")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    shp = gui_seleccionar_shapefile()
    if not shp: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return

    mapas_dir = outdir / "mapas_vc"; graficas_dir = outdir / "graficas"
    mapas_dir.mkdir(parents=True, exist_ok=True); graficas_dir.mkdir(parents=True, exist_ok=True)

    fechas = [extraer_fecha_wrf(a.name) for a in archivos]
    i0, i1 = gui_seleccionar_rango_fechas(fechas)
    archivos = archivos[i0:i1+1]; fechas = fechas[i0:i1+1]

    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    prec, vc_vals, pblh_vals = [], [], []

    for arch, fecha in zip(archivos, fechas):
        ds = xr.open_dataset(arch, decode_times=False)
        mask = mascara_region(ds)
        p = calcular_precipitacion_diaria(ds, mask)
        u10 = ds["U10"]; v10 = ds["V10"]; pblh = ds["PBLH"]
        ws = np.sqrt(u10**2 + v10**2)
        vc_full = ws * pblh
        vc_map = vc_full.mean(dim="Time")
        vc_m = float(vc_full.where(mask).mean())
        pbl_m = float(pblh.where(mask).mean())
        prec.append(p); vc_vals.append(vc_m); pblh_vals.append(pbl_m)
        _mapa_vc_imshow(vc_map, ds, gdf, fecha, mapas_dir)
        ds.close()

    _grafica_serie_pblh_vc(fechas, prec, vc_vals, pblh_vals, graficas_dir)
    messagebox.showinfo("Listo", f"Mapas y gráfica temporal guardados en:\n{outdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 3b – MAPA VC CONTORNOS RELLENOS + SERIE TEMPORAL
# ═════════════════════════════════════════════════════════════════════════════

def _mapa_vc_contorno(vc_2d, ds, gdf, fecha, mapas_dir, vc_max):
    nombre_col = detectar_columna_nombre(gdf)
    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    levels = np.linspace(0, vc_max, 21)
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(lon, lat, vc_2d.values, levels=levels, cmap="coolwarm_r", extend="max")
    ax.contour(lon, lat, vc_2d.values, colors="k", linewidths=0.5, levels=levels)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)
    for _, row in gdf.iterrows():
        nr = row[nombre_col]
        if nr in REEMPLAZOS_SHAPE:
            p = row.geometry.representative_point()
            ax.text(p.x, p.y, REEMPLAZOS_SHAPE[nr], fontsize=9, fontweight="bold", ha="center", va="center")
    ax.text(CALI_LON, CALI_LAT, "Cali", fontsize=9, fontweight="bold", ha="center", va="center")
    ax.text(ROZO_LON, ROZO_LAT, "Rozo", fontsize=9, fontweight="bold", ha="center", va="center")
    ax.set_xlim(LON_MIN-0.1, LON_MAX+0.1); ax.set_ylim(LAT_MIN-0.1, LAT_MAX+0.1)
    fig.colorbar(cf, ax=ax, label="VC (m² s⁻¹)")
    ax.set_title(f"VC promedio diario (contorno) – {fecha}")
    plt.tight_layout()
    plt.savefig(mapas_dir / f"VC_contorno_{fecha}.png", dpi=150); plt.close()


def run_03b_vc_contorno():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(GEO_OK,"geopandas")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    shp = gui_seleccionar_shapefile()
    if not shp: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return
    vc_max = gui_seleccionar_float("Tope VC", "Valor máximo de VC (m² s⁻¹):", 5000.0)
    if vc_max is None: return

    mapas_dir = outdir / "mapas_vc_contorno"; graficas_dir = outdir / "graficas"
    mapas_dir.mkdir(parents=True, exist_ok=True); graficas_dir.mkdir(parents=True, exist_ok=True)

    fechas = [extraer_fecha_wrf(a.name) for a in archivos]
    i0, i1 = gui_seleccionar_rango_fechas(fechas)
    archivos = archivos[i0:i1+1]; fechas = fechas[i0:i1+1]

    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    prec, vc_vals, pblh_vals = [], [], []

    for arch, fecha in zip(archivos, fechas):
        ds = xr.open_dataset(arch, decode_times=False)
        mask = mascara_region(ds)
        p = calcular_precipitacion_diaria(ds, mask)
        u10 = ds["U10"]; v10 = ds["V10"]; pblh = ds["PBLH"]
        ws = np.sqrt(u10**2 + v10**2)
        vc_full = ws * pblh
        vc_map = vc_full.mean(dim="Time")
        vc_m = float(vc_full.where(mask).mean())
        pbl_m = float(pblh.where(mask).mean())
        prec.append(p); vc_vals.append(vc_m); pblh_vals.append(pbl_m)
        _mapa_vc_contorno(vc_map, ds, gdf, fecha, mapas_dir, vc_max)
        ds.close()

    _grafica_serie_pblh_vc(fechas, prec, vc_vals, pblh_vals, graficas_dir, "serie_temporal_contorno.png")
    messagebox.showinfo("Listo", f"Mapas de contorno guardados en:\n{outdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 4 – CICLO HORARIO PROMEDIO: PBLH, VIENTO, VC, PRECIPITACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def _extraer_horario(ds, mask):
    u10 = ds["U10"]; v10 = ds["V10"]; pblh = ds["PBLH"]
    rain = ds["RAINC"] + ds["RAINNC"]
    ws = np.sqrt(u10**2 + v10**2)
    vc = ws * pblh
    def _sp(arr): return arr.where(mask).mean(dim=("south_north","west_east")).values
    pblh_m = _sp(pblh); ws_m = _sp(ws); vc_m = _sp(vc); rain_m = _sp(rain)
    rain_h = np.diff(rain_m, prepend=rain_m[0])
    time_vals = ds["XTIME"].values
    horas = np.array([t/60 % 24 for t in time_vals])
    return horas, pblh_m, ws_m, vc_m, rain_h


def run_04_ciclo_horario():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return
    outdir.mkdir(parents=True, exist_ok=True)

    pblh_list, ws_list, vc_list, rain_list, horas_list = [], [], [], [], []
    fechas = [extraer_fecha_wrf(a.name) for a in archivos]

    for arch in archivos:
        ds = xr.open_dataset(arch, decode_times=False)
        mask = mascara_region(ds)
        horas, pblh_h, ws_h, vc_h, rain_h = _extraer_horario(ds, mask)
        horas_list.append(horas); pblh_list.append(pblh_h)
        ws_list.append(ws_h); vc_list.append(vc_h); rain_list.append(rain_h)
        ds.close()

    hp, pblh_prom = promedio_horario_lista(horas_list, pblh_list)
    _,  ws_prom   = promedio_horario_lista(horas_list, ws_list)
    _,  vc_prom   = promedio_horario_lista(horas_list, vc_list)
    _,  rain_prom = promedio_horario_lista(horas_list, rain_list)
    dias_txt = "Días: " + ", ".join(fechas)

    graficas = [
        ("PBLH (m)", pblh_prom, "green", "o", "PBLH_promedio_horario.png", "bar"),
        ("Viento (m/s)", ws_prom, "blue", "s", "Viento_promedio_horario.png", "line"),
        ("VC (m²/s)", vc_prom, "red", "^", "VC_promedio_horario.png", "line"),
        ("Precipitación (mm/h)", rain_prom, "cyan", None, "Precipitacion_promedio_horario.png", "bar"),
    ]
    for ylabel, datos, color, marker, fname, tipo in graficas:
        plt.figure(figsize=(8,5))
        if tipo == "bar": plt.bar(hp, datos, color=color, alpha=0.7)
        else: plt.plot(hp, datos, marker=marker, color=color)
        plt.xlabel("Hora"); plt.ylabel(ylabel)
        plt.title(f"{ylabel} – promedio horario"); plt.grid(True)
        plt.figtext(0.5, -0.05, dias_txt, ha="center", fontsize=9)
        plt.savefig(outdir/fname, dpi=150, bbox_inches="tight"); plt.close()

    # Gráfica combinada
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(hp, pblh_prom, "o-", color="green", label="PBLH (m)"); ax1.set_ylabel("PBLH (m)", color="green")
    ax2 = ax1.twinx(); ax2.plot(hp, ws_prom, "s-", color="blue", label="Viento (m/s)"); ax2.set_ylabel("Viento (m/s)", color="blue")
    ax3 = ax1.twinx(); ax3.spines["right"].set_position(("axes",1.1))
    ax3.plot(hp, vc_prom, "^-", color="red", label="VC (m²/s)"); ax3.set_ylabel("VC (m²/s)", color="red")
    ax4 = ax1.twinx(); ax4.spines["right"].set_position(("axes",1.2))
    ax4.bar(hp, rain_prom, color="cyan", alpha=0.3, width=0.8, label="Prec (mm/h)"); ax4.set_ylabel("Prec (mm/h)", color="cyan")
    plt.title("Promedio horario – PBLH, Viento, VC y Precipitación")
    plt.figtext(0.5, -0.05, dias_txt, ha="center", fontsize=9)
    plt.grid(True); plt.savefig(outdir/"Variables_combinadas_multiejes.png", dpi=150, bbox_inches="tight"); plt.close()
    messagebox.showinfo("Listo", f"Gráficas horarias guardadas en:\n{outdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 5 – MAPA VC + FLECHAS DE VIENTO PARA UNA HORA
# ═════════════════════════════════════════════════════════════════════════════

def run_05_mapa_vc_viento_hora():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(GEO_OK,"geopandas"),(SCIPY_OK,"scipy")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    shp = gui_seleccionar_shapefile()
    if not shp: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return
    hora = gui_seleccionar_hora()
    if hora is None: return
    outdir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(shp).to_crs(epsg=4326)

    for arch in archivos:
        ds = xr.open_dataset(arch, decode_times=False)
        horas_arr = (ds["XTIME"].values / 60.0) % 24
        idx = int(np.argmin(np.abs(horas_arr - hora)))
        u10 = ds["U10"].isel(Time=idx)
        v10 = ds["V10"].isel(Time=idx)
        pblh = ds["PBLH"].isel(Time=idx)
        ws = np.sqrt(u10**2 + v10**2)
        vc = ws * pblh
        lat = ds["XLAT"].isel(Time=0).values
        lon = ds["XLONG"].isel(Time=0).values

        fig, ax = plt.subplots(figsize=(10,8))
        cf = ax.contourf(lon, lat, vc.values, levels=20, cmap="coolwarm", extend="both")
        ax.contour(lon, lat, vc.values, colors="k", linewidths=0.4)
        plt.colorbar(cf, ax=ax, label="VC (m²/s)")
        quiver_normalizado(ax, ds, u10, v10)
        gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, zorder=7)
        agregar_labels_ciudades(ax)
        ax.set_xlim(LON_MIN-ZOOM_OUT, LON_MAX+ZOOM_OUT); ax.set_ylim(LAT_MIN-ZOOM_OUT, LAT_MAX+ZOOM_OUT)
        ax.set_title(f"VC + viento | Hora {hora:02d}:00")
        plt.tight_layout()
        etiqueta = arch.name.replace("wrfout_","")
        plt.savefig(outdir/f"VC_viento_h{hora:02d}_{etiqueta}.png", dpi=150); plt.close()
        ds.close()

    messagebox.showinfo("Listo", f"Mapas guardados en:\n{outdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 6 – MAPA VC + FLECHAS POR CADA TIMESTEP DE UN DÍA
# ═════════════════════════════════════════════════════════════════════════════

def run_06_vc_viento_por_dia():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(GEO_OK,"geopandas"),(SCIPY_OK,"scipy")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return

    # Agrupar por fecha
    fechas_dict: dict = {}
    for a in archivos:
        f = extraer_fecha_wrf(a.name)
        fechas_dict.setdefault(f, []).append(a)
    fechas = sorted(fechas_dict.keys())

    root = tk.Tk(); root.withdraw()
    opciones = "\n".join([f"{i+1}. {f}" for i,f in enumerate(fechas)])
    sel = simpledialog.askinteger("Seleccione día", f"Días disponibles:\n{opciones}\n\nIngrese número:",
                                   minvalue=1, maxvalue=len(fechas))
    root.destroy()
    if sel is None: return
    fecha_sel = fechas[sel-1]

    shp = gui_seleccionar_shapefile()
    if not shp: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return
    vc_max = gui_seleccionar_float("Tope VC", "Valor máximo de VC (m² s⁻¹):", 5000.0)
    if vc_max is None: return
    outdir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    levels = np.linspace(0, vc_max, 21)

    for arch in fechas_dict[fecha_sel]:
        ds = xr.open_dataset(arch, decode_times=False)
        xtimes = ds["XTIME"].values
        etiqueta = arch.name.replace("wrfout_","")
        lat = ds["XLAT"].isel(Time=0).values
        lon = ds["XLONG"].isel(Time=0).values

        for idx in range(len(xtimes)):
            h, m = formatear_hora_xtime(xtimes[idx])
            u10 = ds["U10"].isel(Time=idx); v10 = ds["V10"].isel(Time=idx)
            pblh = ds["PBLH"].isel(Time=idx)
            ws = np.sqrt(u10**2 + v10**2); vc = ws * pblh

            fig, ax = plt.subplots(figsize=(10,8))
            cf = ax.contourf(lon, lat, vc.values, levels=levels, cmap="coolwarm_r", extend="max")
            ax.contour(lon, lat, vc.values, colors="k", linewidths=0.4, levels=levels)
            plt.colorbar(cf, ax=ax, label="VC (m²/s)")
            quiver_normalizado(ax, ds, u10, v10)
            gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, zorder=7)
            agregar_labels_ciudades(ax)
            ax.set_xlim(LON_MIN-ZOOM_OUT, LON_MAX+ZOOM_OUT); ax.set_ylim(LAT_MIN-ZOOM_OUT, LAT_MAX+ZOOM_OUT)
            ax.set_title(f"VC + viento | {fecha_sel} | {h:02d}:{m:02d}")
            plt.tight_layout()
            plt.savefig(outdir/f"VC_viento_h{h:02d}{m:02d}_{etiqueta}.png", dpi=150); plt.close()

        ds.close()
    messagebox.showinfo("Listo", f"Mapas por timestep guardados en:\n{outdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 7 – EMAX PM10 HORARIO Y DIARIO (MAPAS)
# ═════════════════════════════════════════════════════════════════════════════

def _mapa_emax_viento(emax, u10, v10, ds, gdf, titulo, outpath, cbar_label, emax_max, shp):
    lat = ds["XLAT"].isel(Time=0).values
    lon = ds["XLONG"].isel(Time=0).values
    levels = np.linspace(0, emax_max, 21)
    data = np.asarray(emax.values)
    valid = data[np.isfinite(data)]

    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(lon, lat, emax.values, levels=levels, cmap="coolwarm_r", extend="max")
    ax.contour(lon, lat, emax.values, colors="k", linewidths=0.4, levels=levels)
    plt.colorbar(cf, ax=ax, label=cbar_label)
    if valid.size:
        q1, q2 = np.nanpercentile(valid, [33, 66])
        fig.text(0.02, 0.02, f"Baja < {q1:.2g}\nMedia {q1:.2g}-{q2:.2g}\nAlta > {q2:.2g}",
                 fontsize=9, va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
    quiver_normalizado(ax, ds, u10, v10)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, zorder=7)
    agregar_labels_ciudades(ax)
    ax.set_xlim(LON_MIN-ZOOM_OUT, LON_MAX+ZOOM_OUT); ax.set_ylim(LAT_MIN-ZOOM_OUT, LAT_MAX+ZOOM_OUT)
    ax.set_title(titulo)
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(outpath, dpi=150); plt.close()


def run_07_emax_pm10_mapas():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(GEO_OK,"geopandas"),(SCIPY_OK,"scipy")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return

    # Agrupar por fecha
    fechas_dict: dict = {}
    for a in archivos:
        f = extraer_fecha_wrf(a.name)
        fechas_dict.setdefault(f, []).append(a)
    fechas = sorted(fechas_dict.keys())
    root = tk.Tk(); root.withdraw()
    opciones = "\n".join([f"{i+1}. {f}" for i, f in enumerate(fechas)])
    sel = simpledialog.askinteger("Seleccione día", f"Días disponibles:\n{opciones}\n\nIngrese número:",
                                   minvalue=1, maxvalue=len(fechas))
    root.destroy()
    if sel is None: return
    fecha_sel = fechas[sel-1]

    shp = gui_seleccionar_shapefile()
    if not shp: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return
    emax_max_h = gui_seleccionar_float("Tope Emax horario", "Valor máximo Emax kg/h:", 5000.0)
    if emax_max_h is None: return
    emax_max_d = gui_seleccionar_float("Tope Emax diario", "Valor máximo Emax kg/día:", 50000.0)
    if emax_max_d is None: return

    outdir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(shp).to_crs(epsg=4326)

    for arch in fechas_dict[fecha_sel]:
        ds = xr.open_dataset(arch, decode_times=False)
        L = obtener_longitud_caracteristica(ds)
        u10 = ds["U10"]; v10 = ds["V10"]; pblh = ds["PBLH"]
        ws = np.sqrt(u10**2 + v10**2)
        vc = ws * pblh
        xtimes = ds["XTIME"].values
        etq = arch.name.replace("wrfout_", "")

        # Mapas horarios
        for idx in range(len(xtimes)):
            h, m = formatear_hora_xtime(xtimes[idx])
            emax_h = PM10_LIMITE_UG_M3 * vc.isel(Time=idx) * L * FACTOR_KG_HORA
            titulo = f"Emax PM10 (kg/h) | {fecha_sel} {h:02d}:{m:02d}"
            outpath = outdir / f"Emax_PM10_h{h:02d}{m:02d}_{fecha_sel}_{etq}.png"
            _mapa_emax_viento(emax_h, u10.isel(Time=idx), v10.isel(Time=idx),
                              ds, gdf, titulo, outpath, "Emax PM10 (kg/h)", emax_max_h, shp)

        # Mapa diario
        vc_mean = vc.mean(dim="Time")
        u10_m = u10.mean(dim="Time"); v10_m = v10.mean(dim="Time")
        emax_d = PM10_LIMITE_UG_M3 * vc_mean * L * FACTOR_KG_DIA
        titulo_d = f"Emax PM10 diario (kg/día) | {fecha_sel}"
        _mapa_emax_viento(emax_d, u10_m, v10_m, ds, gdf, titulo_d,
                          outdir / f"Emax_PM10_diario_{fecha_sel}_{etq}.png",
                          "Emax PM10 (kg/día)", emax_max_d, shp)
        ds.close()

    messagebox.showinfo("Listo", f"Mapas Emax PM10 guardados en:\n{outdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 8 – EMAX PM10 SERIE TEMPORAL + DISPERSIÓN
# ═════════════════════════════════════════════════════════════════════════════

def _grafica_serie_emax(fechas, emax, vc, precip, outpath, titulo="Emax PM10"):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fechas)); w = 0.35
    ax1.plot(x, emax, "o-", color="tab:red", linewidth=2, label="Emax (kg/día)")
    ax1.set_ylabel("Emax PM10 (kg/día)", color="tab:red")
    ax2 = ax1.twinx()
    ax2.bar(x-w/2, vc, width=w, color="tab:blue", alpha=0.45, label="VC (m²/s)")
    ax2.set_ylabel("VC (m²/s)", color="tab:blue")
    ax3 = ax1.twinx(); ax3.spines["right"].set_position(("axes", 1.1))
    ax3.bar(x+w/2, precip, width=w, color="tab:green", alpha=0.35, label="Prec (mm)")
    ax3.set_ylabel("Prec (mm)", color="tab:green")
    ax1.set_xticks(x); ax1.set_xticklabels(fechas, rotation=45, ha="right")
    plt.title(titulo); plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()


def _grafica_dispersion(fechas, y, vc, precip, outpath, ylabel, titulo):
    y = np.asarray(y); vc = np.asarray(vc); precip = np.asarray(precip)
    vc_ok = np.isfinite(vc); y_ok = np.isfinite(y); p_ok = np.isfinite(precip)
    fig, ax = plt.subplots(figsize=(7, 6))
    mask_c = vc_ok & y_ok & p_ok
    mask_g = vc_ok & y_ok & ~p_ok
    if mask_c.any():
        sc = ax.scatter(vc[mask_c], y[mask_c], c=precip[mask_c], cmap="viridis",
                        s=70, edgecolor="black", alpha=0.85)
        plt.colorbar(sc, ax=ax, label="Prec diaria (mm)")
    if mask_g.any():
        ax.scatter(vc[mask_g], y[mask_g], color="gray", s=70, edgecolor="black", alpha=0.85)
    for i, f in enumerate(fechas):
        if np.isfinite(vc[i]) and np.isfinite(y[i]):
            ax.text(vc[i], y[i], f, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("VC (m²/s)"); ax.set_ylabel(ylabel); ax.set_title(titulo)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()


def _calcular_vc_emax_prec_por_fecha(archivos, fechas_sel, shp):
    """Retorna listas (fechas, vc, emax, prec) para días seleccionados."""
    try:
        gdf = gpd.read_file(shp).to_crs(epsg=4326)
        geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    except Exception:
        geom = None

    fechas_dict: dict = {}
    for a in archivos:
        f = extraer_fecha_wrf(a.name)
        fechas_dict.setdefault(f, []).append(a)

    res_f, res_vc, res_em, res_pr = [], [], [], []
    for fecha in sorted(fechas_sel):
        archs = fechas_dict.get(fecha, [])
        vc_sum = ntime_sum = L_sum = 0.0; prec_sum = 0.0; prec_ok = False
        for arch in archs:
            ds = xr.open_dataset(arch, decode_times=False)
            mask = mascara_region(ds)
            L = obtener_longitud_caracteristica(ds)
            u10 = ds["U10"]; v10 = ds["V10"]; pblh = ds["PBLH"]
            vc_full = np.sqrt(u10**2 + v10**2) * pblh
            vc_m = float(vc_full.where(mask).mean(dim=("Time","south_north","west_east")))
            nt = int(vc_full.sizes.get("Time", vc_full.shape[0]))
            if np.isfinite(vc_m): vc_sum += vc_m*nt; L_sum += L*nt; ntime_sum += nt
            p = calcular_precipitacion_diaria(ds, mask)
            if np.isfinite(p): prec_sum += p; prec_ok = True
            ds.close()
        if ntime_sum > 0:
            vc_d = vc_sum/ntime_sum; L_d = L_sum/ntime_sum
            emax_d = PM10_LIMITE_UG_M3 * vc_d * L_d * FACTOR_KG_DIA
        else:
            vc_d = emax_d = np.nan
        res_f.append(fecha); res_vc.append(vc_d)
        res_em.append(emax_d); res_pr.append(prec_sum if prec_ok else np.nan)
    return res_f, res_vc, res_em, res_pr


def run_08_emax_pm10_serie():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(GEO_OK,"geopandas")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    shp = gui_seleccionar_shapefile()
    if not shp: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return
    outdir.mkdir(parents=True, exist_ok=True)

    fechas_disp = sorted({extraer_fecha_wrf(a.name) for a in archivos})
    dias_sel = gui_seleccionar_dias(fechas_disp)
    if not dias_sel: return

    f_list, vc_list, em_list, pr_list = _calcular_vc_emax_prec_por_fecha(archivos, dias_sel, shp)

    gdir = outdir / "graficas_emax"; gdir.mkdir(parents=True, exist_ok=True)
    _grafica_serie_emax(f_list, em_list, vc_list, pr_list,
                        gdir/"serie_emax_vc_precip.png", "Serie diaria – Emax PM10, VC, Prec")
    _grafica_dispersion(f_list, em_list, vc_list, pr_list,
                        gdir/"dispersion_emax_vc.png",
                        "Emax PM10 (kg/día)", "Dispersión Emax – VC con precipitación")

    # Imprimir tabla
    print("\nRESUMEN DIARIO – Emax PM10")
    print(f"{'Fecha':<12}{'VC(m²/s)':>12}{'Emax(kg/d)':>14}{'Prec(mm)':>12}")
    for f, vc, em, pr in zip(f_list, vc_list, em_list, pr_list):
        print(f"{f:<12}{vc:12.2f}{em:14.2f}{pr:12.2f}")

    messagebox.showinfo("Listo", f"Gráficas Emax PM10 guardadas en:\n{gdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 9 – q PM10 SERIE TEMPORAL + DISPERSIÓN
# ═════════════════════════════════════════════════════════════════════════════

B_SUG_BAJO = 25.0
B_SUG_ALTO = 70.0


def run_09_q_pm10_serie():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(GEO_OK,"geopandas")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    shp = gui_seleccionar_shapefile()
    if not shp: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return

    c_fondo = gui_seleccionar_float(
        "Concentración de fondo",
        f"Ingrese concentración de fondo B (µg/m³)\nSugerencias: bajo={B_SUG_BAJO}, alto={B_SUG_ALTO}\n(0 si no aplica):",
        B_SUG_ALTO
    )
    if c_fondo is None: return
    outdir.mkdir(parents=True, exist_ok=True)

    fechas_disp = sorted({extraer_fecha_wrf(a.name) for a in archivos})
    dias_sel = gui_seleccionar_dias(fechas_disp)
    if not dias_sel: return

    # Reutilizar cálculo de vc/prec, luego calcular q
    f_list, vc_list, em_list, pr_list = _calcular_vc_emax_prec_por_fecha(archivos, dias_sel, shp)

    # q = (C_limite - C_fondo) * VC / L  → usamos L ≈ L_caracteristica (necesitamos recalcular)
    fechas_dict: dict = {}
    for a in archivos:
        fi = extraer_fecha_wrf(a.name); fechas_dict.setdefault(fi, []).append(a)

    q_list = []
    for fecha, vc_d in zip(f_list, vc_list):
        if not np.isfinite(vc_d):
            q_list.append(np.nan); continue
        archs = fechas_dict.get(fecha, [])
        L_sum = n = 0
        for arch in archs:
            ds = xr.open_dataset(arch, decode_times=False)
            L_sum += obtener_longitud_caracteristica(ds); n += 1
            ds.close()
        L_d = L_sum / n if n else 1.0
        delta_c = PM10_LIMITE_UG_M3 - c_fondo
        q_d = (delta_c * vc_d / L_d) * FACTOR_KG_DIA
        q_list.append(q_d)

    gdir = outdir / "graficas_q"; gdir.mkdir(parents=True, exist_ok=True)
    _grafica_serie_emax(f_list, q_list, vc_list, pr_list,
                        gdir/"serie_q_vc_precip.png",
                        f"Serie q PM10 (kg/m²·día) | B={c_fondo:.1f} µg/m³")
    _grafica_dispersion(f_list, q_list, vc_list, pr_list,
                        gdir/"dispersion_q_vc.png",
                        f"q (kg/m²·día)",
                        f"Dispersión q – VC | B={c_fondo:.1f} µg/m³")

    print("\nRESUMEN DIARIO – q PM10")
    print(f"{'Fecha':<12}{'VC(m²/s)':>12}{'q(kg/m²·d)':>14}{'Prec(mm)':>12}")
    for f, vc, q, pr in zip(f_list, vc_list, q_list, pr_list):
        print(f"{f:<12}{vc:12.2f}{q:14.6f}{pr:12.2f}")

    messagebox.showinfo("Listo", f"Gráficas q PM10 guardadas en:\n{gdir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 10 – MAPA ESPACIAL DE q POR CELDA CON INTERPOLACIÓN IDW
# ═════════════════════════════════════════════════════════════════════════════

def _idw_interpolar(lon_vals, lat_vals, q_vals, lon_q, lat_q, power=2.0):
    vals = q_vals.astype(float); mask = np.isfinite(vals)
    x = lon_vals[mask]; y = lat_vals[mask]; v = vals[mask]
    if v.size == 0: return np.full_like(lon_q, np.nan, dtype=float)
    dx = lon_q[..., None] - x; dy = lat_q[..., None] - y
    dist2 = dx*dx + dy*dy; dist2 = np.where(dist2 == 0, 1e-12, dist2)
    w = 1.0 / np.power(dist2, power/2.0)
    return np.sum(w*v, axis=-1) / np.sum(w, axis=-1)


def _mapa_q_celda(q_2d, lat, lon, gdf, fecha, outpath, q_max, c_fondo, escala_label):
    lat_v = lat.values if hasattr(lat, "values") else lat
    lon_v = lon.values if hasattr(lon, "values") else lon
    q_v = q_2d.values if hasattr(q_2d, "values") else q_2d
    finite = np.isfinite(q_v)

    lon_g = np.linspace(LON_MIN-ZOOM_OUT, LON_MAX+ZOOM_OUT, 200)
    lat_g = np.linspace(LAT_MIN-ZOOM_OUT, LAT_MAX+ZOOM_OUT, 200)
    lon_q, lat_q = np.meshgrid(lon_g, lat_g)
    q_i = _idw_interpolar(lon_v[finite], lat_v[finite], q_v[finite], lon_q, lat_q)

    levels = np.linspace(0, q_max if q_max > 0 else 1, 21)
    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(lon_q, lat_q, q_i, levels=levels, cmap="RdYlBu", extend="max")
    plt.colorbar(cf, ax=ax, label="q (kg/km²·día)")
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8, zorder=7)
    agregar_labels_ciudades(ax)
    ax.set_xlim(LON_MIN-ZOOM_OUT, LON_MAX+ZOOM_OUT); ax.set_ylim(LAT_MIN-ZOOM_OUT, LAT_MAX+ZOOM_OUT)
    ax.set_title(f"q PM10 (kg/km²·día) | {fecha} | B={c_fondo:.1f} µg/m³ | {escala_label}")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()


def run_10_q_mapa_celda():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(GEO_OK,"geopandas")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    shp = gui_seleccionar_shapefile()
    if not shp: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return
    c_fondo = gui_seleccionar_float("Concentración de fondo",
                                     f"B (µg/m³) – Suger: bajo={B_SUG_BAJO}, alto={B_SUG_ALTO}:", B_SUG_ALTO)
    if c_fondo is None: return
    q_max_val = gui_seleccionar_float("Tope q", "Valor máximo de q (kg/km²·día):", 5000.0)
    if q_max_val is None: return

    outdir.mkdir(parents=True, exist_ok=True)
    fechas_disp = sorted({extraer_fecha_wrf(a.name) for a in archivos})
    dias_sel = gui_seleccionar_dias(fechas_disp)
    if not dias_sel: return

    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    fechas_dict: dict = {}
    for a in archivos:
        fi = extraer_fecha_wrf(a.name); fechas_dict.setdefault(fi, []).append(a)

    mapas_dir = outdir / "mapas_q"; mapas_dir.mkdir(parents=True, exist_ok=True)
    delta_c = PM10_LIMITE_UG_M3 - c_fondo

    for fecha in sorted(dias_sel):
        archs = fechas_dict.get(fecha, [])
        vc_sum = None; ntime_sum = 0; L_sum = 0.0; lat_ref = lon_ref = None
        for arch in archs:
            ds = xr.open_dataset(arch, decode_times=False)
            if lat_ref is None:
                lat_ref = ds["XLAT"].isel(Time=0).load()
                lon_ref = ds["XLONG"].isel(Time=0).load()
            L = obtener_longitud_caracteristica(ds)
            u10 = ds["U10"]; v10 = ds["V10"]; pblh = ds["PBLH"]
            vc_f = np.sqrt(u10**2 + v10**2) * pblh
            vc_mean_2d = vc_f.mean(dim="Time").load()
            nt = int(vc_f.sizes.get("Time", vc_f.shape[0]))
            vc_sum = vc_mean_2d * nt if vc_sum is None else vc_sum + vc_mean_2d*nt
            L_sum += L*nt; ntime_sum += nt
            ds.close()
        if ntime_sum == 0: continue
        vc_dia = vc_sum / ntime_sum
        L_dia = L_sum / ntime_sum
        mask = mascara_region(xr.Dataset({"XLAT": lat_ref.expand_dims("Time"),
                                          "XLONG": lon_ref.expand_dims("Time")}))
        q_map = (delta_c * vc_dia / L_dia * FACTOR_KG_DIA * 1e6).where(mask)  # kg/km²
        _mapa_q_celda(q_map, lat_ref, lon_ref, gdf, fecha,
                      mapas_dir/f"q_celda_{fecha}.png", q_max_val, c_fondo, "IDW")

    messagebox.showinfo("Listo", f"Mapas q por celda guardados en:\n{mapas_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# HERRAMIENTA 11 – ÍNDICES ALLWINE-WHITEMAN (CTI)
# ═════════════════════════════════════════════════════════════════════════════

def _aw_mask_fill(arr, attrs):
    out = arr.astype(float, copy=True)
    fill = attrs.get("_FillValue") or attrs.get("missing_value")
    if fill is not None:
        out = np.where(np.isclose(out, fill), np.nan, out)
    return np.where(np.abs(out) > 200.0, np.nan, out)


def _aw_parse_times(ds):
    """Intenta parsear tiempos del WRF como datetimes."""
    try:
        if "Times" in ds:
            times = ["".join(t.astype(str)) for t in ds["Times"].values]
            return pd.to_datetime(times, errors="coerce")
        if "XTIME" in ds:
            ref = pd.Timestamp("2000-01-01")
            return ref + pd.to_timedelta(ds["XTIME"].values, unit="m")
    except Exception:
        pass
    return pd.RangeIndex(ds.sizes.get("Time", 0))


def _aw_infer_dt(times) -> float:
    """Estima dt en segundos a partir de una secuencia de tiempos."""
    try:
        t = pd.to_datetime(times, errors="coerce")
        t = t.dropna() if hasattr(t, "dropna") else t
        if len(t) >= 2:
            diffs = np.diff(t.values).astype(float) / 1e9
            valid = diffs[diffs > 0]
            if len(valid): return float(np.median(valid))
    except Exception:
        pass
    return 3600.0


def _aw_compute_indices(u_arr, v_arr, dt_s, tau_h):
    """Calcula S (recorrido m), L (desplazamiento m), R (recirculación) usando ventana deslizante."""
    n = len(u_arr)
    window = max(1, int(round(tau_h * 3600.0 / dt_s)))
    S_out = np.full(n, np.nan); L_out = np.full(n, np.nan)
    R_out = np.full(n, np.nan); theta_out = np.full(n, np.nan)

    for i in range(window-1, n):
        u_w = u_arr[i-window+1:i+1]; v_w = v_arr[i-window+1:i+1]
        valid = np.isfinite(u_w) & np.isfinite(v_w)
        if valid.sum() < window: continue
        uw = np.where(valid, u_w, 0.0); vw = np.where(valid, v_w, 0.0)
        spd = np.sqrt(uw**2 + vw**2)
        S = spd.sum() * dt_s
        X = vw.sum() * dt_s; Y = uw.sum() * dt_s
        L = np.sqrt(X**2 + Y**2)
        R = 1.0 - (L / S) if S > 0 else np.nan
        R = float(np.clip(R, 0.0, 1.0)) if np.isfinite(R) else np.nan
        theta = float(np.degrees(np.arctan2(X, Y)) % 360) if (np.isfinite(X) and np.isfinite(Y)) else np.nan
        S_out[i] = S; L_out[i] = L; R_out[i] = R; theta_out[i] = theta
    return S_out, L_out, R_out, theta_out


def _aw_clasificar(S_m, R, th):
    S_c = th["S_c"]*1000; R_c = th["R_c"]; S_cv = th["S_cv"]*1000; R_cv = th["R_cv"]
    stag = S_m <= S_c
    recirc = R >= R_c
    vent = (R <= R_cv) & (S_m >= S_cv)
    return stag, recirc, vent


def _aw_plot_timeseries(df, th, outdir, tag):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t = df.index if hasattr(df.index, "freq") else range(len(df))
    colors = {"stagnation": "red", "recirculation": "orange", "ventilation": "green"}
    for ax, col, label in zip(axes,
        ["S_km", "R", "condition"],
        ["Recorrido S (km)", "Recirculación R", "Clasificación"]):
        if col == "condition":
            for cond, color in colors.items():
                if cond in df.columns:
                    mask = df[cond].astype(bool)
                    ax.fill_between(range(len(df)), 0, 1, where=mask.values,
                                    color=color, alpha=0.4, transform=ax.get_xaxis_transform(), label=cond)
            ax.set_ylim(0, 1); ax.set_ylabel("Condición")
        elif col == "S_km":
            if "S_km" in df.columns:
                ax.plot(range(len(df)), df["S_km"].values)
            ax.axhline(th["S_c"], color="red", ls="--", lw=0.8, label=f"S_c={th['S_c']} km")
            ax.axhline(th["S_cv"], color="green", ls="--", lw=0.8, label=f"S_cv={th['S_cv']} km")
            ax.set_ylabel(label)
        elif col == "R":
            if "R" in df.columns:
                ax.plot(range(len(df)), df["R"].values)
            ax.axhline(th["R_c"], color="red", ls="--", lw=0.8, label=f"R_c={th['R_c']}")
            ax.axhline(th["R_cv"], color="green", ls="--", lw=0.8, label=f"R_cv={th['R_cv']}")
            ax.set_ylabel(label)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Allwine-Whiteman – {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"AW_timeseries_{tag}.png", dpi=150); plt.close()


def _aw_plot_scatter(df, th, outdir, tag):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_map = {"stagnation": "red", "recirculation": "orange",
                  "ventilation": "green", "neutral": "gray"}
    for i, row in df.iterrows():
        s = row.get("S_km", np.nan); r = row.get("R", np.nan)
        if not (np.isfinite(s) and np.isfinite(r)): continue
        if row.get("stagnation", False): c = "red"
        elif row.get("recirculation", False): c = "orange"
        elif row.get("ventilation", False): c = "green"
        else: c = "gray"
        ax.scatter(s, r, color=c, s=20, alpha=0.6)
    ax.axvline(th["S_c"], color="red", ls="--", lw=0.8, label=f"S_c={th['S_c']} km")
    ax.axvline(th["S_cv"], color="green", ls="--", lw=0.8, label=f"S_cv={th['S_cv']} km")
    ax.axhline(th["R_c"], color="red", ls=":", lw=0.8, label=f"R_c={th['R_c']}")
    ax.axhline(th["R_cv"], color="green", ls=":", lw=0.8, label=f"R_cv={th['R_cv']}")
    ax.set_xlabel("S (km)"); ax.set_ylabel("R (recirculación)")
    ax.set_title(f"S-R Isopleths – {tag}"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(outdir / f"AW_scatter_{tag}.png", dpi=150); plt.close()


def _aw_plot_pct_diario(df, th, outdir, tag):
    if "Time" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        return
    try:
        if "Time" in df.columns:
            df = df.set_index("Time")
        daily = df[["stagnation", "recirculation", "ventilation"]].resample("D").mean() * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        daily.plot(ax=ax, marker="o", linewidth=1.5)
        ax.set_ylabel("% del día"); ax.set_title(f"% Diario Condiciones AW – {tag}")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(outdir/f"AW_pct_diario_{tag}.png", dpi=150); plt.close()
    except Exception:
        pass


def _aw_mapa_pct(archivos, tau_h, dt_s, th, outdir, tag):
    """Genera mapa de % stagnation, recirculation y ventilation sobre la malla."""
    ds0 = xr.open_dataset(archivos[0], decode_times=False, mask_and_scale=False)
    u_dims = ds0["U10"].dims
    time_dim = "Time" if "Time" in u_dims else u_dims[0]
    yd, xd = u_dims[-2], u_dims[-1]
    lat2d = ds0["XLAT"].isel({time_dim: 0}).values
    lon2d = ds0["XLONG"].isel({time_dim: 0}).values
    shape2d = ds0["U10"].isel({time_dim: 0}).values.shape
    ds0.close()

    window = max(1, int(round(tau_h*3600.0/dt_s)))
    S_c = th["S_c"]*1000; R_c = th["R_c"]; S_cv = th["S_cv"]*1000; R_cv = th["R_cv"]
    cnt = np.zeros(shape2d, np.int32)
    c_stag = np.zeros(shape2d, np.int32)
    c_rec  = np.zeros(shape2d, np.int32)
    c_vent = np.zeros(shape2d, np.int32)
    sum_speed = np.zeros(shape2d); sum_u = np.zeros(shape2d); sum_v = np.zeros(shape2d)
    vc_count = np.zeros(shape2d, np.int16)
    buf_sp = deque(maxlen=window); buf_u = deque(maxlen=window)
    buf_v  = deque(maxlen=window); buf_vld = deque(maxlen=window)

    for arch in archivos:
        ds = xr.open_dataset(arch, decode_times=False, mask_and_scale=False)
        nt = int(ds.sizes.get(time_dim, ds["U10"].shape[0]))
        for i in range(nt):
            u = _aw_mask_fill(ds["U10"].isel({time_dim: i}).values, ds["U10"].attrs)
            v = _aw_mask_fill(ds["V10"].isel({time_dim: i}).values, ds["V10"].attrs)
            vld = np.isfinite(u) & np.isfinite(v)
            u0 = np.where(vld, u, 0.0); v0 = np.where(vld, v, 0.0)
            sp = np.where(vld, np.sqrt(u0**2+v0**2), 0.0); vi = vld.astype(np.int8)
            if len(buf_sp) == window:
                sum_speed -= buf_sp.popleft(); sum_u -= buf_u.popleft()
                sum_v -= buf_v.popleft(); vc_count -= buf_vld.popleft()
            buf_sp.append(sp); buf_u.append(u0); buf_v.append(v0); buf_vld.append(vi)
            sum_speed += sp; sum_u += u0; sum_v += v0; vc_count += vi
            if len(buf_sp) == window:
                full = vc_count == window
                if np.any(full):
                    S = sum_speed * dt_s
                    L = np.sqrt((sum_v*dt_s)**2 + (sum_u*dt_s)**2)
                    R = 1.0 - np.where(S>0, L/S, 0.0)
                    R = np.clip(R, 0.0, 1.0)
                    msk = full & np.isfinite(S) & np.isfinite(R)
                    cnt   += msk.astype(np.int32)
                    c_stag += (msk & (S<=S_c)).astype(np.int32)
                    c_rec  += (msk & (R>=R_c)).astype(np.int32)
                    c_vent += (msk & (R<=R_cv) & (S>=S_cv)).astype(np.int32)
        ds.close()

    with np.errstate(invalid="ignore", divide="ignore"):
        pct_s = np.where(cnt>0, 100.0*c_stag/cnt, np.nan)
        pct_r = np.where(cnt>0, 100.0*c_rec/cnt, np.nan)
        pct_v = np.where(cnt>0, 100.0*c_vent/cnt, np.nan)

    for data, nombre in [(pct_s,"stagnation"), (pct_r,"recirculation"), (pct_v,"ventilation")]:
        fig, ax = plt.subplots(figsize=(9, 7))
        cf = ax.contourf(lon2d, lat2d, data, levels=np.linspace(0,100,21), cmap="YlOrRd", extend="both")
        plt.colorbar(cf, ax=ax, label="% del período")
        ax.set_title(f"% {nombre.capitalize()} – AW | tau={tau_h:.0f}h | {tag}")
        ax.set_xlim(LON_MIN-ZOOM_OUT, LON_MAX+ZOOM_OUT); ax.set_ylim(LAT_MIN-ZOOM_OUT, LAT_MAX+ZOOM_OUT)
        agregar_labels_ciudades(ax)
        plt.tight_layout(); plt.savefig(outdir/f"AW_mapa_pct_{nombre}_{tag}.png", dpi=150); plt.close()


def _ventana_aw_punto(lat2d, lon2d, ds0, time_dim) -> Optional[Tuple[int,int,str]]:
    """Ventana tkinter para elegir una ciudad o coordenada manual."""
    ciudades = list(CITY_LABELS.keys()) + ["Manual"]
    sel = {"res": None}
    win = tk.Tk(); win.title("Seleccionar punto de análisis"); win.attributes("-topmost", True)
    tk.Label(win, text="Ciudad / Punto:").pack(padx=10, pady=5)
    var = tk.StringVar(value=ciudades[0])
    for c in ciudades:
        tk.Radiobutton(win, text=c, variable=var, value=c).pack(anchor="w", padx=20)
    def ok():
        sel["res"] = var.get(); win.destroy()
    tk.Button(win, text="Aceptar", command=ok).pack(pady=8)
    win.mainloop()
    if sel["res"] is None: return None

    nombre = sel["res"]
    if nombre == "Manual":
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        lat_in = simpledialog.askfloat("Latitud", "Ingrese latitud (decimal):", parent=root)
        lon_in = simpledialog.askfloat("Longitud", "Ingrese longitud (decimal):", parent=root)
        root.destroy()
        if lat_in is None or lon_in is None: return None
        lat_pt, lon_pt = lat_in, lon_in
    else:
        lon_pt, lat_pt = CITY_LABELS[nombre]

    dist2 = (lon2d - lon_pt)**2 + (lat2d - lat_pt)**2
    iy, ix = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return int(iy), int(ix), nombre


def run_11_allwine_whiteman():
    if not _check_deps((XR_OK,"xarray"),(MPL_OK,"matplotlib"),(PD_OK,"pandas")): return
    archivos = gui_seleccionar_archivos_wrf()
    if not archivos: return
    outdir = gui_seleccionar_directorio()
    if not outdir: return

    tau_h = gui_seleccionar_float("Tau (ventana)", "Ventana temporal τ (horas):", AW_DEFAULT_TAU_H, 1.0)
    if tau_h is None: tau_h = AW_DEFAULT_TAU_H
    th = dict(AW_DEFAULT_THRESHOLDS)

    outdir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d%H%M")

    # Leer primer archivo para obtener la malla
    ds0 = xr.open_dataset(archivos[0], decode_times=False, mask_and_scale=False)
    u_dims = ds0["U10"].dims
    time_dim = "Time" if "Time" in u_dims else u_dims[0]
    yd, xd = u_dims[-2], u_dims[-1]
    lat2d = ds0["XLAT"].isel({time_dim: 0}).values
    lon2d = ds0["XLONG"].isel({time_dim: 0}).values
    ds0.close()

    # Estimar dt
    all_times = []
    for arch in archivos:
        ds = xr.open_dataset(arch, decode_times=False, mask_and_scale=False)
        all_times.extend(_aw_parse_times(ds).tolist())
        ds.close()
    dt_s = _aw_infer_dt(all_times)
    print(f"τ={tau_h} h | dt={dt_s:.0f} s | ventana={int(round(tau_h*3600/dt_s))} pasos")

    # ── Modo de análisis ──
    root = tk.Tk(); root.withdraw()
    modo = simpledialog.askinteger(
        "Modo de análisis",
        "Seleccione modo:\n1. Punto/ciudad (serie temporal)\n2. Mapa de porcentajes (toda la malla)",
        minvalue=1, maxvalue=2
    )
    root.destroy()
    if modo is None: return

    tag = f"tau{int(tau_h)}h_{run_id}"

    if modo == 1:
        # ── Modo punto ──
        punto = _ventana_aw_punto(lat2d, lon2d, archivos[0], time_dim)
        if punto is None: return
        iy, ix, nombre = punto
        print(f"Punto: {nombre} | iy={iy}, ix={ix} | lat={lat2d[iy,ix]:.4f}, lon={lon2d[iy,ix]:.4f}")
        tag = f"{nombre.lower()}_{tag}"

        u_list, v_list, t_list = [], [], []
        for arch in archivos:
            ds = xr.open_dataset(arch, decode_times=False, mask_and_scale=False)
            times = _aw_parse_times(ds)
            u = _aw_mask_fill(ds["U10"].isel({yd: iy, xd: ix}).values, ds["U10"].attrs)
            v = _aw_mask_fill(ds["V10"].isel({yd: iy, xd: ix}).values, ds["V10"].attrs)
            u_list.extend(u.tolist()); v_list.extend(v.tolist()); t_list.extend(times.tolist())
            ds.close()

        df_raw = pd.DataFrame({"Time": t_list, "U10": u_list, "V10": v_list})
        df_raw = df_raw.dropna(subset=["U10","V10"]).sort_values("Time").drop_duplicates("Time").reset_index(drop=True)
        if df_raw.empty:
            messagebox.showerror("Sin datos", "No hay datos válidos en el punto seleccionado."); return

        S, L, R, theta = _aw_compute_indices(df_raw["U10"].values, df_raw["V10"].values, dt_s, tau_h)
        df_raw["S_m"] = S; df_raw["L_m"] = L; df_raw["R"] = R
        df_raw["S_km"] = S / 1000.0; df_raw["theta_deg"] = theta
        stag, rec, vent = _aw_clasificar(S, R, th)
        df_raw["stagnation"] = stag; df_raw["recirculation"] = rec; df_raw["ventilation"] = vent

        csv_out = outdir / f"AW_punto_{tag}.csv"
        df_raw.to_csv(csv_out, index=False)

        _aw_plot_timeseries(df_raw, th, outdir, tag)
        _aw_plot_scatter(df_raw, th, outdir, tag)
        _aw_plot_pct_diario(df_raw, th, outdir, tag)

        n = len(df_raw.dropna(subset=["S_m","R"]))
        s_ok = stag[np.isfinite(S)]; r_ok = rec[np.isfinite(R)]; v_ok = vent[np.isfinite(S) & np.isfinite(R)]
        msg = (f"Punto: {nombre}\nRegistros válidos: {n}\n\n"
               f"% Estancamiento:  {100*s_ok.mean():.1f}%\n"
               f"% Recirculación:  {100*r_ok.mean():.1f}%\n"
               f"% Ventilación:    {100*v_ok.mean():.1f}%\n\n"
               f"CSV guardado:\n{csv_out}")
        messagebox.showinfo("Allwine-Whiteman – Resultados", msg)

    else:
        # ── Modo mapa ──
        messagebox.showinfo("Modo mapa", "Se generarán mapas de % estancamiento, recirculación y ventilación.\nEsto puede tomar varios minutos.")
        _aw_mapa_pct(archivos, tau_h, dt_s, th, outdir, tag)
        messagebox.showinfo("Listo", f"Mapas AW guardados en:\n{outdir}")


# ═════════════════════════════════════════════════════════════════════════════
# MENÚ PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

TOOLS = [
    ("1. Clasificación precipitación (diaria)",        "Tabla de regímenes R0-R3 por día.",          run_01_clasificacion_prec),
    ("2. Mapas VC por régimen + extremos PBLH",        "VC promedio/max/min/mediodía por régimen.",   run_02_vc_por_regimen),
    ("3a. Mapa VC imshow + serie temporal",            "Mapa coloreado (imshow) + gráfica diaria.",   run_03a_vc_imshow),
    ("3b. Mapa VC contornos + serie temporal",         "Mapas de contorno relleno + gráfica diaria.", run_03b_vc_contorno),
    ("4. Ciclo horario PBLH / viento / VC",            "Promedio horario de variables (0-23 h).",     run_04_ciclo_horario),
    ("5. Mapa VC + viento – hora elegida",             "Contourf + flechas para una hora fija.",      run_05_mapa_vc_viento_hora),
    ("6. Mapa VC + viento – todos los timesteps",      "Un PNG por cada timestep de un día.",         run_06_vc_viento_por_dia),
    ("7. Emax PM10 – mapas horarios y diario",         "Capacidad asimilativa PM10 (kg/h y kg/día).", run_07_emax_pm10_mapas),
    ("8. Emax PM10 – serie + dispersión",              "Serie temporal y scatter Emax/VC/Prec.",      run_08_emax_pm10_serie),
    ("9. q PM10 – serie + dispersión",                 "q (cap. asim. por área) serie y scatter.",    run_09_q_pm10_serie),
    ("10. q PM10 – mapa por celda (IDW)",              "Mapa espacial q interpolado por IDW.",        run_10_q_mapa_celda),
    ("11. Allwine-Whiteman (CTI)",                     "Estancamiento, recirculación, ventilación.",  run_11_allwine_whiteman),
]


def _lanzar_herramienta(func):
    """Llama la herramienta directamente (macOS requiere tkinter en el hilo principal)."""
    func()


def main():
    import tkinter.ttk as ttk

    root = tk.Tk()
    root.title("WRF Urbano – Menú Central v1.0 (Independiente)")
    root.geometry("780x570")
    root.resizable(True, True)

    # ── Estilo ttk (funciona en macOS) ──────────────────────────────────────
    style = ttk.Style(root)
    try:
        style.theme_use("clam")  # 'clam' acepta colores personalizados en macOS
    except Exception:
        pass

    style.configure("Tool.TButton",
                    font=("Arial", 11, "bold"),
                    foreground="#ffffff",
                    background="#2d3150",
                    padding=(6, 5))
    style.map("Tool.TButton",
              background=[("active", "#3d4570"), ("pressed", "#1a1f3a")],
              foreground=[("active", "#ffffff")])

    style.configure("Close.TButton",
                    font=("Arial", 12, "bold"),
                    foreground="#ffffff",
                    background="#c0392b",
                    padding=(10, 6))
    style.map("Close.TButton",
              background=[("active", "#e74c3c"), ("pressed", "#922b21")],
              foreground=[("active", "#ffffff")])

    BG  = "#ffffff"   # blanco – máximo contraste garantizado en macOS
    FG  = "#1a1a1a"   # título casi negro
    FG2 = "#1a1a1a"   # descripción – negro puro

    tk.Label(root, text="Herramientas de análisis WRF post-procesamiento",
             font=("Arial", 15, "bold"), bg=BG, fg=FG).pack(pady=(16, 2))
    tk.Label(root, text="v1.0  ·  script independiente  ·  sin dependencias externas",
             font=("Arial", 9), bg=BG, fg="#555555").pack(pady=(0, 10))
    root.configure(bg=BG)

    canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
    sb = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas, bg=BG)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=5)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    for nombre, desc, func in TOOLS:
        row = tk.Frame(frame, pady=4, bg=BG); row.pack(fill=tk.X, padx=8)
        ttk.Button(row, text=nombre, style="Tool.TButton", width=38,
                   command=lambda f=func: _lanzar_herramienta(f)
                   ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(row, text=desc, font=("Arial", 10, "bold"), fg=FG2, bg=BG, anchor="w",
                 wraplength=340).pack(side=tk.LEFT, fill=tk.X, expand=True)

    ttk.Button(root, text="✕  Cerrar", style="Close.TButton",
               command=root.destroy).pack(side=tk.BOTTOM, pady=14)

    root.mainloop()


if __name__ == "__main__":
    main()

