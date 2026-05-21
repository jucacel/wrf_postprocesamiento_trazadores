import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ======================================================
# Clasificación de precipitación diaria
# ======================================================

def clasificar_regimen_precipitacion(p_mm):
    if p_mm <= 1.0:
        return "R0_seco"
    elif p_mm <= 5.0:
        return "R1_lluvia_debil"
    elif p_mm <= 20.0:
        return "R2_lluvia_moderada"
    else:
        return "R3_lluvia_fuerte"

# ======================================================
# Selección de archivos WRFOUT
# ======================================================

def seleccionar_archivos_gui():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    archivos = filedialog.askopenfilenames(
        title="Seleccionar archivos wrfout diarios",
        filetypes=[("Todos los archivos", "*")]
    )
    root.destroy()
    archivos = [a for a in archivos if Path(a).name.startswith("wrfout")]
    if not archivos:
        raise SystemExit("No se seleccionaron archivos wrfout válidos.")
    return sorted(archivos)

# ======================================================
# Precipitación diaria promedio
# ======================================================

def precipitacion_diaria_media(ds):
    rain_total = ds["RAINC"] + ds["RAINNC"]
    rain_day = rain_total.isel(Time=-1) - rain_total.isel(Time=0)
    rain_mean = rain_day.mean(dim=("south_north","west_east"))
    return float(rain_mean.values)

# ======================================================
# VC diario usando distintos criterios de PBLH
# ======================================================

def calcular_vc_extremos(ds):
    """
    Devuelve VC promedio, VC usando PBLH_max, VC usando PBLH_min y VC mediodía
    """
    u10 = ds["U10"].load()
    v10 = ds["V10"].load()
    pblh = ds["PBLH"].load()

    ws = np.sqrt(u10**2 + v10**2)

    # VC promedio temporal
    vc_promedio = (ws * pblh).mean(dim="Time", keep_attrs=True)

    # Índice del PBLH máximo diario (promedio espacial)
    idx_max = pblh.mean(dim=("south_north","west_east")).argmax(dim="Time")
    vc_max = (ws.isel(Time=idx_max) * pblh.isel(Time=idx_max))

    # Índice del PBLH mínimo diario (promedio espacial)
    idx_min = pblh.mean(dim=("south_north","west_east")).argmin(dim="Time")
    vc_min = (ws.isel(Time=idx_min) * pblh.isel(Time=idx_min))

    # VC mediodía (timestep central)
    t_med = len(ds["Time"]) // 2
    vc_mediodia = (ws.isel(Time=t_med) * pblh.isel(Time=t_med))

    return vc_promedio, vc_max, vc_min, vc_mediodia

# ======================================================
# Generar mapas por régimen
# ======================================================

def mapas_vc_por_regimen(vc_diarios, regimen_diarios, titulo_extra=""):
    """
    vc_diarios: dict con listas de VC 2D por tipo:
       {"promedio": [...], "max": [...], "min": [...], "mediodia": [...]}
    regimen_diarios: lista de regímenes por día
    """
    regimenes = ["R0_seco","R1_lluvia_debil","R2_lluvia_moderada","R3_lluvia_fuerte"]
    tipos_vc = ["promedio","max","min","mediodia"]

    for r in regimenes:
        # Filtrar días de ese régimen
        idx_r = [i for i, reg in enumerate(regimen_diarios) if reg == r]
        if not idx_r:
            print(f"No hay días para régimen {r}, se omite.")
            continue
        for tipo in tipos_vc:
            campos = [vc_diarios[tipo][i] for i in idx_r]
            vc_stack = xr.concat(campos, dim="dia")
            vc_media = vc_stack.mean("dia")
            plt.figure(figsize=(8,6))
            vc_media.plot(cmap="viridis")
            plt.title(f"VC {tipo} promedio - {r} {titulo_extra}")
            plt.tight_layout()
            plt.show()

# ======================================================
# MAIN
# ======================================================

def main():
    print("\nPASO C – MAPAS DE VENTILACIÓN POR RÉGIMEN Y EXTREMOS PBLH")
    print("Valle geográfico del río Cauca")
    print("="*80)

    archivos = seleccionar_archivos_gui()

    vc_diarios = {"promedio":[], "max":[], "min":[], "mediodia":[]}
    regimen_diarios = []
    fechas = []
    p_mm_diarios = []

    for archivo in archivos:
        nombre = Path(archivo).name
        fecha = nombre.replace("wrfout_d01_", "").replace("_00_00_00", "")
        fechas.append(fecha)

        ds = xr.open_dataset(archivo, decode_times=False, mask_and_scale=False)

        # --- Precipitación diaria ---
        p_mm = precipitacion_diaria_media(ds)
        p_mm_diarios.append(p_mm)
        regimen = clasificar_regimen_precipitacion(p_mm)
        regimen_diarios.append(regimen)

        # --- VC diario con extremos ---
        vc_prom, vc_max, vc_min, vc_noon = calcular_vc_extremos(ds)
        vc_diarios["promedio"].append(vc_prom)
        vc_diarios["max"].append(vc_max)
        vc_diarios["min"].append(vc_min)
        vc_diarios["mediodia"].append(vc_noon)

        ds.close()

    # ======================================================
    # IMPRIMIR TABLA EN CONSOLA
    # ======================================================
    print("\nTABLA DE DATOS DIARIOS COMPLETA")
    print("-"*110)
    print(f"{'Fecha':<12} {'P(mm)':>8} {'Régimen':>22} "
          f"{'VC_prom':>12} {'VC_max':>12} {'VC_min':>12} {'VC_mediodia':>12}")
    print("-"*110)
    for i in range(len(fechas)):
        print(f"{fechas[i]:<12} {p_mm_diarios[i]:8.2f} {regimen_diarios[i]:>22} "
              f"{float(vc_diarios['promedio'][i].mean()):12.1f} "
              f"{float(vc_diarios['max'][i].mean()):12.1f} "
              f"{float(vc_diarios['min'][i].mean()):12.1f} "
              f"{float(vc_diarios['mediodia'][i].mean()):12.1f}")

    # ======================================================
    # GENERAR MAPAS
    # ======================================================
    print("\nGenerando mapas de VC por régimen y tipo PBLH...")
    mapas_vc_por_regimen(vc_diarios, regimen_diarios, titulo_extra="(Paso C Ampliado)")

    print("\nPaso C completado correctamente. Mapas y tabla listos para análisis de capacidad asimilativa.")

if __name__ == "__main__":
    main()
