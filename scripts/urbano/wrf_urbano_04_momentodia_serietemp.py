import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ======================================================
# CONFIGURACIÓN DE ZONA
# ======================================================
LAT_MIN, LAT_MAX = 3.35, 3.65
LON_MIN, LON_MAX = -76.546012, -76.25

# ======================================================
# INTERFAZ INTERACTIVA
# ======================================================
def seleccionar_archivos_wrf():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    archivos = filedialog.askopenfilenames(title="Seleccione los archivos wrfout")
    root.destroy()
    if not archivos:
        raise SystemExit("No se seleccionaron archivos wrfout.")
    return sorted(archivos)

def seleccionar_directorio_salida():
    root = tk.Tk()
    root.withdraw()
    out = filedialog.askdirectory(title="Seleccionar carpeta de salida")
    root.destroy()
    return Path(out)

# ======================================================
# MÁSCARA ESPACIAL
# ======================================================
def mascara_region(ds):
    lat = ds["XLAT"].isel(Time=0)
    lon = ds["XLONG"].isel(Time=0)
    return (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)

# ======================================================
# EXTRAER VARIABLES HORARIAS
# ======================================================
def extraer_horario(ds, mask):
    u10 = ds["U10"]
    v10 = ds["V10"]
    pblh = ds["PBLH"]
    rain = ds["RAINC"] + ds["RAINNC"]

    ws = np.sqrt(u10**2 + v10**2)
    vc_full = ws * pblh

    # Promedio espacial sobre la zona
    pblh_mean = pblh.where(mask).mean(dim=("south_north", "west_east")).values
    ws_mean = ws.where(mask).mean(dim=("south_north", "west_east")).values
    vc_mean = vc_full.where(mask).mean(dim=("south_north", "west_east")).values
    rain_mean = rain.where(mask).mean(dim=("south_north", "west_east")).values
    rain_hora = np.diff(rain_mean, prepend=rain_mean[0])  # mm/h por paso

    # Horas
    time_vals = ds["XTIME"].values
    horas = np.array([t/60 % 24 for t in time_vals])
    return horas, pblh_mean, ws_mean, vc_mean, rain_hora

# ======================================================
# PROMEDIO HORARIO SOBRE TODOS LOS DÍAS
# ======================================================
def promedio_horario(horas_list, data_list):
    horas_unicas = np.arange(24)
    promedio = np.zeros(24)
    for h in horas_unicas:
        valores_hora = []
        for horas, data in zip(horas_list, data_list):
            valores_hora.extend(data[horas.astype(int) == h])
        promedio[h] = np.mean(valores_hora)
    return horas_unicas, promedio

# ======================================================
# GRAFICAS INDIVIDUALES
# ======================================================
def graficas_individuales(horas_prom, pblh_prom, ws_prom, vc_prom, rain_prom, outdir, fechas_consideradas):
    dias_texto = "Días considerados: " + ", ".join(fechas_consideradas)

    # PBLH
    plt.figure(figsize=(8,5))
    plt.plot(horas_prom, pblh_prom, marker='o', color='green')
    plt.xlabel("Hora")
    plt.ylabel("PBLH (m)")
    plt.title("Altura capa límite – promedio horario")
    plt.figtext(0.5, -0.05, dias_texto, ha="center", fontsize=9)
    plt.grid(True)
    plt.savefig(outdir / "PBLH_promedio_horario.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Viento
    plt.figure(figsize=(8,5))
    plt.plot(horas_prom, ws_prom, marker='s', color='blue')
    plt.xlabel("Hora")
    plt.ylabel("Velocidad del viento (m/s)")
    plt.title("Velocidad del viento – promedio horario")
    plt.figtext(0.5, -0.05, dias_texto, ha="center", fontsize=9)
    plt.grid(True)
    plt.savefig(outdir / "Viento_promedio_horario.png", dpi=150, bbox_inches='tight')
    plt.close()

    # VC
    plt.figure(figsize=(8,5))
    plt.plot(horas_prom, vc_prom, marker='^', color='red')
    plt.xlabel("Hora")
    plt.ylabel("VC (m²/s)")
    plt.title("Coeficiente de ventilación – promedio horario")
    plt.figtext(0.5, -0.05, dias_texto, ha="center", fontsize=9)
    plt.grid(True)
    plt.savefig(outdir / "VC_promedio_horario.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Precipitación
    plt.figure(figsize=(8,5))
    plt.bar(horas_prom, rain_prom, color='cyan', alpha=0.7)
    plt.xlabel("Hora")
    plt.ylabel("Precipitación (mm/h)")
    plt.title("Precipitación – promedio horario")
    plt.figtext(0.5, -0.05, dias_texto, ha="center", fontsize=9)
    plt.grid(True)
    plt.savefig(outdir / "Precipitacion_promedio_horario.png", dpi=150, bbox_inches='tight')
    plt.close()

# ======================================================
# GRAFICA COMBINADA MULTIEJES CON PRECIPITACIÓN
# ======================================================
def grafica_combinada_multiejes(horas_prom, pblh_prom, ws_prom, vc_prom, rain_prom, outdir, fechas_consideradas):
    dias_texto = "Días considerados: " + ", ".join(fechas_consideradas)

    fig, ax1 = plt.subplots(figsize=(10,5))

    # PBLH
    ax1.plot(horas_prom, pblh_prom, color='green', marker='o', label='PBLH (m)')
    ax1.set_xlabel("Hora")
    ax1.set_ylabel("PBLH (m)", color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    # Viento
    ax2 = ax1.twinx()
    ax2.plot(horas_prom, ws_prom, color='blue', marker='s', label='Viento (m/s)')
    ax2.set_ylabel("Viento (m/s)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # VC
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(horas_prom, vc_prom, color='red', marker='^', label='VC (m²/s)')
    ax3.set_ylabel("VC (m²/s)", color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    # Precipitación
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("axes", 1.2))
    ax4.bar(horas_prom, rain_prom, color='cyan', alpha=0.3, label='Precipitación (mm/h)', width=0.8)
    ax4.set_ylabel("Precipitación (mm/h)", color='cyan')
    ax4.tick_params(axis='y', labelcolor='cyan')

    plt.title("Promedio horario – PBLH, Viento, VC y Precipitación")
    plt.figtext(0.5, -0.05, dias_texto, ha="center", fontsize=9)

    # Combinar leyendas
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2, ax3, ax4]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc='upper left')

    plt.grid(True)
    plt.savefig(outdir / "Variables_combinadas_multiejes.png", dpi=150, bbox_inches='tight')
    plt.close()

# ======================================================
# MAIN
# ======================================================
def main():
    archivos = seleccionar_archivos_wrf()
    outdir = seleccionar_directorio_salida()
    outdir.mkdir(parents=True, exist_ok=True)

    pblh_list, ws_list, vc_list, rain_list, horas_list = [], [], [], [], []
    fechas = [Path(a).name.split("_")[2] for a in archivos]

    for f in archivos:
        ds = xr.open_dataset(f, decode_times=False)
        mask = mascara_region(ds)
        horas, pblh_h, ws_h, vc_h, rain_h = extraer_horario(ds, mask)
        horas_list.append(horas)
        pblh_list.append(pblh_h)
        ws_list.append(ws_h)
        vc_list.append(vc_h)
        rain_list.append(rain_h)
        ds.close()

    # Promedios horarios
    horas_prom, pblh_prom = promedio_horario(horas_list, pblh_list)
    _, ws_prom = promedio_horario(horas_list, ws_list)
    _, vc_prom = promedio_horario(horas_list, vc_list)
    _, rain_prom = promedio_horario(horas_list, rain_list)

    # Graficas individuales
    graficas_individuales(horas_prom, pblh_prom, ws_prom, vc_prom, rain_prom, outdir, fechas)

    # Graficar combinada multiejes con precipitación
    grafica_combinada_multiejes(horas_prom, pblh_prom, ws_prom, vc_prom, rain_prom, outdir, fechas)

    print(f"Gráficas de promedio horario generadas en: {outdir}")

if __name__ == "__main__":
    main()
