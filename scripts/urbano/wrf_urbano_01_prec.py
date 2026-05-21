import numpy as np
import xarray as xr
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ======================================================
# CLASIFICACIÓN DE PRECIPITACIÓN (REGIÓN TROPICAL)
# Basada en WMO / IDEAM – adaptada a ventilación
# ======================================================

def clasificar_regimen_precipitacion(p_mm):
    """
    Clasificación funcional de precipitación diaria (mm)
    Adaptada a región tropical (Valle del Cauca)
    """
    if p_mm <= 1.0:
        return "R0_seco"
    elif p_mm <= 5.0:
        return "R1_lluvia_debil"
    elif p_mm <= 20.0:
        return "R2_lluvia_moderada"
    else:
        return "R3_lluvia_fuerte"


# ======================================================
# SELECCIÓN MÚLTIPLE DE ARCHIVOS WRFOUT
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

    if not archivos:
        raise SystemExit("No se seleccionaron archivos.")

    archivos = [
        a for a in archivos
        if Path(a).name.startswith("wrfout")
    ]

    if not archivos:
        raise SystemExit("No se seleccionaron archivos wrfout válidos.")

    return sorted(archivos)


# ======================================================
# PRECIPITACIÓN DIARIA MEDIA ESPACIAL
# ======================================================

def precipitacion_diaria_media(ds):
    """
    Precipitación diaria media (mm):
    (RAINC + RAINNC)_final - (RAINC + RAINNC)_inicial
    """

    rainc = ds["RAINC"].load()
    rainnc = ds["RAINNC"].load()

    rain_total = rainc + rainnc

    rain_day = rain_total.isel(Time=-1) - rain_total.isel(Time=0)

    rain_mean = rain_day.mean(
        dim=("south_north", "west_east")
    )

    return float(rain_mean.values)


# ======================================================
# MAIN
# ======================================================

def main():
    print("\nCLASIFICACIÓN DIARIA POR REGÍMENES DE PRECIPITACIÓN (WRF)")
    print("Región tropical – Valle geográfico del río Cauca")
    print("=" * 75)

    archivos = seleccionar_archivos_gui()

    print(f"\nArchivos seleccionados: {len(archivos)}\n")

    resultados = []

    for archivo in archivos:
        nombre = Path(archivo).name

        # Extraer fecha desde el nombre WRF
        fecha = (
            nombre
            .replace("wrfout_d01_", "")
            .replace("_00_00_00", "")
        )

        print(f"Procesando {nombre}")

        ds = xr.open_dataset(
            archivo,
            decode_times=False,
            mask_and_scale=False
        )

        p_mm = precipitacion_diaria_media(ds)
        regimen = clasificar_regimen_precipitacion(p_mm)

        resultados.append((fecha, p_mm, regimen))

        ds.close()

    # ==================================================
    # RESUMEN
    # ==================================================

    print("\nRESUMEN DIARIO POR REGÍMENES")
    print("-" * 60)
    print(f"{'Fecha':<12} {'P(mm)':>10} {'Régimen':>30}")
    print("-" * 60)

    for fecha, p, reg in resultados:
        print(f"{fecha:<12} {p:10.2f} {reg:>30}")

    print("\nClasificación completada correctamente.")
    print("Regímenes:")
    print("  R0_seco            : P ≤ 1 mm")
    print("  R1_lluvia_debil    : 1 < P ≤ 5 mm")
    print("  R2_lluvia_moderada : 5 < P ≤ 20 mm")
    print("  R3_lluvia_fuerte   : P > 20 mm")


if __name__ == "__main__":
    main()
