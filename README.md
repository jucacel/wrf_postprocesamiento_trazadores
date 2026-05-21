# WRF Postprocesamiento 

Scripts de postprocesamiento para salidas WRF/WRF-Chem, con foco inicial en diagnosticos de trazadores atmosfericos y pulsos de CO real mediante `wrfchemi`.

El repositorio contiene codigo y documentacion. Los datos pesados, salidas WRF, figuras generadas y ambientes virtuales deben mantenerse fuera de Git.

## Alcance

- Detectar trazadores en archivos `wrfout`.
- Calcular columna integrada, series regionales y fraccion sobre PBLH.
- Generar cortes verticales y mapas de diagnostico.
- Crear pulsos fisicos de CO en `wrfchemi`.
- Validar masa emitida y comparar caso-control.
- Ejecutar diagnosticos meteorologicos urbanos sobre precipitacion, PBLH, viento,
  coeficiente de ventilacion, capacidad asimilativa e indices Allwine-Whiteman.

## Contenido inicial

```text
scripts/trazadores/
├── wrf_trazadores_00_core.py
├── wrf_trazadores_01_presencia_trazador.py
├── wrf_trazadores_02_columna_total.py
├── wrf_trazadores_03_serie_region.py
├── wrf_trazadores_04_fraccion_sobre_pblh.py
├── wrf_trazadores_05_corte_vertical.py
├── wrf_trazadores_06_mapa_penacho_altura_ventilacion.py
├── wrf_trazadores_07_desplazamiento_vertical.py
├── wrf_trazadores_08_generate_wrfchemi_pulse.py
├── wrf_trazadores_09_validate_wrfchemi_mass.py
├── wrf_trazadores_10_check_wrfout_co.py
├── wrf_trazadores_11_patch_namelist_co_real.py
├── wrf_trazadores_13_diagnose_source_signal.py
├── wrf_trazadores_14_visualize_case_control_delta.py
└── wrf_trazadores_16_visualize_delta_planview.py

scripts/urbano/
├── wrf_urbano_01_prec.py
├── ...
├── wrf_urbano_11_allwine_whiteman.py
└── wrf_urbano_ALL_v1.py
```

## Instalacion

```bash
git clone https://github.com/jucacel/wrf_postprocesamiento.git
cd wrf_postprocesamiento
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso general

Los scripts aceptan archivos `wrfout` o una carpeta de corrida mediante argumentos. Ejemplo de serie regional:

```bash
python scripts/trazadores/wrf_trazadores_03_serie_region.py \
  --run-dir /ruta/a/corrida_wrf \
  --domain d03 \
  --tracer-name tr17_1 \
  --lat-min 3.20 --lat-max 3.80 \
  --lon-min -76.80 --lon-max -76.10 \
  --outdir outputs/serie_region
```

Ejemplo de generacion de pulso CO real:

```bash
python scripts/trazadores/wrf_trazadores_08_generate_wrfchemi_pulse.py \
  --wrfinput /ruta/a/wrfinput_d01 \
  --out-dir outputs/wrfchemi \
  --domain d01 \
  --start 2024-09-01_00:00:00 \
  --run-hours 4 \
  --interval-min 60 \
  --pulse-start 2024-09-01_00:00:00 \
  --pulse-hours 1 \
  --mass-ton 1 \
  --source-ij 56 66 \
  --patch-radius 1 \
  --emissions-zdim 2 \
  --output-style single \
  --include-zero-co2 \
  --include-zero-nox
```

## Documentacion

- [Paso a paso de simulacion de trazadores](docs/PASO_A_PASO_SIMULACION_TRAZADORES.md)
- [Scripts de trazadores](docs/trazadores_scripts.md)
- [CO real con wrfchemi](docs/co_real_wrfchemi.md)
- [Scripts urbanos](docs/urbano/README.md)
- [Catalogo de scripts](docs/catalogo_scripts.md)
- [Inventario de migracion](docs/github_migration_inventory.md)

## Datos

No versionar archivos WRF grandes (`wrfout_*`, `wrfinput_*`, `wrfbdy_*`, `wrfchemi_*`, `*.nc`, `*.grib`) ni resultados generados. El `.gitignore` del repo ya los excluye.

## Verificacion

```bash
python3 -m py_compile scripts/trazadores/*.py
```

## Estado del proyecto

Primer corte limpio para GitHub. Falta convertir gradualmente los scripts en un paquete Python formal y agregar pruebas con datos pequenos de ejemplo.

## Licencia

MIT. Ver [LICENSE](LICENSE).
