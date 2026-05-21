# Inventario para migracion a GitHub

Fecha de revision: 2026-05-20

## Diagnostico general

La carpeta fuente no era todavia un repositorio Git en la raiz. Contenia codigo util, respaldos, clones de terceros, ambientes virtuales, salidas WRF, NetCDF/GRIB, figuras, shapes y PDFs. No convenia subir esa carpeta completa.

La estrategia recomendada es crear un repositorio limpio, copiando solo el codigo y documentacion seleccionados, con `data/`, `outputs/`, `Z_WRF_OUT/`, `RESULTADOS_*`, `SHAPES/`, ambientes virtuales y respaldos fuera de Git.

## Repositorios/clones ya existentes dentro de la carpeta

| Ruta | Estado | Recomendacion |
|---|---|---|
| `PyChEmiss/` | Ya es un clone de `https://github.com/quishqa/PyChEmiss.git`; tiene cambios y archivos locales. | No mezclar en el nuevo repo. Si hay mejoras propias, tratarlas como fork o documentar uso como dependencia. |
| `postwrf/` | Ya es un clone de `https://github.com/anikfal/postwrf.git`; tiene cambios y salidas locales. | No subir como codigo propio. Separar wrappers propios si aportan valor. |

## Prioridad 1: llevar primero a GitHub

| Componente | Archivos | Motivo | Accion previa |
|---|---|---|---|
| Diagnosticos de trazadores WRF | `wrf_trazadores_00_core.py`, `wrf_trazadores_01_presencia_trazador.py`, `wrf_trazadores_02_columna_total.py`, `wrf_trazadores_03_serie_region.py`, `wrf_trazadores_04_fraccion_sobre_pblh.py`, `wrf_trazadores_05_corte_vertical.py`, `wrf_trazadores_06_mapa_penacho_altura_ventilacion.py`, `wrf_trazadores_07_desplazamiento_vertical.py` | Es el bloque mas coherente: tiene nucleo comun, CLI, diagnosticos cientificos claros y documentacion existente. | Mover a `src/wrf_trazadores/` o `scripts/trazadores/`, crear `README.md`, `requirements.txt` y ejemplos sin rutas absolutas. |
| Pulso CO real con `wrfchemi` | `wrf_trazadores_08_generate_wrfchemi_pulse.py`, `wrf_trazadores_09_validate_wrfchemi_mass.py`, `wrf_trazadores_10_check_wrfout_co.py`, `wrf_trazadores_11_patch_namelist_co_real.py`, `wrf_trazadores_13_diagnose_source_signal.py`, `wrf_trazadores_14_visualize_case_control_delta.py`, `wrf_trazadores_16_visualize_delta_planview.py` | Tiene valor metodologico alto: generacion fisica de emisiones, validacion de masa y diagnostico caso-control. | Integrar con el mismo paquete o dejar como `scripts/co_real/`; revisar wrappers `.sh` con rutas locales. |
| Documentacion de trazadores | `PASO_A_PASO_SIMULACION_TRAZADORES.md`, `Trazadores_codex/README_wrf_trazadores_scripts.md`, `Trazadores_codex/co_real_wrfchemi/README_CO_REAL_WRFCHEMI.md`, `Trazadores_codex/archivos_MD/*.md` | Ya explica metodologia, arquitectura y uso. | Consolidar en `docs/`, reemplazar rutas personales por placeholders. |

## Prioridad 2: llevar despues, con limpieza

| Componente | Archivos | Motivo | Accion previa |
|---|---|---|---|
| `wrf_urbano` modular | `wrf_urbano_01_prec.py` a `wrf_urbano_12_allwine_whiteman.py` | Tiene productos meteorologicos y de capacidad asimilativa relevantes. | Revisar duplicados, normalizar nombres y parametrizar rutas/shapes. |
| `wrf_urbano_ALL_v1.py` | Script unificado de 1538 lineas | Es demostrativo y autocontenido, pero grande y con GUI `tkinter`. | Subir como release inicial o ejemplo, no como base principal si se va a mantener modularmente. |
| Procesadores historicos WRF | `wrf_processor*.py`, `wrf_vc_resultados*.py`, `wrf_pblh_diaria*.py`, `wrf_precip*.py`, `wrf_rosavientos_vel.py` | Pueden ser utiles, pero hay duplicacion entre raiz, `00_repository_code/` y `01_repository_scripts_from_ubuntu/`. | Elegir una sola version canonica por script y archivar el resto fuera del repo. |

## Prioridad 3: no subir al primer repo

| Tipo | Rutas/ejemplos | Motivo |
|---|---|---|
| Salidas WRF y datos pesados | `Z_WRF_OUT/`, `Trazadores_codex/RESULTADOS_VTRAZ/`, `RESULTADOS_POSTPROC/`, archivos `wrfout_*`, `wrfinput_*`, `wrfbdy_*`, `*.nc`, `*.grib` | Pesados y reproducibles; deben quedar en almacenamiento externo o documentarse. |
| Figuras/resultados generados | `RESULTADOS_*`, `Z_WRF_PROCESSOR_RESULTADOS/`, `resultados_gif_png/`, `*.png`, `*.gif`, `*.csv` | Son productos, no fuente. Subir solo ejemplos pequenos si hacen falta en docs. |
| Ambientes locales | `.venv/`, `.conda/`, `.cache/`, `.mplconfig/` | No deben versionarse. |
| Respaldos completos | `Respaldos namelist/`, `PyChEmiss_2015/`, `wrfout_repo_old/` | Mezclan historico, datos y codigo externo. |
| Scripts de notas/rutas locales | `rutas_links.sh`, `rutas_links_v2.sh`, `01_WRF_scripts/rutas_links.sh` | Contienen rutas locales, IPs o comandos de maquina. Mejor convertir partes utiles a documentacion. |
| Herramientas no relacionadas | `tools_youtubedownloader.py`, scripts sueltos de videos o PDFs si no pertenecen al flujo WRF | Ruido para el repo principal. |

## Estructura recomendada del primer repo

```text
wrf-postprocesamiento/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── docs/
│   ├── trazadores.md
│   ├── co_real_wrfchemi.md
│   └── ejemplos.md
├── scripts/
│   ├── trazadores/
│   └── co_real/
├── src/
│   └── wrf_post/
└── examples/
    ├── configs/
    └── comandos/
```

## `.gitignore` minimo recomendado

```gitignore
__pycache__/
*.pyc
.DS_Store
.venv/
venv/
.conda/
.cache/
.mplconfig/

*.nc
*.grib
*.grb
*.grb2
wrfout_*
wrfinput_*
wrfbdy_*
wrfrst_*
met_em.*
geo_em.*

*.png
*.jpg
*.jpeg
*.gif
*.pdf
*.csv
*.log
*.zip
*.tar.gz

data/
outputs/
output/
results/
Resultados/
RESULTADOS*/
Z_WRF_OUT/
Z_WRF_PROCESSOR_RESULTADOS/
SHAPES/
scratch/
tmp/
Respaldos namelist/
```

## Siguiente paso recomendado

Crear una carpeta limpia paralela, por ejemplo `/ruta/wrf-postprocesamiento-github`, copiar solo la prioridad 1, agregar `README.md`, `.gitignore` y `requirements.txt`, y validar que al menos los scripts importen sin errores de sintaxis.
