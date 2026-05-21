# Catalogo de scripts

Este catalogo resume el proposito de cada script incluido actualmente en el repositorio.

## Scripts de trazadores

| Script | Proposito |
|---|---|
| `scripts/trazadores/wrf_trazadores_00_core.py` | Nucleo comun para los diagnosticos de trazadores. Centraliza seleccion de archivos, validacion minima de `wrfout`, deteccion de variables de trazador, parseo de tiempos, lectura NetCDF, calculo de alturas AGL, espesores verticales, integracion de columna, fraccion sobre PBLH, mascaras lat-lon y utilidades de exportacion. |
| `scripts/trazadores/wrf_trazadores_01_presencia_trazador.py` | Revisa una corrida WRF y verifica si existen trazadores en los `wrfout`. Detecta variables candidatas por dominio, identifica el instante de mayor senal superficial en una ventana geografica y genera reportes/figuras para evidenciar presencia del trazador. |
| `scripts/trazadores/wrf_trazadores_02_columna_total.py` | Calcula la columna vertical integrada del trazador usando espesores derivados de `PH` y `PHB`. Produce serie temporal regional, mapa de columna media y mapa del instante de maxima carga. |
| `scripts/trazadores/wrf_trazadores_03_serie_region.py` | Genera series temporales regionales del trazador, incluyendo estadisticos de superficie, columna integrada y centroide del penacho. Si se indica fuente, estima desplazamientos relativos del centroide. |
| `scripts/trazadores/wrf_trazadores_04_fraccion_sobre_pblh.py` | Calcula que fraccion de la columna del trazador queda por encima y por debajo de la PBLH. Usa `PH`, `PHB` y `PBLH`, y exporta CSV, series, mapas medios y resumen del evento. |
| `scripts/trazadores/wrf_trazadores_05_corte_vertical.py` | Construye cortes verticales del trazador a lo largo de un transecto lat-lon. Puede incluir terreno, contornos meteorologicos opcionales, mapa inserto, punto de emision y trazabilidad de la fuente de datos. |
| `scripts/trazadores/wrf_trazadores_06_mapa_penacho_altura_ventilacion.py` | Genera mapas horarios del penacho a una altura AGL fija. Muestra trazador, shape del area de estudio, fuente, centroide visible, flecha de desplazamiento y punto de maxima concentracion. |
| `scripts/trazadores/wrf_trazadores_07_desplazamiento_vertical.py` | Analiza la evolucion vertical del trazador. Produce panel tiempo-altura, CSV con diagnosticos de altura y fraccion sobre PBLH, cortes verticales por tiempo y GIF opcional. |
| `scripts/trazadores/wrf_trazadores_08_generate_wrfchemi_pulse.py` | Genera un archivo `wrfchemi` con pulso fisico controlado de CO (`E_CO`) en `mol km-2 hr-1`. Distribuye una masa total definida por el usuario sobre un parche de grilla y una ventana temporal. |
| `scripts/trazadores/wrf_trazadores_09_validate_wrfchemi_mass.py` | Valida la masa fisica representada en uno o varios archivos `wrfchemi` con `E_CO`, integrando emisiones con area de celda e intervalo temporal. |
| `scripts/trazadores/wrf_trazadores_10_check_wrfout_co.py` | Comprueba si una corrida WRF-Chem produjo campos de CO utilizables en `wrfout`, buscando variables compatibles y revisando senal basica. |
| `scripts/trazadores/wrf_trazadores_11_patch_namelist_co_real.py` | Modifica un `namelist.input` para pruebas cortas de WRF-Chem con CO real y `wrfchemi`, ajustando campos relevantes de configuracion. |
| `scripts/trazadores/wrf_trazadores_13_diagnose_source_signal.py` | Diagnostica si el pulso de CO generado en `wrfchemi` aparece en el `wrfout` cerca de la fuente esperada. Sirve para validar que la emision entra efectivamente al modelo. |
| `scripts/trazadores/wrf_trazadores_14_visualize_case_control_delta.py` | Visualiza la senal aislada del pulso de CO como diferencia entre caso con emision y control sin emision. Esta orientado a cortes/diagnosticos de delta `caso - control`. |
| `scripts/trazadores/wrf_trazadores_16_visualize_delta_planview.py` | Visualiza en planta la diferencia `caso - control` para aislar el penacho de CO. Genera mapas del delta y filtra componentes conectados para destacar la senal principal. |

## Scripts urbanos

| Script | Proposito |
|---|---|
| `scripts/urbano/wrf_urbano_01_prec.py` | Clasifica dias segun precipitacion media diaria a partir de archivos `wrfout`. Calcula lluvia acumulada diaria y asigna regimenes de precipitacion. |
| `scripts/urbano/wrf_urbano_02_prec_viento_pblh.py` | Relaciona regimenes de precipitacion con ventilacion. Calcula coeficiente de ventilacion usando viento y PBLH bajo varios criterios: promedio, PBLH maxima, PBLH minima y mediodia. |
| `scripts/urbano/wrf_urbano_03_cuadros.py` | Genera mapas diarios de coeficiente de ventilacion y una serie temporal diaria de precipitacion, PBLH y VC. Usa mascara regional y shapefile urbano para contextualizar los mapas. |
| `scripts/urbano/wrf_urbano_03_mapacontorno_vc_v2.py` | Variante de `wrf_urbano_03` que representa el coeficiente de ventilacion con contornos rellenos y escala configurable, ademas de serie temporal diaria. |
| `scripts/urbano/wrf_urbano_04_momentodia_serietemp.py` | Calcula ciclos horarios promedio de PBLH, velocidad del viento, coeficiente de ventilacion y precipitacion. Exporta graficas individuales y una grafica combinada multieje. |
| `scripts/urbano/wrf_urbano_05_mapasviento_vc_wd.py` | Genera un mapa de coeficiente de ventilacion con flechas de viento para una hora seleccionada por el usuario. |
| `scripts/urbano/wrf_urbano_06_vc_viento_por_dia_v2.py` | Genera mapas de coeficiente de ventilacion y viento para todos los timesteps de un dia seleccionado. Produce una secuencia de PNG por hora/timestep. |
| `scripts/urbano/wrf_urbano_07_capacidad_asimilativa_pm10_v2.py` | Calcula capacidad asimilativa maxima para PM10 (`Emax`) usando VC y longitud caracteristica. Produce mapas horarios en kg/h y mapa diario en kg/dia, con opciones de escala manual, percentil o temporada. |
| `scripts/urbano/wrf_urbano_08_capacidad_asimilativa_pm10_scatter.py` | Genera series temporales y graficas de dispersion para `Emax`, VC y precipitacion. Permite comparar capacidad asimilativa con condiciones meteorologicas diarias. |
| `scripts/urbano/wrf_urbano_09_capacidad_asimilativa_q.py` | Calcula `q` de PM10 a partir de concentracion limite, concentracion de fondo, VC y longitud caracteristica. Produce series temporales y dispersiones relacionadas con VC y precipitacion. |
| `scripts/urbano/wrf_urbano_10_q_mapa_por_celda.py` | Calcula y mapea `q` por celda de grilla. Genera productos espaciales de capacidad asimilativa por area, con unidades y escalas configurables. |
| `scripts/urbano/wrf_urbano_11_allwine_whiteman.py` | Calcula indices Allwine-Whiteman de estancamiento, recirculacion y ventilacion a partir de viento superficial. Puede operar por punto, ciudad o mapa, y genera series, diagramas S-R, porcentajes, conteos y mapas de incidencia. |
| `scripts/urbano/wrf_urbano_ALL_v1.py` | Script unificado con menu `tkinter` que integra en una sola interfaz varias herramientas urbanas: precipitacion, VC, ciclos horarios, mapas de viento, capacidad asimilativa PM10, mapas de `q` e indices Allwine-Whiteman. |

## Documentacion auxiliar

| Archivo | Proposito |
|---|---|
| `docs/trazadores_scripts.md` | Guia operativa de los scripts de trazadores, sus dependencias, convenciones de entrada y productos esperados. |
| `docs/co_real_wrfchemi.md` | Explica la ruta metodologica para pasar de trazadores inertes a CO real mediante `wrfchemi`, incluyendo generacion, validacion y comparacion caso-control. |
| `docs/PASO_A_PASO_SIMULACION_TRAZADORES.md` | Paso a paso metodologico para preparar y ejecutar simulaciones con trazadores. |
| `docs/urbano/README.md` | Resumen especifico del bloque urbano y sus dependencias. |
| `docs/urbano/estructura_codigo.md` | Describe la estructura comun de los scripts urbanos: imports, constantes, GUI, calculos y salidas. |
| `docs/urbano/estandar_visualizacion.md` | Resume las convenciones visuales de mapas y graficas urbanas. |
| `docs/github_migration_inventory.md` | Inventario inicial usado para decidir que scripts migrar a GitHub y que archivos excluir. |
