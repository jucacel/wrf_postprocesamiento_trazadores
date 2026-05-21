# Scripts `wrf_urbano`

Este bloque agrupa scripts interactivos para diagnosticos meteorologicos urbanos a partir de salidas WRF.

## Scripts incluidos

| Script | Proposito |
|---|---|
| `wrf_urbano_01_prec.py` | Clasificacion de precipitacion diaria. |
| `wrf_urbano_02_prec_viento_pblh.py` | Precipitacion, viento y PBLH. |
| `wrf_urbano_03_cuadros.py` | Mapas/cuadros de coeficiente de ventilacion. |
| `wrf_urbano_03_mapacontorno_vc_v2.py` | Mapa de contornos de VC. |
| `wrf_urbano_04_momentodia_serietemp.py` | Ciclo horario y series temporales. |
| `wrf_urbano_05_mapasviento_vc_wd.py` | Mapa de VC con vectores de viento. |
| `wrf_urbano_06_vc_viento_por_dia_v2.py` | Mapas por timestep para un dia. |
| `wrf_urbano_07_capacidad_asimilativa_pm10_v2.py` | Capacidad asimilativa PM10 horaria/diaria. |
| `wrf_urbano_08_capacidad_asimilativa_pm10_scatter.py` | Series y dispersion para Emax PM10. |
| `wrf_urbano_09_capacidad_asimilativa_q.py` | Calculo de `q` para PM10. |
| `wrf_urbano_10_q_mapa_por_celda.py` | Mapa espacial de `q` por celda. |
| `wrf_urbano_11_allwine_whiteman.py` | Indices Allwine-Whiteman de estancamiento, recirculacion y ventilacion. |
| `wrf_urbano_ALL_v1.py` | Script unificado/autocontenido con varios diagnosticos urbanos. |

## Nota sobre duplicados

`wrf_urbano_12_allwine_whiteman.py` no se incluyo porque es identico a `wrf_urbano_11_allwine_whiteman.py` en la carpeta fuente revisada.

## Dependencias

Estos scripts usan principalmente `xarray`, `numpy`, `matplotlib`, `tkinter`, `geopandas`, `scipy` y `pandas`.

Varios flujos son interactivos y abren ventanas `tkinter` para seleccionar archivos, carpetas, fechas o parametros. En servidores sin entorno grafico puede ser necesario adaptar los scripts a CLI antes de ejecutarlos.

## Salidas

Las salidas generadas deben mantenerse fuera de Git. Use carpetas como `outputs/`, `results/` o una ruta externa de trabajo.
