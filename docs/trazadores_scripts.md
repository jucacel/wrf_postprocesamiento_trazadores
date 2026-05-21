# Scripts `wrf_trazadores`

Paquete minimo viable para analizar salidas WRF con trazadores atmosfericos,
alineado con la estructura observada en `wrf_urbano`:

- `wrf_trazadores_00_core.py`
  Nucleo reutilizable de lectura, validacion, tiempos, alturas AGL, mascaras y
  exportes comunes.
- `wrf_trazadores_02_columna_total.py`
  Calcula columna integrada, genera serie temporal y mapas medio/pico.
- `wrf_trazadores_03_serie_region.py`
  Genera series regionales de superficie, columna y centroide del penacho.
- `wrf_trazadores_04_fraccion_sobre_pblh.py`
  Calcula la fraccion del trazador sobre y bajo PBLH.
- `wrf_trazadores_07_desplazamiento_vertical.py`
  Genera un panel tiempo-altura, CSV diagnostico y GIF/frames de corte vertical
  para ver como el trazador se desplaza en altura.

## Dependencias usadas en esta implementacion

Esta version usa el stack que ya estaba disponible en el entorno local:

- `python3`
- `netCDF4`
- `numpy`
- `pandas`
- `matplotlib`

No depende de `xarray` para poder correr aun cuando ese paquete no este activo
en el entorno del terminal.

## Convenciones de entrada

Todos los scripts:

- aceptan uno o varios `wrfout`;
- detectan el trazador automaticamente si no se pasa `--tracer-name`;
- validan `Time`, `XLAT`, `XLONG` y las variables criticas del diagnostico;
- pueden trabajar sobre todo el dominio o sobre una caja lat-lon.
- si no se pasan archivos, intentan descubrir una corrida dentro de
  `Z_WRF_OUT` si esa carpeta existe junto al repositorio;
- si no se pasa `--outdir`, escriben por defecto en
  `POSTPROC_TRAZADORES/<script>` dentro de la carpeta de la corrida.

## Prioridad actual de autodeteccion

Cuando existe `Z_WRF_OUT`, el nucleo intenta usar estas carpetas en este orden:

1. `Z_WRF_OUT/1_DIA_VTRAZ`
2. otra carpeta de corrida reciente con `wrfout`
3. seleccion manual por `tkinter`

## Ejemplos de uso

### 1. Columna total en dominio completo

```bash
python3 scripts/trazadores/wrf_trazadores_02_columna_total.py \
  --run-dir /ruta/a/corrida_wrf \
  --domain d03 \
  --tracer-name tr17_1 \
  --outdir /ruta/salida
```

### 2. Serie regional en una caja y con fuente conocida

```bash
python3 scripts/trazadores/wrf_trazadores_03_serie_region.py \
  --run-dir /ruta/a/corrida_wrf \
  --domain d03 \
  --tracer-name tr17_1 \
  --lat-min 3.20 --lat-max 3.80 \
  --lon-min -76.80 --lon-max -76.10 \
  --source-lat 3.50 --source-lon -76.52 \
  --outdir /ruta/salida
```

### 3. Fraccion del trazador sobre PBLH

```bash
python3 scripts/trazadores/wrf_trazadores_04_fraccion_sobre_pblh.py \
  --run-dir /ruta/a/corrida_wrf \
  --domain d03 \
  --tracer-name tr17_1 \
  --outdir /ruta/salida
```

### 4. Desplazamiento vertical del trazador

```bash
python3 scripts/trazadores/wrf_trazadores_07_desplazamiento_vertical.py \
  --run-dir /ruta/a/corrida_wrf \
  --domain d01 \
  --tracer-name N_SLV \
  --i 56 --j 66 \
  --axis x \
  --x-coordinate lon \
  --frame-height-reference msl \
  --max-height 2500 \
  --auto-max-height \
  --auto-height-frac 0.1 \
  --height-bin 25 \
  --log-color \
  --tag caso_trazador_vertical_auto10pct
```

## Seleccion interactiva

Si no se pasan archivos ni carpeta de salida, los scripts intentan abrir
ventanas `tkinter` para seleccionar:

- archivos `wrfout`;
- carpeta de salida.

Ese comportamiento sigue el estilo operativo de `wrf_urbano`, pero la logica de
calculo queda separada de la GUI.

## Productos esperados

### `02_columna_total`

- CSV de serie temporal;
- PNG de serie regional;
- PNG de mapa de columna media;
- PNG de mapa del instante de pico regional.

### `03_serie_region`

- CSV con estadisticos regionales;
- PNG con panel de series;
- distancia fuente-centroide si se indica `--source-lat/--source-lon`.

### `04_fraccion_sobre_pblh`

- CSV con integrales y fracciones por tiempo;
- PNG de series;
- PNG de mapa medio de fraccion sobre PBLH;
- TXT con resumen del evento.

### `07_desplazamiento_vertical`

- CSV con masa proxy, fraccion sobre PBLH y alturas ztop por umbral;
- PNG tiempo-altura de la masa proxy por nivel vertical;
- PNG por tiempo de un corte vertical, con opcion de eje horizontal por
  longitud/latitud y topografia (`HGT`) en gris;
- GIF del corte vertical si hay dos o mas frames y Pillow esta disponible.
- Anotacion del origen del pulso, tipo de trazador, cantidad inicial proxy y
  duracion. Para `N_SLV/S_SLV`, si no se indica otra cosa, se clasifica como
  trazador inerte.
- Mapa inserto en cada frame con `HGT`, transecto y marcador del pulso. Se
  puede ocultar con `--hide-inset-map`.

## Notas fisicas

- La columna integrada se calcula como `trazador * dz` sumado verticalmente.
- La fraccion sobre PBLH usa la altura real AGL derivada de `PH` y `PHB`.
- Si las unidades del trazador son adimensionales, la integral se interpreta
  como una carga proxy consistente para comparacion relativa entre tiempos.
