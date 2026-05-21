# Ejemplos de comandos

## Serie regional

```bash
python scripts/trazadores/wrf_trazadores_03_serie_region.py \
  --run-dir /ruta/a/corrida_wrf \
  --domain d03 \
  --tracer-name tr17_1 \
  --lat-min 3.20 --lat-max 3.80 \
  --lon-min -76.80 --lon-max -76.10 \
  --outdir outputs/serie_region
```

## Validacion de masa `wrfchemi`

```bash
python scripts/trazadores/wrf_trazadores_09_validate_wrfchemi_mass.py \
  --interval-min 60 \
  /ruta/a/wrfchemi_d01
```
