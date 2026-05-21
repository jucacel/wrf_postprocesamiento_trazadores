# Migracion a CO real con `wrfchemi`

Esta ruta reemplaza trazadores adimensionales por una emision fisica de
monoxido de carbono en archivos `wrfchemi`, usando `E_CO` en
`mol km-2 hr-1`.

## Razon metodologica

Para evaluar una respuesta fisica del modelo conviene definir la masa emitida
desde el insumo de emisiones:

```text
M_CO_kg = sum(E_CO * area_km2 * dt_hr * 0.02801)
```

Los trazadores inertes siguen siendo utiles como diagnostico de transporte,
pero no representan una masa fisica si no se les impone una escala externa.

## Scripts

- `wrf_trazadores_08_generate_wrfchemi_pulse.py`: genera `wrfchemi` con pulso CO.
- `wrf_trazadores_09_validate_wrfchemi_mass.py`: valida masa fisica emitida.
- `wrf_trazadores_10_check_wrfout_co.py`: confirma que `wrfout` contiene `CO`, `co` o `CO_T`.
- `wrf_trazadores_11_patch_namelist_co_real.py`: ajusta `namelist.input` para una prueba corta.
- `wrf_trazadores_13_diagnose_source_signal.py`: diagnostica si la senal aparece en la fuente.
- `wrf_trazadores_14_visualize_case_control_delta.py`: grafica diferencia caso-control en corte.
- `wrf_trazadores_16_visualize_delta_planview.py`: grafica diferencia caso-control en planta.

## Ejemplo generico

Dentro o cerca de la carpeta de corrida:

```bash
python scripts/trazadores/wrf_trazadores_08_generate_wrfchemi_pulse.py \
  --wrfinput wrfinput_d01 \
  --out-dir . \
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
  --copy-dated-start \
  --include-zero-co2 \
  --include-zero-nox
```

Ese ejemplo genera:

```text
wrfchemi_d01
wrfchemi_2024-09-01_00:00:00_d01
```

Para una emision de 1 tonelada de CO en un parche 3x3 durante 1 hora.

## Validacion antes de correr WRF

```bash
python scripts/trazadores/wrf_trazadores_09_validate_wrfchemi_mass.py \
  --interval-min 60 \
  wrfchemi_d01
```

El resultado debe reportar una masa equivalente a la masa solicitada.

## Configuracion WRF-Chem

El `namelist.input` debe activar la lectura de `auxinput5` y un mecanismo que
instancie/escriba CO en `wrfout`. Un ejemplo minimo de parametros relevantes:

```text
&time_control
 io_form_auxinput5       = 2,
 auxinput5_inname        = 'wrfchemi_d<domain>',
 auxinput5_interval_m    = 60,
 frames_per_auxinput5    = 5,
/

&chem
 io_style_emissions      = 2,
 emiss_opt               = 3,
 bio_emiss_opt           = 0,
 gaschem_onoff           = 1,
 aerchem_onoff           = 0,
/
```

El parcheador automatiza algunos de esos cambios:

```bash
python scripts/trazadores/wrf_trazadores_11_patch_namelist_co_real.py \
  namelist.input \
  --run-hours 4 \
  --interval-min 60 \
  --io-style-emissions 2
```

El script deja un respaldo `namelist.input.before_co_real_wrfchemi`.

Si se usa `io_style_emissions=1`, generar archivos de 12 horas con:

```bash
--output-style split12
```

y ajustar:

```text
auxinput5_inname      = 'wrfchemi_<hour>z_d<domain>'
frames_per_auxinput5  = 12
```

## Validacion posterior a WRF

```bash
python scripts/trazadores/wrf_trazadores_10_check_wrfout_co.py \
  wrfout_*
```

Si el script reporta `NO_CO_VARIABLE`, WRF leyo emisiones pero el mecanismo no
esta escribiendo o instanciando CO. En ese caso hay que corregir el mecanismo
quimico o el `namelist.input` antes de interpretar resultados.

## Aislar el pulso de CO del fondo

Si el CO absoluto muestra una senal de fondo dominante, correr un control con
la misma configuracion pero `E_CO=0`, y visualizar:

```text
delta_co = co_con_pulso - co_control
```

Ejemplo:

```bash
python scripts/trazadores/wrf_trazadores_16_visualize_delta_planview.py \
  --case-dir /ruta/a/caso_con_pulso \
  --control-dir /ruta/a/control_sin_pulso \
  --outdir outputs/delta_co
```

Si `delta_co` sale exactamente cero, revisar primero que el `wrfchemi` haya
sido regenerado con las dimensiones y especies requeridas por el mecanismo,
y luego relanzar tanto el caso con pulso como el control.

## Interpretacion para capacidad asimilativa

Una vez `CO` aparece en `wrfout`, una corrida con masa conocida puede usarse
como respuesta unitaria:

```text
R = concentracion_modelada / M_emitida
M_admisible = (C_limite - C_fondo) / R
```

Para trazabilidad tecnica, conservar fuera del repositorio publico:

- masa emitida,
- duracion del pulso,
- ubicacion del pulso,
- configuracion PBL,
- mecanismo `chem_opt`,
- variable de salida usada (`CO`, `co` o `CO_T`),
- unidad de esa variable en `wrfout`.
