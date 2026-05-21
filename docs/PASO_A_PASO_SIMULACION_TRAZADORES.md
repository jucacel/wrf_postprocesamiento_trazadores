# Paso A Paso Simulacion De Trazadores En WRF

Este documento resume el flujo completo que seguimos para trabajar trazadores en WRF, separando claramente dos rutas:

- Ruta 1: `WRF-Chem + PyChEmiss + wrfchemi`
- Ruta 2: `WRF tracer_opt=2` con trazador inerte interno

La diferencia es importante:

- La ruta con `PyChEmiss` inyecta especies quimicas, por ejemplo `CO`, usando archivos `wrfchemi_*`.
- La ruta con `tracer_opt=2` usa trazadores internos de WRF, como `N_SLV` y `S_SLV`, sin depender de `wrfchemi`.

## 1. Ruta WRF-Chem + PyChEmiss + wrfchemi

### 1.1 Preparar la corrida meteorologica base

Se requiere una carpeta de corrida con:

- `met_em.d01*`
- `met_em.d02*`
- `met_em.d03*`
- tablas de WRF
- `real.exe`
- `wrf.exe`
- `namelist.input` con quimica activa, por ejemplo `chem_opt = 100`

### 1.2 Ejecutar real.exe

Comando general:

```bash
./real.exe
```

Archivos generados:

- `wrfinput_d01`
- `wrfinput_d02`
- `wrfinput_d03`
- `wrfbdy_d01`

### 1.3 Preparar el pulso para PyChEmiss

En este flujo usamos scripts y configuraciones tipo:

- `generate_emissions_pulse.py`
- archivos `emissions_pulso*.txt`
- YAML o configuraciones de PyChEmiss

Aqui se define:

- especie a emitir, por ejemplo `E_CO`
- coordenadas del pulso
- intensidad
- hora de emision
- dominio objetivo

### 1.4 Ejecutar PyChEmiss

PyChEmiss toma el inventario/configuracion y genera archivos de emisiones WRF:

- `wrfchemi_d01_*`
- `wrfchemi_d02_*`
- `wrfchemi_d03_*`

Ejemplos:

- `wrfchemi_d01_2018-07-10_00:00:00`
- `wrfchemi_d02_2018-07-10_00:00:00`
- `wrfchemi_d03_2018-07-10_00:00:00`

### 1.5 Ajustar namelist.input para leer wrfchemi

En `&time_control` y `&chem` deben quedar activas las opciones de emisiones. En el flujo revisado aparecian configuraciones del tipo:

```text
io_form_auxinput5 = 2
auxinput5_inname = 'wrfchemi_d<domain>'
auxinput5_interval_m = ...
frames_per_auxinput5 = ...
emiss_inpt_opt = ...
chem_opt = 100,100,100
```

### 1.6 Ejecutar wrf.exe

Comando general:

```bash
mpirun -np 1 ./wrf.exe
```

o con mas nucleos si la corrida es estable.

WRF-Chem lee:

- meteorologia desde `wrfinput_d0*`
- emisiones desde `wrfchemi_d0*`

Mensajes tipicos en `rsl.error.0000`:

```text
med_read_wrf_chem_emissions: Open file wrfchemi_d03_2018-07-10_00:00:00
Input data is acceptable to use
```

### 1.7 Validar la salida

Archivos generados:

- `wrfout_d01_*`
- `wrfout_d02_*`
- `wrfout_d03_*`

En esta ruta se valido la variable `co`, y posteriormente se uso `delta_co` en postproceso para intentar separar el pulso del fondo.

### 1.8 Limitacion de esta ruta

`CO` ya existe como especie quimica del sistema, por lo tanto:

- el pulso puede mezclarse con el fondo
- `delta_co` sirve como diagnostico
- pero no elimina totalmente la ambiguedad fisica

Por esa razon se hizo la segunda ruta.

## 2. Ruta con trazador inerte puro usando tracer_opt=2

Esta ruta se uso para eliminar la ambiguedad de `CO`.

### 2.1 Tomar un arbol fuente WRF

Se trabajo sobre:

```text
/ruta/linux/WRFV4.6.1
```

### 2.2 Aplicar el parche de trazadores

Se copiaron archivos desde:

```text
/ruta/linux/WRF_Postprocesamiento_local/Archivos_traceropt/Registry.EM
/ruta/linux/WRF_Postprocesamiento_local/Archivos_traceropt/solve_em.F
/ruta/linux/WRF_Postprocesamiento_local/Archivos_traceropt/myoutfields_d01.txt
```

hacia el arbol fuente:

```text
WRFV4.6.1/Registry/Registry.EM
WRFV4.6.1/dyn_em/solve_em.F
```

### 2.3 Ajustar el punto de emision en solve_em.F

Originalmente el bloque de emision estaba fijo. Se corrigio para ubicar la emision en el punto objetivo.

En una version del parche se dejo algo del estilo:

```fortran
IF ( config_flags%tracer_opt == 2 ) THEN
  IF ( grid%id == 3 ) THEN
    DO j = 50, 54
      DO i = 21, 25
        tracer(i, 1, j, 1) = 1.0   ! N_SLV
        tracer(i, 1, j, 2) = 1.0   ! S_SLV
      END DO
    END DO
  END IF
END IF
```

Luego se comprobo que para el caso estable final convenia una corrida minima de 1 dominio.

### 2.4 Verificar las entradas del Registry

El `Registry.EM` quedo con trazadores como:

```text
state   real    N_SLV      ikjftb  tracer ...
state   real    S_SLV      ikjftb  tracer ...
package tracer_test2 tracer_opt==2 - tracer:N_SLV,S_SLV
```

### 2.5 Configurar y recompilar WRF

En el arbol `WRFV4.6.1` se ejecuto:

```bash
./configure
./compile em_real
```

La opcion seleccionada en `configure` fue:

- `34` = `GNU (gfortran/gcc) (dmpar)`
- nesting basico = `1`

### 2.6 Verificar que el codigo generado incluyo los trazadores

Se comprobo en archivos generados por WRF:

- `inc/scalar_indices.inc`
- `frame/module_configure.f90`

donde aparecian `N_SLV` y `S_SLV`.

### 2.7 Crear una carpeta de corrida nueva

Para no mezclar con la corrida quimica se crearon carpetas nuevas, por ejemplo:

- `WRF_RUN_VTRAZ_INERTE_4H`
- `WRF_RUN_VTRAZ_INERTE_1DOM`

### 2.8 Copiar binarios y meteorologia

Se copiaron:

- `real.exe`
- `wrf.exe`
- `met_em.d01*` o `met_em.d0*`
- `myoutfields_d01.txt`

### 2.9 Ajustar namelist.input

Para la corrida inerte se dejaron las siguientes reglas:

- quitar por completo `&chem`
- quitar referencias a `wrfchemi`
- activar `scalar_adv_opt = 2`
- activar `tracer_opt = 2`
- usar `iofields_filename = 'myoutfields_d01.txt'`

En la corrida minima de 1 dominio:

- `max_dom = 1`
- solo `d01`
- fisica simplificada y coherente
- corrida de `6 h` porque los `met_em` disponibles estaban cada `6 h`

### 2.10 Ejecutar real.exe

Comando:

```bash
./real.exe
```

Archivos generados:

- `wrfinput_d01`
- `wrfbdy_d01`

### 2.11 Ejecutar wrf.exe

La corrida anidada de 3 dominios resulto inestable.

La corrida estable final fue:

```bash
mpirun -np 1 ./wrf.exe
```

en un caso de 1 dominio.

### 2.12 Validar la salida

En el `wrfout_d01` final se verifico:

- `SUCCESS COMPLETE WRF`
- existencia de:
  - `N_SLV`
  - `S_SLV`

Ejemplo de chequeo:

```bash
ncdump -h wrfout_d01_2018-07-10_00:00:00 | grep -iE "N_SLV|S_SLV"
```

## 3. Resultado tecnico

Quedo demostrado que:

- `tracer_opt=2` si fue compilado correctamente
- el mecanismo de trazador inerte interno si funciona
- `N_SLV` y `S_SLV` pueden quedar escritos en el `wrfout`

El problema posterior fue de estabilidad del caso anidado y de contenido de una corrida especifica, no de la compilacion del trazador.

## 4. Archivos involucrados por ruta

### 4.1 Ruta PyChEmiss / wrfchemi

- `met_em.d0*`
- `real.exe`
- `wrf.exe`
- `namelist.input` con quimica
- scripts del pulso
- configuracion de PyChEmiss
- `wrfchemi_d0*`
- `wrfinput_d0*`
- `wrfbdy_d01`
- `wrfout_d0*`

### 4.2 Ruta tracer_opt=2

- `Registry.EM`
- `solve_em.F`
- `myoutfields_d01.txt`
- `configure`
- `compile em_real`
- `namelist.input` sin `&chem`
- `wrfinput_d01`
- `wrfbdy_d01`
- `wrfout_d01*`

## 5. Secuencia minima resumida

### 5.1 Si se quiere repetir la ruta PyChEmiss + CO

1. Preparar `met_em`
2. Ejecutar `real.exe`
3. Definir el pulso
4. Ejecutar PyChEmiss
5. Generar `wrfchemi_d0*`
6. Ajustar `namelist` quimico
7. Ejecutar `wrf.exe`
8. Validar `co` o `delta_co`

### 5.2 Si se quiere repetir la ruta de trazador inerte puro

1. Parchear `Registry.EM` y `solve_em.F`
2. Recompilar WRF
3. Crear nueva carpeta de corrida
4. Ajustar `namelist.input` con `tracer_opt = 2`
5. Ejecutar `real.exe`
6. Ejecutar `wrf.exe`
7. Validar `N_SLV` y `S_SLV`

## 6. Recomendacion practica

Si el objetivo es estudiar penachos sin ambiguedad quimica:

- usar preferentemente la ruta `tracer_opt=2`
- validar primero en un caso minimo de 1 dominio
- luego escalar al caso anidado

Si el objetivo es representar emisiones de una especie quimica real:

- usar `PyChEmiss + wrfchemi`
- pero interpretar con cuidado el fondo quimico

