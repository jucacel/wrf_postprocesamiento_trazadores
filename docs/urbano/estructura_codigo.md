# Estructura de Código: Scripts WRF Urbano

Los scripts de la serie `wrf_urbano` siguen un diseño modular y orientado a la interactividad. A continuación se detallan las coincidencias estructurales encontradas:

## 1. Organización del Archivo
Todos los scripts respetan el siguiente orden:
1.  **Importaciones:** Estándar con `numpy`, `xarray`, `matplotlib.pyplot`, `geopandas` y `tkinter`.
2.  **Configuración Global:** Bloques de constantes (mayúsculas) para límites geográficos (`LAT_MIN`, `LON_MAX`), parámetros físicos y niveles de visualización.
3.  **Interfaz Interactiva:** Funciones específicas que encapsulan popups de `tkinter` para entrada de usuario.
4.  **Cálculos Científicos:** Funciones puras que procesan DataArrays de xarray.
5.  **Generación de Salidas:** Funciones de graficación o guardado de CSV.
6.  **Bloque Main:** Función `main()` que orquesta el flujo y ejecución protegida por `if __name__ == "__main__":`.

## 2. Uso de Ventanas Emergentes (UI)
A diferencia de los scripts tradicionales de consola, estos utilizan `tkinter` para:
-   **Selección de archivos:** `filedialog.askopenfilenames` (frecuentemente filtrando por el prefijo `wrfout`).
-   **Toma de decisiones:** `messagebox.askyesno` para procesos iterativos (ej. "¿Desea agregar otro archivo?").
-   **Entrada de parámetros:** `simpledialog.askinteger` o `askfloat` para definir escalas, fechas disponibles o modos de operación.
-   **Selección de carpetas:** `filedirectory.askdirectory` para definir la ruta de salida de resultados.

## 3. Manejo de Datos y Coordenadas
-   **Xarray:** Es la librería base para abrir NetCDF, permitiendo indexación por nombre de variable y manejo de dimensiones temporales.
-   **Extracción de Fechas:** Uso de expresiones regulares (`re`) para extraer la fecha directamente del nombre del archivo `wrfout_dXX_YYYY-MM-DD`.
-   **Geospatial:** Uso de `geopandas` para cargar shapes decorativos y `griddata` (scipy) para interpolar vectores de viento a grillas regulares de visualización (`quiver`).

## 4. Trazabilidad
-   Uso de una función `run_tag()` basada en el tiempo actual para evitar la sobreescritura de resultados.
-   Mensajes en consola que informan el progreso (ej. "Procesando archivo X...", "✔ Mapas generados").
