# Estándar de Visualización: Scripts WRF Urbano

Los aspectos visuales de los scripts `wrf_urbano` están estandarizados para producir figuras listas para reportes técnicos o artículos científicos.

## 1. Convenciones de Escala y Color
-   **Paletas de Colores:** Uso recurrente de `coolwarm_r`. Esta paleta invertida es estratégica: los colores fríos (azul) representan alta ventilación o capacidad asimilativa, mientras que los cálidos (rojo) indican estancamiento o baja dispersión.
-   **Niveles Fijos:** Se definen `levels` mediante `np.linspace(0, tope, N)` para asegurar que diferentes mapas del mismo día sean comparables visualmente.
-   **Modos de Escala:** El usuario puede elegir entre:
    -   **Manual:** Valor fijo ingresado por el usuario.
    -   **Percentil 95:** Cálculo dinámico basado en los datos para evitar que valores atípicos (outliers) "quemen" la escala.
    -   **Temporada:** Topes predefinidos para época seca o húmeda.

## 2. Formato de Títulos
Los títulos se construyen dinámicamente siguiendo este patrón:
`{Variable} | {Fecha/Hora} | Tope: {ValorMax} | {ModoEscala}`
*Ejemplo: "Emax PM10 (kg/h) | 2018-07-10 14:00 | Tope 5000 | Escala P95"*

## 3. Elementos Decorativos y Anotaciones
Las figuras incluyen siempre estos elementos en posiciones consistentes:
-   **Etiquetas de Ciudades:** Rótulos de texto fijos para **Cali, Yumbo, Rozo y Palmira** en coordenadas hardcodeadas para facilitar la orientación geográfica.
-   **Caja de Categorías:** Un cuadro de texto (bbox) en la esquina inferior izquierda que clasifica los valores en "Baja, Media y Alta" según percentiles (33, 66).
-   **Nota de Resolución:** Un cuadro en la esquina inferior derecha indicando la resolución de la malla (ej. "Resolución: 1.0 km × 1.0 km").
-   **Vectores de Viento:** Flechas (`quiver`) de tamaño y densidad controlada, a menudo normalizadas para mostrar dirección sobre una base de VC.

## 4. Estética de Mapas
-   **Shapes:** Los límites administrativos se dibujan con `edgecolor="black"` y `linewidth=0.8` sobre los contornos de color.
-   **Límites del Eje:** Uso de constantes `LAT_MIN`, `LON_MAX` junto con un factor de `ZOOM_OUT` para dar aire a los bordes del mapa.
-   **Guardado:** Los archivos se guardan con nombres descriptivos que incluyen la hora y la etiqueta del archivo original (ej. `Emax_PM10_h1400_2018-07-10_d03.png`).
