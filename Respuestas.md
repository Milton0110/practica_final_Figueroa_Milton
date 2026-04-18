# Respuestas - Practica Final: Analisis y Modelado de Datos

---

## Ejercicio 1 - Analisis Estadistico Descriptivo

Descripcion y analisis:
Se trabajo con el dataset `video_games_sales.csv` (16598 filas, 11 columnas). Se eligio `global_sales` como variable objetivo continua para regresion. El analisis descriptivo se guardo en `output/ej1_descriptivo.csv` y se generaron los graficos requeridos de histogramas, boxplots por categoria, heatmap y frecuencias categoricas.

Para los histogramas hemos incluido 2 imagenes extra en la que aplico log1p para ver con un poquito mas de detalle la representación gráfica:
-ej1_histogramas_log1p.png: en el que se ven las graficas con el log1p aplicado.
-ej1_histogramas_comparacion_log.png: que es una comparativa de las graficas originales vs las graficas con el log1p aplicado.

Para las categoricas (ej1_categoricas.png) decidí darle más tamaño a la gráfica de "frecuencia - publisher" y no mostrar otros para que se puedan representar mejor los datos, además la muestra tiene los ejes invertidos para que los labels no se solapen ni descoloquen y se pueda visualizar correctamente la gráfica.

### Pregunta 1.1 - Fuente del dataset y variable objetivo
El dataset utilizado es `video_games_sales.csv`, basado en el dataset publico de ventas historicas de videojuegos (conocido como `vgsales`, difundido en Kaggle y repositorios docentes).
URL: https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024

La variable objetivo elegida fue `global_sales` porque:
1. Es numerica continua.
2. Resume el rendimiento total comercial de cada videojuego.
3. Tiene sentido economico/modelado intentar estimarla desde metadatos como `year`, `platform` y `genre`.

### Pregunta 1.2 - Distribucion y outliers
La distribucion de las variables de ventas (`na_sales`, `eu_sales`, `jp_sales`, `other_sales`, `global_sales`) es fuertemente asimetrica a la derecha, con muchos titulos de ventas bajas y pocos superventas. Por esa razón y para ver un poco mejor los datos adjunto un reajuste hecho con log1p y la comparativa de graficas, se ve un poquito más claro.

En `global_sales`:
- IQR = 0.41 (Q1 = 0.06, Q3 = 0.47)
- Skewness = 17.4006
- Kurtosis = 603.9323
- Outliers (IQR): 1893 observaciones (11.40%)

Tambien hay outliers altos en ventas regionales (entre 10% y 15% aprox. segun columna). Se decidio no eliminarlos de forma agresiva, porque representan casos reales de juegos superventas (informacion relevante del dominio).

### Pregunta 1.3 - Top 3 correlaciones con la variable objetivo
Las tres variables con mayor correlacion absoluta con `global_sales` son:
1. `na_sales`: 0.9410
2. `eu_sales`: 0.9028
3. `other_sales`: 0.7483

Esto tiene sentido ya que JP es un país solo y no puede competir con los agrupados en "other"

### Pregunta 1.4 - Valores nulos y tratamiento
Nulos detectados:
- `year`: 1.63%
- `publisher`: 0.35%
- Resto: 0%

Tratamiento aplicado:
Para la detección de outliers utilicé el método IQR porque es robusto ante distribuciones asimétricas y colas largas, características presentes en las variables de ventas. Definí límites por variable como Q1 - 1.5*IQR y Q3 + 1.5*IQR. Como tratamiento apliqué winsorización (capado) de los valores fuera de esos límites en las variables de ventas, evitando eliminar filas y conservando el tamaño muestral. En rank y year solo realicé diagnóstico por su naturaleza (identificador/temporal).

1. En analisis descriptivo (Ejercicio 1), se mantuvieron para no distorsionar el retrato original del dataset.
2. En modelado (Ejercicio 2), `year` se imputo con mediana dentro del pipeline.
3. La columna `publisher` no se uso en el modelo final para evitar inestabilidad por alta cardinalidad.

---

## Ejercicio 2 - Inferencia con Scikit-Learn

Descripcion y analisis:
Se implemento un pipeline de `LinearRegression` con preprocesado reproducible:
1. Split 80/20 con `random_state=42`.
2. Imputacion de numericas con mediana.
3. Escalado de numericas (`StandardScaler`).
4. One-hot encoding de categoricas (`drop='first'`, `handle_unknown='ignore'`).

Para evitar fuga de informacion, se uso como predictores solo: `year`, `platform`, `genre`.

### Pregunta 2.1 - MAE, RMSE, R2 y valoracion del modelo
Metricas sobre test set:
- MAE = 0.590574
- RMSE = 2.032835
- R2 = 0.016410

Valoracion:
El modelo no es muy bueno para hacer predicciones (al descargarlo creía que eran ventas por año y no el año de salida del videojugo y sus ventas)

Al hacer el ejercicio se quedan valores que muestran underfitting porque tanto train como test tienen R2 bajos (0.0649 vs 0.0164), así que el modelo no captura bien el patrón. Además tenemos una brecha en  train-test con un ligero sobreajuste y el problema principal es la falta de capacidad explicativa. Al no poder explicar bien porqué la variable objetivo (global_sales) cambia en el modo que lo hace con las variables dadas. Y no podemos incluir las otras variables de _sales ya que crearian leakage con mucha fuga de información.
Una regresion lineal simple con pocas variables no captura bien la dinamica real de ventas globales.

Variables mas influyentes (por magnitud de coeficiente): dummies de plataforma, especialmente `GB`, `NES`, `PS4`, `XOne`, `WiiU` y en el caso de los Generos `Aventura` y `Estrategia`

---

## Ejercicio 3 - Regresion Lineal Multiple en NumPy
