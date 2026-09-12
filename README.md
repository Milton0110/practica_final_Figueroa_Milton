# Evolve-estadistica-Milton-Figueroa
# Estadística y Modelado con Python

Práctica final del posgrado (bloque de Estadística Aplicada / Python), realizada sobre el dataset
[`video_games_sales.csv`](data/video_games_sales.csv). Consta de 4 ejercicios independientes que cubren
estadística descriptiva, inferencia con Scikit-Learn, regresión lineal múltiple implementada desde cero
en NumPy y análisis de series temporales.

Las respuestas razonadas a cada pregunta del enunciado están en [`Respuestas.md`](Respuestas.md).

## Estructura del repositorio

```
.
├── data/
│   └── video_games_sales.csv          # Dataset original (16598 filas, ventas de videojuegos)
├── output/                            # Figuras y ficheros generados por los scripts (no editar a mano)
├── utils_proyecto.py                  # Utilidades compartidas (carga de CSV, guardado de figuras)
├── ejercicio1_descriptivo.py          # Ejercicio 1 — Análisis estadístico descriptivo
├── ejercicio2_inferencia.py           # Ejercicio 2 — Inferencia con Scikit-Learn
├── ejercicio3_regresion_multiple.py   # Ejercicio 3 — Regresión lineal múltiple en NumPy (desde cero)
├── ejercicio4_series_temporales.py    # Ejercicio 4 — Descomposición y análisis de series temporales
└── Respuestas.md                      # Respuestas escritas a las preguntas del enunciado
```

## Ejercicios

### 1. Análisis Estadístico Descriptivo (`ejercicio1_descriptivo.py`)
Explora `video_games_sales.csv`: resumen estructural del dataset, estadísticos descriptivos por
variable numérica, histogramas (incluyendo versión `log1p` para las variables de ventas, muy sesgadas),
boxplots de la variable objetivo por categoría, heatmap de correlaciones, distribución de variables
categóricas (plataforma, género, publisher), detección de outliers por IQR con capado (winsorización)
y revisión de multicolinealidad.

**Salidas** (`output/`): `ej1_resumen_estructural.txt`, `ej1_descriptivo.csv`, `ej1_histogramas.png`,
`ej1_histogramas_log1p.png`, `ej1_histogramas_comparacion_log.png`, `ej1_boxplots.png`,
`ej1_heatmap_correlacion.png`, `ej1_categoricas.png`, `ej1_outliers.txt`, `ej1_datos_tratados_iqr.csv`,
`ej1_top3_correlaciones.txt`.

### 2. Inferencia con Scikit-Learn (`ejercicio2_inferencia.py`)
Entrena un pipeline de `LinearRegression` para predecir `global_sales` a partir de `year`, `platform`
y `genre` (variables elegidas deliberadamente para evitar data leakage frente a las ventas por región).
El pipeline incluye imputación, escalado de numéricas y one-hot encoding de categóricas, con
split train/test 80/20 (`random_state=42`).

**Salidas** (`output/`): `ej2_metricas_regresion.txt` (MAE, RMSE y R² en train y test),
`ej2_residuos.png` (residuos vs. predicción), `ej2_importancia_coeficientes.csv` (coeficientes
ordenados por magnitud absoluta).

### 3. Regresión Lineal Múltiple en NumPy (`ejercicio3_regresion_multiple.py`)
Implementación manual de regresión lineal múltiple por mínimos cuadrados (OLS), sin usar
Scikit-Learn: cálculo de coeficientes vía `np.linalg.lstsq` sobre la matriz de diseño con columna
de intercepto, y funciones propias de MAE, RMSE y R². Se valida contra un dataset sintético con
coeficientes reales conocidos (β₀=5, β₁=2, β₂=-1, β₃=0.5) para comprobar que la implementación
recupera esos valores de forma aproximada.

**Salidas** (`output/`): `ej3_coeficientes.txt` (coeficientes ajustados vs. reales),
`ej3_metricas.txt`, `ej3_predicciones.png` (real vs. predicho).

### 4. Series Temporales (`ejercicio4_series_temporales.py`)
Genera una serie temporal sintética diaria (2018–2023) con tendencia lineal, estacionalidad anual,
un ciclo de largo plazo (~4 años) y ruido gaussiano. Aplica `seasonal_decompose` (modelo aditivo,
periodo 365) y analiza el residuo resultante: media, desviación típica, asimetría, curtosis,
test de normalidad de Jarque-Bera, test de estacionariedad ADF y gráficos ACF/PACF, para valorar
si el residuo se comporta como ruido ideal.

**Salidas** (`output/`): `ej4_serie_original.png`, `ej4_descomposicion.png`, `ej4_acf_pacf.png`,
`ej4_histograma_ruido.png`, `ej4_analisis.txt`.

## Cómo ejecutar

Cada ejercicio es un script independiente que se ejecuta desde la raíz del repositorio (usan rutas
relativas como `data/` y `output/`):

```bash
python ejercicio1_descriptivo.py
python ejercicio2_inferencia.py
python ejercicio3_regresion_multiple.py
python ejercicio4_series_temporales.py
```

### Dependencias

El proyecto no incluye `requirements.txt`. Las librerías usadas a lo largo de los 4 scripts son:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels
```

## Dataset

`data/video_games_sales.csv` — ventas de videojuegos por plataforma, género, publisher y región
(NA, EU, JP, otros), con ventas globales (`global_sales`) como variable objetivo. Dataset de tipo
"vgsales", de origen público (Kaggle).
