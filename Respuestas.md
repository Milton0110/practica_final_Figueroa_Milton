# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Documento combinado entre la plantilla del ejercicio y mis respuestas.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

Descripción y análisis:
Trabajé con `video_games_sales.csv` (16598 filas, 11 columnas).  
La variable objetivo fue `global_sales` porque representa las ventas globales del juego y tiene sentido intentar estimarla con variables como año, plataforma y género.

Además de los histogramas normales, añadí:
- `ej1_histogramas_log1p.png`
- `ej1_histogramas_comparacion_log.png`

Esto lo hice para ver mejor la distribución, porque las ventas están muy concentradas en valores bajos y unos pocos juegos venden muchísimo.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

El dataset es `video_games_sales.csv` (dataset público tipo vgsales, difundido en Kaggle).  
Referencia: https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024

El target es `global_sales` porque:
1. Es numérico continuo.
2. Resume el resultado comercial total.
3. Tiene lógica predecirlo con características del juego que suelen estar relacionadas con cuánto termina vendiendo. Datos como año de salida, plataforma, género o publisher.

---

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

Las variables de ventas (`na_sales`, `eu_sales`, `jp_sales`, `other_sales`, `global_sales`) están muy sesgadas a la derecha: muchos juegos venden poco y unos pocos venden muchísimo.

Sí hay outliers (sobre todo en variables de ventas).  
Ejemplo en `global_sales`: alrededor de 11.40% por IQR.

Decisión: no eliminarlos en bloque porque son casos reales importantes (superventas).  
Apliqué IQR para detectar y capado (winsorización) en variables de ventas para tratamiento, sin borrar filas.

---

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

Top 3:
1. `na_sales`: 0.9410
2. `eu_sales`: 0.9028
3. `other_sales`: 0.7483

Tiene sentido porque son ventas por regiones y están muy relacionadas con las ventas globales.

---

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

Sí, pero pocos:
- `year`: 1.63%
- `publisher`: 0.35%
- resto: 0%

Tratamiento:
1. En descriptivo los mantuve para no alterar la foto real del dataset.
2. En modelado, `year` se imputa con mediana dentro del pipeline.
3. `publisher` no se usó en el modelo final por alta cardinalidad e inestabilidad.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

Descripción y análisis:
Hice un pipeline de `LinearRegression` con:
1. split 80/20 (`random_state=42`)
2. imputación (mediana en numéricas)
3. escalado (`StandardScaler`)
4. one-hot en categóricas (`drop='first'`, `handle_unknown='ignore'`)

Para evitar fuga de información usé como features: `year`, `platform`, `genre`.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

Resultados en test:
- MAE = 0.590574
- RMSE = 2.032835
- R² = 0.016410

Conclusión corta: no funciona muy bien para predecir.

Lo que pasa es que el modelo se queda corto (underfitting): incluso en train el R² es bajo, y en test también.  
Además no usamos las otras `_sales` porque eso provocaría leakage (sería casi darle la respuesta al modelo).

Variables más influyentes (por coeficientes): sobre todo plataformas (`GB`, `NES`, `PS4`, `XOne`, `WiiU`) y en géneros destacan `Adventure` y `Strategy`.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

Descripción y análisis:
En este ejercicio armamos una regresión lineal "a mano" con NumPy, sin apoyarnos en scikit-learn ni en fórmulas predefinidas. La idea era entender qué pasa por dentro: calcular coeficientes, predecir y evaluar métricas.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

Esa formula busca los coeficientes que mejor encajan con los datos, intentando que el error total sea lo mas pequeño posible.

Y la columna de unos se mete para que exista el intercepto β₀.
Si no la pones, obligas al modelo a pasar por cero si o si, y eso normalmente te empeora el ajuste.

---

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parámetro | Valor real | Valor ajustado |
|-----------|-----------:|---------------:|
| β₀        | 5.0        | 4.864995       |
| β₁        | 2.0        | 2.063618       |
| β₂        | -1.0       | -1.117038      |
| β₃        | 0.5        | 0.438517       |

Quedaron bastante cerca, asi que la implementacion esta bien.

---

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

- MAE = 1.166462
- RMSE = 1.461243
- R² = 0.689672

MAE y RMSE sí quedan cerca de la referencia. El R² queda algo por debajo del valor orientativo del enunciado, pero el ajuste sigue siendo coherente con un modelo lineal con ruido.

---

**Pregunta 3.4** — Compara los resultados con la regresión logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido.

No, no sale parecido: en el Ejercicio 3 el resultado es bastante mejor que en el 2.

¿Que cambia?
1. En el 3 los datos están generados para tener una relación lineal clara.
2. En el 2 son datos reales de mercado (mucho más desordenados, con ruido y factores no observados) y además no usamos columnas que provocarían leakage.
3. Por eso en el 2 el modelo explica poquito y en el 3 le va bastante mejor.

---

## Ejercicio 4 — Series Temporales

Descripción y análisis:
Se ejecutó el flujo completo: serie original, descomposición, ACF/PACF, histograma de residuo y tests.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

Si, se ve una tendencia creciente bastante clara.

A ojo con los valores de la descomposicion:
- arranca cerca de ~64.14 (en la parte válida de tendencia)
- termina cerca de ~155.25
- sube aprox. ~91.10 en 6 años

---

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

Si, hay estacionalidad anual.

- periodo: ~365 dias (al hacer la creacion 365.25, los calculos los hago con 365)
- amplitud aprox.: ~18.40
- pico a pico aprox.: ~36.80

---

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

Si, se notan ciclos de largo plazo.

Para mi, la forma simple de diferenciarlo:
Los ciclos presentan subidas y bajadas alrededor de la tendencia. La grafica muestra una subida y una bajada que no llega hasta su punto minimo (ciclos).
Por esto la Tendencia es a aumentar ya que se acumulan los aumentos de los ciclos en la tendencia anual.

---

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

Resultados del residuo:
- media = 0.127078
- desviación típica = 3.222043
- p-value Jarque-Bera = 0.576561
- p-value ADF = 0.000000

Interpretación rápida:
Se comporta bastante parecido a ruido ideal: media cercana a 0, normalidad no rechazada (JB) y estacionariedad clara (ADF).

---

*Fin del documento de respuestas*
