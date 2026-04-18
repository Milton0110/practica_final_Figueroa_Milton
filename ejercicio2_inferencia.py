"""
=============================================================================
PRACTICA FINAL - EJERCICIO 2
Inferencia con Scikit-Learn sobre video_games_sales.csv
=============================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils_proyecto import cargar_csv_con_year_numerico, guardar_figura

SEED = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DATASET_FILENAME = "video_games_sales.csv"
TARGET_COLUMN = "global_sales"

# Columnas elegidas para evitar fuga de informacion directa sobre la variable objetivo.
FEATURE_COLUMNS = ["year", "platform", "genre"]

sns.set_theme(style="whitegrid")


def cargar_dataset() -> pd.DataFrame:
    """
    Descripcion:
    Carga el dataset desde disco y normaliza la columna `year`.

    Parametros:
    Esta funcion no recibe parametros.

    Retorna:
    pd.DataFrame: Dataset cargado.
    """
    dataset_path = DATA_DIR / DATASET_FILENAME
    return cargar_csv_con_year_numerico(dataset_path)


def construir_preprocesador(df_x: pd.DataFrame) -> ColumnTransformer:
    """
    Descripcion:
    Construye el preprocesador para variables numericas y categoricas.

    Parametros:
    df_x (pd.DataFrame): Matriz de variables predictoras.

    Retorna:
    ColumnTransformer: Transformador compuesto listo para entrenamiento.
    """
    num_cols = df_x.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_x.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def extraer_importancias_lineales(model: Pipeline) -> pd.DataFrame:
    """
    Descripcion:
    Extrae coeficientes del modelo lineal y calcula su magnitud absoluta.

    Parametros:
    model (Pipeline): Pipeline entrenado con pasos `preprocess` y `regressor`.

    Retorna:
    pd.DataFrame: Tabla ordenada por importancia absoluta de coeficientes.
    """
    pre = model.named_steps["preprocess"]
    reg = model.named_steps["regressor"]

    feature_names = pre.get_feature_names_out()
    coefs = reg.coef_.ravel()

    imp = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs),
        }
    ).sort_values("abs_coef", ascending=False)
    return imp


def entrenar_modelo(df: pd.DataFrame) -> dict[str, object]:
    """
    Descripcion:
    Entrena una regresion lineal con preprocesado y calcula metricas.

    Parametros:
    df (pd.DataFrame): Dataset de entrada con features y target.

    Retorna:
    dict[str, object]: Diccionario con modelo, metricas, particiones, predicciones y residuos.
    """
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    faltantes = [c for c in required if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    local = df[required].copy()
    local = local.dropna(subset=[TARGET_COLUMN])

    x = local[FEATURE_COLUMNS]
    y = local[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
    )

    preprocessor = construir_preprocesador(x_train)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    met = {
        "mae_train": float(mean_absolute_error(y_train, pred_train)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "r2_train": float(r2_score(y_train, pred_train)),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "r2_test": float(r2_score(y_test, pred_test)),
    }

    resid = y_test - pred_test

    return {
        "model": model,
        "metrics": met,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "pred_test": pred_test,
        "resid": resid,
    }


def guardar_metricas(resultados: dict[str, object]) -> None:
    """
    Descripcion:
    Guarda en texto las metricas de train y test del modelo.

    Parametros:
    resultados (dict[str, object]): Estructura de salida de `entrenar_modelo`.

    Retorna:
    None: Escribe `output/ej2_metricas_regresion.txt`.
    """
    met = resultados["metrics"]
    ruta = OUTPUT_DIR / "ej2_metricas_regresion.txt"

    with ruta.open("w", encoding="utf-8") as f:
        f.write("Regresion Lineal - Metricas\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {DATASET_FILENAME}\n")
        f.write(f"Target: {TARGET_COLUMN}\n")
        f.write(f"Features usadas: {', '.join(FEATURE_COLUMNS)}\n")
        f.write("\nTrain:\n")
        f.write(f"  MAE  = {met['mae_train']:.6f}\n")
        f.write(f"  RMSE = {met['rmse_train']:.6f}\n")
        f.write(f"  R2   = {met['r2_train']:.6f}\n")
        f.write("\nTest:\n")
        f.write(f"  MAE  = {met['mae_test']:.6f}\n")
        f.write(f"  RMSE = {met['rmse_test']:.6f}\n")
        f.write(f"  R2   = {met['r2_test']:.6f}\n")


def guardar_residuos(resultados: dict[str, object]) -> None:
    """
    Descripcion:
    Genera el grafico de residuos frente a valores predichos.

    Parametros:
    resultados (dict[str, object]): Estructura de salida de `entrenar_modelo`.

    Retorna:
    None: Guarda `output/ej2_residuos.png`.
    """
    y_pred = resultados["pred_test"]
    resid = resultados["resid"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=resid, alpha=0.6, color="#264653", ax=ax)
    ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Valores predichos")
    ax.set_ylabel("Residuo (y_real - y_pred)")
    ax.set_title("Grafico de residuos - Regresion lineal")
    fig.tight_layout()
    guardar_figura(fig, OUTPUT_DIR / "ej2_residuos.png")


def guardar_importancias(resultados: dict[str, object]) -> None:
    """
    Descripcion:
    Exporta ranking de importancias lineales basado en coeficientes absolutos.

    Parametros:
    resultados (dict[str, object]): Estructura de salida de `entrenar_modelo`.

    Retorna:
    None: Escribe `output/ej2_importancia_coeficientes.csv`.
    """
    model = resultados["model"]
    imp = extraer_importancias_lineales(model)
    imp.to_csv(OUTPUT_DIR / "ej2_importancia_coeficientes.csv", index=False, encoding="utf-8")


def main() -> None:
    """
    Descripcion:
    Ejecuta el pipeline completo del ejercicio 2 y genera sus salidas.

    Parametros:
    Esta funcion no recibe parametros.

    Retorna:
    None: Entrena, evalua y guarda artefactos en `output/`.
    """
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_dataset()
    resultados = entrenar_modelo(df)

    guardar_metricas(resultados)
    guardar_residuos(resultados)
    guardar_importancias(resultados)

    met = resultados["metrics"]

    print("=" * 60)
    print("EJERCICIO 2 completado")
    print("=" * 60)
    print(f"Target: {TARGET_COLUMN}")
    print(f"MAE test:  {met['mae_test']:.4f}")
    print(f"RMSE test: {met['rmse_test']:.4f}")
    print(f"R2 test:   {met['r2_test']:.4f}")
    print("Salidas:")
    print("  - output/ej2_metricas_regresion.txt")
    print("  - output/ej2_residuos.png")
    print("  - output/ej2_importancia_coeficientes.csv")


if __name__ == "__main__":
    main()
