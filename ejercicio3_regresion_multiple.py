"""
=============================================================================
PRACTICA FINAL - EJERCICIO 3
Regresion Lineal Multiple implementada desde cero con NumPy
=============================================================================
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils_proyecto import guardar_figura

# Creamos output si todavia no existe.
os.makedirs("output", exist_ok=True)


def _agregar_intercepto(X):
    """
    Descripcion:
    Anade la columna de unos para incluir el intercepto en el modelo.

    Parametros:
    X (np.ndarray): Matriz de entrada con forma (n, p).

    Retorna:
    np.ndarray: Matriz extendida con forma (n, p+1).
    """
    return np.column_stack([np.ones(X.shape[0]), X])


def _a_arrays(y_real, y_pred):
    """
    Descripcion:
    Convierte y_real e y_pred a arrays de NumPy para operar sin sorpresas.

    Parametros:
    y_real: Valores reales.
    y_pred: Valores predichos.

    Retorna:
    tuple[np.ndarray, np.ndarray]: Par de arrays (y_real, y_pred).
    """
    return np.asarray(y_real), np.asarray(y_pred)


def regresion_lineal_multiple(X_train, y_train, X_test):
    """
    Descripcion:
    Ajusta una regresion lineal multiple por OLS y predice sobre X_test.

    Parametros:
    X_train (np.ndarray): Features de entrenamiento con forma (n_train, p).
    y_train (np.ndarray): Objetivo de entrenamiento con forma (n_train,).
    X_test (np.ndarray): Features de test con forma (n_test, p).

    Retorna:
    tuple[np.ndarray, np.ndarray]:
    - coefs: Vector [beta0, beta1, ..., betap].
    - y_pred: Predicciones para X_test.
    """
    X_train_b = _agregar_intercepto(X_train)
    coefs, *_ = np.linalg.lstsq(X_train_b, y_train, rcond=None)

    X_test_b = _agregar_intercepto(X_test)
    y_pred = X_test_b @ coefs

    return coefs, y_pred


def calcular_mae(y_real, y_pred):
    """
    Descripcion:
    Calcula el Mean Absolute Error (MAE).

    Parametros:
    y_real (np.ndarray): Valores reales.
    y_pred (np.ndarray): Valores predichos.

    Retorna:
    float: Valor MAE.
    """
    y_real, y_pred = _a_arrays(y_real, y_pred)
    return float(np.mean(np.abs(y_real - y_pred)))


def calcular_rmse(y_real, y_pred):
    """
    Descripcion:
    Calcula el Root Mean Squared Error (RMSE).

    Parametros:
    y_real (np.ndarray): Valores reales.
    y_pred (np.ndarray): Valores predichos.

    Retorna:
    float: Valor RMSE.
    """
    y_real, y_pred = _a_arrays(y_real, y_pred)
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def calcular_r2(y_real, y_pred):
    """
    Descripcion:
    Calcula el coeficiente de determinacion R2.

    Parametros:
    y_real (np.ndarray): Valores reales.
    y_pred (np.ndarray): Valores predichos.

    Retorna:
    float: Valor de R2.
    """
    y_real, y_pred = _a_arrays(y_real, y_pred)
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def graficar_real_vs_predicho(y_real, y_pred, ruta_salida="output/ej3_predicciones.png"):
    """
    Descripcion:
    Genera el scatter de valores reales vs predichos.

    Parametros:
    y_real (np.ndarray): Valores reales en test.
    y_pred (np.ndarray): Predicciones del modelo.
    ruta_salida (str): Ruta donde guardar la figura.

    Retorna:
    None: Guarda la figura en disco.
    """
    y_real, y_pred = _a_arrays(y_real, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_real, y_pred, alpha=0.65, color="#2a9d8f", edgecolor="none")

    vmin = min(np.min(y_real), np.min(y_pred))
    vmax = max(np.max(y_real), np.max(y_pred))
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="crimson", linewidth=1.5)

    ax.set_xlabel("Valores reales")
    ax.set_ylabel("Valores predichos")
    ax.set_title("Real vs. Predicho")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    guardar_figura(fig, os.fspath(ruta_salida))


def guardar_coeficientes(coefs, coefs_reales, n_features):
    """
    Descripcion:
    Guarda coeficientes ajustados y reales en un txt.

    Parametros:
    coefs (np.ndarray): Coeficientes ajustados del modelo.
    coefs_reales (np.ndarray): Coeficientes reales de referencia.
    n_features (int): Numero de variables explicativas.

    Retorna:
    None: Escribe `output/ej3_coeficientes.txt`.
    """
    with open("output/ej3_coeficientes.txt", "w", encoding="utf-8") as f:
        f.write("Regresion Lineal Multiple - Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres = ["Intercepto (beta0)"] + [f"beta{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres, coefs):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia:\n")
        for nombre, valor in zip(nombres, coefs_reales):
            f.write(f"  {nombre}: {valor:.6f}\n")


def guardar_metricas(mae, rmse, r2):
    """
    Descripcion:
    Guarda MAE, RMSE y R2 en un txt.

    Parametros:
    mae (float): Mean Absolute Error.
    rmse (float): Root Mean Squared Error.
    r2 (float): Coeficiente R2.

    Retorna:
    None: Escribe `output/ej3_metricas.txt`.
    """
    with open("output/ej3_metricas.txt", "w", encoding="utf-8") as f:
        f.write("Regresion Lineal Multiple - Metricas de evaluacion\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R2   : {r2:.6f}\n")


# =============================================================================
# MAIN — NO MODIFIQUES ESTE BLOQUE (es la prueba de referencia del profesor)
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Datos sintéticos con semilla fija para reproducibilidad
    # -------------------------------------------------------------------------
    SEMILLA = 42
    rng = np.random.default_rng(SEMILLA)

    n_muestras = 200
    n_features = 3

    # Generamos features aleatorias
    X = rng.standard_normal((n_muestras, n_features))

    # Coeficientes "reales" conocidos: β₀=5, β₁=2, β₂=-1, β₃=0.5
    coefs_reales = np.array([5.0, 2.0, -1.0, 0.5])

    # Variable objetivo con ruido gaussiano (σ=1.5)
    ruido = rng.normal(0, 1.5, n_muestras)
    y = coefs_reales[0] + X @ coefs_reales[1:] + ruido

    # -------------------------------------------------------------------------
    # Split Train / Test (80% / 20%) — sin mezclar aleatoriamente
    # -------------------------------------------------------------------------
    corte = int(0.8 * n_muestras)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    # -------------------------------------------------------------------------
    # Ajuste del modelo
    # -------------------------------------------------------------------------
    coefs, y_pred = regresion_lineal_multiple(X_train, y_train, X_test)

    # -------------------------------------------------------------------------
    # Métricas
    # -------------------------------------------------------------------------
    mae  = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2   = calcular_r2(y_test, y_pred)

    # -------------------------------------------------------------------------
    # Mostrar resultados en consola
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("RESULTADOS — Regresión Lineal Múltiple (NumPy)")
    print("=" * 50)
    print(f"\nCoeficientes reales:   {coefs_reales}")
    print(f"Coeficientes ajustados: {coefs}")
    print(f"\nMétricas sobre test set:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # -------------------------------------------------------------------------
    # RESULTADO DE REFERENCIA DEL PROFESOR
    # Con SEMILLA=42, los resultados esperados aproximados son:
    #   Coefs ajustados ≈ [5.0, 2.0, -1.0, 0.5]  (cercanos a los reales)
    #   MAE  ≈ 1.20  (±0.20 según implementación)
    #   RMSE ≈ 1.50  (±0.20 según implementación)
    #   R²   ≈ 0.80  (±0.05 según implementación)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Guardar salidas
    # -------------------------------------------------------------------------

    # Fichero de coeficientes
    with open("output/ej3_coeficientes.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal Múltiple — Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres = ["Intercepto (β₀)"] + [f"β{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres, coefs):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia:\n")
        for nombre, valor in zip(nombres, coefs_reales):
            f.write(f"  {nombre}: {valor:.6f}\n")

    # Fichero de métricas
    with open("output/ej3_metricas.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal Múltiple — Métricas de evaluación\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R²   : {r2:.6f}\n")

    # Gráfico
    graficar_real_vs_predicho(y_test, y_pred)

    print("\nSalidas guardadas en la carpeta output/")
    print("  → output/ej3_coeficientes.txt")
    print("  → output/ej3_metricas.txt")
    print("  → output/ej3_predicciones.png")
