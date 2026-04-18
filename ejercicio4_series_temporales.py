"""
=============================================================================
PRACTICA FINAL - EJERCICIO 4
Analisis y Descomposicion de Series Temporales
=============================================================================
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from utils_proyecto import guardar_figura

# Creamos output si hace falta.
os.makedirs("output", exist_ok=True)


def generar_serie_temporal(semilla=42):
    """
    Descripcion:
    Genera una serie sintetica con tendencia, estacionalidad, ciclo y ruido.

    Parametros:
    semilla (int): Semilla aleatoria para reproducibilidad.

    Retorna:
    pd.Series: Serie diaria entre 2018-01-01 y 2023-12-31.
    """
    rng = np.random.default_rng(semilla)

    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    tendencia = 0.05 * t + 50
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) + 6 * np.cos(4 * np.pi * t / 365.25)
    ciclo = 8 * np.sin(2 * np.pi * t / 1461)
    ruido = rng.normal(loc=0, scale=3.5, size=n)

    valores = tendencia + estacionalidad + ciclo + ruido
    return pd.Series(valores, index=fechas, name="valor")


def visualizar_serie(serie):
    """
    Descripcion:
    Dibuja la serie completa y la guarda como imagen.

    Parametros:
    serie (pd.Series): Serie temporal a visualizar.

    Retorna:
    None: Guarda `output/ej4_serie_original.png`.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(serie.index, serie.values, color="#1f77b4", linewidth=1.0)
    ax.set_title("Serie temporal original")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    guardar_figura(fig, "output/ej4_serie_original.png")


def descomponer_serie(serie):
    """
    Descripcion:
    Descompone la serie en componentes aditivos y guarda el grafico.

    Parametros:
    serie (pd.Series): Serie temporal de entrada.

    Retorna:
    DecomposeResult: Resultado con trend, seasonal y resid.
    """
    resultado = seasonal_decompose(serie, model="additive", period=365)
    fig = resultado.plot()
    fig.set_size_inches(12, 9)
    fig.tight_layout()
    guardar_figura(fig, "output/ej4_descomposicion.png")
    return resultado


def _calcular_estadisticas_residuo(residuo_limpio):
    """
    Descripcion:
    Calcula estadisticos basicos y tests para el residuo.

    Parametros:
    residuo_limpio (pd.Series): Residuo sin NaN.

    Retorna:
    dict[str, float]: Diccionario con media, desviacion, asimetria, curtosis, JB y ADF.
    """
    media = float(residuo_limpio.mean())
    std = float(residuo_limpio.std())
    asimetria = float(residuo_limpio.skew())
    curtosis = float(residuo_limpio.kurtosis())

    jb_stat, jb_p = jarque_bera(residuo_limpio)
    adf_stat, p_adf, _, _, _, _ = adfuller(residuo_limpio)

    return {
        "media": media,
        "std": std,
        "asimetria": asimetria,
        "curtosis": curtosis,
        "jb_stat": float(jb_stat),
        "jb_p": float(jb_p),
        "adf_stat": float(adf_stat),
        "p_adf": float(p_adf),
    }


def _graficar_acf_pacf(residuo_limpio, ruta_salida):
    """
    Descripcion:
    Genera el grafico conjunto ACF/PACF del residuo.

    Parametros:
    residuo_limpio (pd.Series): Residuo sin NaN.
    ruta_salida (str): Ruta para guardar la figura.

    Retorna:
    None: Guarda la imagen de ACF/PACF.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(residuo_limpio, lags=60, ax=axes[0])
    plot_pacf(residuo_limpio, lags=60, ax=axes[1], method="ywm")
    axes[0].set_title("ACF del residuo")
    axes[1].set_title("PACF del residuo")
    fig.tight_layout()
    guardar_figura(fig, ruta_salida)


def _graficar_histograma_con_normal(residuo_limpio, media, std, ruta_salida):
    """
    Descripcion:
    Dibuja el histograma del residuo y superpone una normal ajustada.

    Parametros:
    residuo_limpio (pd.Series): Residuo sin NaN.
    media (float): Media del residuo.
    std (float): Desviacion tipica del residuo.
    ruta_salida (str): Ruta para guardar la figura.

    Retorna:
    None: Guarda el histograma con la curva normal.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    _, bins, _ = ax.hist(
        residuo_limpio,
        bins=35,
        density=True,
        alpha=0.65,
        color="#90be6d",
        edgecolor="white",
    )

    x = np.linspace(min(bins), max(bins), 300)
    y = norm.pdf(x, loc=media, scale=std if std > 0 else 1e-12)

    ax.plot(x, y, color="crimson", linewidth=2, label="Normal ajustada")
    ax.set_title("Histograma del residuo + curva normal")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    guardar_figura(fig, ruta_salida)


def _guardar_analisis_residuo(stats, ruta_salida):
    """
    Descripcion:
    Guarda en txt los estadisticos y una lectura rapida del residuo.

    Parametros:
    stats (dict[str, float]): Estadisticos del residuo.
    ruta_salida (str): Ruta del archivo de salida.

    Retorna:
    None: Escribe el informe de analisis.
    """
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("Analisis del residuo de la serie temporal\n")
        f.write("=" * 60 + "\n")
        f.write(f"Media: {stats['media']:.6f}\n")
        f.write(f"Desviacion tipica: {stats['std']:.6f}\n")
        f.write(f"Asimetria: {stats['asimetria']:.6f}\n")
        f.write(f"Curtosis (exceso): {stats['curtosis']:.6f}\n")
        f.write(f"Jarque-Bera stat: {stats['jb_stat']:.6f}\n")
        f.write(f"Jarque-Bera p-value: {stats['jb_p']:.6f}\n")
        f.write(f"ADF stat: {stats['adf_stat']:.6f}\n")
        f.write(f"ADF p-value: {stats['p_adf']:.6f}\n")
        f.write("\nInterpretacion automatica:\n")
        f.write(
            "- Normalidad: "
            + ("no se rechaza" if stats["jb_p"] > 0.05 else "se rechaza")
            + " (alpha=0.05)\n"
        )
        f.write(
            "- Estacionariedad (ADF): "
            + ("se rechaza raiz unitaria" if stats["p_adf"] < 0.05 else "no se rechaza raiz unitaria")
            + " (alpha=0.05)\n"
        )


def analizar_residuo(residuo):
    """
    Descripcion:
    Analiza si el residuo se parece a un ruido ideal.

    Parametros:
    residuo (pd.Series): Componente residuo de la descomposicion.

    Retorna:
    None: Guarda ACF/PACF, histograma y archivo de analisis.
    """
    residuo_limpio = residuo.dropna()
    if residuo_limpio.empty:
        raise ValueError("El residuo esta vacio tras eliminar NaN.")

    stats = _calcular_estadisticas_residuo(residuo_limpio)

    _graficar_acf_pacf(residuo_limpio, "output/ej4_acf_pacf.png")
    _graficar_histograma_con_normal(
        residuo_limpio,
        stats["media"],
        stats["std"],
        "output/ej4_histograma_ruido.png",
    )
    _guardar_analisis_residuo(stats, "output/ej4_analisis.txt")


if __name__ == "__main__":
    print("=" * 55)
    print("EJERCICIO 4 - Analisis de Series Temporales")
    print("=" * 55)

    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print("\nSerie generada:")
    print(f"  Periodo:      {serie.index[0].date()} -> {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media:         {serie.mean():.2f}")
    print(f"  Std:           {serie.std():.2f}")
    print(f"  Min / Max:     {serie.min():.2f} / {serie.max():.2f}")

    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    print("[3/3] Analizando el residuo...")
    if resultado is not None:
        analizar_residuo(resultado.resid)

    print("\nSalidas esperadas en output/:")
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "OK" if existe else "PENDIENTE"
        print(f"  [{estado}] output/{s}")

    print("\nRecuerda completar las respuestas en Respuestas.md")
