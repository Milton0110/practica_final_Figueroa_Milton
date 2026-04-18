"""
Utilidades compartidas para los ejercicios de la práctica final.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def cargar_csv_con_year_numerico(dataset_path: Path) -> pd.DataFrame:
    """
    Descripcion:
    Carga un CSV y, si existe, convierte `year` a numérico.

    Parametros:
    dataset_path (Path): Ruta completa al CSV.

    Retorna:
    pd.DataFrame: DataFrame cargado y con `year` normalizado cuando aplica.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset esperado: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def guardar_figura(fig, ruta: Path, dpi: int = 150) -> None:
    """
    Descripcion:
    Guarda una figura y la cierra para liberar memoria.

    Parametros:
    fig: Figura de Matplotlib.
    ruta (Path): Ruta donde se guardará la imagen.
    dpi (int): Resolución de salida.

    Retorna:
    None: Guarda la figura en disco y cierra el objeto.
    """
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
