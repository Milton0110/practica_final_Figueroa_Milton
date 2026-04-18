"""
=============================================================================
PRACTICA FINAL - EJERCICIO 1
Analisis Estadistico Descriptivo sobre video_games_sales.csv
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DATASET_FILENAME = "video_games_sales.csv"
TARGET_COLUMN = "global_sales"

sns.set_theme(style="whitegrid")


def cargar_dataset() -> pd.DataFrame:
    """Carga el dataset"""
    dataset_path = DATA_DIR / DATASET_FILENAME
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset esperado: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def columnas_numericas_y_categoricas(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Devuelve la separacion de columnas numericas y categoricas."""
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numericas, categoricas


def guardar_resumen_estructural(
    df: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str],
) -> None:
    """Guarda informacion estructural."""
    ruta = OUTPUT_DIR / "ej1_resumen_estructural.txt"

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    nulos = (df.isna().mean() * 100).sort_values(ascending=False)

    with ruta.open("w", encoding="utf-8") as f:
        f.write("Resumen estructural del dataset\n")
        f.write("=" * 60 + "\n")
        f.write(f"Filas: {df.shape[0]}\n")
        f.write(f"Columnas: {df.shape[1]}\n")
        f.write(f"Tamano en memoria: {memory_mb:.2f} MB\n")
        f.write(f"Variables numericas: {len(numericas)}\n")
        f.write(f"Variables categoricas: {len(categoricas)}\n\n")

        f.write("Dtypes por columna:\n")
        for col, dtype in df.dtypes.items():
            f.write(f"- {col}: {dtype}\n")

        f.write("\nPorcentaje de nulos por columna:\n")
        for col, pct in nulos.items():
            f.write(f"- {col}: {pct:.2f}%\n")


def guardar_descriptivo(df: pd.DataFrame, numericas: list[str]) -> None:
    """Genera descriptivo de variables numericas"""
    if not numericas:
        raise ValueError("No hay variables numericas para generar descriptivo")

    rows = []
    for col in numericas:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        moda = s.mode()
        rows.append(
            {
                "variable": col,
                "count": s.count(),
                "mean": s.mean(),
                "median": s.median(),
                "mode": float(moda.iloc[0]) if not moda.empty else np.nan,
                "std": s.std(ddof=1),
                "var": s.var(ddof=1),
                "min": s.min(),
                "q1": s.quantile(0.25),
                "q2": s.quantile(0.50),
                "q3": s.quantile(0.75),
                "max": s.max(),
                "iqr": s.quantile(0.75) - s.quantile(0.25),
                "skew": s.skew(),
                "kurtosis": s.kurtosis(),
            }
        )

    desc = pd.DataFrame(rows).set_index("variable").sort_index()
    desc.to_csv(OUTPUT_DIR / "ej1_descriptivo.csv", encoding="utf-8")


def graficar_histogramas(df: pd.DataFrame, numericas: list[str]) -> None:
    """Genera histogramas de variables numericas"""
    if not numericas:
        return

    n = len(numericas)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.6))
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(numericas):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i], color="#2a9d8f")
        axes[i].set_title(col)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Histogramas de variables numericas", y=1.01)
    fig.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_DIR / "ej1_histogramas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _columnas_ventas_para_log(df: pd.DataFrame, numericas: list[str]) -> list[str]:
    """Selecciona columnas de ventas para visualizaciones logaritmicas."""
    cols = [c for c in numericas if c.lower().endswith("_sales")]
    if cols:
        return cols
    # Fallback: evitar variables con escala/semantica no comparable.
    return [c for c in numericas if c not in {"rank", "year"}]


def graficar_histogramas_log1p(df: pd.DataFrame, numericas: list[str]) -> None:
    """Genera ej1_histogramas_log1p.png con transformacion log1p."""
    cols = _columnas_ventas_para_log(df, numericas)
    if not cols:
        return

    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.8, nrows * 3.6))
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(cols):
        s = pd.to_numeric(df[col], errors="coerce")
        s_log = np.log1p(s.clip(lower=0))
        sns.histplot(s_log.dropna(), kde=True, bins=30, ax=axes[i], color="#577590")
        axes[i].set_title(f"log1p({col})")
        axes[i].set_xlabel("log1p(valor)")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Histogramas con transformacion log1p", y=1.01)
    fig.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_DIR / "ej1_histogramas_log1p.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficar_histogramas_comparacion_log(df: pd.DataFrame, numericas: list[str]) -> None:
    """Genera ej1_histogramas_comparacion_log.png (original vs log1p)."""
    cols = _columnas_ventas_para_log(df, numericas)
    if not cols:
        return

    n = len(cols)
    fig, axes = plt.subplots(n, 2, figsize=(12, max(3.2 * n, 4.8)))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(cols):
        s = pd.to_numeric(df[col], errors="coerce")
        s_log = np.log1p(s.clip(lower=0))

        sns.histplot(s.dropna(), kde=True, bins=30, ax=axes[i, 0], color="#2a9d8f")
        axes[i, 0].set_title(f"{col} (original)")
        axes[i, 0].set_xlabel(col)

        sns.histplot(s_log.dropna(), kde=True, bins=30, ax=axes[i, 1], color="#577590")
        axes[i, 1].set_title(f"{col} (log1p)")
        axes[i, 1].set_xlabel(f"log1p({col})")

    fig.suptitle("Comparacion de distribuciones: original vs log1p", y=1.002)
    fig.tight_layout(pad=1.8, h_pad=1.6, w_pad=1.1)
    fig.savefig(OUTPUT_DIR / "ej1_histogramas_comparacion_log.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _top_categories(series: pd.Series, top_n: int = 12) -> pd.Series:
    """Agrupa categorias raras en 'Otros' para mejorar legibilidad."""
    s = series.astype("string").fillna("<NA>")
    top = s.value_counts().head(top_n).index
    return s.where(s.isin(top), other="Otros")


def graficar_boxplots_target_por_categorica(df: pd.DataFrame, categoricas: list[str]) -> None:
    """Genera ej1_boxplots.png (target por variables categoricas)."""
    if TARGET_COLUMN not in df.columns:
        return

    candidatas = [c for c in categoricas if c != "name"]
    if not candidatas:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No hay variables categoricas para boxplots", ha="center", va="center")
        ax.axis("off")
        fig.savefig(OUTPUT_DIR / "ej1_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(candidatas)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.2, rows * 4.8))
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(candidatas):
        local = df[[col, TARGET_COLUMN]].copy()
        local[col] = _top_categories(local[col], top_n=12)
        sns.boxplot(data=local, x=col, y=TARGET_COLUMN, ax=axes[i], color="#8ecae6")
        axes[i].set_title(f"{TARGET_COLUMN} por {col}")
        axes[i].tick_params(axis="x", rotation=35)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ej1_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficar_heatmap_correlacion(df: pd.DataFrame, numericas: list[str]) -> pd.Series:
    """Genera ej1_heatmap_correlacion.png y devuelve correlaciones con target."""
    if len(numericas) < 2:
        return pd.Series(dtype=float)

    corr = df[numericas].corr(method="pearson", numeric_only=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Matriz de correlacion de Pearson")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ej1_heatmap_correlacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if TARGET_COLUMN in corr.columns:
        return corr[TARGET_COLUMN].drop(index=TARGET_COLUMN).sort_values(key=np.abs, ascending=False)
    return pd.Series(dtype=float)


def graficar_categoricas(df: pd.DataFrame, categoricas: list[str]) -> None:
    """Genera ej1_categoricas.png con frecuencias de variables categoricas."""
    if not categoricas:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No hay variables categoricas", ha="center", va="center")
        ax.axis("off")
        fig.savefig(OUTPUT_DIR / "ej1_categoricas.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    cols_plot = [c for c in categoricas if c != "name"] or categoricas[:1]

    preferred_order = ["platform", "genre", "publisher"]
    ordered = [c for c in preferred_order if c in cols_plot] + [c for c in cols_plot if c not in preferred_order]
    cols_plot = ordered

    # Caso principal del dataset actual:
    # - platform y genre arriba
    # - publisher abajo ocupando doble ancho
    if all(c in cols_plot for c in ["platform", "genre", "publisher"]):
        fig = plt.figure(figsize=(15, 14))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.6])

        ax_platform = fig.add_subplot(gs[0, 0])
        ax_genre = fig.add_subplot(gs[0, 1])
        ax_publisher = fig.add_subplot(gs[1, :])

        for col, ax in [("platform", ax_platform), ("genre", ax_genre), ("publisher", ax_publisher)]:
            top_n = 30 if col == "publisher" else 15
            if col == "publisher":
                # Top puro sin agrupar categorias restantes en "Otros"
                freq = (
                    df[col]
                    .astype("string")
                    .fillna("<NA>")
                    .value_counts(dropna=False)
                    .head(top_n)
                )
                # Etiquetas largas: usar barras horizontales para evitar solapamiento.
                sns.barplot(y=freq.index.astype(str), x=freq.values, ax=ax, color="#f4a261")
                ax.set_title(f"Frecuencia - {col} (top {top_n})")
                ax.set_xlabel("conteo")
                ax.set_ylabel(col)
                ax.tick_params(axis="y", labelsize=9)
            else:
                s = _top_categories(df[col], top_n=top_n)
                freq = s.value_counts(dropna=False)
                sns.barplot(x=freq.index.astype(str), y=freq.values, ax=ax, color="#f4a261")
                ax.set_title(f"Frecuencia - {col} (top {top_n})")
                ax.set_xlabel(col)
                ax.set_ylabel("conteo")
                ax.tick_params(axis="x", rotation=35)
    else:
        # Fallback generico para cualquier otro dataset
        n = len(cols_plot)
        cols = 2
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.2, rows * 4.6))
        axes = np.atleast_1d(axes).ravel()

        for i, col in enumerate(cols_plot):
            top_n = 30 if col == "publisher" else 15
            if col == "publisher":
                # Top puro sin agrupar categorias restantes en "Otros"
                freq = (
                    df[col]
                    .astype("string")
                    .fillna("<NA>")
                    .value_counts(dropna=False)
                    .head(top_n)
                )
                sns.barplot(y=freq.index.astype(str), x=freq.values, ax=axes[i], color="#f4a261")
                axes[i].set_title(f"Frecuencia - {col} (top {top_n})")
                axes[i].set_xlabel("conteo")
                axes[i].set_ylabel(col)
                axes[i].tick_params(axis="y", labelsize=9)
            else:
                s = _top_categories(df[col], top_n=top_n)
                freq = s.value_counts(dropna=False)
                sns.barplot(x=freq.index.astype(str), y=freq.values, ax=axes[i], color="#f4a261")
                axes[i].set_title(f"Frecuencia - {col} (top {top_n})")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("conteo")
                axes[i].tick_params(axis="x", rotation=35)

        for j in range(n, len(axes)):
            axes[j].axis("off")

    fig.tight_layout(pad=1.4, h_pad=2.0, w_pad=1.4)
    fig.savefig(OUTPUT_DIR / "ej1_categoricas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def guardar_outliers_y_multicolinealidad(df: pd.DataFrame, numericas: list[str]) -> pd.DataFrame:
    """Detecta outliers (IQR), aplica tratamiento y analiza multicolinealidad.

    Metodo elegido: IQR (robusto ante distribuciones asimetricas y colas largas).
    Tratamiento elegido: capado/winsorizacion a [Q1-1.5*IQR, Q3+1.5*IQR].
    """
    ruta = OUTPUT_DIR / "ej1_outliers.txt"
    ruta_dataset_tratado = OUTPUT_DIR / "ej1_datos_tratados_iqr.csv"

    # Copia para tratamiento sin alterar el dataset original
    df_tratado = df.copy()

    corr = df[numericas].corr(numeric_only=True) if numericas else pd.DataFrame()
    pares_altos = []
    if not corr.empty:
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i + 1 :]:
                r = corr.loc[c1, c2]
                if abs(r) > 0.9:
                    pares_altos.append((c1, c2, r))

    # En este dataset, rank es un identificador ordinal y year es temporal:
    # se diagnostican, pero no se capan automaticamente.
    cols_solo_diagnostico = {"rank", "year"}
    total_capeados = 0

    with ruta.open("w", encoding="utf-8") as f:
        f.write("Deteccion y tratamiento de outliers (metodo IQR)\n")
        f.write("=" * 60 + "\n")
        f.write(
            "Justificacion metodo: IQR es robusto con distribuciones sesgadas y "
            "valores extremos, algo esperable en variables de ventas.\n"
        )
        f.write(
            "Tratamiento aplicado: capado (winsorizacion) a limites IQR; "
            "no elimina filas.\n\n"
        )
        for col in numericas:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            n_out = int(((s < low) | (s > high)).sum())
            pct = (n_out / len(s)) * 100
            accion = "solo diagnostico (sin capado)"

            if col not in cols_solo_diagnostico and n_out > 0:
                # Capado sobre la columna completa preservando NaN.
                col_num = pd.to_numeric(df_tratado[col], errors="coerce")
                df_tratado[col] = col_num.clip(lower=low, upper=high)
                total_capeados += n_out
                accion = f"capado IQR aplicado ({n_out} valores)"

            f.write(
                f"- {col}: outliers={n_out} ({pct:.2f}%), "
                f"limites=[{low:.4f}, {high:.4f}] -> {accion}\n"
            )

        f.write(f"\nTotal de valores capados: {total_capeados}\n")

        f.write("\nPosible multicolinealidad (|r| > 0.9):\n")
        if pares_altos:
            for c1, c2, r in sorted(pares_altos, key=lambda x: abs(x[2]), reverse=True):
                f.write(f"- {c1} vs {c2}: r={r:.4f}\n")
        else:
            f.write("- No se detectaron pares por encima de |r| > 0.9\n")

    # Exportar dataset tratado para trazabilidad y reutilizacion.
    df_tratado.to_csv(ruta_dataset_tratado, index=False, encoding="utf-8")
    return df_tratado


def guardar_top3_correlaciones(corr_target: pd.Series) -> None:
    """Guarda las 3 variables con mayor correlacion absoluta con el target."""
    ruta = OUTPUT_DIR / "ej1_top3_correlaciones.txt"
    top3 = corr_target.head(3)

    with ruta.open("w", encoding="utf-8") as f:
        f.write(f"Top-3 correlaciones (abs) con {TARGET_COLUMN}\n")
        f.write("=" * 60 + "\n")
        if top3.empty:
            f.write("No se pudieron calcular correlaciones con la variable objetivo.\n")
            return
        for col, val in top3.items():
            f.write(f"- {col}: {val:.4f}\n")


def main() -> None:
    """Ejecuta el pipeline completo del ejercicio 1."""
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_dataset()
    numericas, categoricas = columnas_numericas_y_categoricas(df)

    guardar_resumen_estructural(df, numericas, categoricas)
    guardar_descriptivo(df, numericas)
    graficar_histogramas(df, numericas)
    graficar_histogramas_log1p(df, numericas)
    graficar_histogramas_comparacion_log(df, numericas)
    graficar_boxplots_target_por_categorica(df, categoricas)
    corr_target = graficar_heatmap_correlacion(df, numericas)
    graficar_categoricas(df, categoricas)
    guardar_outliers_y_multicolinealidad(df, numericas)
    guardar_top3_correlaciones(corr_target)

    print("=" * 60)
    print("EJERCICIO 1 completado")
    print("=" * 60)
    print(f"Dataset: {DATA_DIR / DATASET_FILENAME}")
    print(f"Filas x columnas: {df.shape[0]} x {df.shape[1]}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Numericas: {len(numericas)} | Categoricas: {len(categoricas)}")


if __name__ == "__main__":
    main()
