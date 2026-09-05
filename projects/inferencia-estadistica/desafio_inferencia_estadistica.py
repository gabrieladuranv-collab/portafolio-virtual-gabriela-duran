# -*- coding: utf-8 -*-
"""
Desafío: Inferencia estadística
Autora: Gabriela Durán Vidal

Este programa:
1. Simula un DataFrame con 200 estudiantes.
2. Realiza análisis descriptivo y tratamiento de valores nulos.
3. Grafica la distribución de los puntajes de satisfacción.
4. Calcula un intervalo de confianza del 95%.
5. Realiza una prueba t de una muestra para contrastar si la media es igual a 7.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def crear_datos(cantidad: int = 200, semilla: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    edades = np.clip(np.rint(rng.normal(loc=29, scale=8, size=cantidad)), 18, 60).astype(float)
    generos = rng.choice(["Mujer", "Hombre", "No binario", "Prefiere no responder"], size=cantidad, p=[0.48, 0.44, 0.05, 0.03]).astype(object)
    puntajes_satisfaccion = np.clip(rng.normal(loc=7.25, scale=1.25, size=cantidad), 1, 10).round(2)
    horas_estudio = np.clip(rng.normal(loc=8.5, scale=3.0, size=cantidad), 1, 20).round(1)
    df = pd.DataFrame({"Edad": edades, "Genero": generos, "Puntaje_satisfaccion": puntajes_satisfaccion, "Horas_estudio_semanales": horas_estudio})
    df.loc[rng.choice(df.index, size=3, replace=False), "Edad"] = np.nan
    df.loc[rng.choice(df.index, size=2, replace=False), "Genero"] = np.nan
    df.loc[rng.choice(df.index, size=4, replace=False), "Horas_estudio_semanales"] = np.nan
    return df


def mostrar_titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def explorar_y_tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    mostrar_titulo("1. CARGA Y EXPLORACIÓN DE DATOS")
    print(f"Cantidad de registros: {df.shape[0]}")
    print(f"Cantidad de variables: {df.shape[1]}")
    print("\nPrimeros cinco registros:")
    print(df.head().to_string(index=False))
    columnas_numericas = ["Edad", "Puntaje_satisfaccion", "Horas_estudio_semanales"]
    estadisticas = df[columnas_numericas].agg(["mean", "median", "std"]).T.rename(columns={"mean": "Media", "median": "Mediana", "std": "Desviacion_estandar"})
    print("\nEstadísticas descriptivas de las variables numéricas:")
    print(estadisticas.round(2).to_string())
    print("\nValores nulos detectados por variable:")
    print(df.isna().sum().to_string())
    print("\nTratamiento aplicado:\n- Edad y horas de estudio: mediana.\n- Género: moda.\n- Puntaje de satisfacción: no presenta nulos.")
    df_limpio = df.copy()
    for columna in ["Edad", "Horas_estudio_semanales"]:
        df_limpio[columna] = df_limpio[columna].fillna(df_limpio[columna].median())
    df_limpio["Genero"] = df_limpio["Genero"].fillna(df_limpio["Genero"].mode(dropna=True)[0])
    return df_limpio


def analizar_distribucion(df: pd.DataFrame, carpeta_salida: Path) -> None:
    mostrar_titulo("2. DISTRIBUCIÓN Y VISUALIZACIÓN")
    puntajes = df["Puntaje_satisfaccion"].to_numpy()
    media_numpy = np.mean(puntajes)
    varianza_numpy = np.var(puntajes, ddof=1)
    asimetria = stats.skew(puntajes, bias=False)
    estadistico_shapiro, p_shapiro = stats.shapiro(puntajes)
    print(f"Media: {media_numpy:.3f}")
    print(f"Varianza muestral: {varianza_numpy:.3f}")
    print(f"Asimetría: {asimetria:.3f}")
    print(f"Shapiro-Wilk: W = {estadistico_shapiro:.4f}, p = {p_shapiro:.4f}")
    plt.figure(figsize=(9, 5))
    plt.hist(puntajes, bins=12, edgecolor="black")
    plt.axvline(media_numpy, linestyle="--", linewidth=2, label=f"Media = {media_numpy:.2f}")
    plt.title("Distribución de los puntajes de satisfacción")
    plt.xlabel("Puntaje de satisfacción")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    ruta_histograma = carpeta_salida / "histograma_satisfaccion_gabriela_duran.png"
    plt.savefig(ruta_histograma, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Histograma guardado en: {ruta_histograma}")


def calcular_intervalo_confianza(df: pd.DataFrame) -> tuple[float, float]:
    mostrar_titulo("3. INTERVALO DE CONFIANZA DEL 95%")
    puntajes = df["Puntaje_satisfaccion"].dropna().to_numpy()
    n = len(puntajes)
    media = np.mean(puntajes)
    desviacion = np.std(puntajes, ddof=1)
    error_estandar = desviacion / np.sqrt(n)
    valor_t_critico = stats.t.ppf(0.975, df=n - 1)
    margen_error = valor_t_critico * error_estandar
    inferior, superior = media - margen_error, media + margen_error
    print(f"Intervalo de confianza del 95%: [{inferior:.3f}, {superior:.3f}]")
    return inferior, superior


def realizar_prueba_hipotesis(df: pd.DataFrame) -> None:
    mostrar_titulo("4. PRUEBA DE HIPÓTESIS")
    puntajes = df["Puntaje_satisfaccion"].dropna().to_numpy()
    estadistico_t, valor_p = stats.ttest_1samp(puntajes, popmean=7)
    print("H0: media poblacional = 7")
    print("H1: media poblacional ≠ 7")
    print(f"t = {estadistico_t:.4f}")
    print(f"p = {valor_p:.6f}")
    if valor_p < 0.05:
        print("Se rechaza H0 al nivel de significancia de 0,05.")
    else:
        print("No se rechaza H0 al nivel de significancia de 0,05.")


def main() -> None:
    carpeta_salida = Path(__file__).resolve().parent
    datos = crear_datos()
    datos_limpios = explorar_y_tratar_nulos(datos)
    analizar_distribucion(datos_limpios, carpeta_salida)
    calcular_intervalo_confianza(datos_limpios)
    realizar_prueba_hipotesis(datos_limpios)


if __name__ == "__main__":
    main()
