from dataclasses import replace
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import t

from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW
from src.barw.graficas import (
    graficar_boxplot_semillas,
    graficar_balance_bifurcacion_terminacion,
)


def ejecutar_una_semilla(config_base, semilla, usar_kdtree=True):
    """
    Ejecuta una simulación BARW para una semilla dada y devuelve sus métricas.
    """

    config = replace(config_base, semilla=semilla)

    simulacion = SimulacionBARW(config=config, usar_kdtree=usar_kdtree)
    resultado = simulacion.ejecutar()

    historial = resultado["historial"]
    conducto = resultado["conducto"]

    if conducto:
        x_max = max(max(seg[0], seg[2]) for seg in conducto)
    else:
        x_max = config.x0

    bifurcaciones = historial["num_bifurcaciones"][-1]
    terminaciones = historial["num_terminaciones"][-1]
    puntas_finales = historial["num_puntas_activas"][-1]

    return {
        "semilla": semilla,
        "segmentos": len(conducto),
        "bifurcaciones": bifurcaciones,
        "terminaciones": terminaciones,
        "puntas_activas_finales": puntas_finales,
        "tiempo_final": historial["tiempo"][-1],
        "x_max": x_max,
        "balance_term_bif": terminaciones - bifurcaciones,
    }


def ejecutar_muchas_semillas(num_semillas=100, usar_kdtree=True):
    """
    Ejecuta el modelo para muchas semillas y devuelve una tabla con métricas.
    """

    config_base = BARWConfig()

    resultados = []

    for semilla in range(1, num_semillas + 1):
        print(f"Ejecutando semilla {semilla}/{num_semillas}...")

        metricas = ejecutar_una_semilla(
            config_base=config_base,
            semilla=semilla,
            usar_kdtree=usar_kdtree,
        )

        resultados.append(metricas)

    return pd.DataFrame(resultados)


def intervalo_confianza_95(serie):
    """
    Calcula media, desviación típica e intervalo de confianza al 95%.
    """

    datos = np.asarray(serie, dtype=float)
    n = len(datos)

    media = np.mean(datos)
    desviacion = np.std(datos, ddof=1)

    if n > 1:
        error = t.ppf(0.975, df=n - 1) * desviacion / sqrt(n)
    else:
        error = 0.0

    return {
        "media": media,
        "desviacion": desviacion,
        "ic95_inf": media - error,
        "ic95_sup": media + error,
    }


def resumen_estadistico(df):
    """
    Calcula resumen estadístico para las métricas principales.
    """

    metricas = [
        "segmentos",
        "bifurcaciones",
        "terminaciones",
        "tiempo_final",
        "x_max",
        "balance_term_bif",
    ]

    filas = []

    for metrica in metricas:
        resumen = intervalo_confianza_95(df[metrica])
        resumen["metrica"] = metrica
        filas.append(resumen)

    columnas = ["metrica", "media", "desviacion", "ic95_inf", "ic95_sup"]

    return pd.DataFrame(filas)[columnas]


def main():
    num_semillas = 100

    df = ejecutar_muchas_semillas(
        num_semillas=num_semillas,
        usar_kdtree=True,
    )

    resumen = resumen_estadistico(df)

    print("\nResultados por semilla:")
    print(df)

    print("\nResumen estadístico:")
    print(resumen)

    df.to_csv("resultados/validacion_semillas.csv", index=False)
    resumen.to_csv("resultados/resumen_validacion_semillas.csv", index=False)

    graficar_boxplot_semillas(
        df,
        guardar="resultados/boxplot_validacion_semillas.png",
    )

    graficar_balance_bifurcacion_terminacion(
        df,
        guardar="resultados/balance_bifurcacion_terminacion.png",
    )


if __name__ == "__main__":
    main()