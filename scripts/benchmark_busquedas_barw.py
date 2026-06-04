from dataclasses import replace
from math import sqrt
import time
import os
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy.stats import t

from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW
from src.barw.graficas import (
    graficar_tiempo_busquedas,
    graficar_tiempo_vs_segmentos
)


def ejecutar_simulacion(config, usar_kdtree):
    """
    Ejecuta una simulación BARW y mide el tiempo de ejecución.
    """

    simulacion = SimulacionBARW(
        config=config,
        usar_kdtree=usar_kdtree
    )

    inicio = time.perf_counter()

    # Evita que los print de ejecutar() contaminen la salida del benchmark
    with redirect_stdout(io.StringIO()):
        resultado = simulacion.ejecutar()

    fin = time.perf_counter()

    tiempo_ejecucion = fin - inicio

    historial = resultado["historial"]
    conducto = resultado["conducto"]

    if conducto:
        x_max = max(max(seg[0], seg[2]) for seg in conducto)
    else:
        x_max = config.x0

    return {
        "tiempo_ejecucion": tiempo_ejecucion,
        "segmentos": len(conducto),
        "bifurcaciones": historial["num_bifurcaciones"][-1],
        "terminaciones": historial["num_terminaciones"][-1],
        "puntas_activas_finales": historial["num_puntas_activas"][-1],
        "tiempo_final": historial["tiempo"][-1],
        "x_max": x_max,
    }


def ejecutar_benchmark(num_semillas=50):
    """
    Compara búsqueda exhaustiva y búsqueda mediante cKDTree
    para varias semillas aleatorias.
    """

    config_base = BARWConfig()
    resultados = []

    for semilla in range(1, num_semillas + 1):
        print(f"Ejecutando semilla {semilla}/{num_semillas}...")

        for usar_kdtree, metodo in [(False, "exhaustiva"), (True, "cKDTree")]:
            config = replace(config_base, semilla=semilla)

            metricas = ejecutar_simulacion(
                config=config,
                usar_kdtree=usar_kdtree
            )

            metricas["semilla"] = semilla
            metricas["metodo"] = metodo

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

    return media, desviacion, media - error, media + error


def resumen_por_metodo(df):
    """
    Calcula resumen estadístico del tiempo de ejecución para cada método.
    """

    filas = []

    for metodo, grupo in df.groupby("metodo"):
        media, desviacion, ic_inf, ic_sup = intervalo_confianza_95(
            grupo["tiempo_ejecucion"]
        )

        filas.append({
            "metodo": metodo,
            "tiempo_medio": media,
            "desviacion": desviacion,
            "ic95_inf": ic_inf,
            "ic95_sup": ic_sup,
            "segmentos_medios": grupo["segmentos"].mean(),
            "x_max_medio": grupo["x_max"].mean(),
            "num_simulaciones": len(grupo)
        })

    return pd.DataFrame(filas)


def comprobar_equivalencia(df):
    """
    Comprueba si ambas búsquedas generan las mismas métricas para cada semilla.
    """

    diferencias = []

    for semilla, grupo in df.groupby("semilla"):
        if len(grupo) != 2:
            continue

        fila_exh = grupo[grupo["metodo"] == "exhaustiva"].iloc[0]
        fila_kdt = grupo[grupo["metodo"] == "cKDTree"].iloc[0]

        diferencias.append({
            "semilla": semilla,
            "dif_segmentos": fila_kdt["segmentos"] - fila_exh["segmentos"],
            "dif_bifurcaciones": fila_kdt["bifurcaciones"] - fila_exh["bifurcaciones"],
            "dif_terminaciones": fila_kdt["terminaciones"] - fila_exh["terminaciones"],
            "dif_x_max": fila_kdt["x_max"] - fila_exh["x_max"],
        })

    return pd.DataFrame(diferencias)


def main():
    os.makedirs("resultados", exist_ok=True)

    num_semillas = 50

    df = ejecutar_benchmark(num_semillas=num_semillas)

    resumen = resumen_por_metodo(df)
    diferencias = comprobar_equivalencia(df)

    print("\nResultados completos:")
    print(df)

    print("\nResumen por método:")
    print(resumen)

    print("\nDiferencias entre métodos por semilla:")
    print(diferencias)

    print("\nMáximas diferencias absolutas:")
    print(diferencias.drop(columns=["semilla"]).abs().max())

    df.to_csv("resultados/benchmark_busquedas_barw.csv", index=False)
    resumen.to_csv("resultados/resumen_benchmark_busquedas_barw.csv", index=False)
    diferencias.to_csv("resultados/diferencias_busquedas_barw.csv", index=False)

    graficar_tiempo_busquedas(
        resumen,
        guardar="resultados/benchmark_tiempo_busquedas.png"
    )

    graficar_tiempo_vs_segmentos(
        df,
        guardar="resultados/benchmark_tiempo_vs_segmentos.png"
    )


if __name__ == "__main__":
    main()