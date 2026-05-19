from dataclasses import replace
import pandas as pd

from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW
from src.barw.graficas import graficar_comparacion_semillas


def ejecutar_semillas(semillas, config_base, usar_kdtree=True):
    resultados = []

    for semilla in semillas:
        config = replace(config_base, semilla=semilla)

        simulacion = SimulacionBARW(config=config, usar_kdtree=usar_kdtree)
        resultado = simulacion.ejecutar()

        historial = resultado["historial"]
        conducto = resultado["conducto"]

        if conducto:
            x_max = max(max(seg[0], seg[2]) for seg in conducto)
        else:
            x_max = config.x0

        resultados.append({
            "semilla": semilla,
            "segmentos": len(conducto),
            "bifurcaciones": historial["num_bifurcaciones"][-1],
            "terminaciones": historial["num_terminaciones"][-1],
            "puntas_activas_finales": historial["num_puntas_activas"][-1],
            "tiempo_final": historial["tiempo"][-1],
            "x_max": x_max
        })

    return pd.DataFrame(resultados)


def main():
    semillas = [1, 2, 3, 4, 5, 10, 42, 100]

    config_base = BARWConfig()

    df = ejecutar_semillas(
        semillas=semillas,
        config_base=config_base,
        usar_kdtree=True
    )

    print(df)
    print()
    print("Resumen:")
    print(df.describe())

    df.to_csv("resultados/comparacion_semillas.csv", index=False)

    graficar_comparacion_semillas(
        df,
        guardar="resultados/comparacion_semillas_xmax.png"
    )


if __name__ == "__main__":
    main()