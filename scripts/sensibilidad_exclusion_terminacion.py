
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW


def ejecutar_sensibilidad_exclusion(
    semillas: list[int],
    valores_gracia: list[int],
    config_base: BARWConfig,
    metodo_busqueda: int = 2,
) -> pd.DataFrame:
    """
    Evalúa la sensibilidad de la simulación respecto al número de pasos
    de gracia aplicado entre ramas madre e hija tras una bifurcación.

    Las mismas semillas se reutilizan para cada valor de gracia.
    """

    resultados = []
    umbral_avance = 0.90 * config_base.Lx

    for gracia in valores_gracia:
        for semilla in semillas:
            config = replace(
                config_base,
                semilla=semilla,
                pasos_exclusion_aniquilacion=gracia,
            )

            simulacion = SimulacionBARW(
                config=config,
                metodo_busqueda=metodo_busqueda,
            )

            resultado = simulacion.ejecutar()
            historial = resultado["historial"]
            puntas = resultado["puntas"]

            if historial["tiempo"]:
                tiempo_final = historial["tiempo"][-1]
                x_max = historial["x_max"][-1]
            else:
                tiempo_final = 0.0
                x_max = config.x0

            puntas_activas_finales = sum(
                punta.activa for punta in puntas
            )

            resultados.append(
                {
                    "gracia_madre_hija": gracia,
                    "semilla": semilla,
                    "segmentos": len(resultado["conducto"]),
                    "bifurcaciones": simulacion.contador_bifurcaciones,
                    "colisiones": simulacion.contador_colisiones,
                    "salidas_frontera": simulacion.contador_salidas,
                    "terminaciones_totales": simulacion.contador_terminaciones,
                    "puntas_activas_finales": puntas_activas_finales,
                    "extinta": puntas_activas_finales == 0,
                    "tiempo_final": tiempo_final,
                    "x_max": x_max,
                    "alcanza_90pct": x_max >= umbral_avance,
                }
            )

    return pd.DataFrame(resultados)


def main():
    semillas = list(range(1, 31))
    valores_gracia = [16, 18, 20, 22, 24]

    config_base = BARWConfig(
        pasos_exclusion_propia=6,
    )

    df = ejecutar_sensibilidad_exclusion(
        semillas=semillas,
        valores_gracia=valores_gracia,
        config_base=config_base,
        metodo_busqueda=2,  # 0: exhaustiva, 1: KDTree, 2: QuadTree
    )

    resumen = (
        df.groupby("gracia_madre_hija")
        .agg(
            simulaciones=("semilla", "size"),
            proporcion_alcanza_90pct=("alcanza_90pct", "mean"),
            x_max_medio=("x_max", "mean"),
            x_max_mediana=("x_max", "median"),
            x_max_desv=("x_max", "std"),
            segmentos_medios=("segmentos", "mean"),
            bifurcaciones_medias=("bifurcaciones", "mean"),
            colisiones_medias=("colisiones", "mean"),
            salidas_medias=("salidas_frontera", "mean"),
            tiempo_final_medio=("tiempo_final", "mean"),
            tiempo_final_mediano=("tiempo_final", "median"),
        )
        .reset_index()
    )

    carpeta = Path("resultados") / "sensibilidad_exclusion_terminacion"
    carpeta.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        carpeta / "resultados_por_semilla.csv",
        index=False,
    )

    resumen.to_csv(
        carpeta / "resumen.csv",
        index=False,
    )

    print("\nResumen por valor de gracia:")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
