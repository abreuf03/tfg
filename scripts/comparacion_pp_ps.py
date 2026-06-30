
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW


def extraer_metricas(
    resultado: dict,
    simulacion: SimulacionBARW,
    semilla: int,
    modo_colision: str,
) -> dict:
    """Extrae las métricas finales de una ejecución."""

    historial = resultado["historial"]

    x_max = historial["x_max"][-1] if historial["x_max"] else 0.0
    tiempo_final = historial["tiempo"][-1] if historial["tiempo"] else 0.0

    puntas_activas_finales = (
        historial["num_puntas_activas"][-1]
        if historial["num_puntas_activas"]
        else 0
    )

    max_puntas_activas = (
        max(historial["num_puntas_activas"])
        if historial["num_puntas_activas"]
        else 0
    )

    return {
        "semilla": semilla,
        "modo_colision": modo_colision,
        "x_max": x_max,
        "alcanza_90pct": int(x_max >= 0.90 * simulacion.config.Lx),
        "segmentos": len(resultado["conducto"]),
        "bifurcaciones": simulacion.contador_bifurcaciones,
        "colisiones": simulacion.contador_colisiones,
        "salidas_frontera": simulacion.contador_salidas,
        "tiempo_final": tiempo_final,
        "max_puntas_activas": max_puntas_activas,
        "puntas_activas_finales": puntas_activas_finales,
        "extinta": int(puntas_activas_finales == 0),
    }


def ejecutar_comparacion(
    semillas: list[int],
    config_base: BARWConfig,
    metodo_busqueda: int = 0,
) -> pd.DataFrame:
    """
    Ejecuta, para cada semilla, una simulación punto-punto y otra
    punto-segmento con idénticos parámetros.
    """

    resultados = []

    for semilla in semillas:
        print(f"\nSemilla {semilla}")

        for modo in ("punto_punto", "punto_segmento"):
            config = replace(
                config_base,
                semilla=semilla,
                modo_colision=modo,
            )

            simulacion = SimulacionBARW(
                config=config,
                metodo_busqueda=metodo_busqueda,
            )

            resultado = simulacion.ejecutar()

            resultados.append(
                extraer_metricas(
                    resultado=resultado,
                    simulacion=simulacion,
                    semilla=semilla,
                    modo_colision=modo,
                )
            )

    return pd.DataFrame(resultados)


def construir_tabla_pareada(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una fila por semilla con las métricas de ambos modos y las
    diferencias: punto_segmento - punto_punto.
    """

    metricas = [
        "x_max",
        "alcanza_90pct",
        "segmentos",
        "bifurcaciones",
        "colisiones",
        "salidas_frontera",
        "tiempo_final",
        "max_puntas_activas",
        "puntas_activas_finales",
        "extinta",
    ]

    pareadas = datos.pivot(
        index="semilla",
        columns="modo_colision",
        values=metricas,
    )

    pareadas.columns = [
        f"{metrica}_{modo}"
        for metrica, modo in pareadas.columns
    ]

    pareadas = pareadas.reset_index()

    for metrica in metricas:
        pareadas[f"delta_{metrica}"] = (
            pareadas[f"{metrica}_punto_segmento"]
            - pareadas[f"{metrica}_punto_punto"]
        )

    return pareadas


def crear_resumen(
    datos: pd.DataFrame,
    pareadas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resume los resultados por modo y las diferencias emparejadas."""

    resumen_modos = (
        datos.groupby("modo_colision")
        .agg(
            simulaciones=("semilla", "count"),
            proporcion_alcanza_90pct=("alcanza_90pct", "mean"),
            x_max_medio=("x_max", "mean"),
            x_max_mediana=("x_max", "median"),
            segmentos_medios=("segmentos", "mean"),
            bifurcaciones_medias=("bifurcaciones", "mean"),
            colisiones_medias=("colisiones", "mean"),
            salidas_medias=("salidas_frontera", "mean"),
            tiempo_final_medio=("tiempo_final", "mean"),
            max_puntas_activas_medio=("max_puntas_activas", "mean"),
        )
        .reset_index()
    )

    columnas_delta = [
        columna
        for columna in pareadas.columns
        if columna.startswith("delta_")
    ]

    resumen_deltas = pd.DataFrame(
        {
            "media": pareadas[columnas_delta].mean(),
            "mediana": pareadas[columnas_delta].median(),
            "desviacion_tipica": pareadas[columnas_delta].std(),
            "minimo": pareadas[columnas_delta].min(),
            "maximo": pareadas[columnas_delta].max(),
        }
    )

    resumen_deltas.index.name = "metrica"
    resumen_deltas = resumen_deltas.reset_index()

    return resumen_modos, resumen_deltas


def main() -> None:
    semillas = list(range(1, 31))

    # Esta es la configuración de referencia.
    # Ambos modos comparten exactamente estos parámetros.
    config_base = BARWConfig(
        pasos_exclusion_propia=6,
        pasos_exclusion_madre_hija=10,
    )

    # Se utiliza búsqueda exhaustiva en ambos modos para no mezclar
    # el efecto geométrico con el rendimiento de KDTree o QuadTree.
    datos = ejecutar_comparacion(
        semillas=semillas,
        config_base=config_base,
        metodo_busqueda=0,
    )

    pareadas = construir_tabla_pareada(datos)

    resumen_modos, resumen_deltas = crear_resumen(
        datos=datos,
        pareadas=pareadas,
    )

    directorio_salida = Path(
        "resultados/comparacion_punto_punto_segmento"
    )
    directorio_salida.mkdir(parents=True, exist_ok=True)

    datos.to_csv(
        directorio_salida / "ejecuciones_individuales.csv",
        index=False,
    )

    pareadas.to_csv(
        directorio_salida / "resultados_pareados_por_semilla.csv",
        index=False,
    )

    resumen_modos.to_csv(
        directorio_salida / "resumen_por_modo.csv",
        index=False,
    )

    resumen_deltas.to_csv(
        directorio_salida / "resumen_diferencias_pareadas.csv",
        index=False,
    )

    print("\nResumen por modo de colisión:")
    print(resumen_modos.to_string(index=False))

    print("\nDiferencias pareadas: punto_segmento - punto_punto")
    print(resumen_deltas.to_string(index=False))

    print(
        "\nArchivos guardados en:",
        directorio_salida,
    )


if __name__ == "__main__":
    main()
