
from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW


def ejecutar_prueba(modo_colision: str) -> None:
    config = BARWConfig(
        Lx=10,
        Ly=10,
        x0=0,
        theta0=0.0,
        long_paso=1.0,
        tiempo_paso=1.0,
        tiempo_total=20,
        pb=0.0,
        ang_amplitud=0.0,
        Ra=3.0,
        pasos_exclusion_propia=6,
        pasos_exclusion_aniquilacion=10,
        modo_colision=modo_colision,
        semilla=1,
    )

    simulacion = SimulacionBARW(
        config=config,
        metodo_busqueda=0,
    )
    resultado = simulacion.ejecutar()

    historial = resultado["historial"]

    print(f"\nModo: {modo_colision}")
    print(f"Segmentos: {len(resultado['conducto'])}")
    print(f"Bifurcaciones: {simulacion.contador_bifurcaciones}")
    print(f"Colisiones: {simulacion.contador_colisiones}")
    print(f"Salidas: {simulacion.contador_salidas}")
    print(f"x_max final: {historial['x_max'][-1]}")
    print(f"Puntas activas finales: {historial['num_puntas_activas'][-1]}")


if __name__ == "__main__":
    ejecutar_prueba("punto_punto")
    ejecutar_prueba("punto_segmento")
