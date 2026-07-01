
import pytest

from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW


def ejecutar_simulacion_corta(metodo_busqueda: int) -> dict:
    """Ejecuta una simulación pequeña y devuelve sus métricas finales."""
    config = BARWConfig(
        Lx=80.0,
        Ly=80.0,
        pb=0.15,
        Ra=3.0,
        tiempo_total=30.0,
        semilla=2026,
        max_puntas=500,
        max_pasos=100,
        pasos_exclusion_propia=6,
        pasos_exclusion_madre_hija=10,
        modo_colision="punto_punto",
    )

    resultado = SimulacionBARW(
        config=config,
        metodo_busqueda=metodo_busqueda,
    ).ejecutar()

    historial = resultado["historial"]

    return {
        "motivo_parada": resultado["motivo_parada"],
        "pasos_ejecutados": resultado["pasos_ejecutados"],
        "segmentos": len(resultado["conducto"]),
        "puntas_totales": len(resultado["puntas"]),
        "bifurcaciones": historial["num_bifurcaciones"][-1],
        "terminaciones": historial["num_terminaciones"][-1],
        "colisiones": historial["num_colisiones"][-1],
        "salidas_frontera": historial["num_salidas_frontera"][-1],
        "puntas_activas_finales": historial["num_puntas_activas"][-1],
        "x_max": historial["x_max"][-1],
    }


def test_estrategias_de_busqueda_producen_las_mismas_metricas():
    referencia = ejecutar_simulacion_corta(metodo_busqueda=0)
    resultado_kdtree = ejecutar_simulacion_corta(metodo_busqueda=1)
    resultado_quadtree = ejecutar_simulacion_corta(metodo_busqueda=2)

    assert resultado_kdtree == referencia
    assert resultado_quadtree == referencia


def test_metodo_de_busqueda_invalido_lanza_error():
    config = BARWConfig()

    with pytest.raises(ValueError, match="metodo_busqueda debe ser"):
        SimulacionBARW(config=config, metodo_busqueda=99)
