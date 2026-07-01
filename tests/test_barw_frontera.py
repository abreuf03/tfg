
import pytest

from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW


def test_salida_por_frontera_desactiva_la_punta_sin_depositar_segmento():
    config = BARWConfig(
        Lx=0.5,
        Ly=10.0,
        long_paso=1.0,
        pb=0.0,
        ang_amplitud=0.0,
        semilla=7,
    )
    simulacion = SimulacionBARW(config=config, metodo_busqueda=0)
    simulacion.inicializar()

    punta_inicial = simulacion.puntas[0]
    simulacion.paso()

    assert not punta_inicial.activa
    assert punta_inicial.x == pytest.approx(config.x0)
    assert punta_inicial.y == pytest.approx(config.Ly / 2.0)
    assert simulacion.contador_salidas == 1
    assert simulacion.contador_colisiones == 0
    assert simulacion.contador_terminaciones == 1
    assert simulacion.conducto == []
