
import pytest

from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW


def test_bifurcacion_crea_una_hija_y_mantiene_activa_a_la_madre():
    config = BARWConfig(
        Lx=20.0,
        Ly=20.0,
        long_paso=1.0,
        pb=1.0,
        Ra=0.1,
        ang_amplitud=0.0,
        angulo_bifurcacion=0.5,
        semilla=13,
    )
    simulacion = SimulacionBARW(config=config, metodo_busqueda=0)
    simulacion.inicializar()

    madre = simulacion.puntas[0]
    simulacion.paso()

    assert simulacion.contador_bifurcaciones == 1
    assert len(simulacion.conducto) == 1
    assert len(simulacion.puntas) == 2
    assert madre.activa
    assert madre.edad == 1

    hija = simulacion.puntas[1]
    assert hija.activa
    assert hija.id_padre == madre.id
    assert hija.id_rama_padre == madre.id_rama
    assert hija.generacion == madre.generacion + 1
    assert hija.edad == 0
    assert hija.x == pytest.approx(madre.x)
    assert hija.y == pytest.approx(madre.y)
    assert abs(hija.theta - madre.theta) == pytest.approx(
        config.angulo_bifurcacion
    )
