
import math

import pytest

from src.barw.config import BARWConfig
from src.barw.punta import Punta
from src.barw.simulacion import SimulacionBARW


def crear_simulacion_geometrica():
    """Crea una simulación mínima para probar solo la geometría."""
    config = BARWConfig(
        Ra=1.0,
        modo_colision="punto_segmento",
        pasos_exclusion_propia=0,
        pasos_exclusion_madre_hija=0,
    )
    return SimulacionBARW(config=config, metodo_busqueda=0)


def test_distancia_punto_segmento_con_proyeccion_interior():
    simulacion = crear_simulacion_geometrica()

    distancia = simulacion.distancia_punto_segmento(
        px=2.0,
        py=3.0,
        x0=0.0,
        y0=0.0,
        x1=4.0,
        y1=0.0,
    )

    assert distancia == pytest.approx(3.0)


def test_distancia_punto_segmento_mas_cerca_del_extremo():
    simulacion = crear_simulacion_geometrica()

    distancia = simulacion.distancia_punto_segmento(
        px=5.0,
        py=2.0,
        x0=0.0,
        y0=0.0,
        x1=4.0,
        y1=0.0,
    )

    assert distancia == pytest.approx(math.sqrt(5.0))


def test_distancia_punto_segmento_degenerado():
    simulacion = crear_simulacion_geometrica()

    distancia = simulacion.distancia_punto_segmento(
        px=3.0,
        py=4.0,
        x0=0.0,
        y0=0.0,
        x1=0.0,
        y1=0.0,
    )

    assert distancia == pytest.approx(5.0)


def test_colision_punto_segmento_detecta_el_interior_del_segmento():
    simulacion = crear_simulacion_geometrica()

    # Segmento horizontal de x=0 a x=10.
    simulacion.conducto = [
        (0.0, 0.0, 10.0, 0.0, 0, 0),
    ]

    punta = Punta(
        x=5.0,
        y=0.0,
        theta=0.0,
        id=1,
        id_rama=1,
    )

    # Está a distancia 0.5 del interior del segmento.
    assert simulacion.hay_colision(
        punta,
        x_nueva=5.0,
        y_nueva=0.5,
        paso_actual=20,
    )


def test_colision_punto_segmento_respeta_el_radio_de_aniquilacion():
    simulacion = crear_simulacion_geometrica()

    simulacion.conducto = [
        (0.0, 0.0, 10.0, 0.0, 0, 0),
    ]

    punta = Punta(
        x=5.0,
        y=0.0,
        theta=0.0,
        id=1,
        id_rama=1,
    )

    # Justo en el radio: debe colisionar porque la condición es <= Ra.
    assert simulacion.hay_colision(
        punta,
        x_nueva=5.0,
        y_nueva=1.0,
        paso_actual=20,
    )

    # Fuera del radio: no debe colisionar.
    assert not simulacion.hay_colision(
        punta,
        x_nueva=5.0,
        y_nueva=1.01,
        paso_actual=20,
    )
