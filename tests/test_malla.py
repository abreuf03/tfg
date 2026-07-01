
import numpy as np
import pytest

from src.malla import crear_malla


def test_crear_malla_generar_nodos_y_pasos_correctos():
    malla = crear_malla(a=0.0, b=2.0, nx=5, t0=1.0, tf=3.0, nt=9)

    assert malla.nx == 5
    assert malla.nt == 9

    np.testing.assert_allclose(
        malla.x,
        [0.0, 0.5, 1.0, 1.5, 2.0],
    )
    np.testing.assert_allclose(
        malla.t,
        np.linspace(1.0, 3.0, 9),
    )

    assert malla.dx == pytest.approx(0.5)
    assert malla.dt == pytest.approx(0.25)


@pytest.mark.parametrize( #para ejecutar el mismo test con distintos parametros
    ("parametros", "mensaje"),
    [
        ((1.0, 1.0, 5, 0.0, 1.0, 5), "b > a"),
        ((0.0, 1.0, 1, 0.0, 1.0, 5), "al menos 2 nodos espaciales"),
        ((0.0, 1.0, 5, 1.0, 1.0, 5), "tf > t0"),
        ((0.0, 1.0, 5, 0.0, 1.0, 1), "al menos 2 instantes de tiempo"),
    ],
)

def test_crear_malla_rechaza_parametros_invalidos(parametros, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        crear_malla(*parametros)
