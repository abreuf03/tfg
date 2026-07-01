
import numpy as np

from src.campo_medio.config import CampoMedioConfig
from src.campo_medio.modelo import terminos_reaccion
from src.campo_medio.solvers import resolver_imex_cn
from src.malla import crear_malla


def test_terminos_reaccion_del_campo_medio_coinciden_con_la_formula():
    config = CampoMedioConfig(D=1.0, rb=0.1, re=1.0, n0=2.0)
    a = np.array([0.5, 1.0])
    i = np.array([0.5, 0.0])

    reaccion_a, reaccion_i = terminos_reaccion(a, i, config)

    esperado_a = config.rb * a * (1.0 - (a + i) / config.n0)
    esperado_i = config.re * a + (config.rb / config.n0) * a * (a + i)

    np.testing.assert_allclose(reaccion_a, esperado_a)
    np.testing.assert_allclose(reaccion_i, esperado_i)


def test_solver_imex_cn_preserva_fronteras_y_genera_densidad_inactiva():
    malla = crear_malla(0.0, 1.0, 11, 0.0, 0.02, 21)
    config = CampoMedioConfig(
        D=1.0,
        rb=0.1,
        re=1.0,
        n0=1.0,
        a_in=0.2,
        a_out=0.0,
        a0=0.05,
        i0=0.0,
    )

    A, I = resolver_imex_cn(malla, config)

    assert A.shape == (malla.nt, malla.nx)
    assert I.shape == (malla.nt, malla.nx)
    assert np.isfinite(A).all()
    assert np.isfinite(I).all()

    np.testing.assert_allclose(A[:, 0], config.a_in)
    np.testing.assert_allclose(A[:, -1], config.a_out)

    assert np.min(A) >= -1e-12
    assert np.min(I) >= -1e-12
    assert np.max(I[-1, :]) > 0.0

    # La variable inactiva no debe decrecer.
    assert np.all(np.diff(I, axis=0) >= -1e-12)
