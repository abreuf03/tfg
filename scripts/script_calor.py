import numpy as np

from src.malla import crear_malla
from src.calor import resolver_calor_crank_nicolson


def test_crank_nicolson_calor_error_maximo():
    D = 1.0
    Tf = 0.1

    malla = crear_malla(0.0, 1.0, 21, 0.0, Tf, 161)

    u0 = np.sin(np.pi * malla.x)
    U = resolver_calor_crank_nicolson(malla, D, u0)

    u_exacta = np.exp(-D * np.pi**2 * Tf) * np.sin(np.pi * malla.x)
    error = np.max(np.abs(U[-1, :] - u_exacta))

    assert error < 1e-3