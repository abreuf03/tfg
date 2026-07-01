
import numpy as np

from src.alg_thomas import thomas


def construir_matriz_tridiagonal(a, b, c):
    """Construye la matriz completa solo para verificar el resultado."""
    return (
        np.diag(b)
        + np.diag(a, k=-1)
        + np.diag(c, k=1)
    )


def test_thomas_coincide_con_numpy_linalg_solve():
    # Diagonal inferior, principal y superior.
    a = np.array([-1.0, -1.0, -1.0])
    b = np.array([4.0, 4.0, 4.0, 4.0])
    c = np.array([-1.0, -1.0, -1.0])

    matriz = construir_matriz_tridiagonal(a, b, c)

    solucion_esperada = np.array([1.0, 2.0, -1.0, 3.0])
    d = matriz @ solucion_esperada

    solucion_thomas = thomas(a, b, c, d)
    solucion_numpy = np.linalg.solve(matriz, d)

    np.testing.assert_allclose(solucion_thomas, solucion_esperada)
    np.testing.assert_allclose(solucion_thomas, solucion_numpy)


def test_thomas_no_modifica_los_vectores_de_entrada():
    a = np.array([-1.0, -1.0])
    b = np.array([3.0, 4.0, 3.0])
    c = np.array([-1.0, -1.0])
    d = np.array([1.0, 2.0, 3.0])

    a_original = a.copy()
    b_original = b.copy()
    c_original = c.copy()
    d_original = d.copy()

    thomas(a, b, c, d)

    np.testing.assert_array_equal(a, a_original)
    np.testing.assert_array_equal(b, b_original)
    np.testing.assert_array_equal(c, c_original)
    np.testing.assert_array_equal(d, d_original)
