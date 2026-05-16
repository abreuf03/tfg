import numpy as np
from dataclasses import dataclass


@dataclass
class Malla:
    """Almacena la discretización espacial y temporal de un problema numérico."""

    a: float
    b: float
    nx: int
    t0: float
    tf: float
    nt: int
    x: np.ndarray
    t: np.ndarray
    dx: float
    dt: float


def crear_malla(a, b, nx, t0, tf, nt) -> Malla:
    """
    Crea una malla uniforme espacio-temporal.

    Parámetros:
        a: Extremo izquierdo del intervalo espacial.
        b: Extremo derecho del intervalo espacial.
        nx: Número de nodos espaciales.
        t0: Tiempo inicial.
        tf: Tiempo final.
        nt: Número de instantes temporales.

    Devuelve:
        Una instancia de Malla con los nodos espaciales, los instantes temporales
        y los pasos de discretización dx y dt.
    """

    # Comprobaciones de entrada
    if b <= a:
        raise ValueError("Se requiere b > a.")
    if tf <= t0:
        raise ValueError("Se requiere tf > t0.")
    if nx < 2:
        raise ValueError("Se requieren al menos 2 nodos espaciales.")
    if nt < 2:
        raise ValueError("Se requieren al menos 2 instantes de tiempo.")

    x = np.linspace(a, b, nx)
    t = np.linspace(t0, tf, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    return Malla(a, b, nx, t0, tf, nt, x, t, dx, dt)