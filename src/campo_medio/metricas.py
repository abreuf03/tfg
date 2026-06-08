
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from src.campo_medio.config import CampoMedioConfig
from src.malla import Malla


# -----------------------------------------------------------------------------
# Métricas generales para la validación numérica del Capítulo 2
# -----------------------------------------------------------------------------


def norma_l2_discreta(valores: np.ndarray, dx: float) -> float:
    """Norma L2 discreta: sqrt(dx * sum_j |u_j|^2)."""
    valores = np.asarray(valores, dtype=float)
    return float(np.sqrt(dx * np.sum(valores**2)))


def resumen_funcional(
    A: np.ndarray,
    I: np.ndarray,
    malla: Malla,
    config: CampoMedioConfig,
    tolerancia: float = 1.0e-12,
) -> dict[str, float | bool]:
    """Comprueba propiedades estructurales básicas de una simulación.

    Se verifican:
    - dimensiones de las matrices;
    - ausencia de NaN e infinitos;
    - condiciones de Dirichlet de la variable activa;
    - no negatividad numérica de A e I;
    - carácter no decreciente de I.
    """
    forma_esperada = (malla.nt, malla.nx)
    incremento_minimo_i = float(np.min(np.diff(I, axis=0)))
    minimo_a = float(np.min(A))
    minimo_i = float(np.min(I))

    return {
        "forma_correcta": A.shape == forma_esperada and I.shape == forma_esperada,
        "solucion_finita": bool(np.isfinite(A).all() and np.isfinite(I).all()),
        "frontera_izquierda": bool(
            np.allclose(A[:, 0], config.a_in, atol=tolerancia, rtol=0.0)
        ),
        "frontera_derecha": bool(
            np.allclose(A[:, -1], config.a_out, atol=tolerancia, rtol=0.0)
        ),
        "min_A": minimo_a,
        "max_A": float(np.max(A)),
        "min_I": minimo_i,
        "max_I": float(np.max(I)),
        "no_negatividad": minimo_a >= -tolerancia and minimo_i >= -tolerancia,
        "incremento_minimo_I": incremento_minimo_i,
        "I_no_decreciente": incremento_minimo_i >= -tolerancia,
    }


def comparar_soluciones_finales(
    A_1: np.ndarray,
    I_1: np.ndarray,
    A_2: np.ndarray,
    I_2: np.ndarray,
    dx: float,
) -> dict[str, float]:
    """Compara dos soluciones en el instante final."""
    diferencia_a = np.asarray(A_1[-1] - A_2[-1], dtype=float)
    diferencia_i = np.asarray(I_1[-1] - I_2[-1], dtype=float)

    max_a_ref = max(float(np.max(np.abs(A_2[-1]))), np.finfo(float).eps)
    max_i_ref = max(float(np.max(np.abs(I_2[-1]))), np.finfo(float).eps)

    error_a_inf = float(np.max(np.abs(diferencia_a)))
    error_i_inf = float(np.max(np.abs(diferencia_i)))

    return {
        "error_A_inf": error_a_inf,
        "error_A_l2": norma_l2_discreta(diferencia_a, dx),
        "error_I_inf": error_i_inf,
        "error_I_l2": norma_l2_discreta(diferencia_i, dx),
        "error_rel_A_inf": error_a_inf / max_a_ref,
        "error_rel_I_inf": error_i_inf / max_i_ref,
    }


def error_entre_mallas(
    x_gruesa: np.ndarray,
    u_gruesa: np.ndarray,
    x_fina: np.ndarray,
    u_fina: np.ndarray,
) -> dict[str, float]:
    """Compara dos perfiles mediante autorrefinamiento.

    La solución fina se interpola linealmente sobre los nodos de la malla
    gruesa. La función devuelve el error en norma infinito y en norma L2
    discreta.
    """
    x_gruesa = np.asarray(x_gruesa, dtype=float)
    u_gruesa = np.asarray(u_gruesa, dtype=float)
    x_fina = np.asarray(x_fina, dtype=float)
    u_fina = np.asarray(u_fina, dtype=float)

    referencia = np.interp(x_gruesa, x_fina, u_fina)
    diferencia = u_gruesa - referencia
    dx = float(x_gruesa[1] - x_gruesa[0])

    return {
        "inf": float(np.max(np.abs(diferencia))),
        "l2": norma_l2_discreta(diferencia, dx),
    }


def orden_observado(
    error_h: float,
    error_h2: float,
    razon_refinamiento: float = 2.0,
) -> float:
    """Calcula p = log(E_h/E_{h/2}) / log(razón de refinamiento)."""
    if error_h <= 0.0 or error_h2 <= 0.0:
        return float("nan")
    return float(np.log(error_h / error_h2) / np.log(razon_refinamiento))


def ultimo_punto_significativo(
    x: np.ndarray,
    perfil: np.ndarray,
    tolerancia: float = 1.0e-4,
) -> float:
    """Última posición donde |perfil| supera una tolerancia absoluta."""
    x = np.asarray(x, dtype=float)
    perfil = np.asarray(perfil, dtype=float)
    indices = np.flatnonzero(np.abs(perfil) > tolerancia)
    return float(x[indices[-1]]) if indices.size else float(x[0])


def serie_residuo_balance(
    A: np.ndarray,
    I: np.ndarray,
    malla: Malla,
    config: CampoMedioConfig,
    metodo: Literal["euler", "imex"],
) -> np.ndarray:
    """Residuo máximo por paso de la identidad discreta de balance.

    Al sumar las dos ecuaciones continuas se obtiene

        (a+i)_t = D a_xx + (rb+re) a.

    Para Euler se usa el laplaciano en t^n. Para IMEX--Crank--Nicolson se
    usa el promedio de los laplacianos en t^n y t^{n+1}.
    """
    A = np.asarray(A, dtype=float)
    I = np.asarray(I, dtype=float)

    lhs = (
        (A[1:, 1:-1] - A[:-1, 1:-1])
        + (I[1:, 1:-1] - I[:-1, 1:-1])
    ) / malla.dt

    lap_n = (
        A[:-1, :-2] - 2.0 * A[:-1, 1:-1] + A[:-1, 2:]
    ) / malla.dx**2

    if metodo == "euler":
        difusion = config.D * lap_n
    elif metodo == "imex":
        lap_np1 = (
            A[1:, :-2] - 2.0 * A[1:, 1:-1] + A[1:, 2:]
        ) / malla.dx**2
        difusion = 0.5 * config.D * (lap_n + lap_np1)
    else:
        raise ValueError("metodo debe ser 'euler' o 'imex'.")

    rhs = difusion + (config.rb + config.re) * A[:-1, 1:-1]
    return np.max(np.abs(lhs - rhs), axis=1)


def residuo_balance_discreto(
    A: np.ndarray,
    I: np.ndarray,
    malla: Malla,
    config: CampoMedioConfig,
    metodo: Literal["euler", "imex"],
) -> float:
    """Máximo global del residuo de balance discreto."""
    return float(np.max(serie_residuo_balance(A, I, malla, config, metodo)))
