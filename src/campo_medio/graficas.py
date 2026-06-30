from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from src.malla import Malla


def _preparar_ruta(ruta: str | Path) -> Path:
    """Crea la carpeta de destino y devuelve la ruta como Path."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return ruta


# -----------------------------------------------------------------------------
# Gráficas para la validación del Capítulo 2
# -----------------------------------------------------------------------------


def graficar_comparacion_metodos(
    malla: Malla,
    A_euler: np.ndarray,
    I_euler: np.ndarray,
    A_imex: np.ndarray,
    I_imex: np.ndarray,
    ruta: str | Path,
) -> None:
    """
    Genera una figura comparativa de las soluciones obtenidas con Euler
    explícito e IMEX--Crank--Nicolson en el instante temporal final.

    La figura contiene dos subgráficas. La primera compara los perfiles de la
    variable activa y la segunda compara los perfiles de la variable inactiva.
    El resultado se guarda en la ruta indicada.

    Parámetros:
        malla: Malla espacio-temporal empleada en las simulaciones.
        A_euler: Matriz con la solución de la variable activa obtenida mediante
            Euler explícito.
        I_euler: Matriz con la solución de la variable inactiva obtenida mediante
            Euler explícito.
        A_imex: Matriz con la solución de la variable activa obtenida mediante
            IMEX--Crank--Nicolson.
        I_imex: Matriz con la solución de la variable inactiva obtenida mediante
            IMEX--Crank--Nicolson.
        ruta: Ruta del archivo en el que se guardará la figura generada.

    Devuelve:
        No devuelve ningún valor. La figura se guarda en la ruta indicada.
    """
    ruta = _preparar_ruta(ruta)
    fig, ejes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ejes[0].plot(malla.x, A_euler[-1], label="Euler explícito")
    ejes[0].plot(
        malla.x,
        A_imex[-1],
        "--",
        label="IMEX--Crank--Nicolson",
    )
    ejes[0].set_ylabel(r"$a(x,T)$")
    ejes[0].set_title("Densidad activa en el instante final")
    ejes[0].legend()
    ejes[0].grid(alpha=0.25)

    ejes[1].plot(malla.x, I_euler[-1], label="Euler explícito")
    ejes[1].plot(
        malla.x,
        I_imex[-1],
        "--",
        label="IMEX--Crank--Nicolson",
    )
    ejes[1].set_xlabel(r"$x$")
    ejes[1].set_ylabel(r"$i(x,T)$")
    ejes[1].set_title("Densidad inactiva en el instante final")
    ejes[1].legend()
    ejes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def graficar_evolucion_campo_medio(
    malla: Malla,
    A: np.ndarray,
    I: np.ndarray,
    tiempos: Sequence[float],
    ruta: str | Path,
    nombre_metodo: str = "IMEX--Crank--Nicolson",
) -> None:
    """
    Representa la evolución temporal de ambas especies.

    Parámetros:
        malla: Malla espacio-temporal empleada en las simulaciones.
        A: Matriz con los valores de la variable activa. Cada fila corresponde
            a un instante temporal y cada columna a un nodo espacial.
        I: Matriz con los valores de la variable inactiva. Cada fila corresponde
            a un instante temporal y cada columna a un nodo espacial.
        tiempos: Secuencia de tiempos en los que se quieren representar los perfiles.
        ruta: Ruta del archivo en el que se guardará la figura generada.
        nombre_metodo: Nombre del método numérico utilizado.

    Devuelve:
        No devuelve ningún valor. La figura se guarda en la ruta indicada.
    """
    ruta = _preparar_ruta(ruta)
    fig, ejes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for tiempo in tiempos:
        indice = int(np.argmin(np.abs(malla.t - tiempo)))
        etiqueta = fr"$t={malla.t[indice]:.1f}$"
        ejes[0].plot(malla.x, A[indice], label=etiqueta)
        ejes[1].plot(malla.x, I[indice], label=etiqueta)

    ejes[0].set_ylabel(r"$a(x,t)$")
    ejes[0].set_title(f"Evolución de la densidad activa ({nombre_metodo})")
    ejes[0].legend()
    ejes[0].grid(alpha=0.25)

    ejes[1].set_xlabel(r"$x$")
    ejes[1].set_ylabel(r"$i(x,t)$")
    ejes[1].set_title(f"Evolución de la densidad inactiva ({nombre_metodo})")
    ejes[1].legend()
    ejes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def graficar_balance_discreto(
    t: np.ndarray,
    residuo_euler: np.ndarray,
    residuo_imex: np.ndarray,
    ruta: str | Path,
) -> None:
    """
    Representa el residuo de la identidad de balance en cada paso.

    Parámetros:
        t: Array con los tiempos de los pasos temporales.
        residuo_euler: Array con los residuos obtenidos mediante Euler explícito.
        residuo_imex: Array con los residuos obtenidos mediante IMEX--Crank--Nicolson.
        ruta: Ruta del archivo en el que se guardará la figura generada.

    Devuelve:
        No devuelve ningún valor. La figura se guarda en la ruta indicada.
    """
    ruta = _preparar_ruta(ruta)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(
        t[1:],
        np.maximum(residuo_euler, 1.0e-18),
        label="Euler explícito",
    )
    ax.semilogy(
        t[1:],
        np.maximum(residuo_imex, 1.0e-18),
        label="IMEX--Crank--Nicolson",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("residuo máximo")
    ax.set_title("Identidad discreta de balance")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def graficar_convergencia(
    pasos: np.ndarray,
    errores: dict[str, np.ndarray | list[float]],
    ruta: str | Path,
) -> None:
    """
    Gráfica log--log de los errores de autorrefinamiento.

    Parámetros:
        pasos: Array con los tamaños de paso espacial.
        errores: Diccionario con los errores de autorrefinamiento para cada método.
        ruta: Ruta del archivo en el que se guardará la figura generada.

    Devuelve:
        No devuelve ningún valor. La figura se guarda en la ruta indicada.
    """
    ruta = _preparar_ruta(ruta)
    pasos = np.asarray(pasos, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for etiqueta, valores in errores.items():
        ax.loglog(pasos, np.asarray(valores, dtype=float), "o-", label=etiqueta)

    primer_conjunto = np.asarray(next(iter(errores.values())), dtype=float)
    referencia = pasos**2
    referencia *= primer_conjunto[0] / referencia[0]
    ax.loglog(pasos, referencia, "--", label=r"referencia $O(\Delta x^2)$")

    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$E_h=\|U_h-U_{h/2}\|_\infty$")
    ax.set_title("Autorrefinamiento del sistema de campo medio")
    ax.legend()
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
