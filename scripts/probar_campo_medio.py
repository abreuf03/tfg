import gc
from pathlib import Path
import sys
from typing import Literal

# Permite ejecutar el archivo directamente desde la carpeta scripts.
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campo_medio.config import CampoMedioConfig
from src.campo_medio.metricas import (
    comparar_soluciones_finales,
    error_entre_mallas,
    orden_observado,
    resumen_funcional,
)
from src.campo_medio.solvers import resolver_euler_explicito, resolver_imex_cn
from src.malla import Malla, crear_malla


RESULTADOS = Path("resultados/campo_medio_cap2")
FIGURAS = RESULTADOS / "figuras"
RESULTADOS.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

TOLERANCIA = 1.0e-10


def crear_malla_por_pasos(
    longitud: float,
    tiempo_final: float,
    dx: float,
    dt_maximo: float,
) -> Malla:
    """Crea una malla usando exactamente el dominio y un dt <= dt_maximo."""
    nx = int(round(longitud / dx)) + 1
    nt = int(np.ceil(tiempo_final / dt_maximo)) + 1
    return crear_malla(0.0, longitud, nx, 0.0, tiempo_final, nt)


def ultimo_punto_significativo(
    x: np.ndarray,
    perfil: np.ndarray,
    tolerancia: float = 1.0e-4,
) -> float:
    """Última posición donde |perfil| supera una tolerancia absoluta."""
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

    Euler:
        ((A^{n+1}+I^{n+1})-(A^n+I^n))/dt
        = D Delta_h A^n + (rb+re) A^n.

    IMEX--CN:
        ((A^{n+1}+I^{n+1})-(A^n+I^n))/dt
        = D/2 (Delta_h A^n + Delta_h A^{n+1}) + (rb+re) A^n.
    """
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


def graficar_comparacion_final(
    malla: Malla,
    A_e: np.ndarray,
    I_e: np.ndarray,
    A_c: np.ndarray,
    I_c: np.ndarray,
) -> None:
    fig, ejes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ejes[0].plot(malla.x, A_e[-1], label="Euler explícito")
    ejes[0].plot(malla.x, A_c[-1], "--", label="IMEX--Crank--Nicolson")
    ejes[0].set_ylabel(r"$a(x,T)$")
    ejes[0].set_title("Densidad activa en el instante final")
    ejes[0].legend()

    ejes[1].plot(malla.x, I_e[-1], label="Euler explícito")
    ejes[1].plot(malla.x, I_c[-1], "--", label="IMEX--Crank--Nicolson")
    ejes[1].set_xlabel(r"$x$")
    ejes[1].set_ylabel(r"$i(x,T)$")
    ejes[1].set_title("Densidad inactiva en el instante final")
    ejes[1].legend()

    fig.tight_layout()
    fig.savefig(
        FIGURAS / "campo_medio_comparacion_metodos.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def graficar_evolucion(
    malla: Malla,
    A: np.ndarray,
    I: np.ndarray,
    tiempos: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0),
) -> None:
    fig, ejes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for tiempo in tiempos:
        n = int(np.argmin(np.abs(malla.t - tiempo)))
        ejes[0].plot(malla.x, A[n], label=fr"$t={malla.t[n]:.0f}$")
        ejes[1].plot(malla.x, I[n], label=fr"$t={malla.t[n]:.0f}$")

    ejes[0].set_ylabel(r"$a(x,t)$")
    ejes[0].set_title("Evolución de la densidad activa (IMEX--CN)")
    ejes[0].legend()

    ejes[1].set_xlabel(r"$x$")
    ejes[1].set_ylabel(r"$i(x,t)$")
    ejes[1].set_title("Evolución de la densidad inactiva (IMEX--CN)")
    ejes[1].legend()

    fig.tight_layout()
    fig.savefig(
        FIGURAS / "campo_medio_evolucion.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def graficar_balance(
    malla: Malla,
    residuo_euler: np.ndarray,
    residuo_imex: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(malla.t[1:], np.maximum(residuo_euler, 1.0e-18), label="Euler")
    ax.semilogy(malla.t[1:], np.maximum(residuo_imex, 1.0e-18), label="IMEX--CN")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("residuo máximo")
    ax.set_title("Identidad discreta de balance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURAS / "campo_medio_balance_discreto.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def prueba_funcional_comparacion_y_balance(
    config: CampoMedioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta las pruebas 1, 2 y 4 sobre una simulación común."""

    # Dominio deliberadamente holgado. Con estos parámetros, al final del
    # experimento la parte significativa de A queda muy lejos de x=L.
    longitud = 60.0
    tiempo_final = 15.0
    dx = 0.2
    dt = 0.005
    malla = crear_malla_por_pasos(longitud, tiempo_final, dx, dt)

    lambda_euler = config.D * malla.dt / malla.dx**2
    if lambda_euler > 0.5:
        raise RuntimeError(
            f"Euler no satisface la condición difusiva: lambda={lambda_euler:.4f}."
        )

    print("\n=== PRUEBAS 1, 2 Y 4 ===")
    print(
        f"Dominio [0,{longitud:g}], T={tiempo_final:g}, "
        f"dx={malla.dx:g}, dt={malla.dt:g}, lambda={lambda_euler:.4f}"
    )

    A_e, I_e = resolver_euler_explicito(malla, config)
    A_c, I_c = resolver_imex_cn(malla, config)

    filas_funcionales: list[dict[str, object]] = []
    for nombre, A, I in (
        ("Euler explícito", A_e, I_e),
        ("IMEX--Crank--Nicolson", A_c, I_c),
    ):
        resumen = resumen_funcional(A, I, malla, config, tolerancia=TOLERANCIA)
        x_ultimo_a = ultimo_punto_significativo(malla.x, A[-1])
        x_ultimo_i = ultimo_punto_significativo(malla.x, I[-1])
        resumen.update(
            {
                "metodo": nombre,
                "L": longitud,
                "T": tiempo_final,
                "dx": malla.dx,
                "dt": malla.dt,
                "lambda_difusivo": lambda_euler,
                "x_ultimo_A_mayor_1e-4": x_ultimo_a,
                "x_ultimo_I_mayor_1e-4": x_ultimo_i,
                "distancia_A_a_frontera_derecha": longitud - x_ultimo_a,
            }
        )
        filas_funcionales.append(resumen)

        comprobaciones = (
            resumen["forma_correcta"],
            resumen["solucion_finita"],
            resumen["frontera_izquierda"],
            resumen["frontera_derecha"],
            resumen["no_negatividad"],
            resumen["I_no_decreciente"],
        )
        if not all(comprobaciones):
            raise AssertionError(f"Ha fallado una comprobación funcional en {nombre}.")
        if longitud - x_ultimo_a < 10.0:
            raise AssertionError(
                f"La solución activa se acerca demasiado a la frontera derecha en {nombre}."
            )

    tabla_funcional = pd.DataFrame(filas_funcionales)
    tabla_funcional.to_csv(RESULTADOS / "validacion_funcional.csv", index=False)

    comparacion = comparar_soluciones_finales(A_e, I_e, A_c, I_c, malla.dx)
    comparacion.update(
        {
            "T": tiempo_final,
            "dx": malla.dx,
            "dt": malla.dt,
            "error_rel_A_inf": comparacion["error_A_inf"]
            / max(float(np.max(np.abs(A_c[-1]))), np.finfo(float).eps),
            "error_rel_I_inf": comparacion["error_I_inf"]
            / max(float(np.max(np.abs(I_c[-1]))), np.finfo(float).eps),
        }
    )
    tabla_comparacion = pd.DataFrame([comparacion])
    tabla_comparacion.to_csv(RESULTADOS / "comparacion_metodos.csv", index=False)

    residuo_e = serie_residuo_balance(A_e, I_e, malla, config, "euler")
    residuo_c = serie_residuo_balance(A_c, I_c, malla, config, "imex")
    tabla_balance = pd.DataFrame(
        {
            "t": malla.t[1:],
            "residuo_euler": residuo_e,
            "residuo_imex": residuo_c,
        }
    )
    tabla_balance.to_csv(RESULTADOS / "balance_discreto.csv", index=False)

    if float(np.max(residuo_e)) > 1.0e-9 or float(np.max(residuo_c)) > 1.0e-9:
        raise AssertionError("El residuo de balance es mayor de lo esperado.")

    graficar_comparacion_final(malla, A_e, I_e, A_c, I_c)
    graficar_evolucion(malla, A_c, I_c)
    graficar_balance(malla, residuo_e, residuo_c)

    print("\nValidación funcional:")
    print(
        tabla_funcional[
            [
                "metodo",
                "min_A",
                "max_A",
                "min_I",
                "max_I",
                "incremento_minimo_I",
                "distancia_A_a_frontera_derecha",
            ]
        ].to_string(index=False)
    )
    print("\nComparación de métodos en T:")
    print(tabla_comparacion.to_string(index=False))
    print(
        "\nResiduo máximo de balance: "
        f"Euler={np.max(residuo_e):.3e}, IMEX={np.max(residuo_c):.3e}"
    )

    del A_e, I_e, A_c, I_c, residuo_e, residuo_c
    gc.collect()
    return tabla_funcional, tabla_comparacion, tabla_balance


def prueba_autorrefinamiento(config: CampoMedioConfig) -> pd.DataFrame:
    """Prueba 3: convergencia por comparación de mallas sucesivas."""
    print("\n=== PRUEBA 3: AUTORREFINAMIENTO ===")

    longitud = 60.0
    tiempo_final = 5.0
    pasos = [0.8, 0.4, 0.2, 0.1]
    factor_dt = 0.1  # dt <= 0.1 dx^2 / D; lambda <= 0.1 para Euler.

    soluciones: dict[str, dict[float, dict[str, np.ndarray | float]]] = {
        "Euler": {},
        "IMEX": {},
    }

    for dx in pasos:
        dt_maximo = factor_dt * dx**2 / config.D
        malla = crear_malla_por_pasos(longitud, tiempo_final, dx, dt_maximo)

        A_e, I_e = resolver_euler_explicito(malla, config)
        A_c, I_c = resolver_imex_cn(malla, config)

        soluciones["Euler"][dx] = {
            "x": malla.x.copy(),
            "A": A_e[-1].copy(),
            "I": I_e[-1].copy(),
            "dt": malla.dt,
        }
        soluciones["IMEX"][dx] = {
            "x": malla.x.copy(),
            "A": A_c[-1].copy(),
            "I": I_c[-1].copy(),
            "dt": malla.dt,
        }

        print(
            f"dx={dx:.3f}, dt={malla.dt:.6f}, "
            f"nx={malla.nx}, nt={malla.nt}"
        )
        del A_e, I_e, A_c, I_c
        gc.collect()

    filas: list[dict[str, float | str]] = []
    errores_figura: dict[str, list[float]] = {
        "Euler, A": [],
        "Euler, I": [],
        "IMEX, A": [],
        "IMEX, I": [],
    }

    for metodo in ("Euler", "IMEX"):
        errores_a_inf: list[float] = []
        errores_i_inf: list[float] = []
        errores_a_l2: list[float] = []
        errores_i_l2: list[float] = []

        for h, h2 in zip(pasos[:-1], pasos[1:]):
            gruesa = soluciones[metodo][h]
            fina = soluciones[metodo][h2]

            err_a = error_entre_mallas(
                gruesa["x"], gruesa["A"], fina["x"], fina["A"]
            )
            err_i = error_entre_mallas(
                gruesa["x"], gruesa["I"], fina["x"], fina["I"]
            )
            errores_a_inf.append(err_a["inf"])
            errores_i_inf.append(err_i["inf"])
            errores_a_l2.append(err_a["l2"])
            errores_i_l2.append(err_i["l2"])

            errores_figura[f"{metodo}, A"].append(err_a["inf"])
            errores_figura[f"{metodo}, I"].append(err_i["inf"])

        for j, h in enumerate(pasos[:-1]):
            filas.append(
                {
                    "metodo": metodo,
                    "dx": h,
                    "dt": soluciones[metodo][h]["dt"],
                    "error_A_inf": errores_a_inf[j],
                    "orden_A_inf": (
                        np.nan
                        if j == 0
                        else orden_observado(errores_a_inf[j - 1], errores_a_inf[j])
                    ),
                    "error_A_l2": errores_a_l2[j],
                    "orden_A_l2": (
                        np.nan
                        if j == 0
                        else orden_observado(errores_a_l2[j - 1], errores_a_l2[j])
                    ),
                    "error_I_inf": errores_i_inf[j],
                    "orden_I_inf": (
                        np.nan
                        if j == 0
                        else orden_observado(errores_i_inf[j - 1], errores_i_inf[j])
                    ),
                    "error_I_l2": errores_i_l2[j],
                    "orden_I_l2": (
                        np.nan
                        if j == 0
                        else orden_observado(errores_i_l2[j - 1], errores_i_l2[j])
                    ),
                }
            )

    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "convergencia.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    h_grafica = np.asarray(pasos[:-1])
    for etiqueta, valores in errores_figura.items():
        ax.loglog(h_grafica, valores, "o-", label=etiqueta)

    referencia = h_grafica**2
    primer_error = errores_figura["Euler, A"][0]
    referencia *= primer_error / referencia[0]
    ax.loglog(h_grafica, referencia, "--", label=r"referencia $O(\Delta x^2)$")
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$E_h=\|U_h-U_{h/2}\|_\infty$")
    ax.set_title("Autorrefinamiento del sistema de campo medio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURAS / "campo_medio_convergencia.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nTabla de convergencia:")
    print(
        tabla[
            [
                "metodo",
                "dx",
                "dt",
                "error_A_inf",
                "orden_A_inf",
                "error_I_inf",
                "orden_I_inf",
            ]
        ].to_string(index=False)
    )
    return tabla


def main() -> None:
    config = CampoMedioConfig(
        D=1.0,
        rb=0.1,
        re=1.0,
        n0=1.0,
        a_in=1.0,
        a_out=0.0,
        a0=0.0,
        i0=0.0,
    )

    prueba_funcional_comparacion_y_balance(config)
    prueba_autorrefinamiento(config)
    print(f"\nResultados guardados en: {RESULTADOS.resolve()}")


if __name__ == "__main__":
    main()
