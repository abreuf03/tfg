import gc
from pathlib import Path
import sys
from typing import Literal

from scripts.pde_barw.pulso_reducido import gaussiana_inicial

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
    trayectoria_pico,
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




def ajustar_velocidad( tiempos: np.ndarray, posiciones: np.ndarray, t_inicio: float, t_fin: float, longitud: float,) -> dict[str, float]:
   
    mascara = (
        np.isfinite(posiciones)
        & (tiempos >= t_inicio)
        & (tiempos <= t_fin)
    )

    if np.count_nonzero(mascara) < 3:
        raise ValueError(
            "No hay suficientes puntos válidos para ajustar la velocidad."
        )

    velocidad, ordenada = np.polyfit(
        tiempos[mascara],
        posiciones[mascara],
        deg=1,
    )

    distancia_minima = float(
        np.min(longitud - posiciones[mascara])
    )

    return {
        "t_inicio": float(t_inicio),
        "t_fin": float(t_fin),
        "velocidad_numerica": float(velocidad),
        "ordenada_ajuste": float(ordenada),
        "n_puntos_ajuste": int(np.count_nonzero(mascara)),
        "distancia_minima_frontera": distancia_minima,
    }

def margen_frontera_en_ventana(malla: Malla,A: np.ndarray,t_inicio: float,t_fin: float,umbral: float,) -> float:
    
    mascara_tiempo = (
        (malla.t >= t_inicio)
        & (malla.t <= t_fin)
    )

    perfiles = A[mascara_tiempo]

    indices = np.flatnonzero(
        np.any(perfiles > umbral, axis=0)
    )

    if indices.size == 0:
        return float("nan")

    x_derecha = float(malla.x[indices[-1]])
    return float(malla.x[-1] - x_derecha)

def simular_y_medir_velocidad(config: CampoMedioConfig,longitud: float,tiempo_final: float,dx: float,
                              dt: float,ventanas: tuple[tuple[float, float], ...],umbral: float = 1.0e-6,) -> list[dict[str, float]]:
    
    malla = crear_malla_por_pasos(
        longitud=longitud,
        tiempo_final=tiempo_final,
        dx=dx,
        dt_maximo=dt,
    )

    # La condición inicial debe reconstruirse para cada malla.
    a0 = gaussiana_inicial(
        malla.x,
        x0=8.0,
        sigma=0.8,
        A0=0.2,
    )

    a0[0] = 0.0
    a0[-1] = 0.0

    i0 = np.zeros_like(malla.x)

    config_experimento = CampoMedioConfig(
        D=config.D,
        rb=config.rb,
        re=config.re,
        n0=config.n0,
        a_in=0.0,
        a_out=0.0,
        a0=a0,
        i0=i0,
    )

    A, I = resolver_imex_cn(malla, config_experimento)

    if np.max(A) <= 1.0e-12:
        raise RuntimeError(
            "No se ha generado el pulso: max(A) es prácticamente cero."
        )

    # Mismo observable empleado en el experimento original.
    posiciones = trayectoria_pico(malla.x, A)

    filas: list[dict[str, float]] = []

    for t_inicio, t_fin in ventanas:
        if not (0.0 <= t_inicio < t_fin <= tiempo_final):
            raise ValueError(
                f"Ventana inválida [{t_inicio}, {t_fin}] para "
                f"T={tiempo_final}."
            )

        ajuste = ajustar_velocidad(
            tiempos=malla.t,
            posiciones=posiciones,
            t_inicio=t_inicio,
            t_fin=t_fin,
            longitud=longitud,
        )

        ajuste["distancia_minima_frontera"] = (
            margen_frontera_en_ventana(
                malla=malla,
                A=A,
                t_inicio=t_inicio,
                t_fin=t_fin,
                umbral=umbral,
            )
        )

        filas.append(
            {
                "L": float(longitud),
                "T": float(tiempo_final),
                "dx": float(malla.dx),
                "dt": float(malla.dt),
                "lambda_difusivo": float(
                    config.D * malla.dt / malla.dx**2
                ),
                "umbral_margen": float(umbral),
                **ajuste,
            }
        )

    print(
        f"L={longitud:g}, dx={malla.dx:g}, dt={malla.dt:g}, "
        f"max(A)={np.max(A):.4e}, "
        f"x_max(T)={posiciones[-1]:.4f}"
    )

    del A, I, posiciones
    gc.collect()

    return filas

def valores_unicos(referencia: float,valores: tuple[float, ...],) -> list[float]:
    """Incluye la referencia y evita duplicados numéricos."""
    salida: list[float] = []

    for valor in (referencia, *valores):
        if not any(np.isclose(valor, previo) for previo in salida):
            salida.append(float(valor))

    return salida

def estudio_sensibilidad_velocidad(config: CampoMedioConfig,longitud_base: float,tiempo_final: float,
                                   dx_base: float,dt_base: float,ventana_referencia: tuple[float, float],
                                   pasos_espaciales: tuple[float, ...],pasos_temporales: tuple[float, ...],
                                   longitudes: tuple[float, ...],ventanas: tuple[tuple[float, float], ...],
                                   umbral: float = 1.0e-4,) -> pd.DataFrame:
    
    t0_ref, tf_ref = ventana_referencia

    ventanas_completas = list(ventanas)
    if not any(
        np.isclose(t0, t0_ref) and np.isclose(tf, tf_ref)
        for t0, tf in ventanas_completas
    ):
        ventanas_completas.append(ventana_referencia)

    # Simulación base: se reutiliza para comparar ventanas.
    filas_base = simular_y_medir_velocidad(
        config=config,
        longitud=longitud_base,
        tiempo_final=tiempo_final,
        dx=dx_base,
        dt=dt_base,
        ventanas=tuple(ventanas_completas),
        umbral=umbral,
    )

    base_por_ventana = {
        (fila["t_inicio"], fila["t_fin"]): fila
        for fila in filas_base
    }

    referencia = base_por_ventana[(float(t0_ref), float(tf_ref))]
    velocidad_referencia = float(referencia["velocidad_numerica"])

    filas: list[dict[str, object]] = []

    def agregar_fila(
        factor: str,
        valor: str,
        fila_resultado: dict[str, float],
    ) -> None:
        fila = dict(fila_resultado)
        fila["factor"] = factor
        fila["valor_variado"] = valor
        filas.append(fila)

    # Sensibilidad a la malla espacial.
    for dx in valores_unicos(dx_base, pasos_espaciales):
        if np.isclose(dx, dx_base):
            fila = referencia
        else:
            fila = simular_y_medir_velocidad(
                config=config,
                longitud=longitud_base,
                tiempo_final=tiempo_final,
                dx=dx,
                dt=dt_base,
                ventanas=(ventana_referencia,),
                umbral=umbral,
            )[0]

        agregar_fila(
            factor="Malla espacial",
            valor=rf"$\Delta x={fila['dx']:.4g}$",
            fila_resultado=fila,
        )

    # Sensibilidad al paso temporal.
    for dt in valores_unicos(dt_base, pasos_temporales):
        if np.isclose(dt, dt_base):
            fila = referencia
        else:
            fila = simular_y_medir_velocidad(
                config=config,
                longitud=longitud_base,
                tiempo_final=tiempo_final,
                dx=dx_base,
                dt=dt,
                ventanas=(ventana_referencia,),
                umbral=umbral,
            )[0]

        agregar_fila(
            factor="Paso temporal",
            valor=rf"$\Delta t={fila['dt']:.4g}$",
            fila_resultado=fila,
        )

    # Sensibilidad al tamaño del dominio.
    for longitud in valores_unicos(longitud_base, longitudes):
        if np.isclose(longitud, longitud_base):
            fila = referencia
        else:
            fila = simular_y_medir_velocidad(
                config=config,
                longitud=longitud,
                tiempo_final=tiempo_final,
                dx=dx_base,
                dt=dt_base,
                ventanas=(ventana_referencia,),
                umbral=umbral,
            )[0]

        agregar_fila(
            factor="Longitud del dominio",
            valor=rf"$L={fila['L']:.4g}$",
            fila_resultado=fila,
        )

    # Sensibilidad a la ventana de ajuste.
    for t_inicio, t_fin in ventanas_completas:
        fila = base_por_ventana[(float(t_inicio), float(t_fin))]

        agregar_fila(
            factor="Ventana de ajuste",
            valor=rf"$[{t_inicio:.0f},{t_fin:.0f}]$",
            fila_resultado=fila,
        )

    tabla = pd.DataFrame(filas)

    velocidad_asintotica = 2.0 * np.sqrt(config.D * config.rb)

    tabla["variacion_vs_referencia_pct"] = (
        100.0
        * (tabla["velocidad_numerica"] - velocidad_referencia)
        / velocidad_referencia
    )

    tabla["error_vs_asintotica_pct"] = (
        100.0
        * np.abs(tabla["velocidad_numerica"] - velocidad_asintotica)
        / velocidad_asintotica
    )

    orden_factores = {
        "Malla espacial": 0,
        "Paso temporal": 1,
        "Longitud del dominio": 2,
        "Ventana de ajuste": 3,
    }

    tabla["_orden"] = tabla["factor"].map(orden_factores)

    tabla = (
        tabla.sort_values(["_orden", "dx", "dt", "L", "t_inicio"])
        .drop(columns="_orden")
        .reset_index(drop=True)
    )

    tabla.to_csv(
        RESULTADOS / "sensibilidad_velocidad.csv",
        index=False,
    )

    tabla_memoria = tabla[
        [
            "factor",
            "valor_variado",
            "velocidad_numerica",
            "variacion_vs_referencia_pct",
            "distancia_minima_frontera",
        ]
    ].rename(
        columns={
            "factor": "Factor",
            "valor_variado": "Valor",
            "velocidad_numerica": r"$V_{\mathrm{num}}$",
            "variacion_vs_referencia_pct": r"$\Delta V$ (\%)",
            "distancia_minima_frontera": r"Margen a frontera",
        }
    )

    

    print("\n=== SENSIBILIDAD DE LA VELOCIDAD ===")
    print(tabla_memoria.to_string(index=False))

    print(
        "\nVelocidad asintótica de referencia: "
        f"v*=2 sqrt(D rb)={velocidad_asintotica:.6f}"
    )

    print(
        "\nArchivos creados:\n"
        f" - {RESULTADOS / 'sensibilidad_velocidad.csv'}\n"
        
    )

    return tabla


def main() -> None:
    # ------------------------------------------------------------
    # Configuración base: misma que el experimento del pulso
    # ------------------------------------------------------------
    L_BASE = 280.0
    T_BASE = 300.0
    DX_BASE = 0.2
    DT_BASE = 0.02

    # Este umbral NO se usa para calcular la velocidad.
    # Solo sirve para comprobar que el perfil activo no alcanza
    # la frontera derecha.
    UMBRAL_MARGEN = 1.0e-6

    
    config = CampoMedioConfig(
        D=1.0,
        rb=0.1,
        re=1.0,
        n0=1.0,
        a_in=0.0,
        a_out=0.0,
        a0=0.0,
        i0=0.0,
    )

    
    tabla = estudio_sensibilidad_velocidad(
        config=config,
        longitud_base=L_BASE,
        tiempo_final=T_BASE,
        dx_base=DX_BASE,
        dt_base=DT_BASE,
        ventana_referencia=(200.0, 300.0),

        
        pasos_espaciales=(
            0.4,
            0.2,
            0.1,
        ),

        
        pasos_temporales=(
            0.04,
            0.02,
            0.015,
        ),

        
        longitudes=(
            240.0,
            280.0,
            320.0,
        ),

        
        ventanas=(
            (160.0, 300.0),
            (180.0, 300.0),
            (200.0, 300.0),
            (220.0, 300.0),
        ),

        umbral=UMBRAL_MARGEN,
    )

    print("\nTabla final de sensibilidad:")
    print(
        tabla[
            [
                "factor",
                "valor_variado",
                "velocidad_numerica",
                "variacion_vs_referencia_pct",
                "distancia_minima_frontera",
            ]
        ].to_string(index=False)
    )

    print(f"\nResultados guardados en: {RESULTADOS.resolve()}")


if __name__ == "__main__":
    main()