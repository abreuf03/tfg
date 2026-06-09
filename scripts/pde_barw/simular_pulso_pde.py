import gc
from pathlib import Path

import sys

# Permite ejecutar el archivo directamente desde la carpeta scripts.
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))


from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.campo_medio.config import CampoMedioConfig

from src.campo_medio.solvers import resolver_euler_explicito, resolver_imex_cn
from src.malla import Malla, crear_malla
from src.campo_medio.metricas import trayectoria_pico

RESULTADOS = Path("resultados/campo_medio_cap4")
FIGURAS = RESULTADOS / "figuras"
RESULTADOS.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)



def crear_malla_por_pasos(
    longitud: float,
    tiempo_final: float,
    dx: float,
    dt_maximo: float,) -> Malla:
    """Crea una malla usando exactamente el dominio y un dt <= dt_maximo."""
    nx = int(round(longitud / dx)) + 1
    nt = int(np.ceil(tiempo_final / dt_maximo)) + 1
    return crear_malla(0.0, longitud, nx, 0.0, tiempo_final, nt)

def gaussiana_inicial(x: np.ndarray, x0: float, sigma: float, A0: float) -> np.ndarray:
    """Perfil gaussiano."""
    return A0 * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


# ------------------------------------------------------------------------- 
# # Figuras # 
# ------------------------------------------------------------------------- 
def graficar_evolucion_perfiles( malla: Malla, A: np.ndarray, I: np.ndarray, tiempos: tuple[float, ...], ruta: Path, ) -> None: 
    """Representa los perfiles activo e inactivo en distintos tiempos.""" 
    fig, ejes = plt.subplots( 2, 1, figsize=(9, 8), sharex=True, ) 
    for tiempo in tiempos: 
        indice = int(np.argmin(np.abs(malla.t - tiempo))) 
        etiqueta = fr"$t={malla.t[indice]:.0f}$" 
        ejes[0].plot( malla.x, A[indice], label=etiqueta, ) 
        ejes[1].plot( malla.x, I[indice], label=etiqueta, ) 
    ejes[0].set_ylabel(r"$a(x,t)$") 
    ejes[0].set_title("Evolución de la densidad de puntas activas") 
    ejes[0].legend(ncol=2) 
    ejes[0].grid(alpha=0.25) 
    ejes[1].set_xlabel(r"$x$") 
    ejes[1].set_ylabel(r"$i(x,t)$") 
    ejes[1].set_title("Evolución de la densidad de conductos inactivos") 
    ejes[1].legend(ncol=2) 
    ejes[1].grid(alpha=0.25) 
    fig.tight_layout() 
    fig.savefig(ruta, dpi=300, bbox_inches="tight") 
    plt.close(fig)


def graficar_marco_movil( malla: Malla, A: np.ndarray, posiciones: np.ndarray, tiempos: tuple[float, ...], ruta: Path, ventana: tuple[float, float] = (-25.0, 25.0), ) -> None: 
    """
    Representa perfiles activos normalizados respecto de su máximo. La coordenada móvil es xi = x - x_max(t). 
    Si los perfiles tardíos se superponen, existe evidencia de que el pulso ha alcanzado una forma aproximadamente estacionaria. 
    """ 
    fig, ax = plt.subplots(figsize=(9, 5.5)) 
    for tiempo in tiempos: 
        indice = int(np.argmin(np.abs(malla.t - tiempo))) 
        perfil = A[indice] 
        maximo = float(np.max(perfil)) 
        if maximo <= 0.0: 
            continue 
        xi = malla.x - posiciones[indice] 
        perfil_normalizado = perfil / maximo 
        ax.plot( xi, perfil_normalizado, label=fr"$t={malla.t[indice]:.0f}$", ) 
    ax.set_xlim(*ventana) 
    ax.set_xlabel(r"$\xi=x-x_{\max}(t)$") 
    ax.set_ylabel(r"$a(x,t)/a_{\max}(t)$") 
    ax.set_title("Perfiles activos en el marco móvil") 
    ax.legend(ncol=2) 
    ax.grid(alpha=0.25) 
    fig.tight_layout() 
    fig.savefig(ruta, dpi=300, bbox_inches="tight") 
    plt.close(fig)



def main() -> None:
    # Parámetros del dominio y de la discretización
    longitud = 280.0
    tiempo_final = 300.0
    dx = 0.2
    dt_maximo = 0.02

    # Parámetros del dato inicial gaussiano
    x_0 = 8.0
    A0 = 0.2
    sigma = 0.8

    # Construcción de la malla
    malla = crear_malla_por_pasos(
        longitud=longitud,
        tiempo_final=tiempo_final,
        dx=dx,
        dt_maximo=dt_maximo,
    )

    # Datos iniciales
    a0 = gaussiana_inicial(
        malla.x,
        x0=x_0,
        sigma=sigma,
        A0=A0,
    )
    i0 = np.zeros_like(malla.x)


    a0[0] = 0.0
    a0[-1] = 0.0

    config = CampoMedioConfig(
        D=1.0,
        rb=0.1,
        re=1.0,
        n0=1.0,
        a_in=0.0,
        a_out=0.0,
        a0=a0,
        i0=i0,
    )

    print("=== EXPERIMENTO DEL PULSO DE CAMPO MEDIO ===")
    print(f"Dominio espacial: [0, {longitud:g}]")
    print(f"Tiempo final: {tiempo_final:g}")
    print(f"nx = {malla.nx}")
    print(f"nt = {malla.nt}")
    print(f"dx = {malla.dx:.6f}")
    print(f"dt = {malla.dt:.6f}")
    print(
        "D dt / dx^2 = "
        f"{config.D * malla.dt / malla.dx**2:.6f}"
    )

    print("\nResolviendo el sistema mediante IMEX--Crank--Nicolson...")

    A, I = resolver_imex_cn(malla, config)

    # ------------------------------------------------------------------
    # Comprobaciones preliminares
    # ------------------------------------------------------------------

    forma_esperada = (malla.nt, malla.nx)

    if A.shape != forma_esperada:
        raise ValueError(
            f"A tiene forma {A.shape}, pero se esperaba {forma_esperada}."
        )

    if I.shape != forma_esperada:
        raise ValueError(
            f"I tiene forma {I.shape}, pero se esperaba {forma_esperada}."
        )

    if not np.isfinite(A).all() or not np.isfinite(I).all():
        raise ValueError("La solución contiene NaN o valores infinitos.")

    if np.min(A) < -1.0e-10:
        raise ValueError("La densidad activa presenta valores negativos relevantes.")

    if np.min(I) < -1.0e-10:
        raise ValueError("La densidad inactiva presenta valores negativos relevantes.")

    if not np.allclose(A[:, 0], 0.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("No se mantiene la condición a(0,t)=0.")

    if not np.allclose(A[:, -1], 0.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("No se mantiene la condición a(L,t)=0.")

    # Posición del máximo activo en cada instante
    posiciones = trayectoria_pico(malla.x, A)
    maximos_A = np.max(A, axis=1)

    # Última posición en la que el perfil final es significativo
    umbral = 1.0e-6
    indices_significativos = np.flatnonzero(A[-1] > umbral)

    if indices_significativos.size > 0:
        x_significativo_final = float(
            malla.x[indices_significativos[-1]]
        )
    else:
        x_significativo_final = 0.0

    distancia_frontera = longitud - x_significativo_final

    print("\nComprobaciones preliminares:")
    print(f"Forma de A: {A.shape}")
    print(f"Forma de I: {I.shape}")
    print(f"min(A) = {np.min(A):.6e}")
    print(f"max(A) = {np.max(A):.6e}")
    print(f"min(I) = {np.min(I):.6e}")
    print(f"max(I) = {np.max(I):.6e}")
    print(f"x_max(0) = {posiciones[0]:.6f}")
    print(f"x_max(T) = {posiciones[-1]:.6f}")
    print(
        f"Última posición con A(x,T) > {umbral:.0e}: "
        f"{x_significativo_final:.6f}"
    )
    print(
        "Distancia del perfil activo a la frontera derecha: "
        f"{distancia_frontera:.6f}"
    )

    if distancia_frontera < 20.0:
        print(
            "ADVERTENCIA: el perfil activo está relativamente cerca "
            "de la frontera derecha."
        )

    # ------------------------------------------------------------------
    # Guardado de los resultados
    # ------------------------------------------------------------------

    np.savez(
        RESULTADOS / "solucion_pulso_campo_medio.npz",
        x=malla.x,
        t=malla.t,
        A=A,
        I=I,
        posiciones=posiciones,
    )

    tabla_trayectoria = pd.DataFrame(
        {
            "t": malla.t,
            "x_max": posiciones,
            "A_max": maximos_A,
        }
    )

    tabla_trayectoria.to_csv(
        RESULTADOS / "trayectoria_pulso.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Generación de figuras
    # ------------------------------------------------------------------

    graficar_evolucion_perfiles(
        malla=malla,
        A=A,
        I=I,
        tiempos=(
            0.0,
            50.0,
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
        ),
        ruta=FIGURAS / "campo_medio_evolucion_pulso.png",
    )

    graficar_marco_movil(
        malla=malla,
        A=A,
        posiciones=posiciones,
        tiempos=(
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
        ),
        ruta=FIGURAS / "campo_medio_marco_movil.png",
    )

    print(f"\nResultados guardados en: {RESULTADOS.resolve()}")






if __name__ == "__main__":
    main()