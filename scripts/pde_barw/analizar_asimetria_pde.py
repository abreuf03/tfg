from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# El archivo se guarda en scripts/pde_barw/, por eso la raíz está en parents[2].
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))


from src.campo_medio.metricas import trayectoria_pico

#ampliación de los tutores
from paquetes_cerrados.hannezo2017_fig3_pde_barw.reproducir_fig3_hannezo_pde_barw import fit_pulse_tails, collapse_pde_profiles_moving_frame



RESULTADOS = RAIZ_PROYECTO / "resultados" / "campo_medio_cap4"
FIGURAS = RESULTADOS / "figuras"
ARCHIVO_SOLUCION = RESULTADOS / "solucion_pulso_campo_medio.npz"

RESULTADOS.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)


def ajuste_lineal(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """Devuelve pendiente, ordenada y R^2 del ajuste y = m*x + b."""
    pendiente, ordenada = np.polyfit(x, y, deg=1)

    prediccion = pendiente * x + ordenada
    suma_residuos = float(np.sum((y - prediccion) ** 2))
    suma_total = float(np.sum((y - np.mean(y)) ** 2))

    r2 = 1.0 - suma_residuos / suma_total if suma_total > 0.0 else 1.0

    return float(pendiente), float(ordenada), float(r2)


def ajustar_colas_pulso(
    x: np.ndarray,
    perfil: np.ndarray,
    x_pico: float,
    fraccion_min: float,
    fraccion_max: float,
    minimo_puntos: int = 5,
) -> dict:
    """Ajusta las dos colas exponenciales del pulso normalizado.

    Cola trasera:
        log(a/a_max) = gamma_- * xi + c_-,  xi < 0

    Cola delantera:
        log(a/a_max) = -gamma_+ * xi + c_+, xi > 0
    """
    x = np.asarray(x, dtype=float)
    perfil = np.asarray(perfil, dtype=float)

    if x.ndim != 1 or perfil.ndim != 1:
        raise ValueError("x y perfil deben ser vectores unidimensionales.")

    if x.size != perfil.size:
        raise ValueError("x y perfil deben tener la misma longitud.")

    if not 0.0 < fraccion_min < fraccion_max < 1.0:
        raise ValueError(
            "Debe cumplirse 0 < fraccion_min < fraccion_max < 1."
        )

    a_max = float(np.max(perfil))

    if a_max <= 0.0:
        raise ValueError("El perfil no tiene un máximo positivo.")

    xi = x - x_pico
    normalizado = perfil / a_max

    banda = (
        np.isfinite(normalizado)
        & (normalizado >= fraccion_min)
        & (normalizado <= fraccion_max)
    )

    mascara_trasera = banda & (xi < 0.0)
    mascara_delantera = banda & (xi > 0.0)

    n_trasera = int(np.count_nonzero(mascara_trasera))
    n_delantera = int(np.count_nonzero(mascara_delantera))

    if n_trasera < minimo_puntos:
        raise ValueError(
            "No hay suficientes puntos en la cola trasera "
            f"({n_trasera} < {minimo_puntos})."
        )

    if n_delantera < minimo_puntos:
        raise ValueError(
            "No hay suficientes puntos en la cola delantera "
            f"({n_delantera} < {minimo_puntos})."
        )

    xi_trasera = xi[mascara_trasera]
    log_trasera = np.log(normalizado[mascara_trasera])

    xi_delantera = xi[mascara_delantera]
    log_delantera = np.log(normalizado[mascara_delantera])

    pendiente_trasera, ordenada_trasera, r2_trasera = ajuste_lineal(
        xi_trasera,
        log_trasera,
    )

    pendiente_delantera, ordenada_delantera, r2_delantera = ajuste_lineal(
        xi_delantera,
        log_delantera,
    )

    gamma_menos = pendiente_trasera
    gamma_mas = -pendiente_delantera

    if gamma_menos <= 0.0 or gamma_mas <= 0.0:
        raise ValueError(
            "Los ajustes no producen tasas de decaimiento positivas."
        )

    cociente = gamma_menos / gamma_mas
    cociente_teorico = np.sqrt(2.0) - 1.0
    error_relativo = abs(cociente - cociente_teorico) / cociente_teorico

    return {
        "x_pico": float(x_pico),
        "a_max": a_max,
        "fraccion_min": float(fraccion_min),
        "fraccion_max": float(fraccion_max),
        "gamma_menos": float(gamma_menos),
        "gamma_mas": float(gamma_mas),
        "cociente": float(cociente),
        "cociente_teorico": float(cociente_teorico),
        "error_relativo": float(error_relativo),
        "error_porcentual": float(100.0 * error_relativo),
        "r2_menos": float(r2_trasera),
        "r2_mas": float(r2_delantera),
        "n_menos": n_trasera,
        "n_mas": n_delantera,
        "ordenada_menos": float(ordenada_trasera),
        "ordenada_mas": float(ordenada_delantera),
        "xi": xi,
        "normalizado": normalizado,
        "mascara_menos": mascara_trasera,
        "mascara_mas": mascara_delantera,
    }


def graficar_ajuste_asimetria(
    x: np.ndarray,
    perfil: np.ndarray,
    ajuste: dict,
    tiempo: float,
    ruta: Path,
) -> None:
    """Representa el perfil normalizado y los ajustes exponenciales."""
    xi = ajuste["xi"]
    normalizado = ajuste["normalizado"]

    mascara_positiva = normalizado > 0.0

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.semilogy(
        xi[mascara_positiva],
        normalizado[mascara_positiva],
        label="Perfil numérico normalizado",
    )

    mascara_menos = ajuste["mascara_menos"]
    mascara_mas = ajuste["mascara_mas"]

    ax.scatter(
        xi[mascara_menos],
        normalizado[mascara_menos],
        s=18,
        label="Puntos de la cola trasera",
    )

    ax.scatter(
        xi[mascara_mas],
        normalizado[mascara_mas],
        s=18,
        label="Puntos de la cola delantera",
    )

    xi_menos = xi[mascara_menos]
    ajuste_menos = np.exp(
        ajuste["gamma_menos"] * xi_menos
        + ajuste["ordenada_menos"]
    )

    xi_mas = xi[mascara_mas]
    ajuste_mas = np.exp(
        -ajuste["gamma_mas"] * xi_mas
        + ajuste["ordenada_mas"]
    )

    ax.semilogy(
        xi_menos,
        ajuste_menos,
        "--",
        label=(
            fr"Ajuste trasero: "
            fr"$\gamma_-={ajuste['gamma_menos']:.5f}$"
        ),
    )

    ax.semilogy(
        xi_mas,
        ajuste_mas,
        "--",
        label=(
            fr"Ajuste delantero: "
            fr"$\gamma_+={ajuste['gamma_mas']:.5f}$"
        ),
    )

    ax.set_xlim(-35.0, 35.0)
    ax.set_ylim(1.0e-4, 1.2)

    ax.set_xlabel(r"$\xi=x-x_{\max}(t)$")
    ax.set_ylabel(r"$a(\xi,t)/a_{\max}(t)$")
    ax.set_title(
        "Ajuste exponencial de las colas del pulso "
        fr"en $t={tiempo:.0f}$"
    )

    ax.legend()
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)

#adaptación de la función plot_pde_stationary_pulse de los tutores
def graficar_pulso_colapsado(
    z: np.ndarray,
    perfil_normalizado: np.ndarray,
    sem_normalizado: np.ndarray,
    ajuste: dict,
    ruta: Path,
) -> None:
    """Representa el perfil medio y los ajustes exponenciales de sus colas."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.fill_between(
        z,
        np.maximum(perfil_normalizado - sem_normalizado, 0.0),
        perfil_normalizado + sem_normalizado,
        alpha=0.2,
        label="SEM entre tiempos",
    )

    ax.plot(
        z,
        perfil_normalizado,
        label="Perfil medio normalizado",
    )

    z_pico = ajuste["z_peak"]

    mascara_delantera = (
        (z > z_pico)
        & (perfil_normalizado > 0.05)
        & (perfil_normalizado < 0.65)
    )

    mascara_trasera = (
        (z < z_pico)
        & (perfil_normalizado > 0.05)
        & (perfil_normalizado < 0.65)
    )

    if "front_slope" in ajuste:
        ajuste_delantero = np.exp(
            ajuste["front_slope"] * z[mascara_delantera]
            + ajuste["front_intercept"]
        )

        ax.plot(
            z[mascara_delantera],
            ajuste_delantero,
            "--",
            label=(
                "Ajuste delantero: "
                fr"$\gamma_+={abs(ajuste['front_slope']):.5f}$"
            ),
        )

    if "back_slope" in ajuste:
        ajuste_trasero = np.exp(
            ajuste["back_slope"] * z[mascara_trasera]
            + ajuste["back_intercept"]
        )

        ax.plot(
            z[mascara_trasera],
            ajuste_trasero,
            ":",
            label=(
                "Ajuste trasero: "
                fr"$\gamma_-={abs(ajuste['back_slope']):.5f}$"
            ),
        )

    ax.axvline(
        z_pico,
        linestyle=":",
        label="Máximo del perfil medio",
    )

    ax.set_xlim(-60.0, 40.0)
    ax.set_ylim(0.0, 1.05)

    ax.set_xlabel(r"$z=x-x_{\max}(t)$")
    ax.set_ylabel(r"$\overline{a}(z)/\overline{a}_{\max}$")
    ax.set_title("Perfil activo medio en el marco móvil")

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)



def main() -> None:
    if not ARCHIVO_SOLUCION.exists():
        raise FileNotFoundError(
            "No se encuentra la solución del pulso:\n"
            f"{ARCHIVO_SOLUCION}\n"
            "Ejecuta primero el script de simulación del pulso."
        )

    print("=== ANÁLISIS DE LA ASIMETRÍA DEL PULSO ===")
    print(f"Cargando: {ARCHIVO_SOLUCION}")

    with np.load(
        ARCHIVO_SOLUCION,
        allow_pickle=False,
    ) as datos:
        x = np.asarray(datos["x"], dtype=float)
        t = np.asarray(datos["t"], dtype=float)
        A = np.asarray(datos["A"], dtype=float)
        I = np.asarray(datos["I"], dtype=float)

        if "posiciones" in datos.files:
            posiciones = np.asarray(
                datos["posiciones"],
                dtype=float,
            )
        else:
            posiciones = trayectoria_pico(x, A)

    tiempos_analisis = (
        200.0,
        250.0,
        300.0,
    )

    bandas = (
        (0.02, 0.20),
        (0.02, 0.25),
        (0.05, 0.30),
    )

    banda_principal = (0.02, 0.25)

    filas = []
    ajuste_figura = None
    perfil_figura = None
    tiempo_figura = 300.0

    for tiempo in tiempos_analisis:
        indice = int(np.argmin(np.abs(t - tiempo)))
        tiempo_real = float(t[indice])
        perfil = A[indice]
        x_pico = float(posiciones[indice])

        for fraccion_min, fraccion_max in bandas:
            ajuste = ajustar_colas_pulso(
                x=x,
                perfil=perfil,
                x_pico=x_pico,
                fraccion_min=fraccion_min,
                fraccion_max=fraccion_max,
                minimo_puntos=5,
            )

            filas.append(
                {
                    "t": tiempo_real,
                    "fraccion_min": fraccion_min,
                    "fraccion_max": fraccion_max,
                    "x_pico": ajuste["x_pico"],
                    "a_max": ajuste["a_max"],
                    "gamma_menos": ajuste["gamma_menos"],
                    "gamma_mas": ajuste["gamma_mas"],
                    "cociente": ajuste["cociente"],
                    "cociente_teorico": ajuste["cociente_teorico"],
                    "error_relativo": ajuste["error_relativo"],
                    "error_porcentual": ajuste["error_porcentual"],
                    "r2_menos": ajuste["r2_menos"],
                    "r2_mas": ajuste["r2_mas"],
                    "n_menos": ajuste["n_menos"],
                    "n_mas": ajuste["n_mas"],
                }
            )

            if (
                np.isclose(tiempo_real, tiempo_figura)
                and np.isclose(fraccion_min, banda_principal[0])
                and np.isclose(fraccion_max, banda_principal[1])
            ):
                ajuste_figura = ajuste
                perfil_figura = perfil.copy()

    tabla = pd.DataFrame(filas)

    tabla.to_csv(
        RESULTADOS / "asimetria_pulso.csv",
        index=False,
    )

    tabla_principal = tabla[
        np.isclose(
            tabla["fraccion_min"],
            banda_principal[0],
        )
        & np.isclose(
            tabla["fraccion_max"],
            banda_principal[1],
        )
    ].copy()

    tabla_principal.to_csv(
        RESULTADOS / "asimetria_pulso_principal.csv",
        index=False,
    )

    print("\nResultados para la banda principal [0.02, 0.25]:")
    print(
        tabla_principal[
            [
                "t",
                "gamma_menos",
                "gamma_mas",
                "cociente",
                "cociente_teorico",
                "error_porcentual",
                "r2_menos",
                "r2_mas",
                "n_menos",
                "n_mas",
            ]
        ].to_string(index=False)
    )

    if ajuste_figura is None or perfil_figura is None:
        raise RuntimeError(
            "No se pudo recuperar el ajuste principal para la figura."
        )

    graficar_ajuste_asimetria(
        x=x,
        perfil=perfil_figura,
        ajuste=ajuste_figura,
        tiempo=tiempo_figura,
        ruta=FIGURAS / "campo_medio_asimetria_pulso.png",
    )

    print(
        "\nPredicción teórica: "
        f"sqrt(2)-1 = {np.sqrt(2.0)-1.0:.8f}"
    )

    print(
        "\nResultados guardados en: "
        f"{RESULTADOS.resolve()}"
    )

        # ------------------------------------------------------------------
    # Ampliación basada en el código proporcionado por los tutores
    # ------------------------------------------------------------------

    tiempos_colapso = (
        200,
        220,
        240,
        260,
        280,
        300,
    )

    # Malla común en la coordenada móvil.
    z_centers = np.arange(
        -60.0,
        40.0 + 0.1,
        0.2,
    )

    # Estructura esperada por collapse_pde_profiles_moving_frame.
    pde = {
        "x": x,
        "times": t,
        "active": A,
        "inactive": I,
    }

    perfiles_colapsados = collapse_pde_profiles_moving_frame(
        pde=pde,
        selected_times=tiempos_colapso,
        z_centers=z_centers,
        anchor_mode="active_peak",
    )

    z = np.asarray(
        [fila["z"] for fila in perfiles_colapsados],
        dtype=float,
    )

    perfil_medio = np.asarray(
        [
            fila["active_density_mean"]
            for fila in perfiles_colapsados
        ],
        dtype=float,
    )

    perfil_sem = np.asarray(
        [
            fila["active_density_sem"]
            for fila in perfiles_colapsados
        ],
        dtype=float,
    )

    escala = float(np.max(perfil_medio))

    if escala <= 0.0:
        raise ValueError(
            "El perfil activo medio no tiene un máximo positivo."
        )

    perfil_normalizado = perfil_medio / escala
    sem_normalizado = perfil_sem / escala

    ajuste_colapsado = fit_pulse_tails(
        z,
        perfil_normalizado,
    )

    gamma_mas = abs(
        float(ajuste_colapsado["front_slope"])
    )

    gamma_menos = abs(
        float(ajuste_colapsado["back_slope"])
    )

    longitud_mas = float(
        ajuste_colapsado["front_length"]
    )

    longitud_menos = float(
        ajuste_colapsado["back_length"]
    )

    # Esta razón equivale a gamma_menos / gamma_mas.
    cociente = float(
        ajuste_colapsado["asymmetry_ratio"]
    )

    cociente_teorico = np.sqrt(2.0) - 1.0

    error_relativo = (
        abs(cociente - cociente_teorico)
        / cociente_teorico
    )

    print("\n=== PERFIL COLAPSADO EN EL MARCO MÓVIL ===")
    print(f"Tiempos utilizados: {tiempos_colapso}")

    print(
        "gamma_- = "
        f"{gamma_menos:.8f}"
    )

    print(
        "gamma_+ = "
        f"{gamma_mas:.8f}"
    )

    print(
        "longitud trasera = "
        f"{longitud_menos:.8f}"
    )

    print(
        "longitud delantera = "
        f"{longitud_mas:.8f}"
    )

    print(
        "gamma_- / gamma_+ = "
        f"{cociente:.8f}"
    )

    print(
        "sqrt(2) - 1 = "
        f"{cociente_teorico:.8f}"
    )

    print(
        "Error porcentual = "
        f"{100.0 * error_relativo:.4f}%"
    )

    print(
        "R² cola trasera = "
        f"{ajuste_colapsado['back_r2_log']:.8f}"
    )

    print(
        "R² cola delantera = "
        f"{ajuste_colapsado['front_r2_log']:.8f}"
    )

    print(
        "RMSE log cola trasera = "
        f"{ajuste_colapsado['back_rmse_log']:.8f}"
    )

    print(
        "RMSE log cola delantera = "
        f"{ajuste_colapsado['front_rmse_log']:.8f}"
    )

    # Tabla resumen del análisis colapsado.
    tabla_colapsada = pd.DataFrame(
        [
            {
                "tiempos": ";".join(
                    str(valor)
                    for valor in tiempos_colapso
                ),
                "banda_min": 0.05,
                "banda_max": 0.65,
                "gamma_menos": gamma_menos,
                "gamma_mas": gamma_mas,
                "longitud_menos": longitud_menos,
                "longitud_mas": longitud_mas,
                "cociente": cociente,
                "cociente_teorico": cociente_teorico,
                "error_relativo": error_relativo,
                "error_porcentual": 100.0 * error_relativo,
                "r2_menos": ajuste_colapsado[
                    "back_r2_log"
                ],
                "r2_mas": ajuste_colapsado[
                    "front_r2_log"
                ],
                "rmse_log_menos": ajuste_colapsado[
                    "back_rmse_log"
                ],
                "rmse_log_mas": ajuste_colapsado[
                    "front_rmse_log"
                ],
                "n_menos": ajuste_colapsado[
                    "back_n"
                ],
                "n_mas": ajuste_colapsado[
                    "front_n"
                ],
                "sem_maximo": float(
                    np.max(sem_normalizado)
                ),
            }
        ]
    )

    tabla_colapsada.to_csv(
        RESULTADOS / "asimetria_pulso_colapsado.csv",
        index=False,
    )

    # Guarda también el perfil medio completo.
    tabla_perfil = pd.DataFrame(
        {
            "z": z,
            "activo_medio": perfil_medio,
            "activo_medio_normalizado": perfil_normalizado,
            "sem": perfil_sem,
            "sem_normalizado": sem_normalizado,
        }
    )

    tabla_perfil.to_csv(
        RESULTADOS / "perfil_pulso_colapsado.csv",
        index=False,
    )

    graficar_pulso_colapsado(
        z=z,
        perfil_normalizado=perfil_normalizado,
        sem_normalizado=sem_normalizado,
        ajuste=ajuste_colapsado,
        ruta=(
            FIGURAS
            / "campo_medio_asimetria_pulso_colapsado.png"
        ),
    )

    # Tabla reducida para el análisis de sensibilidad de la memoria.  
    casos_sensibilidad = ( ("Tiempo", "t=200", 200.0, 0.02, 0.25), 
                          ("Tiempo", "t=250", 250.0, 0.02, 0.25), 
                          ("Tiempo", "t=300", 300.0, 0.02, 0.25), 
                          ("Banda de ajuste", "[0.02, 0.20]", 300.0, 0.02, 0.20), 
                          ("Banda de ajuste", "[0.05, 0.30]", 300.0, 0.05, 0.30), ) 
    
    filas_sensibilidad = [] 
    for factor, caso, tiempo_objetivo, fraccion_min, fraccion_max in casos_sensibilidad: 
        mascara = ( np.isclose(tabla["t"], tiempo_objetivo) & np.isclose(tabla["fraccion_min"], 
                    fraccion_min) & np.isclose(tabla["fraccion_max"], fraccion_max) ) 
        fila = tabla.loc[mascara].copy() 
        if len(fila) != 1: 
            raise RuntimeError( "No se encontró una fila única para el caso de sensibilidad: " f"{factor}, {caso}." ) 
       
        fila.insert(0, "caso", caso) 
        fila.insert(0, "factor", factor) 
        filas_sensibilidad.append(fila) 
        
        tabla_sensibilidad = pd.concat( filas_sensibilidad, ignore_index=True, ) 
        
        columnas_sensibilidad = [ "factor", "caso", "gamma_menos", "gamma_mas", "cociente", 
                                 "cociente_teorico", "error_porcentual", "r2_menos", "r2_mas", "n_menos", 
                                 "n_mas", ] 
        
        tabla_sensibilidad.to_csv( RESULTADOS / "asimetria_pulso_sensibilidad.csv", index=False, ) 
        
        print("\nTabla de sensibilidad de la asimetría:") 
        print( tabla_sensibilidad[ columnas_sensibilidad ].to_string(index=False) )



if __name__ == "__main__":
    main()
