

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))


from src.campo_medio.metricas import (
    ajustar_velocidad_trayectoria,
    trayectoria_pico,
)


RESULTADOS = RAIZ_PROYECTO / "resultados" / "campo_medio_cap4"
FIGURAS = RESULTADOS / "figuras"
ARCHIVO_SOLUCION = RESULTADOS / "solucion_pulso_campo_medio.npz"

FIGURAS.mkdir(parents=True, exist_ok=True)


def graficar_trayectoria_velocidad(
    t: np.ndarray,
    posiciones: np.ndarray,
    ajuste: dict,
    velocidad_teorica: float,
    ruta: Path,
) -> None:
    mascara = ajuste["mascara"]
    t_ajuste = ajuste["t_ajuste"]
    prediccion = ajuste["prediccion"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        t,
        posiciones,
        label=r"Trayectoria numérica $x_{\max}(t)$",
    )

    ax.plot(
        t_ajuste,
        prediccion,
        "--",
        label=(
            fr"Ajuste: $V_{{\mathrm{{num}}}}={ajuste['velocidad']:.6f}$, "
            fr"$R^2={ajuste['r2']:.6f}$"
        ),
    )

    t_inicial = float(t_ajuste[0])
    x_inicial = float(posiciones[mascara][0])

    recta_teorica = (
        x_inicial
        + velocidad_teorica * (t_ajuste - t_inicial)
    )

    ax.plot(
        t_ajuste,
        recta_teorica,
        ":",
        label=fr"Pendiente teórica $V^*={velocidad_teorica:.6f}$",
    )

    ax.axvspan(
        ajuste["t_min"],
        ajuste["t_max"],
        alpha=0.1,
        label="Ventana de ajuste",
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_{\max}(t)$")
    ax.set_title("Trayectoria del máximo del pulso activo")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def graficar_residuos(
    ajuste: dict,
    ruta: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(
        ajuste["t_ajuste"],
        ajuste["residuos"],
    )

    ax.axhline(
        0.0,
        linestyle="--",
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(
        r"$x_{\max}(t)-(V_{\mathrm{num}}t+b)$"
    )
    ax.set_title(
        "Residuos del ajuste lineal de la trayectoria"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not ARCHIVO_SOLUCION.exists():
        raise FileNotFoundError(
            "No se encuentra la solución del pulso:\n"
            f"{ARCHIVO_SOLUCION}\n"
            "Ejecuta primero simular_pulso_campo_medio.py."
        )

    print("=== ANÁLISIS DE LA VELOCIDAD DEL PULSO ===")
    print(f"Cargando: {ARCHIVO_SOLUCION}")

    with np.load(
        ARCHIVO_SOLUCION,
        allow_pickle=False,
    ) as datos:
        t = np.asarray(datos["t"], dtype=float)
        x = np.asarray(datos["x"], dtype=float)

        if "posiciones" in datos.files:
            posiciones = np.asarray(
                datos["posiciones"],
                dtype=float,
            )
        else:
            A = np.asarray(datos["A"], dtype=float)
            posiciones = trayectoria_pico(x, A)

    D = 1.0
    rb = 0.1

    velocidad_teorica = 2.0 * np.sqrt(D * rb)

    ventanas = (
        (100.0, 300.0),
        (150.0, 300.0),
        (200.0, 300.0),
    )

    filas = []
    ajustes = {}

    for t_min, t_max in ventanas:
        ajuste = ajustar_velocidad_trayectoria(
            t=t,
            posiciones=posiciones,
            t_min=t_min,
            t_max=t_max,
        )

        ajustes[(t_min, t_max)] = ajuste

        error_absoluto = abs(
            ajuste["velocidad"] - velocidad_teorica
        )
        error_relativo = error_absoluto / velocidad_teorica

        filas.append(
            {
                "t_min": t_min,
                "t_max": t_max,
                "velocidad_numerica": ajuste["velocidad"],
                "velocidad_teorica": velocidad_teorica,
                "error_absoluto": error_absoluto,
                "error_relativo": error_relativo,
                "error_porcentual": 100.0 * error_relativo,
                "ordenada": ajuste["ordenada"],
                "r2": ajuste["r2"],
                "n_puntos": ajuste["n_puntos"],
            }
        )

    tabla = pd.DataFrame(filas)

    tabla.to_csv(
        RESULTADOS / "velocidad_pulso.csv",
        index=False,
    )

    print("\nResultados de los ajustes:")
    print(
        tabla[
            [
                "t_min",
                "t_max",
                "velocidad_numerica",
                "velocidad_teorica",
                "error_porcentual",
                "r2",
                "n_puntos",
            ]
        ].to_string(index=False)
    )

    ventana_principal = (200.0, 300.0)
    ajuste_principal = ajustes[ventana_principal]

    graficar_trayectoria_velocidad(
        t=t,
        posiciones=posiciones,
        ajuste=ajuste_principal,
        velocidad_teorica=velocidad_teorica,
        ruta=FIGURAS / "campo_medio_velocidad_pulso.png",
    )

    graficar_residuos(
        ajuste=ajuste_principal,
        ruta=FIGURAS / "campo_medio_residuos_velocidad.png",
    )

    error_principal = (
        abs(
            ajuste_principal["velocidad"]
            - velocidad_teorica
        )
        / velocidad_teorica
    )

    print(
        "\nVentana principal: "
        f"[{ajuste_principal['t_min']:.0f}, "
        f"{ajuste_principal['t_max']:.0f}]"
    )
    print(
        "Velocidad numérica: "
        f"{ajuste_principal['velocidad']:.8f}"
    )
    print(
        "Velocidad teórica: "
        f"{velocidad_teorica:.8f}"
    )
    print(
        "Ordenada del ajuste: "
        f"{ajuste_principal['ordenada']:.8f}"
    )
    print(
        "R²: "
        f"{ajuste_principal['r2']:.8f}"
    )
    print(
        "Error relativo: "
        f"{error_principal:.8f}"
    )
    print(
        "Error porcentual: "
        f"{100.0 * error_principal:.4f}%"
    )
    print(
        "\nResultados guardados en: "
        f"{RESULTADOS.resolve()}"
    )


if __name__ == "__main__":
    main()
