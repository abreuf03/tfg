from paquetes_cerrados.hannezo2017_fig3_pde_barw.reproducir_fig3_hannezo_pde_barw import (
    closest_index,
    fit_speed,
    fit_speed_with_mask,
    pde_front_from_active,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

RESULTADOS = (
    RAIZ_PROYECTO
    / "resultados"
    / "comparacion_barw_pde"
)

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True,
)

ARCHIVO_FRENTE_BARW = ( RESULTADOS / "barw_estadisticas_frente.csv" ) 
ARCHIVO_PICO_BARW = ( RESULTADOS / "barw_estadisticas_pico.csv" ) 
ARCHIVO_PDE = ( RAIZ_PROYECTO / "resultados" / "campo_medio_cap4" / "solucion_pulso_campo_medio.npz" ) 
FIGURAS = RESULTADOS / "figuras" 
FIGURAS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    archivo_frente_barw = (
        RESULTADOS
        / "barw_estadisticas_frente.csv"
    )

    archivo_pico_barw = (
        RESULTADOS
        / "barw_estadisticas_pico.csv"
    )

    archivo_pde = (
        RAIZ_PROYECTO
        / "resultados"
        / "campo_medio_cap4"
        / "solucion_pulso_campo_medio.npz"
    )

    figuras = RESULTADOS / "figuras"
    figuras.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Comprobación de archivos
    # ----------------------------------------------------------
    for archivo in (
        archivo_frente_barw,
        archivo_pico_barw,
        archivo_pde,
    ):
        if not archivo.exists():
            raise FileNotFoundError(
                "No se encuentra el archivo:\n"
                f"{archivo}"
            )

    print(
        "=== COMPARACIÓN DE VELOCIDADES BARW--PDE ==="
    )
    print(f"Frente BARW: {archivo_frente_barw}")
    print(f"Pico BARW: {archivo_pico_barw}")
    print(f"Solución PDE: {archivo_pde}")

    # ----------------------------------------------------------
    # Carga de resultados BARW
    # ----------------------------------------------------------
    front_stats = pd.read_csv(
        archivo_frente_barw
    )

    peak_stats = pd.read_csv(
        archivo_pico_barw
    )

    columnas_frente = {
        "time",
        "front_mean_all",
        "front_mean_alive",
        "front_sem_alive",
        "n_alive",
    }

    columnas_pico = {
        "time",
        "peak_mean_alive",
        "peak_sem_alive",
        "n_alive",
    }

    if not columnas_frente.issubset(
        front_stats.columns
    ):
        faltantes = (
            columnas_frente
            - set(front_stats.columns)
        )

        raise ValueError(
            "Faltan columnas en el archivo del frente: "
            + ", ".join(sorted(faltantes))
        )

    if not columnas_pico.issubset(
        peak_stats.columns
    ):
        faltantes = (
            columnas_pico
            - set(peak_stats.columns)
        )

        raise ValueError(
            "Faltan columnas en el archivo del pico: "
            + ", ".join(sorted(faltantes))
        )

    # ----------------------------------------------------------
    # Carga de la solución PDE
    # ----------------------------------------------------------
    with np.load(
        archivo_pde,
        allow_pickle=False,
    ) as datos:
        x_pde = np.asarray(
            datos["x"],
            dtype=float,
        )

        t_pde = np.asarray(
            datos["t"],
            dtype=float,
        )

        A_pde = np.asarray(
            datos["A"],
            dtype=float,
        )

        if "posiciones" in datos.files:
            pico_pde_completo = np.asarray(
                datos["posiciones"],
                dtype=float,
            )
        else:
            indices_maximos = np.argmax(
                A_pde,
                axis=1,
            )

            pico_pde_completo = x_pde[
                indices_maximos
            ]

    forma_esperada = (
        t_pde.size,
        x_pde.size,
    )

    if A_pde.shape != forma_esperada:
        raise ValueError(
            f"A tiene forma {A_pde.shape}, "
            f"pero se esperaba {forma_esperada}."
        )

    if pico_pde_completo.size != t_pde.size:
        raise ValueError(
            "La trayectoria del pico PDE no tiene "
            "la misma longitud que el vector temporal."
        )

    # ----------------------------------------------------------
    # Trayectoria del frente BARW
    # ----------------------------------------------------------
    tiempos_frente_barw = front_stats[
        "time"
    ].to_numpy(dtype=float)

    frente_barw_todas = front_stats[
        "front_mean_all"
    ].to_numpy(dtype=float)

    frente_barw_vivas = front_stats[
        "front_mean_alive"
    ].to_numpy(dtype=float)

    frente_barw_sem = front_stats[
        "front_sem_alive"
    ].to_numpy(dtype=float)

    n_vivas_frente = front_stats[
        "n_alive"
    ].to_numpy(dtype=int)

    # ----------------------------------------------------------
    # Frente PDE en los mismos tiempos del BARW
    # ----------------------------------------------------------
    indices_frente_pde = np.asarray(
        [
            closest_index(
                t_pde,
                tiempo,
            )
            for tiempo in tiempos_frente_barw
        ],
        dtype=int,
    )

    tiempos_frente_pde = t_pde[
        indices_frente_pde
    ]

    frente_pde = np.asarray(
        [
            pde_front_from_active(
                x_pde,
                A_pde[indice],
                relative_threshold=0.08,
            )
            for indice in indices_frente_pde
        ],
        dtype=float,
    )

    # ----------------------------------------------------------
    # Trayectoria del pico BARW
    # ----------------------------------------------------------
    tiempos_pico_barw = peak_stats[
        "time"
    ].to_numpy(dtype=float)

    pico_barw = peak_stats[
        "peak_mean_alive"
    ].to_numpy(dtype=float)

    pico_barw_sem = peak_stats[
        "peak_sem_alive"
    ].to_numpy(dtype=float)

    n_vivas_pico = peak_stats[
        "n_alive"
    ].to_numpy(dtype=int)

    # ----------------------------------------------------------
    # Pico PDE en los mismos tiempos del BARW
    # ----------------------------------------------------------
    indices_pico_pde = np.asarray(
        [
            closest_index(
                t_pde,
                tiempo,
            )
            for tiempo in tiempos_pico_barw
        ],
        dtype=int,
    )

    tiempos_pico_pde = t_pde[
        indices_pico_pde
    ]

    pico_pde = pico_pde_completo[
        indices_pico_pde
    ]

    # ----------------------------------------------------------
    # Configuración de los ajustes
    # ----------------------------------------------------------
    ventanas = (
        (60.0, 300.0),
        (100.0, 300.0),
        (150.0, 300.0),
        (200.0, 300.0),
    )

    minimo_supervivientes = 5

    longitud = float(x_pde[-1])
    limite_espacial = 0.94 * longitud

    D = 1.0
    rb = 0.1

    velocidad_teorica = (
        2.0 * np.sqrt(D * rb)
    )

    filas = []
    ajustes = {}

    # ----------------------------------------------------------
    # Ajustes en las distintas ventanas temporales
    # ----------------------------------------------------------
    for t_min, t_max in ventanas:
        # ======================================================
        # Frente BARW
        # ======================================================
        mascara_frente_barw = (
            (n_vivas_frente >= minimo_supervivientes)
            & (tiempos_frente_barw <= t_max)
        )

        ajuste_frente_barw = (
            fit_speed_with_mask(
                tiempos_frente_barw,
                frente_barw_vivas,
                mascara_frente_barw,
                t_min=t_min,
                x_max=limite_espacial,
            )
        )

        # ======================================================
        # Frente PDE
        # ======================================================
        mascara_frente_pde = (
            tiempos_frente_pde <= t_max
        )

        ajuste_frente_pde = fit_speed(
            tiempos_frente_pde[
                mascara_frente_pde
            ],
            frente_pde[
                mascara_frente_pde
            ],
            t_min=t_min,
            x_max=limite_espacial,
        )

        ajustes[
            ("frente", t_min, t_max, "barw")
        ] = ajuste_frente_barw

        ajustes[
            ("frente", t_min, t_max, "pde")
        ] = ajuste_frente_pde

        velocidad_barw = float(
            ajuste_frente_barw["speed"]
        )

        velocidad_pde = float(
            ajuste_frente_pde["speed"]
        )

        error_absoluto = abs(
            velocidad_pde
            - velocidad_barw
        )

        if (
            np.isfinite(velocidad_barw)
            and velocidad_barw != 0.0
        ):
            error_relativo = (
                error_absoluto
                / abs(velocidad_barw)
            )
        else:
            error_relativo = np.nan

        filas.append(
            {
                "observable": "frente",
                "t_min": t_min,
                "t_max": t_max,
                "velocidad_barw": velocidad_barw,
                "velocidad_pde": velocidad_pde,
                "velocidad_teorica": velocidad_teorica,
                "r2_barw": ajuste_frente_barw["r2"],
                "r2_pde": ajuste_frente_pde["r2"],
                "ci95_barw": ajuste_frente_barw[
                    "speed_ci95"
                ],
                "ci95_pde": ajuste_frente_pde[
                    "speed_ci95"
                ],
                "error_absoluto_barw_pde": (
                    error_absoluto
                ),
                "error_relativo_barw_pde": (
                    error_relativo
                ),
                "error_porcentual_barw_pde": (
                    100.0 * error_relativo
                ),
                "error_barw_teoria_porcentual": (
                    100.0
                    * abs(
                        velocidad_barw
                        - velocidad_teorica
                    )
                    / velocidad_teorica
                ),
                "error_pde_teoria_porcentual": (
                    100.0
                    * abs(
                        velocidad_pde
                        - velocidad_teorica
                    )
                    / velocidad_teorica
                ),
                "n_barw": ajuste_frente_barw["n"],
                "n_pde": ajuste_frente_pde["n"],
            }
        )

        # ======================================================
        # Pico BARW
        # ======================================================
        mascara_pico_barw = (
            (n_vivas_pico >= minimo_supervivientes)
            & (tiempos_pico_barw <= t_max)
        )

        ajuste_pico_barw = (
            fit_speed_with_mask(
                tiempos_pico_barw,
                pico_barw,
                mascara_pico_barw,
                t_min=t_min,
                x_max=limite_espacial,
            )
        )

        # ======================================================
        # Pico PDE
        # ======================================================
        mascara_pico_pde = (
            tiempos_pico_pde <= t_max
        )

        ajuste_pico_pde = fit_speed(
            tiempos_pico_pde[
                mascara_pico_pde
            ],
            pico_pde[
                mascara_pico_pde
            ],
            t_min=t_min,
            x_max=limite_espacial,
        )

        ajustes[
            ("pico", t_min, t_max, "barw")
        ] = ajuste_pico_barw

        ajustes[
            ("pico", t_min, t_max, "pde")
        ] = ajuste_pico_pde

        velocidad_barw = float(
            ajuste_pico_barw["speed"]
        )

        velocidad_pde = float(
            ajuste_pico_pde["speed"]
        )

        error_absoluto = abs(
            velocidad_pde
            - velocidad_barw
        )

        if (
            np.isfinite(velocidad_barw)
            and velocidad_barw != 0.0
        ):
            error_relativo = (
                error_absoluto
                / abs(velocidad_barw)
            )
        else:
            error_relativo = np.nan

        filas.append(
            {
                "observable": "pico",
                "t_min": t_min,
                "t_max": t_max,
                "velocidad_barw": velocidad_barw,
                "velocidad_pde": velocidad_pde,
                "velocidad_teorica": velocidad_teorica,
                "r2_barw": ajuste_pico_barw["r2"],
                "r2_pde": ajuste_pico_pde["r2"],
                "ci95_barw": ajuste_pico_barw[
                    "speed_ci95"
                ],
                "ci95_pde": ajuste_pico_pde[
                    "speed_ci95"
                ],
                "error_absoluto_barw_pde": (
                    error_absoluto
                ),
                "error_relativo_barw_pde": (
                    error_relativo
                ),
                "error_porcentual_barw_pde": (
                    100.0 * error_relativo
                ),
                "error_barw_teoria_porcentual": (
                    100.0
                    * abs(
                        velocidad_barw
                        - velocidad_teorica
                    )
                    / velocidad_teorica
                ),
                "error_pde_teoria_porcentual": (
                    100.0
                    * abs(
                        velocidad_pde
                        - velocidad_teorica
                    )
                    / velocidad_teorica
                ),
                "n_barw": ajuste_pico_barw["n"],
                "n_pde": ajuste_pico_pde["n"],
            }
        )

    # ----------------------------------------------------------
    # Guardado de la tabla comparativa
    # ----------------------------------------------------------
    tabla = pd.DataFrame(filas)

    tabla.to_csv(
        RESULTADOS
        / "comparacion_velocidades_barw_pde.csv",
        index=False,
    )

    print("\nResultados de la comparación:")
    print(
        tabla[
            [
                "observable",
                "t_min",
                "t_max",
                "velocidad_barw",
                "velocidad_pde",
                "error_porcentual_barw_pde",
                "r2_barw",
                "r2_pde",
                "n_barw",
                "n_pde",
            ]
        ].to_string(index=False)
    )

    # ----------------------------------------------------------
    # Ventana principal para las figuras
    # ----------------------------------------------------------
    ventana_principal = (
        60.0,
        300.0,
    )

    # ==========================================================
    # Figura del frente
    # ==========================================================
    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.plot(
        tiempos_frente_barw,
        frente_barw_todas,
        ":",
        label="BARW: media global",
    )

    ax.plot(
        tiempos_frente_barw,
        frente_barw_vivas,
        "--",
        label="BARW: supervivientes",
    )

    ax.fill_between(
        tiempos_frente_barw,
        frente_barw_vivas - frente_barw_sem,
        frente_barw_vivas + frente_barw_sem,
        alpha=0.2,
        label="BARW: error estándar",
    )

    ax.plot(
        tiempos_frente_pde,
        frente_pde,
        label=(
            r"PDE: frente "
            r"$a\geq0{,}08a_{\max}$"
        ),
    )

    ajuste_barw = ajustes[
        (
            "frente",
            ventana_principal[0],
            ventana_principal[1],
            "barw",
        )
    ]

    ajuste_pde = ajustes[
        (
            "frente",
            ventana_principal[0],
            ventana_principal[1],
            "pde",
        )
    ]

    mascara_fit_barw = (
        (tiempos_frente_barw >= ventana_principal[0])
        & (tiempos_frente_barw <= ventana_principal[1])
        & (n_vivas_frente >= minimo_supervivientes)
        & np.isfinite(frente_barw_vivas)
        & (frente_barw_vivas > 10.0)
        & (frente_barw_vivas < limite_espacial)
    )

    mascara_fit_pde = (
        (tiempos_frente_pde >= ventana_principal[0])
        & (tiempos_frente_pde <= ventana_principal[1])
        & np.isfinite(frente_pde)
        & (frente_pde > 10.0)
        & (frente_pde < limite_espacial)
    )

    ax.plot(
        tiempos_frente_barw[
            mascara_fit_barw
        ],
        (
            ajuste_barw["speed"]
            * tiempos_frente_barw[
                mascara_fit_barw
            ]
            + ajuste_barw["intercept"]
        ),
        "--",
        label=(
            "Ajuste BARW: "
            f"V={ajuste_barw['speed']:.4f}"
        ),
    )

    ax.plot(
        tiempos_frente_pde[
            mascara_fit_pde
        ],
        (
            ajuste_pde["speed"]
            * tiempos_frente_pde[
                mascara_fit_pde
            ]
            + ajuste_pde["intercept"]
        ),
        "-.",
        label=(
            "Ajuste PDE: "
            f"V={ajuste_pde['speed']:.4f}"
        ),
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_{\mathrm{front}}(t)$")
    ax.set_title(
        "Comparación de la posición del frente"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figuras
        / "comparacion_velocidad_frente_barw_pde.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ==========================================================
    # Figura del pico activo
    # ==========================================================
    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.plot(
        tiempos_pico_barw,
        pico_barw,
        "--",
        label="BARW: pico activo medio",
    )

    ax.fill_between(
        tiempos_pico_barw,
        pico_barw - pico_barw_sem,
        pico_barw + pico_barw_sem,
        alpha=0.2,
        label="BARW: error estándar",
    )

    ax.plot(
        tiempos_pico_pde,
        pico_pde,
        label="PDE: máximo activo",
    )

    ajuste_barw = ajustes[
        (
            "pico",
            ventana_principal[0],
            ventana_principal[1],
            "barw",
        )
    ]

    ajuste_pde = ajustes[
        (
            "pico",
            ventana_principal[0],
            ventana_principal[1],
            "pde",
        )
    ]

    mascara_fit_barw = (
        (tiempos_pico_barw >= ventana_principal[0])
        & (tiempos_pico_barw <= ventana_principal[1])
        & (n_vivas_pico >= minimo_supervivientes)
        & np.isfinite(pico_barw)
        & (pico_barw > 10.0)
        & (pico_barw < limite_espacial)
    )

    mascara_fit_pde = (
        (tiempos_pico_pde >= ventana_principal[0])
        & (tiempos_pico_pde <= ventana_principal[1])
        & np.isfinite(pico_pde)
        & (pico_pde > 10.0)
        & (pico_pde < limite_espacial)
    )

    ax.plot(
        tiempos_pico_barw[
            mascara_fit_barw
        ],
        (
            ajuste_barw["speed"]
            * tiempos_pico_barw[
                mascara_fit_barw
            ]
            + ajuste_barw["intercept"]
        ),
        "--",
        label=(
            "Ajuste BARW: "
            f"V={ajuste_barw['speed']:.4f}"
        ),
    )

    ax.plot(
        tiempos_pico_pde[
            mascara_fit_pde
        ],
        (
            ajuste_pde["speed"]
            * tiempos_pico_pde[
                mascara_fit_pde
            ]
            + ajuste_pde["intercept"]
        ),
        "-.",
        label=(
            "Ajuste PDE: "
            f"V={ajuste_pde['speed']:.4f}"
        ),
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_{\mathrm{peak}}(t)$")
    ax.set_title(
        "Comparación de la posición del pico activo"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figuras
        / "comparacion_velocidad_pico_barw_pde.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nResultados guardados en: "
        f"{RESULTADOS.resolve()}"
    )


if __name__ == "__main__":
    main()



