import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paquetes_cerrados.hannezo2017_fig3_pde_barw.reproducir_fig3_hannezo_pde_barw import (
    collapse_barw_profiles_moving_frame,
    collapse_pde_profiles_moving_frame,
    compute_barw_pde_comparison,
    normalized,
    smooth_profile,
)


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

ARCHIVO_BARW_HISTORIAL = ( RESULTADOS / "barw_historial.csv" )
ARCHIVO_BARW_CRUDOS = ( RESULTADOS / "barw_perfiles_crudos.csv" ) 
ARCHIVO_BARW_MEDIOS = ( RESULTADOS / "barw_perfiles_medios_vivas.csv" ) 
ARCHIVO_PDE = ( RAIZ_PROYECTO / "resultados" / "campo_medio_cap4" / "solucion_pulso_campo_medio.npz" ) 
FIGURAS = RESULTADOS / "figuras" 
FIGURAS.mkdir(parents=True, exist_ok=True)


ARCHIVO_BARW_RESUMEN = (
    RESULTADOS / "barw_resumen_semillas.csv"
)



def convertir_a_booleano( serie: pd.Series, ) -> pd.Series: 
    """ 
    Convierte de forma segura una columna leída desde CSV a booleanos. 
    Admite valores booleanos reales y cadenas como 'True' o 'False'. 
    """ 
    if pd.api.types.is_bool_dtype(serie): 
        return serie.astype(bool) 
    
    mapa = { "true": True, "false": False, "1": True, "0": False, } 
    resultado = ( serie .astype(str) .str.strip() .str.lower() .map(mapa) ) 
    if resultado.isna().any():
        valores_invalidos = serie[ resultado.isna() ].unique() 
        raise ValueError( "No se pudo interpretar la columna 'alive'. " f"Valores problemáticos: {valores_invalidos}" ) 
    
    return resultado.astype(bool)

def main() -> None:
    # ------------------------------------------------------------------
    # Comprobación de archivos
    # ------------------------------------------------------------------
    archivos_necesarios = (
        ARCHIVO_BARW_HISTORIAL,
        ARCHIVO_BARW_CRUDOS,
        ARCHIVO_PDE,
        ARCHIVO_BARW_RESUMEN,
    )

    for archivo in archivos_necesarios:
        if not archivo.exists():
            raise FileNotFoundError(
                "No se encuentra el archivo necesario:\n"
                f"{archivo}"
            )

    print("=== COMPARACIÓN DE DENSIDADES BARW--PDE ===")
    print(f"Historial BARW: {ARCHIVO_BARW_HISTORIAL}")
    print(f"Perfiles BARW: {ARCHIVO_BARW_CRUDOS}")
    print(f"Solución PDE: {ARCHIVO_PDE}")

    # ------------------------------------------------------------------
    # Carga de resultados BARW
    # ------------------------------------------------------------------
    historial_df = pd.read_csv(
        ARCHIVO_BARW_HISTORIAL
    )

    perfiles_crudos_df = pd.read_csv(
        ARCHIVO_BARW_CRUDOS
    )


    resumen_semillas_df = pd.read_csv(
        ARCHIVO_BARW_RESUMEN
    )


    columnas_historial = {
        "seed",
        "time",
        "x_front",
        "alive",
    }

    columnas_perfiles = {
        "seed",
        "time",
        "x",
        "active_density",
        "duct_density",
        "alive",
    }

    if not columnas_historial.issubset(
        historial_df.columns
    ):
        faltantes = (
            columnas_historial
            - set(historial_df.columns)
        )

        raise ValueError(
            "Faltan columnas en el historial BARW: "
            + ", ".join(sorted(faltantes))
        )

    if not columnas_perfiles.issubset(
        perfiles_crudos_df.columns
    ):
        faltantes = (
            columnas_perfiles
            - set(perfiles_crudos_df.columns)
        )

        raise ValueError(
            "Faltan columnas en los perfiles BARW: "
            + ", ".join(sorted(faltantes))
        )

    historial_df["alive"] = convertir_a_booleano(
        historial_df["alive"]
    )

    perfiles_crudos_df["alive"] = convertir_a_booleano(
        perfiles_crudos_df["alive"]
    )


    columnas_resumen = {
        "seed",
        "survived_to_T",
    }

    if not columnas_resumen.issubset(
        resumen_semillas_df.columns
    ):
        faltantes = (
            columnas_resumen
            - set(resumen_semillas_df.columns)
        )
        raise ValueError(
            "Faltan columnas en el resumen por semilla: "
            + ", ".join(sorted(faltantes))
        )

    resumen_semillas_df["survived_to_T"] = convertir_a_booleano(
        resumen_semillas_df["survived_to_T"]
    )

    semillas_cohorte = (
        resumen_semillas_df.loc[
            resumen_semillas_df["survived_to_T"],
            "seed",
        ]
        .to_numpy(dtype=int)
    )

    if semillas_cohorte.size < 2:
        raise ValueError(
            "La cohorte fija debe contener al menos dos semillas."
        )

    historial_cohorte_df = historial_df[
        historial_df["seed"].isin(semillas_cohorte)
    ].copy()

    perfiles_cohorte_df = perfiles_crudos_df[
        perfiles_crudos_df["seed"].isin(semillas_cohorte)
    ].copy()

    conteo_semillas_por_tiempo = (
        perfiles_cohorte_df
        .groupby("time")["seed"]
        .nunique()
    )

    if not np.all(
        conteo_semillas_por_tiempo.to_numpy()
        == semillas_cohorte.size
    ):
        raise RuntimeError(
            "Faltan perfiles de alguna semilla de la cohorte "
            "en uno o más tiempos."
        )

    if not perfiles_cohorte_df["alive"].all():
        raise RuntimeError(
            "Se ha incluido una realización no activa dentro "
            "de la cohorte fija."
        )

    print(
        "Cohorte fija para perfiles: "
        f"{semillas_cohorte.size} semillas supervivientes "
        "hasta T=300."
    )

    barw = {
        "history": historial_cohorte_df.to_dict(
            orient="records"
        ),
        "profile_rows_raw": perfiles_cohorte_df.to_dict(
            orient="records"
        ),
    }


    # ------------------------------------------------------------------
    # Carga de la solución PDE
    # ------------------------------------------------------------------
    with np.load(
        ARCHIVO_PDE,
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

        I_pde = np.asarray(
            datos["I"],
            dtype=float,
        )

    forma_esperada = (
        t_pde.size,
        x_pde.size,
    )

    if A_pde.shape != forma_esperada:
        raise ValueError(
            f"A tiene forma {A_pde.shape}, "
            f"pero se esperaba {forma_esperada}."
        )

    if I_pde.shape != forma_esperada:
        raise ValueError(
            f"I tiene forma {I_pde.shape}, "
            f"pero se esperaba {forma_esperada}."
        )

    if (
        not np.isfinite(A_pde).all()
        or not np.isfinite(I_pde).all()
    ):
        raise ValueError(
            "La solución PDE contiene valores no finitos."
        )

    pde = {
        "x": x_pde,
        "times": t_pde,
        "active": A_pde,
        "inactive": I_pde,
    }

    # ------------------------------------------------------------------
    # Colapso en el marco móvil y comparación
    # ------------------------------------------------------------------
    tiempos_colapso = (
        140,
        180,
        220,
        260,
        300,
    )

    comparacion = compute_barw_pde_comparison(
        pde=pde,
        barw=barw,
        selected_times=tiempos_colapso,
    )

    claves_necesarias = {
        "rows",
        "barw_collapse_rows",
        "pde_collapse_rows",
        "barw_duct_collapse_rows",
        "pde_duct_collapse_rows",
        "metrics",
    }

    faltantes = (
        claves_necesarias
        - set(comparacion.keys())
    )

    if faltantes:
        raise KeyError(
            "La función compute_barw_pde_comparison "
            "no ha devuelto las claves esperadas: "
            + ", ".join(sorted(faltantes))
        )

    tabla_comparacion = pd.DataFrame(
        comparacion["rows"]
    )

    tabla_barw_activo = pd.DataFrame(
        comparacion["barw_collapse_rows"]
    )

    tabla_pde_activo = pd.DataFrame(
        comparacion["pde_collapse_rows"]
    )

    tabla_barw_conductos = pd.DataFrame(
        comparacion["barw_duct_collapse_rows"]
    )

    tabla_pde_conductos = pd.DataFrame(
        comparacion["pde_duct_collapse_rows"]
    )

    metricas = comparacion["metrics"]

    if tabla_comparacion.empty:
        raise ValueError(
            "La comparación no ha generado ningún perfil."
        )

    # ------------------------------------------------------------------
    # Guardado de resultados numéricos
    # ------------------------------------------------------------------
    tabla_comparacion.to_csv(
        RESULTADOS
        / "comparacion_perfiles_barw_pde.csv",
        index=False,
    )

    tabla_barw_activo.to_csv(
        RESULTADOS
        / "barw_perfil_activo_colapsado.csv",
        index=False,
    )

    tabla_pde_activo.to_csv(
        RESULTADOS
        / "pde_perfil_activo_colapsado.csv",
        index=False,
    )

    tabla_barw_conductos.to_csv(
        RESULTADOS
        / "barw_perfil_conductos_colapsado.csv",
        index=False,
    )

    tabla_pde_conductos.to_csv(
        RESULTADOS
        / "pde_perfil_conductos_colapsado.csv",
        index=False,
    )

    pd.DataFrame(
        [metricas]
    ).to_csv(
        RESULTADOS
        / "metricas_comparacion_perfiles.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Comprobación de columnas
    # ------------------------------------------------------------------
    columnas_comparacion = {
        "z_from_front",
        "barw_active_raw_norm",
        "barw_active_norm",
        "barw_active_sem_norm",
        "pde_active_norm",
        "barw_duct_raw_norm",
        "barw_duct_norm",
        "barw_duct_sem_norm",
        "pde_inactive_norm",
    }

    if not columnas_comparacion.issubset(
        tabla_comparacion.columns
    ):
        faltantes = (
            columnas_comparacion
            - set(tabla_comparacion.columns)
        )

        raise ValueError(
            "Faltan columnas en la tabla de comparación: "
            + ", ".join(sorted(faltantes))
        )

    # ------------------------------------------------------------------
    # Extracción de arrays
    # ------------------------------------------------------------------
    z = tabla_comparacion[
        "z_from_front"
    ].to_numpy(dtype=float)

    activo_barw_crudo = tabla_comparacion[
        "barw_active_raw_norm"
    ].to_numpy(dtype=float)

    activo_barw = tabla_comparacion[
        "barw_active_norm"
    ].to_numpy(dtype=float)

    activo_barw_sem = tabla_comparacion[
        "barw_active_sem_norm"
    ].to_numpy(dtype=float)

    activo_pde = tabla_comparacion[
        "pde_active_norm"
    ].to_numpy(dtype=float)

    conductos_barw_crudo = tabla_comparacion[
        "barw_duct_raw_norm"
    ].to_numpy(dtype=float)

    conductos_barw = tabla_comparacion[
        "barw_duct_norm"
    ].to_numpy(dtype=float)

    conductos_barw_sem = tabla_comparacion[
        "barw_duct_sem_norm"
    ].to_numpy(dtype=float)

    conductos_pde = tabla_comparacion[
        "pde_inactive_norm"
    ].to_numpy(dtype=float)

    

    # ------------------------------------------------------------------
    # Figura de la densidad activa
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.plot(
        z,
        activo_barw_crudo,
        linestyle=":",
        alpha=0.65,
        label="BARW sin suavizar",
    )

    ax.plot(
        z,
        activo_barw,
        linestyle="--",
        label="BARW suavizado",
    )

    ax.fill_between(
        z,
        np.maximum(
            activo_barw - activo_barw_sem,
            0.0,
        ),
        activo_barw + activo_barw_sem,
        alpha=0.20,
        label="BARW: error estándar",
    )

    ax.plot(
        z,
        activo_pde,
        label="PDE",
    )

    ax.axvline(
        0.0,
        linestyle=":",
        linewidth=1.2,
        label="Máximo activo",
    )

    ax.set_xlim(
        -75.0,
        35.0,
    )

    ax.set_xlabel(
        r"$z=x-x_{\mathrm{peak}}(t)$"
    )

    ax.set_ylabel(
        "Densidad activa normalizada"
    )

    ax.set_title(
        "Comparación de la densidad de puntas activas"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURAS
        / "comparacion_densidad_activa_barw_pde.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura de la densidad de conductos
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.plot(
        z,
        conductos_barw_crudo,
        linestyle=":",
        alpha=0.65,
        label="BARW sin suavizar",
    )

    ax.plot(
        z,
        conductos_barw,
        linestyle="--",
        label="BARW suavizado",
    )

    ax.fill_between(
        z,
        np.maximum(
            conductos_barw - conductos_barw_sem,
            0.0,
        ),
        conductos_barw + conductos_barw_sem,
        alpha=0.20,
        label="BARW: error estándar",
    )

    ax.plot(
        z,
        conductos_pde,
        label="PDE",
    )

    ax.axvline(
        0.0,
        linestyle=":",
        linewidth=1.2,
        label="Frente de conductos",
    )

    ax.set_xlim(
        -155.0,
        15.0,
    )

    ax.set_xlabel(
        r"$z=x-x_{\mathrm{front}}^{\mathrm{duct}}(t)$"
    )

    ax.set_ylabel(
        "Densidad de conductos normalizada"
    )

    ax.set_title(
        "Comparación de la densidad de conductos"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURAS
        / "comparacion_densidad_conductos_barw_pde.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------------------
    # Resumen por consola
    # ------------------------------------------------------------------
    print(
        "\nTiempos utilizados en el colapso: "
        f"{tiempos_colapso}"
    )

    print(
        "\n=== MÉTRICAS DE LOS PERFILES NORMALIZADOS ==="
    )

    print("\nPerfil activo:")
    print(
        "  RMSE = "
        f"{metricas['rmse_active_norm']:.6f}"
    )
    print(
        "  Correlación = "
        f"{metricas['corr_active_norm']:.6f}"
    )

    print("\nPerfil de conductos:")
    print(
        "  RMSE = "
        f"{metricas['rmse_duct_norm']:.6f}"
    )
    print(
        "  Correlación = "
        f"{metricas['corr_duct_norm']:.6f}"
    )

    print(
        "\nBins utilizados para el perfil activo: "
        f"{metricas['n_active_bins_used']}"
    )

    print(
        "Bins utilizados para los conductos: "
        f"{metricas['n_duct_bins_used']}"
    )

    print(
        "\nResultados guardados en: "
        f"{RESULTADOS.resolve()}"
    )


if __name__ == "__main__":
    main()



