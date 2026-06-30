import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from paquetes_cerrados.hannezo2017_fig3_pde_barw.reproducir_fig3_hannezo_pde_barw import (
    BARWEnsembleConfig,
    aggregate_front_statistics,
    barw_profile_from_state,
    fit_speed,
    fit_speed_with_mask,
    grouped_profile_rows,
)
from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW


from pathlib import Path

minimo_supervivientes = 5


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


def guardar_resultados_barw(
    resultados: dict[str, object],
) -> None:
    """Guarda los resultados principales del ensemble BARW."""

    pd.DataFrame(
        resultados["history"]
    ).to_csv(
        RESULTADOS / "barw_historial.csv",
        index=False,
    )

    pd.DataFrame(
        resultados["front_stats"]
    ).to_csv(
        RESULTADOS / "barw_estadisticas_frente.csv",
        index=False,
    )

    pd.DataFrame(
        resultados["peak_stats"]
    ).to_csv(
        RESULTADOS / "barw_estadisticas_pico.csv",
        index=False,
    )

    pd.DataFrame(
        resultados["profile_rows_raw"]
    ).to_csv(
        RESULTADOS / "barw_perfiles_crudos.csv",
        index=False,
    )

    pd.DataFrame(
        resultados["profile_rows_mean"]
    ).to_csv(
        RESULTADOS / "barw_perfiles_medios_todas.csv",
        index=False,
    )

    pd.DataFrame(
        resultados["profile_rows_mean_alive"]
    ).to_csv(
        RESULTADOS / "barw_perfiles_medios_vivas.csv",
        index=False,
    )

    pd.DataFrame(
        resultados["seed_summary"]
    ).to_csv(
        RESULTADOS / "barw_resumen_semillas.csv",
        index=False,
    )

    pd.DataFrame(
        resultados["final_segments"]
    ).to_csv(
        RESULTADOS / "barw_segmentos_semilla_referencia.csv",
        index=False,
    )

    pd.DataFrame(
        resultados["final_tips"]
    ).to_csv(
        RESULTADOS / "barw_puntas_semilla_referencia.csv",
        index=False,
    )

    # Los diccionarios de ajuste pueden contener vectores.
    # Se guardan únicamente sus valores escalares.
    for nombre in (
        "speed_fit_front_all",
        "speed_fit_front_alive",
        "speed_fit_peak_alive",
    ):
        ajuste = resultados[nombre]

        ajuste_escalar = {
            clave: valor
            for clave, valor in ajuste.items()
            if np.isscalar(valor)
        }

        pd.DataFrame(
            [ajuste_escalar]
        ).to_csv(
            RESULTADOS / f"{nombre}.csv",
            index=False,
        )



def posicion_pico_activo(
    sim: SimulacionBARW,
    cfg: BARWEnsembleConfig,
) -> float:
    """Calcula la posición del máximo del perfil activo longitudinal."""

    posiciones_x = np.asarray(
        [
            punta.x
            for punta in sim.puntas
            if punta.activa
        ],
        dtype=float,
    )

    if posiciones_x.size == 0:
        return np.nan

    bordes = np.linspace(
        0.0,
        cfg.Lx,
        cfg.n_bins + 1,
    )

    centros = 0.5 * (
        bordes[:-1]
        + bordes[1:]
    )

    conteos, _ = np.histogram(
        posiciones_x,
        bins=bordes,
    )

    indice_pico = int(np.argmax(conteos))

    return float(centros[indice_pico])


def estadisticas_pico_activo(
    historial: list[dict[str, object]],
    cfg: BARWEnsembleConfig,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    """Calcula la trayectoria media del pico activo y ajusta su velocidad."""

    historial_df = pd.DataFrame(historial)

    datos_vivos = historial_df[
        historial_df["alive"]
        & historial_df["x_peak"].notna()
    ].copy()

    if datos_vivos.empty:
        raise ValueError(
            "No hay realizaciones vivas con una posición de pico válida."
        )

    peak_stats_df = (
        datos_vivos
        .groupby("time", as_index=False)
        .agg(
            peak_mean_alive=("x_peak", "mean"),
            peak_std_alive=("x_peak", "std"),
            n_alive=("x_peak", "count"),
        )
    )

    # Para un único dato la desviación estándar de pandas es NaN.
    peak_stats_df["peak_std_alive"] = (
        peak_stats_df["peak_std_alive"]
        .fillna(0.0)
    )

    peak_stats_df["peak_sem_alive"] = (
        peak_stats_df["peak_std_alive"]
        / np.sqrt(peak_stats_df["n_alive"])
    )

    peak_time = peak_stats_df[
        "time"
    ].to_numpy(dtype=float)

    peak_mean_alive = peak_stats_df[
        "peak_mean_alive"
    ].to_numpy(dtype=float)

    valid_peak = (
        peak_stats_df["n_alive"]
        .to_numpy(dtype=int)
        >= minimo_supervivientes
    )

    speed_fit_peak_alive = fit_speed_with_mask(
        peak_time,
        peak_mean_alive,
        valid_peak,
        t_min=60.0,
        x_max=0.94 * cfg.Lx,
    )

    peak_stats = peak_stats_df.to_dict(
        orient="records",
    )

    return peak_stats, speed_fit_peak_alive



def ejecutar_barw_comparacion(cfg:BARWEnsembleConfig,) -> dict[str, object]:
    #adaptación de la función run_barw_ensemble del código proporcionado por los tutores

    all_history: list[dict[str, object]] = []
    profile_rows_raw: list[dict[str, object]] = []
    final_segments: list[dict[str, object]] = []
    final_tips: list[dict[str, object]] = []
    seed_summary: list[dict[str, object]] = []

    snapshot_steps = {
        int(round(tiempo / cfg.step_time))
        for tiempo in cfg.snapshot_times
    }
    max_steps = int(round(cfg.total_time / cfg.step_time))

    for seed in cfg.seeds:
        barw_cfg = BARWConfig( 
            Lx=cfg.Lx, 
            Ly=cfg.Ly, 
            pb=cfg.rb, 
            Ra=cfg.Ra, 
            long_paso=cfg.step_length, 
            tiempo_paso=cfg.step_time, 
            tiempo_total=cfg.total_time, 
            ang_amplitud=np.pi / 10, 
            angulo_bifurcacion=np.pi / 6, 
            pasos_exclusion_propia=6, 
            pasos_exclusion_madre_hija=10, 
            modo_colision="punto_punto", 
            semilla=int(seed), 
            max_puntas=100_000, 
            max_pasos=max_steps, )
        
        sim = SimulacionBARW(barw_cfg, metodo_busqueda=1)
        sim.inicializar()
        active = sum(p.activa for p in sim.puntas)
        extinction_time = math.nan

        for step in range(max_steps + 1):
            time = float(step * cfg.step_time)
            x_front = max((max(seg[0], seg[2]) for seg in sim.conducto), default=0.0)
            x_peak = posicion_pico_activo(sim,cfg,)
            alive = active > 0
            all_history.append(
                {
                    "seed": int(seed),
                    "time": time,
                    "x_front": float(x_front),
                    "x_peak": float(x_peak),
                    "active_tips": int(active),
                    "alive": bool(alive),
                    "segments": int(len(sim.conducto)),
                    "bifurcations": int(sim.contador_bifurcaciones),
                    "terminations": int(sim.contador_terminaciones),
                }
            )
            if step in snapshot_steps:
                for row in barw_profile_from_state(sim, cfg, time):
                    row["seed"] = int(seed)
                    row["alive"] = bool(alive)
                    profile_rows_raw.append(row)
            if step == max_steps:
                break
            if alive:
                sim.paso()
                active = sum(p.activa for p in sim.puntas)
                if active == 0 and not np.isfinite(extinction_time):
                    extinction_time = float((step + 1) * cfg.step_time)

        if seed == cfg.seeds[0]:
            
            for idx, segmento in enumerate(sim.conducto):
                x0, y0, x1, y1, id_rama, *resto = segmento

                paso_deposito = (
                    int(resto[0])
                    if resto
                    else np.nan
                )

                final_segments.append(
                    {
                        "segment_id": idx,
                        "seed": int(seed),
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x1),
                        "y1": float(y1),
                        "branch_id": int(id_rama),
                        "deposit_step": paso_deposito,
                    }
                )

            for p in sim.puntas:
                final_tips.append(
                    {
                        "seed": int(seed),
                        "tip_id": int(p.id),
                        "branch_id": int(p.id_rama),
                        "x": float(p.x),
                        "y": float(p.y),
                        "theta": float(p.theta),
                        "generation": int(p.generacion),
                        "active": bool(p.activa),
                    }
                )
        final_peak = posicion_pico_activo(sim,cfg,)
        seed_summary.append(
            {
                "seed": int(seed),
                "extinction_time": float(extinction_time) if np.isfinite(extinction_time) else "",
                "survived_to_T": bool(active > 0),
                "final_front": float(max((max(seg[0], seg[2]) for seg in sim.conducto), default=0.0)),
                "final_peak": float(final_peak),
                "final_active_tips": int(active),
                "final_segments": int(len(sim.conducto)),
                "bifurcations": int(sim.contador_bifurcaciones),
                "terminations": int(sim.contador_terminaciones),
            }
        )

    front_stats = aggregate_front_statistics(all_history)
    stats_time = np.asarray([r["time"] for r in front_stats], dtype=float)
    front_all = np.asarray([r["front_mean_all"] for r in front_stats], dtype=float)
    front_alive = np.asarray([r["front_mean_alive"] for r in front_stats], dtype=float)
    valid_alive = np.asarray([r["n_alive"] >= minimo_supervivientes for r in front_stats], dtype=bool)
    speed_fit_front_all = fit_speed(stats_time, front_all, t_min=60.0, x_max=0.94 * cfg.Lx)
    speed_fit_front_alive = fit_speed_with_mask(stats_time, front_alive, valid_alive, t_min=60.0, x_max=0.94 * cfg.Lx)
    peak_stats, speed_fit_peak_alive = (estadisticas_pico_activo(all_history,cfg,)
)

    grouped_all: dict[tuple[float, float], dict[str, list[float]]] = {}
    grouped_alive: dict[tuple[float, float], dict[str, list[float]]] = {}
    for row in profile_rows_raw:
        key = (float(row["time"]), float(row["x"]))
        grouped_all.setdefault(key, {"active_density": [], "duct_density": [], "active_count": [], "duct_length": []})
        for field in grouped_all[key]:
            grouped_all[key][field].append(float(row[field]))
        if bool(row["alive"]):
            grouped_alive.setdefault(key, {"active_density": [], "duct_density": [], "active_count": [], "duct_length": []})
            for field in grouped_alive[key]:
                grouped_alive[key][field].append(float(row[field]))

    profile_rows_mean_all = grouped_profile_rows(grouped_all)
    profile_rows_mean_alive = grouped_profile_rows(grouped_alive)

    return {
        "history": all_history,
        "front_stats": front_stats,
        "peak_stats": peak_stats,
        "profile_rows_raw": profile_rows_raw,
        "profile_rows_mean": profile_rows_mean_all,
        "profile_rows_mean_alive": profile_rows_mean_alive,
        "final_segments": final_segments,
        "final_tips": final_tips,
        "seed_summary": seed_summary,
        "speed_fit_front_all": speed_fit_front_all,
        "speed_fit_front_alive": speed_fit_front_alive,
        "speed_fit_peak_alive": speed_fit_peak_alive,
    }




def main() -> None:
    # Primera prueba: usar solo cinco semillas.
    # Cuando el resultado sea correcto, sustituir 1005 por 1100.
    semillas = tuple(range(1000, 1100))

    cfg = BARWEnsembleConfig(
        seeds=semillas,
        Lx=280.0,
        Ly=150.0,
        rb=0.1,
        Ra=3.0,
        total_time=300.0,
        step_time=1.0,
        step_length=1.0,
        n_bins=120,
        snapshot_times=(
            120,
            140,
            160,
            180,
            200,
            220,
            240,
            260,
            280,
            300,
        ),
        collapse_times=(
            140,
            180,
            220,
            260,
            300,
        ),
    )

    print("=== GENERACIÓN DEL ENSEMBLE BARW ===")
    print(f"Número de semillas: {len(cfg.seeds)}")
    print(f"Dominio: [0, {cfg.Lx}] x [0, {cfg.Ly}]")
    print(f"Tiempo final: {cfg.total_time}")
    print(f"Probabilidad de bifurcación: {cfg.rb}")
    print(f"Radio de aniquilación: {cfg.Ra}")
    print(f"Número de bins: {cfg.n_bins}")

    resultados = ejecutar_barw_comparacion(cfg)

    guardar_resultados_barw(resultados)

    resumen_semillas = pd.DataFrame(
        resultados["seed_summary"]
    )

    supervivientes = int(
        resumen_semillas["survived_to_T"].sum()
    )

    fraccion_supervivencia = (
        supervivientes
        / len(resumen_semillas)
    )

    print("\n=== RESUMEN DEL ENSEMBLE ===")
    print(
        "Realizaciones supervivientes: "
        f"{supervivientes}/{len(resumen_semillas)}"
    )
    print(
        "Fracción de supervivencia: "
        f"{fraccion_supervivencia:.4f}"
    )

    print("\nAjuste del frente acumulado, todas las realizaciones:")
    print(resultados["speed_fit_front_all"])

    print("\nAjuste del frente acumulado, realizaciones vivas:")
    print(resultados["speed_fit_front_alive"])

    print("\nAjuste del pico activo, realizaciones vivas:")
    print(resultados["speed_fit_peak_alive"])

    print(
        "\nResultados guardados en: "
        f"{RESULTADOS.resolve()}"
    )

    diagnostico_frente = pd.DataFrame(
        resultados["front_stats"]
    )

    print(
        diagnostico_frente.loc[
            diagnostico_frente["time"].isin(
                [0.0, 30.0, 60.0, 100.0, 200.0, 300.0]
            ),
            [
                "time",
                "n_alive",
                "survival_fraction",
                "front_mean_alive",
            ],
        ].to_string(index=False)
    )

    diagnostico_pico = pd.DataFrame(
        resultados["peak_stats"]
    )

    print(
        diagnostico_pico.loc[
            diagnostico_pico["time"].isin(
                [0.0, 30.0, 60.0, 100.0, 200.0, 300.0]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

