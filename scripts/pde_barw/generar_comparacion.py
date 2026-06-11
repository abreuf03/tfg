import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from paquetes_cerrados.hannezo2017_fig3_pde_barw.reproducir_fig3_hannezo_pde_barw import (
    BARWEnsembleConfig,
    aggregate_front_statistics,
    barw_profile_from_state,
    collapse_barw_profiles_moving_frame,
    collapse_pde_profiles_moving_frame,
    compute_barw_pde_comparison,
    fit_speed,
    fit_speed_with_mask,
    grouped_profile_rows,
    normalized,
    sem,
    smooth_profile,
)
from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW


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
        >= 5
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
            semilla=int(seed),
            max_puntas=100000,
            max_pasos=max_steps,
        )
        sim = SimulacionBARW(barw_cfg, usar_kdtree=True)
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
            for idx, (x0, y0, x1, y1, id_rama) in enumerate(sim.conducto):
                final_segments.append(
                    {
                        "segment_id": idx,
                        "seed": int(seed),
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x1),
                        "y1": float(y1),
                        "branch_id": int(id_rama),
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
    valid_alive = np.asarray([r["n_alive"] >= 5 for r in front_stats], dtype=bool)
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
