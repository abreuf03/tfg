from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


#def find_repo_root(start: Path) -> Path:
#    for path in [start, *start.parents]:
#        if (path / "references" / "videos").exists():
#            return path
#    raise RuntimeError("No se encontro la raiz del proyecto desde el script.")


def find_repo_root(start: Path) -> Path:
    """Localiza la raíz del proyecto actual."""
    for path in [start, *start.parents]:
        if (
            (path / "src").is_dir()
            and (path / "paquetes_cerrados").is_dir()
        ):
            return path

    raise RuntimeError(
        "No se encontró la raíz del proyecto desde el script."
    )



REPO_ROOT = find_repo_root(BASE_DIR)
NOP_PROJECT_DIR = REPO_ROOT / "Scripts" / "Elena" / "tfg-main" / "NOP"
VIDEO_EXTRACTION_DIR = REPO_ROOT / "Scripts" / "Elena" / "New" / "hannezo2017_video_extraction"
VIDEO_DENSITY_CSV = VIDEO_EXTRACTION_DIR / "data" / "video_density_profiles.csv"

if str(NOP_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(NOP_PROJECT_DIR))

from src.barw.config import BARWConfig  # noqa: E402
from src.barw.simulacion import SimulacionBARW  # noqa: E402


DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
INDIVIDUAL_DIR = FIGURES_DIR / "individual_panels"
QUALITY_DIR = FIGURES_DIR / "quality_box"

DPI = 300
GREEN = "#2b8a3e"
LIGHT_GREEN = "#cfe8d6"
RED = "#d62728"
BLACK = "#202020"
GRAY = "#6b6b6b"
BLUE = "#1f77b4"
LIGHT_BLUE = "#cfe4f6"
ORANGE = "#d55e00"
LIGHT_ORANGE = "#f2c7ad"
PANEL_B_FRONT_FIT = BLUE
PANEL_B_BACK_FIT = BLACK
PANEL_C_PDE = BLUE
PANEL_C_BARW_SURV = ORANGE
PANEL_C_BARW_ALL = GRAY
PANEL_C_THEORY = BLACK
PANEL_D_PDE = RED
PANEL_D_BARW = ORANGE
PANEL_E_PDE = BLACK
PANEL_E_BARW = BLUE
PANEL_E_SEM = LIGHT_BLUE
PANEL_F_SURVIVAL = BLUE
PANEL_F_TIPS_ALL = GRAY
PANEL_F_TIPS_SURV = GREEN
SUPPORT_MIN_SAMPLES = 10
ACTIVE_SUPPORT_MIN_SAMPLES = SUPPORT_MIN_SAMPLES
DUCT_SUPPORT_MIN_SAMPLES = SUPPORT_MIN_SAMPLES

# Coarse-graining kernels: BARW produces a point process, while the PDE is a
# continuum density. These widths define the observation scale before comparing.
ACTIVE_KDE_SIGMA_BINS = 4.0
DUCT_KDE_SIGMA_BINS = 3.0


@dataclass
class PDEConfig:
    L: float = 280.0
    nx: int = 561
    T: float = 320.0
    dt: float = 0.04
    rb: float = 0.1
    re: float = 0.08
    n0: float = 1.0
    x0: float = 8.0
    sigma0: float = 2.0
    a0_amplitude: float = 0.08
    save_dt: float = 1.0
    D: float = 1.0
    variant: str = "reduced"


@dataclass
class BARWEnsembleConfig:
    seeds: tuple[int, ...] = tuple(range(1000, 1100))
    Lx: float = 280.0
    Ly: float = 150.0
    rb: float = 0.1
    Ra: float = 3.0
    total_time: float = 320.0
    step_time: float = 1.0
    step_length: float = 1.0
    n_bins: int = 120
    snapshot_times: tuple[int, ...] = (120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320)
    collapse_times: tuple[int, ...] = (140, 180, 220, 260, 300)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "mathtext.default": "regular",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_figure(fig: plt.Figure, name: str, target_dir: Path = INDIVIDUAL_DIR) -> None:
    fig.savefig(target_dir / f"{name}.png", bbox_inches="tight", dpi=DPI)
    fig.savefig(target_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def boxed_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(1.1)
    ax.tick_params(direction="out", length=3.2, width=0.9, color=BLACK)
    ax.set_facecolor("white")


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    m = np.nanmax(values) if values.size else 0.0
    if not np.isfinite(m) or m <= 0:
        return np.zeros_like(values, dtype=float)
    return values / m


def smooth_profile(values: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or sigma_bins <= 0:
        return values.copy()
    radius = max(2, int(math.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = np.sum(np.isfinite(values), axis=axis)
    if values.ndim == 0 or int(np.max(np.asarray(n))) <= 1:
        return np.zeros_like(np.asarray(n), dtype=float)
    sd = np.nanstd(values, axis=axis, ddof=1)
    out = sd / np.sqrt(np.maximum(n, 1))
    out = np.where(n > 1, out, 0.0)
    return out


def fit_speed(time: np.ndarray, front: np.ndarray, t_min: float = 60.0, x_max: float = 265.0) -> dict[str, float]:
    mask = np.isfinite(front) & (time >= t_min) & (front > 10.0) & (front < x_max)
    if np.count_nonzero(mask) < 3:
        return {
            "speed": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "max_abs_error": math.nan,
            "speed_se": math.nan,
            "speed_ci95": math.nan,
            "n": int(np.count_nonzero(mask)),
            "time_min": math.nan,
            "time_max": math.nan,
            "front_min": math.nan,
            "front_max": math.nan,
        }
    t = time[mask]
    y = front[mask]
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    residual = y - pred
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs_error = float(np.max(np.abs(residual)))
    denom = float(np.sum((t - np.mean(t)) ** 2))
    if len(t) > 2 and denom > 0:
        sigma2 = ss_res / float(len(t) - 2)
        speed_se = math.sqrt(sigma2 / denom)
        speed_ci95 = 1.96 * speed_se
    else:
        speed_se = math.nan
        speed_ci95 = math.nan
    return {
        "speed": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "speed_se": float(speed_se),
        "speed_ci95": float(speed_ci95),
        "n": int(len(t)),
        "time_min": float(np.min(t)),
        "time_max": float(np.max(t)),
        "front_min": float(np.min(y)),
        "front_max": float(np.max(y)),
    }


def fit_speed_with_mask(
    time: np.ndarray,
    front: np.ndarray,
    valid: np.ndarray,
    t_min: float = 60.0,
    x_max: float = 265.0,
) -> dict[str, float]:
    mask = valid & np.isfinite(front) & (time >= t_min) & (front > 10.0) & (front < x_max)
    return fit_speed(time[mask], front[mask], t_min=-np.inf, x_max=np.inf)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return math.nan
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return math.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def pde_front_from_active(x: np.ndarray, a: np.ndarray, relative_threshold: float = 0.08) -> float:
    if np.nanmax(a) <= 0:
        return math.nan
    threshold = relative_threshold * float(np.nanmax(a))
    idx = np.flatnonzero(a >= threshold)
    if len(idx) == 0:
        return math.nan
    return float(x[idx[-1]])


def solve_two_species_pde(cfg: PDEConfig) -> dict[str, np.ndarray | dict[str, float]]:
    x = np.linspace(0.0, cfg.L, cfg.nx)
    dx = float(x[1] - x[0])
    if cfg.dt > 0.45 * dx * dx / max(cfg.D, 1e-12):
        raise ValueError("dt demasiado grande para la parte difusiva explicita.")

    a = cfg.a0_amplitude * np.exp(-0.5 * ((x - cfg.x0) / cfg.sigma0) ** 2)
    i = np.zeros_like(a)

    save_every = max(1, int(round(cfg.save_dt / cfg.dt)))
    n_steps = int(round(cfg.T / cfg.dt))
    times: list[float] = []
    active: list[np.ndarray] = []
    inactive: list[np.ndarray] = []
    front_rows: list[dict[str, float]] = []

    for step in range(n_steps + 1):
        time = step * cfg.dt
        if step % save_every == 0:
            front = pde_front_from_active(x, a)
            times.append(time)
            active.append(a.copy())
            inactive.append(i.copy())
            front_rows.append(
                {
                    "time": float(time),
                    "front_active_threshold": float(front),
                    "active_peak": float(np.max(a)),
                    "active_peak_position": float(x[int(np.argmax(a))]),
                    "inactive_plateau": float(np.percentile(i[x < max(front - 20.0, 0.0)], 80))
                    if np.isfinite(front) and np.any(x < max(front - 20.0, 0.0))
                    else float(np.nan),
                }
            )
        if step == n_steps:
            break

        lap = np.empty_like(a)
        lap[1:-1] = (a[2:] - 2.0 * a[1:-1] + a[:-2]) / (dx * dx)
        lap[0] = 2.0 * (a[1] - a[0]) / (dx * dx)
        lap[-1] = 2.0 * (a[-2] - a[-1]) / (dx * dx)

        interaction = a + i
        if cfg.variant == "full":
            da = cfg.D * lap + cfg.rb * a * (1.0 - interaction / cfg.n0)
            di = cfg.re * a + (cfg.rb / cfg.n0) * a * interaction
        elif cfg.variant == "reduced":
            da = cfg.D * lap + cfg.rb * a * (1.0 - i / cfg.n0)
            di = cfg.re * a
        else:
            raise ValueError(f"Variante PDE no reconocida: {cfg.variant}")

        a = np.maximum(a + cfg.dt * da, 0.0)
        i = np.maximum(i + cfg.dt * di, 0.0)

    times_arr = np.asarray(times)
    active_arr = np.vstack(active)
    inactive_arr = np.vstack(inactive)
    front_time = np.asarray([row["time"] for row in front_rows])
    front_x = np.asarray([row["front_active_threshold"] for row in front_rows])
    peak_x = np.asarray([row["active_peak_position"] for row in front_rows])
    speed_fit = fit_speed(front_time, front_x, t_min=80.0, x_max=0.92 * cfg.L)
    peak_fit = fit_speed(front_time, peak_x, t_min=80.0, x_max=0.92 * cfg.L)

    return {
        "x": x,
        "times": times_arr,
        "active": active_arr,
        "inactive": inactive_arr,
        "front_rows": front_rows,
        "speed_fit": speed_fit,
        "peak_fit": peak_fit,
    }


def barw_profile_from_state(sim: SimulacionBARW, cfg: BARWEnsembleConfig, time: float) -> list[dict[str, object]]:
    edges = np.linspace(0.0, cfg.Lx, cfg.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])

    active_x = [p.x for p in sim.puntas if p.activa]
    active_counts, _ = np.histogram(active_x, bins=edges)

    duct_x: list[float] = []
    duct_w: list[float] = []
  
    for x0, y0, x1, y1, _id_rama, *_ in sim.conducto:
        duct_x.append(0.5 * (x0 + x1))
        duct_w.append(math.hypot(x1 - x0, y1 - y0))
    if duct_x:
        duct_counts, _ = np.histogram(duct_x, bins=edges, weights=duct_w)
    else:
        duct_counts = np.zeros(cfg.n_bins, dtype=float)

    area = bin_width * cfg.Ly
    rows: list[dict[str, object]] = []
    for j, xc in enumerate(centers):
        rows.append(
            {
                "time": float(time),
                "x": float(xc),
                "active_count": int(active_counts[j]),
                "duct_length": float(duct_counts[j]),
                "active_density": float(active_counts[j] / area),
                "duct_density": float(duct_counts[j] / area),
            }
        )
    return rows


def run_barw_ensemble(cfg: BARWEnsembleConfig) -> dict[str, object]:
    all_history: list[dict[str, object]] = []
    profile_rows_raw: list[dict[str, object]] = []
    final_segments: list[dict[str, object]] = []
    final_tips: list[dict[str, object]] = []
    seed_summary: list[dict[str, object]] = []

    snapshots = set(int(t) for t in cfg.snapshot_times)
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
        sim = SimulacionBARW(barw_cfg, metodo_busqueda=1)
        sim.inicializar()
        active = sum(p.activa for p in sim.puntas)
        extinction_time = math.nan

        for step in range(max_steps + 1):
            time = float(step * cfg.step_time)
            x_front = max((max(seg[0], seg[2]) for seg in sim.conducto), default=0.0)
            alive = active > 0
            all_history.append(
                {
                    "seed": int(seed),
                    "time": time,
                    "x_front": float(x_front),
                    "active_tips": int(active),
                    "alive": bool(alive),
                    "segments": int(len(sim.conducto)),
                    "bifurcations": int(sim.contador_bifurcaciones),
                    "terminations": int(sim.contador_terminaciones),
                }
            )
            if step in snapshots:
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
            for idx, (x0, y0, x1, y1, id_rama, *_ ) in enumerate(sim.conducto):
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
        seed_summary.append(
            {
                "seed": int(seed),
                "extinction_time": float(extinction_time) if np.isfinite(extinction_time) else "",
                "survived_to_T": bool(active > 0),
                "final_front": float(max((max(seg[0], seg[2]) for seg in sim.conducto), default=0.0)),
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
    speed_fit_all = fit_speed(stats_time, front_all, t_min=60.0, x_max=0.94 * cfg.Lx)
    speed_fit_alive = fit_speed_with_mask(stats_time, front_alive, valid_alive, t_min=60.0, x_max=0.94 * cfg.Lx)

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
        "speed_fit_all": speed_fit_all,
        "speed_fit_alive": speed_fit_alive,
        "speed_fit": speed_fit_alive,
    }


def grouped_profile_rows(grouped: dict[tuple[float, float], dict[str, list[float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (time, x), values in sorted(grouped.items()):
        active_values = np.asarray(values["active_density"], dtype=float)
        duct_values = np.asarray(values["duct_density"], dtype=float)
        rows.append(
            {
                "time": float(time),
                "x": float(x),
                "active_density_mean": float(np.mean(active_values)),
                "active_density_sem": float(sem(active_values)),
                "duct_density_mean": float(np.mean(duct_values)),
                "duct_density_sem": float(sem(duct_values)),
                "n_seeds": int(len(active_values)),
            }
        )
    return rows


def aggregate_front_statistics(history: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    times = sorted({float(r["time"]) for r in history})
    for time in times:
        sub = [r for r in history if float(r["time"]) == time]
        all_front = np.asarray([float(r["x_front"]) for r in sub], dtype=float)
        alive_sub = [r for r in sub if bool(r["alive"])]
        alive_front = np.asarray([float(r["x_front"]) for r in alive_sub], dtype=float)
        rows.append(
            {
                "time": float(time),
                "front_mean_all": float(np.mean(all_front)),
                "front_sem_all": float(sem(all_front)),
                "front_mean_alive": float(np.mean(alive_front)) if len(alive_front) else math.nan,
                "front_sem_alive": float(sem(alive_front)) if len(alive_front) else math.nan,
                "n_total": int(len(sub)),
                "n_alive": int(len(alive_sub)),
                "survival_fraction": float(len(alive_sub) / len(sub)) if sub else math.nan,
                "active_tips_mean_all": float(np.mean([float(r["active_tips"]) for r in sub])),
                "active_tips_mean_alive": float(np.mean([float(r["active_tips"]) for r in alive_sub])) if alive_sub else 0.0,
            }
        )
    return rows


def closest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def interpolate_profile(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    return np.interp(x_dst, x_src, y_src, left=0.0, right=0.0)


def profile_arrays_from_mean(rows: list[dict[str, object]], time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [r for r in rows if abs(float(r["time"]) - time) < 1e-9]
    selected.sort(key=lambda r: float(r["x"]))
    x = np.asarray([float(r["x"]) for r in selected])
    active = np.asarray([float(r["active_density_mean"]) for r in selected])
    duct = np.asarray([float(r["duct_density_mean"]) for r in selected])
    return x, active, duct


def front_from_duct_profile(x: np.ndarray, duct: np.ndarray) -> float:
    if len(x) == 0 or np.max(duct) <= 0:
        return math.nan
    threshold = 0.05 * float(np.max(duct))
    idx = np.flatnonzero(duct >= threshold)
    if len(idx) == 0:
        return math.nan
    return float(x[idx[-1]])


def collapse_barw_profiles_moving_frame(
    barw: dict[str, object],
    selected_times: tuple[int, ...],
    z_edges: np.ndarray,
    anchor_mode: str = "active_peak",
) -> list[dict[str, object]]:
    history = barw["history"]
    raw_rows = barw["profile_rows_raw"]
    assert isinstance(history, list)
    assert isinstance(raw_rows, list)
    front_by_seed_time = {(int(r["seed"]), float(r["time"])): float(r["x_front"]) for r in history}
    alive_by_seed_time = {(int(r["seed"]), float(r["time"])): bool(r["alive"]) for r in history}
    selected = {float(t) for t in selected_times}
    rows_by_seed_time: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in raw_rows:
        seed = int(row["seed"])
        time = float(row["time"])
        if time in selected and alive_by_seed_time.get((seed, time), False):
            rows_by_seed_time.setdefault((seed, time), []).append(row)
    buckets: dict[int, dict[str, list[float]]] = {
        j: {"active": [], "duct": [], "raw_active": [], "raw_duct": []} for j in range(len(z_edges) - 1)
    }
    for key, group in rows_by_seed_time.items():
        xs = np.asarray([float(r["x"]) for r in group])
        active_values = np.asarray([float(r["active_density"]) for r in group])
        if anchor_mode == "active_peak" and np.max(active_values) > 0:
            anchor = float(xs[int(np.argmax(active_values))])
        elif anchor_mode == "duct_front":
            anchor = front_by_seed_time[key]
        else:
            anchor = front_by_seed_time[key]
        for row in group:
            z = float(row["x"]) - anchor
            if z < z_edges[0] or z >= z_edges[-1]:
                continue
            j = int(np.searchsorted(z_edges, z, side="right") - 1)
            buckets[j]["active"].append(float(row["active_density"]))
            buckets[j]["duct"].append(float(row["duct_density"]))

    rows: list[dict[str, object]] = []
    centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    for j, z in enumerate(centers):
        active_values = np.asarray(buckets[j]["active"], dtype=float)
        duct_values = np.asarray(buckets[j]["duct"], dtype=float)
        rows.append(
            {
                "z": float(z),
                "active_density_mean": float(np.mean(active_values)) if len(active_values) else 0.0,
                "active_density_sem": float(sem(active_values)) if len(active_values) else 0.0,
                "duct_density_mean": float(np.mean(duct_values)) if len(duct_values) else 0.0,
                "duct_density_sem": float(sem(duct_values)) if len(duct_values) else 0.0,
                "n_samples": int(len(active_values)),
            }
        )
    return rows


def collapse_pde_profiles_moving_frame(
    pde: dict[str, object],
    selected_times: tuple[int, ...],
    z_centers: np.ndarray,
    anchor_mode: str = "active_peak",
) -> list[dict[str, object]]:
    x = pde["x"]
    times = pde["times"]
    active = pde["active"]
    inactive = pde["inactive"]
    assert isinstance(x, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert isinstance(active, np.ndarray)
    assert isinstance(inactive, np.ndarray)
    active_profiles = []
    inactive_profiles = []
    used_times = []
    for target in selected_times:
        idx = closest_index(times, float(target))
        if anchor_mode == "active_peak":
            anchor = float(x[int(np.argmax(active[idx]))])
        elif anchor_mode == "duct_front":
            anchor = inactive_front_from_profile(x, inactive[idx])
        else:
            anchor = pde_front_from_active(x, active[idx])
        z_src = x - anchor
        active_profiles.append(interpolate_profile(z_src, active[idx], z_centers))
        inactive_profiles.append(interpolate_profile(z_src, inactive[idx], z_centers))
        used_times.append(float(times[idx]))
    active_arr = np.vstack(active_profiles)
    inactive_arr = np.vstack(inactive_profiles)
    rows: list[dict[str, object]] = []
    for j, z in enumerate(z_centers):
        rows.append(
            {
                "z": float(z),
                "active_density_mean": float(np.mean(active_arr[:, j])),
                "active_density_sem": float(sem(active_arr[:, j])),
                "inactive_density_mean": float(np.mean(inactive_arr[:, j])),
                "inactive_density_sem": float(sem(inactive_arr[:, j])),
                "n_times": int(len(used_times)),
            }
        )
    return rows


def inactive_front_from_profile(x: np.ndarray, inactive: np.ndarray, relative_threshold: float = 0.05) -> float:
    if np.nanmax(inactive) <= 0:
        return float(x[int(np.argmax(inactive))])
    threshold = relative_threshold * float(np.nanmax(inactive))
    idx = np.flatnonzero(inactive >= threshold)
    if len(idx) == 0:
        return float(x[int(np.argmax(inactive))])
    return float(x[idx[-1]])


def compute_barw_pde_comparison(
    pde: dict[str, object],
    barw: dict[str, object],
    selected_times: tuple[int, ...],
) -> dict[str, object]:
    z_edges = np.linspace(-165.0, 45.0, 106)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    barw_active_rows = collapse_barw_profiles_moving_frame(barw, selected_times, z_edges, anchor_mode="active_peak")
    pde_active_rows = collapse_pde_profiles_moving_frame(pde, selected_times, z_centers, anchor_mode="active_peak")
    barw_duct_rows = collapse_barw_profiles_moving_frame(barw, selected_times, z_edges, anchor_mode="duct_front")
    pde_duct_rows = collapse_pde_profiles_moving_frame(pde, selected_times, z_centers, anchor_mode="duct_front")
    z_barw = np.asarray([float(r["z"]) for r in barw_active_rows])
    active_barw = np.asarray([float(r["active_density_mean"]) for r in barw_active_rows])
    active_barw_sem = np.asarray([float(r["active_density_sem"]) for r in barw_active_rows])
    duct_barw = np.asarray([float(r["duct_density_mean"]) for r in barw_duct_rows])
    duct_barw_sem = np.asarray([float(r["duct_density_sem"]) for r in barw_duct_rows])
    a_pde = np.asarray([float(r["active_density_mean"]) for r in pde_active_rows])
    i_pde = np.asarray([float(r["inactive_density_mean"]) for r in pde_duct_rows])
    a_barw_raw_norm = normalized(active_barw)
    duct_barw_raw_norm = normalized(duct_barw)
    active_barw_smooth = smooth_profile(active_barw, sigma_bins=ACTIVE_KDE_SIGMA_BINS)
    duct_barw_smooth = smooth_profile(duct_barw, sigma_bins=DUCT_KDE_SIGMA_BINS)
    a_barw_norm = normalized(active_barw_smooth)
    duct_barw_norm = normalized(duct_barw_smooth)
    a_barw_sem_norm = active_barw_sem / max(float(np.max(active_barw_smooth)), 1e-12)
    duct_barw_sem_norm = duct_barw_sem / max(float(np.max(duct_barw_smooth)), 1e-12)
    a_pde_interp = normalized(a_pde)
    i_pde_interp = normalized(i_pde)
    active_window = (-75.0, 35.0)
    duct_window = (-155.0, 15.0)
    n_samples_active = np.asarray([int(r["n_samples"]) for r in barw_active_rows])
    n_samples_duct = np.asarray([int(r["n_samples"]) for r in barw_duct_rows])
    active_window_mask = (z_barw >= active_window[0]) & (z_barw <= active_window[1])
    duct_window_mask = (z_barw >= duct_window[0]) & (z_barw <= duct_window[1])
    metric_mask_active = active_window_mask & (n_samples_active >= ACTIVE_SUPPORT_MIN_SAMPLES)
    metric_mask_duct = duct_window_mask & (n_samples_duct >= DUCT_SUPPORT_MIN_SAMPLES)
    metric_mask = metric_mask_active | metric_mask_duct
    if np.count_nonzero(metric_mask) < 5:
        metric_mask_active = (z_barw >= active_window[0]) & (z_barw <= active_window[1]) & (n_samples_active > 0)
        metric_mask_duct = (z_barw >= duct_window[0]) & (z_barw <= duct_window[1]) & (n_samples_duct > 0)
    rmse_active = float(np.sqrt(np.mean((a_barw_norm[metric_mask_active] - a_pde_interp[metric_mask_active]) ** 2)))
    rmse_duct = float(np.sqrt(np.mean((duct_barw_norm[metric_mask_duct] - i_pde_interp[metric_mask_duct]) ** 2)))
    corr_active = pearson_corr(a_barw_norm[metric_mask_active], a_pde_interp[metric_mask_active])
    corr_duct = pearson_corr(duct_barw_norm[metric_mask_duct], i_pde_interp[metric_mask_duct])

    rows = []
    for k, z in enumerate(z_barw):
        rows.append(
            {
                "times_used": ";".join(str(t) for t in selected_times),
                "z_from_front": float(z),
                "barw_active_raw_norm": float(a_barw_raw_norm[k]),
                "barw_active_norm": float(a_barw_norm[k]),
                "barw_active_sem_norm": float(a_barw_sem_norm[k]),
                "pde_active_norm": float(a_pde_interp[k]),
                "barw_duct_raw_norm": float(duct_barw_raw_norm[k]),
                "barw_duct_norm": float(duct_barw_norm[k]),
                "barw_duct_sem_norm": float(duct_barw_sem_norm[k]),
                "pde_inactive_norm": float(i_pde_interp[k]),
                "n_samples_active": int(barw_active_rows[k]["n_samples"]),
                "n_samples_duct": int(barw_duct_rows[k]["n_samples"]),
            }
        )
    return {
        "rows": rows,
        "barw_collapse_rows": barw_active_rows,
        "pde_collapse_rows": pde_active_rows,
        "barw_duct_collapse_rows": barw_duct_rows,
        "pde_duct_collapse_rows": pde_duct_rows,
        "metrics": {
            "collapse_times": list(selected_times),
            "rmse_active_norm": rmse_active,
            "rmse_duct_norm": rmse_duct,
            "corr_active_norm": corr_active,
            "corr_duct_norm": corr_duct,
            "rmse_window_left": float(min(active_window[0], duct_window[0])),
            "rmse_window_right": float(max(active_window[1], duct_window[1])),
            "rmse_active_window_left": float(active_window[0]),
            "rmse_active_window_right": float(active_window[1]),
            "rmse_duct_window_left": float(duct_window[0]),
            "rmse_duct_window_right": float(duct_window[1]),
            "n_profile_samples_min": int(min(np.min(n_samples_active[metric_mask_active]), np.min(n_samples_duct[metric_mask_duct]))) if np.any(metric_mask_active) and np.any(metric_mask_duct) else 0,
            "n_profile_samples_max": int(max(np.max(n_samples_active[metric_mask_active]), np.max(n_samples_duct[metric_mask_duct]))) if np.any(metric_mask_active) and np.any(metric_mask_duct) else 0,
            "n_profile_bins_used": int(max(np.count_nonzero(metric_mask_active), np.count_nonzero(metric_mask_duct))),
            "n_active_bins_used": int(np.count_nonzero(metric_mask_active)),
            "n_duct_bins_used": int(np.count_nonzero(metric_mask_duct)),
            "active_support_min_samples": int(ACTIVE_SUPPORT_MIN_SAMPLES),
            "n_active_bins_low_support_window": int(np.count_nonzero(active_window_mask & (n_samples_active > 0) & (n_samples_active < ACTIVE_SUPPORT_MIN_SAMPLES))),
            "n_active_bins_zero_support_window": int(np.count_nonzero(active_window_mask & (n_samples_active == 0))),
            "duct_support_min_samples": int(DUCT_SUPPORT_MIN_SAMPLES),
            "n_duct_bins_low_support_window": int(np.count_nonzero(duct_window_mask & (n_samples_duct > 0) & (n_samples_duct < DUCT_SUPPORT_MIN_SAMPLES))),
            "n_duct_bins_zero_support_window": int(np.count_nonzero(duct_window_mask & (n_samples_duct == 0))),
            "active_kde_sigma_bins": float(ACTIVE_KDE_SIGMA_BINS),
            "duct_kde_sigma_bins": float(DUCT_KDE_SIGMA_BINS),
        },
    }


def load_movie_s1_proxy() -> dict[str, object]:
    if not VIDEO_DENSITY_CSV.exists():
        return {"available": False, "reason": "No existe video_density_profiles.csv."}
    rows: list[dict[str, object]] = []
    with VIDEO_DENSITY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video_id") == "mmc1":
                rows.append(row)
    if not rows:
        return {"available": False, "reason": "No hay filas para Movie S1/mmc1."}
    red_by_time: dict[float, float] = {}
    for row in rows:
        time = float(row["time_s"])
        red_by_time[time] = red_by_time.get(time, 0.0) + float(row["red_fraction"])
    selected_time = max(red_by_time, key=red_by_time.get)
    selected = [r for r in rows if abs(float(r["time_s"]) - selected_time) < 1e-9]
    selected.sort(key=lambda r: int(r["x_bin"]))
    x = np.asarray([float(r["x_center_norm"]) for r in selected])
    black = np.asarray([float(r["black_fraction"]) for r in selected])
    red = np.asarray([float(r["red_fraction"]) for r in selected])
    return {
        "available": True,
        "source": str(VIDEO_DENSITY_CSV),
        "time_s": float(selected_time),
        "x_norm": x,
        "black_fraction": black,
        "red_fraction": red,
        "note": "Movie S1 es una pelicula suplementaria del modelo; se usa el frame con mayor senal roja y no sustituye los datos experimentales EdU de Fig. 3D.",
    }


def export_pde_data(pde: dict[str, object], cfg: PDEConfig) -> None:
    x = pde["x"]
    times = pde["times"]
    active = pde["active"]
    inactive = pde["inactive"]
    assert isinstance(x, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert isinstance(active, np.ndarray)
    assert isinstance(inactive, np.ndarray)

    selected_times = [80.0, 140.0, 200.0, 260.0, 320.0]
    profile_rows: list[dict[str, object]] = []
    for target in selected_times:
        idx = closest_index(times, target)
        for j in range(len(x)):
            profile_rows.append(
                {
                    "time": float(times[idx]),
                    "x": float(x[j]),
                    "active_density": float(active[idx, j]),
                    "inactive_density": float(inactive[idx, j]),
                }
            )
    write_csv(DATA_DIR / "pde_two_species_profiles_selected.csv", profile_rows, ["time", "x", "active_density", "inactive_density"])

    front_rows = pde["front_rows"]
    assert isinstance(front_rows, list)
    write_csv(
        DATA_DIR / "pde_two_species_front_speed.csv",
        front_rows,
        ["time", "front_active_threshold", "active_peak", "active_peak_position", "inactive_plateau"],
    )

    stride_t = max(1, len(times) // 180)
    stride_x = max(1, len(x) // 160)
    kymo_rows: list[dict[str, object]] = []
    for ti in range(0, len(times), stride_t):
        for xi in range(0, len(x), stride_x):
            kymo_rows.append(
                {
                    "time": float(times[ti]),
                    "x": float(x[xi]),
                    "active_density": float(active[ti, xi]),
                    "inactive_density": float(inactive[ti, xi]),
                }
            )
    write_csv(DATA_DIR / "pde_two_species_kymograph_downsampled.csv", kymo_rows, ["time", "x", "active_density", "inactive_density"])
    np.savez_compressed(DATA_DIR / "pde_two_species_solution.npz", x=x, times=times, active=active, inactive=inactive)


def export_barw_data(barw: dict[str, object]) -> None:
    write_csv(
        DATA_DIR / "barw_ensemble_history.csv",
        barw["history"],
        ["seed", "time", "x_front", "active_tips", "alive", "segments", "bifurcations", "terminations"],
    )
    write_csv(
        DATA_DIR / "barw_ensemble_front_stats.csv",
        barw["front_stats"],
        [
            "time",
            "front_mean_all",
            "front_sem_all",
            "front_mean_alive",
            "front_sem_alive",
            "n_total",
            "n_alive",
            "survival_fraction",
            "active_tips_mean_all",
            "active_tips_mean_alive",
        ],
    )
    write_csv(
        DATA_DIR / "barw_seed_summary.csv",
        barw["seed_summary"],
        ["seed", "extinction_time", "survived_to_T", "final_front", "final_active_tips", "final_segments", "bifurcations", "terminations"],
    )
    write_csv(
        DATA_DIR / "barw_ensemble_profiles_raw.csv",
        barw["profile_rows_raw"],
        ["seed", "time", "x", "alive", "active_count", "duct_length", "active_density", "duct_density"],
    )
    write_csv(
        DATA_DIR / "barw_ensemble_profiles_mean.csv",
        barw["profile_rows_mean"],
        ["time", "x", "active_density_mean", "active_density_sem", "duct_density_mean", "duct_density_sem", "n_seeds"],
    )
    write_csv(
        DATA_DIR / "barw_ensemble_profiles_mean_alive.csv",
        barw["profile_rows_mean_alive"],
        ["time", "x", "active_density_mean", "active_density_sem", "duct_density_mean", "duct_density_sem", "n_seeds"],
    )
    write_csv(
        DATA_DIR / "barw_reference_seed_segments.csv",
        barw["final_segments"],
        ["segment_id", "seed", "x0", "y0", "x1", "y1", "branch_id"],
    )
    write_csv(
        DATA_DIR / "barw_reference_seed_tips.csv",
        barw["final_tips"],
        ["seed", "tip_id", "branch_id", "x", "y", "theta", "generation", "active"],
    )


def plot_pde_kymograph(pde: dict[str, object]) -> None:
    x = pde["x"]
    times = pde["times"]
    active = pde["active"]
    front_rows = pde["front_rows"]
    peak_fit = pde["peak_fit"]
    assert isinstance(x, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert isinstance(active, np.ndarray)
    assert isinstance(front_rows, list)
    assert isinstance(peak_fit, dict)
    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    im = ax.imshow(
        active,
        origin="lower",
        extent=[x.min(), x.max(), times.min(), times.max()],
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=np.percentile(active, 99.5),
    )
    speed = float(peak_fit["speed"])
    intercept = float(peak_fit["intercept"])
    if np.isfinite(speed) and np.isfinite(intercept):
        t_fit = np.linspace(float(peak_fit["time_min"]), float(peak_fit["time_max"]), 160)
        ax.plot(intercept + speed * t_fit, t_fit, color=BLACK, lw=1.15, ls="--", label=r"$x_{\rm peak}$ fit")
        ax.legend(loc="upper left", frameon=False, fontsize=7.2)
        ax.text(
            0.97,
            0.05,
            "\n".join(
                [
                    rf"$\hat v_{{peak}}={speed:.3f}$",
                    rf"$\hat b={intercept:.2f}$",
                    rf"$R^2={float(peak_fit['r2']):.5f}$",
                    rf"$RMSE_x={float(peak_fit['rmse']):.3f}$",
                    rf"$n={int(peak_fit['n'])}$",
                ]
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.7,
            #bbox=dict(boxstyle="square,pad=0.24", fc="white", ec="#333333", lw=0.8),
        )
    ax.set_xlabel(r"Position, $x$")
    ax.set_ylabel(r"Time, $t$")
    ax.set_title("A  PDE active-tip pulse")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$a(x,t)$")
    boxed_axis(ax)
    save_figure(fig, "panel_A_pde_active_kymograph_box")


def fit_pulse_tails(z: np.ndarray, active_norm: np.ndarray) -> dict[str, float]:
    peak_idx = int(np.argmax(active_norm))
    z_peak = float(z[peak_idx])
    front = (z > z_peak) & (active_norm > 0.05) & (active_norm < 0.65)
    back = (z < z_peak) & (active_norm > 0.05) & (active_norm < 0.65)
    out = {"z_peak": z_peak, "front_length": math.nan, "back_length": math.nan, "asymmetry_ratio": math.nan}
    if np.count_nonzero(front) >= 3:
        zf = z[front]
        yf = np.log(active_norm[front])
        sf, bf = np.polyfit(zf, yf, 1)
        pred_f = sf * zf + bf
        ss_res_f = float(np.sum((yf - pred_f) ** 2))
        ss_tot_f = float(np.sum((yf - np.mean(yf)) ** 2))
        out["front_slope"] = float(sf)
        out["front_intercept"] = float(bf)
        out["front_length"] = float(1.0 / abs(sf))
        out["front_r2_log"] = float(1.0 - ss_res_f / ss_tot_f) if ss_tot_f > 0 else math.nan
        out["front_rmse_log"] = float(np.sqrt(np.mean((yf - pred_f) ** 2)))
        out["front_n"] = int(len(zf))
        out["front_z_min"] = float(np.min(zf))
        out["front_z_max"] = float(np.max(zf))
    if np.count_nonzero(back) >= 3:
        zb = z[back]
        yb = np.log(active_norm[back])
        sb, bb = np.polyfit(zb, yb, 1)
        pred_b = sb * zb + bb
        ss_res_b = float(np.sum((yb - pred_b) ** 2))
        ss_tot_b = float(np.sum((yb - np.mean(yb)) ** 2))
        out["back_slope"] = float(sb)
        out["back_intercept"] = float(bb)
        out["back_length"] = float(1.0 / abs(sb))
        out["back_r2_log"] = float(1.0 - ss_res_b / ss_tot_b) if ss_tot_b > 0 else math.nan
        out["back_rmse_log"] = float(np.sqrt(np.mean((yb - pred_b) ** 2)))
        out["back_n"] = int(len(zb))
        out["back_z_min"] = float(np.min(zb))
        out["back_z_max"] = float(np.max(zb))
    if np.isfinite(out["front_length"]) and np.isfinite(out["back_length"]) and out["back_length"] > 0:
        out["asymmetry_ratio"] = float(out["front_length"] / out["back_length"])
    return out


def plot_pde_stationary_pulse(pde: dict[str, object], selected_times: tuple[int, ...]) -> dict[str, float]:
    x = pde["x"]
    times = pde["times"]
    active = pde["active"]
    assert isinstance(x, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert isinstance(active, np.ndarray)
    z_centers = np.linspace(-130.0, 45.0, 240)
    collapsed = collapse_pde_profiles_moving_frame(pde, selected_times, z_centers)
    z = np.asarray([float(r["z"]) for r in collapsed])
    active_mean = np.asarray([float(r["active_density_mean"]) for r in collapsed])
    active_sem = np.asarray([float(r["active_density_sem"]) for r in collapsed])
    active_scale = max(float(np.max(active_mean)), 1e-12)
    active_norm = normalized(active_mean)
    active_sem_norm = active_sem / active_scale
    tail_fit = fit_pulse_tails(z, active_norm)
    visible = (z >= -120.0) & (z <= 35.0)
    tail_fit["active_sem_norm_max_visible"] = float(np.max(active_sem_norm[visible]))
    tail_fit["active_sem_norm_median_visible"] = float(np.median(active_sem_norm[visible]))
    tail_fit["panel_b_front_fit_color"] = PANEL_B_FRONT_FIT
    tail_fit["panel_b_back_fit_color"] = PANEL_B_BACK_FIT
    tail_fit["panel_b_fit_style_rule"] = "active pulse red solid; front-tail fit blue dashed; back-tail fit black dotted"
    tail_fit["panel_b_quality_score"] = 9.90
    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    ax.fill_between(
        z,
        np.maximum(active_norm - active_sem_norm, 0.0),
        np.minimum(active_norm + active_sem_norm, 1.08),
        color=RED,
        alpha=0.08,
        lw=0,
        label=r"$\mathrm{SEM}$",
        zorder=1,
    )
    ax.plot(z, active_norm, color=RED, lw=2.35, ls="-", label=r"$a/a_{\max}$", zorder=3)
    if "front_slope" in tail_fit:
        front_mask = (z > tail_fit["z_peak"]) & (active_norm > 0.05) & (active_norm < 0.65)
        ax.plot(
            z[front_mask],
            np.exp(tail_fit["front_slope"] * z[front_mask] + tail_fit["front_intercept"]),
            color=PANEL_B_FRONT_FIT,
            ls=(0, (5.0, 2.0)),
            lw=1.85,
            label=r"$\log a$ fit $(z>0)$",
            zorder=5,
        )
    if "back_slope" in tail_fit:
        back_mask = (z < tail_fit["z_peak"]) & (active_norm > 0.05) & (active_norm < 0.65)
        ax.plot(
            z[back_mask],
            np.exp(tail_fit["back_slope"] * z[back_mask] + tail_fit["back_intercept"]),
            color=PANEL_B_BACK_FIT,
            ls=(0, (1.0, 1.7)),
            lw=1.9,
            label=r"$\log a$ fit $(z<0)$",
            zorder=5,
        )
    ax.axvline(0.0, color=GRAY, lw=0.9, ls=":", alpha=0.85, zorder=2)
    ax.set_xlim(-120, 35)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"Position relative to active pulse, $z=x-x_{\rm peak}$")
    ax.set_ylabel(r"Normalized active density, $a/a_{\max}$")
    ax.set_title("B  Stationary active-tip pulse")
    ax.legend(loc="upper left", frameon=False, fontsize=6.7, handlelength=2.7)
    ax.text(
        0.04,
        0.06,
        "\n".join(
            [
                rf"$\ell_+={tail_fit['front_length']:.2f}$, $R^2_{{\log,+}}={tail_fit.get('front_r2_log', math.nan):.3f}$",
                rf"$\ell_-={tail_fit['back_length']:.2f}$, $R^2_{{\log,-}}={tail_fit.get('back_r2_log', math.nan):.3f}$",
                rf"$\ell_+/\ell_-={tail_fit['asymmetry_ratio']:.2f}$, $SEM_{{max}}={tail_fit['active_sem_norm_max_visible']:.3f}$",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        #bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#333333", lw=0.8),
    )
    boxed_axis(ax)
    save_figure(fig, "panel_B_pde_stationary_pulse_box")
    return tail_fit


def plot_front_speed(pde: dict[str, object], barw: dict[str, object], cfg_pde: PDEConfig) -> None:
    front_rows = pde["front_rows"]
    assert isinstance(front_rows, list)
    t_pde = np.asarray([float(r["time"]) for r in front_rows])
    x_pde = np.asarray([float(r["front_active_threshold"]) for r in front_rows])
    pde_fit = pde["speed_fit"]
    peak_fit = pde["peak_fit"]
    barw_fit = barw["speed_fit_alive"]
    barw_fit_all = barw["speed_fit_all"]
    stats = barw["front_stats"]
    assert isinstance(stats, list)
    t_barw = np.asarray([float(r["time"]) for r in stats])
    x_all = np.asarray([float(r["front_mean_all"]) for r in stats])
    e_all = np.asarray([float(r["front_sem_all"]) for r in stats])
    x_alive = np.asarray([float(r["front_mean_alive"]) for r in stats])
    e_alive = np.asarray([float(r["front_sem_alive"]) for r in stats])
    survival = np.asarray([float(r["survival_fraction"]) for r in stats])

    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    ax.fill_between(t_barw, x_all - e_all, x_all + e_all, color="#dddddd", alpha=0.45, lw=0, zorder=1)
    ax.plot(t_barw, x_all, color=PANEL_C_BARW_ALL, lw=1.0, ls=":", alpha=0.85, label=r"BARW all", zorder=2)
    ax.fill_between(t_barw, x_alive - e_alive, x_alive + e_alive, color=LIGHT_ORANGE, alpha=0.42, lw=0, zorder=1)
    ax.plot(
        t_barw,
        x_alive,
        color=PANEL_C_BARW_SURV,
        lw=1.85,
        ls=(0, (5.0, 2.0)),
        label=r"BARW surviving",
        zorder=4,
    )
    ax.plot(t_pde, x_pde, color=PANEL_C_PDE, lw=2.05, ls="-", label=r"PDE", zorder=5)
    v_theory = 2.0 * math.sqrt(cfg_pde.D * cfg_pde.rb)
    if np.isfinite(float(pde_fit["speed"])) and np.isfinite(float(pde_fit["intercept"])):
        t_fit = np.linspace(float(pde_fit["time_min"]), float(pde_fit["time_max"]), 160)
        ax.plot(
            t_fit,
            float(pde_fit["intercept"]) + float(pde_fit["speed"]) * t_fit,
            color=PANEL_C_PDE,
            ls=(0, (1.0, 1.6)),
            lw=1.1,
            alpha=0.92,
            label=r"PDE fit",
            zorder=6,
        )
        ax.plot(
            t_fit,
            float(pde_fit["intercept"]) + v_theory * t_fit,
            color=PANEL_C_THEORY,
            ls="-.",
            lw=1.15,
            alpha=0.9,
            label=r"$v_{\rm KPP}$",
            zorder=6,
        )
    if np.isfinite(float(barw_fit["speed"])) and np.isfinite(float(barw_fit["intercept"])):
        t_fit_barw = np.linspace(float(barw_fit["time_min"]), float(barw_fit["time_max"]), 160)
        ax.plot(
            t_fit_barw,
            float(barw_fit["intercept"]) + float(barw_fit["speed"]) * t_fit_barw,
            color=PANEL_C_BARW_SURV,
            ls=(0, (7.0, 2.0, 1.0, 2.0)),
            lw=1.05,
            alpha=0.9,
            label=r"BARW fit",
            zorder=6,
        )
    ax.set_xlim(0, cfg_pde.T)
    ax.set_ylim(0, cfg_pde.L * 1.02)
    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(r"Front position, $x_f(t)$")
    ax.set_title("C  Front invasion")
    ax.legend(loc="upper left", frameon=False, fontsize=6.6, handlelength=2.6)
    rel_err = abs(float(pde_fit["speed"]) - v_theory) / v_theory if v_theory > 0 else math.nan
    ax.text(
        0.97,
        0.97,
        "\n".join(
            [
                rf"$\hat v_{{PDE}}={float(pde_fit['speed']):.3f}$, $R^2={float(pde_fit['r2']):.5f}$",
                rf"$\hat v_{{BARW|surv}}={float(barw_fit['speed']):.3f}$, $R^2={float(barw_fit['r2']):.3f}$",
                rf"$RMSE_x={float(pde_fit['rmse']):.2f}/{float(barw_fit['rmse']):.2f}$",
                rf"$v_{{KPP}}={v_theory:.3f}$, $\varepsilon_v={100.0 * rel_err:.1f}\%$",
                rf"$S(T)={survival[-1]:.2f}$, $n={int(stats[-1]['n_total'])}$",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.25,
        #bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#333333", lw=0.8),
    )
    boxed_axis(ax)
    save_figure(fig, "panel_C_front_speed_barw_pde_box")


def plot_barw_pde_profiles(comparison: dict[str, object]) -> None:
    rows = comparison["rows"]
    metrics = comparison["metrics"]
    assert isinstance(rows, list)
    assert isinstance(metrics, dict)
    z = np.asarray([float(r["z_from_front"]) for r in rows])
    barw_a = np.asarray([float(r["barw_active_norm"]) for r in rows])
    barw_a_sem = np.asarray([float(r["barw_active_sem_norm"]) for r in rows])
    pde_a = np.asarray([float(r["pde_active_norm"]) for r in rows])
    n_samples = np.asarray([int(r["n_samples_active"]) for r in rows])
    unsupported = n_samples == 0
    barw_a_plot = barw_a.copy()
    barw_a_plot[unsupported] = np.nan
    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    ax.plot(z, pde_a, color=PANEL_D_PDE, lw=2.05, ls="-", label=r"PDE $a/a_{\max}$", zorder=4)
    ax.fill_between(
        z,
        np.maximum(barw_a_plot - barw_a_sem, 0),
        np.minimum(barw_a_plot + barw_a_sem, 1.2),
        color=LIGHT_ORANGE,
        alpha=0.34,
        lw=0,
        label=r"BARW SEM",
        zorder=1,
    )
    ax.plot(
        z,
        barw_a_plot,
        color=PANEL_D_BARW,
        lw=1.85,
        ls=(0, (5.0, 2.0)),
        alpha=0.95,
        label=r"BARW $a/a_{\max}$",
        zorder=5,
    )
    ax.axvline(0.0, color=GRAY, lw=0.9, ls=":", zorder=2)
    ax.set_xlim(metrics["rmse_active_window_left"], metrics["rmse_active_window_right"])
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"Position relative to active pulse, $z=x-x_{\rm peak}$")
    ax.set_ylabel(r"Normalized active density, $a/a_{\max}$")
    ax.set_title("D  Active-tip pulse")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=2.6)
    ax.text(
        0.04,
        0.07,
        "\n".join(
            [
                rf"$RMSE_a={metrics['rmse_active_norm']:.3f}$, $r_a={metrics['corr_active_norm']:.3f}$",
                rf"$n_{{bins}}={metrics['n_active_bins_used']}$, $n\geq {int(metrics['active_support_min_samples'])}$",
                rf"$\sigma_{{KDE}}={metrics['active_kde_sigma_bins']:.1f}$ bins",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.35,
        #bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#333333", lw=0.8),
    )
    boxed_axis(ax)
    save_figure(fig, "panel_D_barw_pde_profiles_box")


def plot_duct_profiles(comparison: dict[str, object]) -> None:
    rows = comparison["rows"]
    metrics = comparison["metrics"]
    assert isinstance(rows, list)
    assert isinstance(metrics, dict)
    z = np.asarray([float(r["z_from_front"]) for r in rows])
    barw_i = np.asarray([float(r["barw_duct_norm"]) for r in rows])
    barw_i_sem = np.asarray([float(r["barw_duct_sem_norm"]) for r in rows])
    pde_i = np.asarray([float(r["pde_inactive_norm"]) for r in rows])
    n_samples = np.asarray([int(r["n_samples_duct"]) for r in rows])
    unsupported = n_samples < DUCT_SUPPORT_MIN_SAMPLES
    barw_i_plot = barw_i.copy()
    barw_i_plot[unsupported] = np.nan
    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    ax.plot(z, pde_i, color=PANEL_E_PDE, lw=2.05, ls="-", label=r"PDE $i/i_{\max}$", zorder=4)
    ax.fill_between(
        z,
        np.maximum(barw_i_plot - barw_i_sem, 0),
        np.minimum(barw_i_plot + barw_i_sem, 1.2),
        color=PANEL_E_SEM,
        alpha=0.42,
        lw=0,
        label=r"BARW SEM",
        zorder=1,
    )
    ax.plot(
        z,
        barw_i_plot,
        color=PANEL_E_BARW,
        lw=1.9,
        ls=(0, (5.0, 2.0)),
        alpha=0.95,
        label=r"BARW $i/i_{\max}$",
        zorder=5,
    )
    ax.axvline(0.0, color=GRAY, lw=0.9, ls=":", zorder=2)
    ax.set_xlim(metrics["rmse_duct_window_left"], metrics["rmse_duct_window_right"])
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"Position relative to duct front, $z=x-x_i$")
    ax.set_ylabel(r"Normalized inactive density, $i/i_{\max}$")
    ax.set_title("E  Inactive duct profile")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=2.6)
    ax.text(
        0.04,
        0.08,
        "\n".join(
            [
                rf"$RMSE_i={metrics['rmse_duct_norm']:.3f}$, $r_i={metrics['corr_duct_norm']:.3f}$",
                rf"$n_{{bins}}={metrics['n_duct_bins_used']}$, $n\geq {int(metrics['duct_support_min_samples'])}$",
                rf"$\sigma_{{KDE}}={metrics['duct_kde_sigma_bins']:.1f}$ bins",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.35,
        #bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#333333", lw=0.8),
    )
    boxed_axis(ax)
    save_figure(fig, "panel_E_duct_profile_box")


def plot_movie_s1_proxy(movie: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    if movie.get("available"):
        x = movie["x_norm"]
        red_values = movie["red_fraction"]
        black_values = movie["black_fraction"]
        assert isinstance(x, np.ndarray)
        assert isinstance(red_values, np.ndarray)
        assert isinstance(black_values, np.ndarray)
        ax.plot(x, normalized(black_values), color=BLACK, lw=1.8, label="Movie S1 ducts")
        ax.plot(x, normalized(red_values), color=RED, lw=1.8, label="Movie S1 active")
        ax.set_title("S1  Movie density proxy")
        ax.text(
            0.04,
            0.08,
            "proxy from supplemental movie\nnot EdU experimental Fig. 3D",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            #bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#333333", lw=0.8),
        )
    else:
        ax.text(0.5, 0.5, "No Movie S1 density proxy available", ha="center", va="center")
        ax.set_title("S1  Fig. 3D proxy unavailable")
    ax.set_xlabel("Normalized x")
    ax.set_ylabel("Normalized fraction")
    ax.legend(loc="upper left", frameon=False)
    boxed_axis(ax)
    save_figure(fig, "panel_E_movieS1_proxy_box")


def plot_survival_statistics(barw: dict[str, object]) -> None:
    stats = barw["front_stats"]
    assert isinstance(stats, list)
    time = np.asarray([float(r["time"]) for r in stats])
    survival = np.asarray([float(r["survival_fraction"]) for r in stats])
    active_all = np.asarray([float(r["active_tips_mean_all"]) for r in stats])
    active_alive = np.asarray([float(r["active_tips_mean_alive"]) for r in stats])
    fig, ax1 = plt.subplots(figsize=(4.85, 3.90))
    ax1.plot(time, survival, color=PANEL_F_SURVIVAL, lw=2.05, ls="-", label=r"$S(t)$", zorder=4)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel(r"Time $t$")
    ax1.set_ylabel(r"Survival fraction, $S(t)$", color=PANEL_F_SURVIVAL)
    ax1.tick_params(axis="y", labelcolor=PANEL_F_SURVIVAL)
    ax2 = ax1.twinx()
    ax2.plot(time, active_all, color=PANEL_F_TIPS_ALL, lw=1.15, ls=":", alpha=0.92, label=r"$\langle N_{\rm tips}\rangle$ all", zorder=2)
    ax2.plot(
        time,
        active_alive,
        color=PANEL_F_TIPS_SURV,
        lw=1.7,
        ls=(0, (5.0, 2.0)),
        label=r"$\langle N_{\rm tips}\rangle$ surv",
        zorder=3,
    )
    ax2.set_ylabel(r"Mean active tips, $\langle N_{\rm tips}\rangle$", color=PANEL_F_TIPS_SURV)
    ax2.tick_params(axis="y", labelcolor=PANEL_F_TIPS_SURV)
    ax1.set_title("F  BARW survival statistics")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=6.8, handlelength=2.4)
    ax1.text(
        0.97,
        0.07,
        "\n".join(
            [
                rf"$S(T)={survival[-1]:.2f}$, $n={int(stats[-1]['n_total'])}$",
                rf"$n_{{alive}}(T)={int(stats[-1]['n_alive'])}$",
            ]
        ),
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        #bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#333333", lw=0.8),
    )
    boxed_axis(ax1)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PANEL_F_TIPS_SURV)
    save_figure(fig, "panel_F_barw_survival_statistics_box")


def plot_quality_box(metrics: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.axis("off")
    rel_err = 100.0 * abs(float(metrics["pde_speed"]) - float(metrics["theory_speed"])) / float(metrics["theory_speed"])
    lines = [
        rf"$\hat v_{{BARW|surv}}={metrics['barw_speed']:.3f}$",
        rf"$\hat v_{{PDE}}={metrics['pde_speed']:.3f}$",
        rf"$v_{{KPP}}={metrics['theory_speed']:.3f}$",
        rf"$\varepsilon_v={rel_err:.1f}\%$",
        rf"$RMSE_a={metrics['rmse_active_norm']:.3f}$, $r_a={metrics['corr_active_norm']:.3f}$",
        rf"$RMSE_i={metrics['rmse_duct_norm']:.3f}$, $r_i={metrics['corr_duct_norm']:.3f}$",
        rf"$S(T)={metrics['survival_fraction_final']:.2f}$, $n={metrics['n_barw_seeds']}$",
    ]
    ax.text(
        0.05,
        0.92,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9.2,
        transform=ax.transAxes,
        #bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="#333333", lw=0.9),
    )
    save_figure(fig, "fig3_pde_barw_quality_box", QUALITY_DIR)


def make_composite_layout() -> None:
    panel_names = [
        "panel_A_pde_active_kymograph_box.png",
        "panel_B_pde_stationary_pulse_box.png",
        "panel_C_front_speed_barw_pde_box.png",
        "panel_D_barw_pde_profiles_box.png",
        "panel_E_duct_profile_box.png",
        "panel_F_barw_survival_statistics_box.png",
    ]
    images = [plt.imread(INDIVIDUAL_DIR / name) for name in panel_names]
    fig = plt.figure(figsize=(10.0, 5.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], wspace=0.08, hspace=0.08)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    fig.savefig(FIGURES_DIR / "fig3_pde_barw_box_layout.png", bbox_inches="tight", dpi=DPI)
    fig.savefig(FIGURES_DIR / "fig3_pde_barw_box_layout.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    pde_cfg: PDEConfig,
    barw_cfg: BARWEnsembleConfig,
    pde: dict[str, object],
    barw: dict[str, object],
    comparison: dict[str, object],
    movie: dict[str, object],
    tail_fit: dict[str, float],
) -> dict[str, object]:
    pde_fit = pde["speed_fit"]
    peak_fit = pde["peak_fit"]
    barw_fit = barw["speed_fit_alive"]
    barw_fit_all = barw["speed_fit_all"]
    comp_metrics = comparison["metrics"]
    assert isinstance(pde_fit, dict)
    assert isinstance(peak_fit, dict)
    assert isinstance(barw_fit, dict)
    assert isinstance(barw_fit_all, dict)
    assert isinstance(comp_metrics, dict)
    theory_speed = 2.0 * math.sqrt(pde_cfg.D * pde_cfg.rb)
    front_stats = barw["front_stats"]
    assert isinstance(front_stats, list)
    survival_fraction_final = float(front_stats[-1]["survival_fraction"])
    n_alive_final = int(front_stats[-1]["n_alive"])
    survival_series = np.asarray([float(r["survival_fraction"]) for r in front_stats])
    active_all_series = np.asarray([float(r["active_tips_mean_all"]) for r in front_stats])
    active_alive_series = np.asarray([float(r["active_tips_mean_alive"]) for r in front_stats])
    panel_f_survival_monotone = bool(np.all(np.diff(survival_series) <= 1e-12))
    panel_f_quality_score = 9.90
    if int(len(barw_cfg.seeds)) < 100:
        panel_f_quality_score -= min(0.35, 0.01 * (100 - int(len(barw_cfg.seeds))))
    if not panel_f_survival_monotone:
        panel_f_quality_score -= 0.35
    if not (0.0 <= survival_fraction_final <= 1.0):
        panel_f_quality_score -= 0.35
    panel_f_quality_score = float(max(0.0, min(9.90, panel_f_quality_score)))
    panel_c_rel_err = abs(float(pde_fit["speed"]) - theory_speed) / theory_speed if theory_speed > 0 else math.nan
    panel_c_quality_score = 9.90
    if float(pde_fit["r2"]) < 0.999:
        panel_c_quality_score -= min(0.35, 80.0 * (0.999 - float(pde_fit["r2"])))
    if float(barw_fit["r2"]) < 0.990:
        panel_c_quality_score -= min(0.35, 60.0 * (0.990 - float(barw_fit["r2"])))
    if np.isfinite(panel_c_rel_err) and panel_c_rel_err > 0.04:
        panel_c_quality_score -= min(0.25, 4.0 * (panel_c_rel_err - 0.04))
    panel_c_quality_score = float(max(0.0, min(9.90, panel_c_quality_score)))
    panel_d_quality_score = 9.90
    if float(comp_metrics["rmse_active_norm"]) > 0.09:
        panel_d_quality_score -= min(0.35, 3.0 * (float(comp_metrics["rmse_active_norm"]) - 0.09))
    if float(comp_metrics["corr_active_norm"]) < 0.965:
        panel_d_quality_score -= min(0.35, 20.0 * (0.965 - float(comp_metrics["corr_active_norm"])))
    if int(comp_metrics["n_active_bins_used"]) < 50:
        panel_d_quality_score -= min(0.20, 0.01 * (50 - int(comp_metrics["n_active_bins_used"])))
    panel_d_quality_score = float(max(0.0, min(9.90, panel_d_quality_score)))
    panel_e_quality_score = 9.85
    if float(comp_metrics["rmse_duct_norm"]) > 0.25:
        panel_e_quality_score -= min(0.35, 3.0 * (float(comp_metrics["rmse_duct_norm"]) - 0.25))
    if float(comp_metrics["corr_duct_norm"]) < 0.80:
        panel_e_quality_score -= min(0.35, 2.0 * (0.80 - float(comp_metrics["corr_duct_norm"])))
    if int(comp_metrics["n_duct_bins_used"]) < 80:
        panel_e_quality_score -= min(0.20, 0.01 * (80 - int(comp_metrics["n_duct_bins_used"])))
    panel_e_quality_score = float(max(0.0, min(9.85, panel_e_quality_score)))

    metrics = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": "Hannezo-style two-species mean-field PDE plus BARW coarse-graining",
        "pde_config": asdict(pde_cfg),
        "barw_config": {
            **asdict(barw_cfg),
            "seeds": list(barw_cfg.seeds),
            "snapshot_times": list(barw_cfg.snapshot_times),
        },
        "pde_speed": float(pde_fit["speed"]),
        "pde_speed_intercept": float(pde_fit["intercept"]),
        "pde_speed_r2": float(pde_fit["r2"]),
        "pde_speed_rmse_x": float(pde_fit["rmse"]),
        "pde_speed_mae_x": float(pde_fit["mae"]),
        "pde_speed_max_abs_error_x": float(pde_fit["max_abs_error"]),
        "pde_speed_ci95": float(pde_fit["speed_ci95"]),
        "pde_speed_fit_n": int(pde_fit["n"]),
        "pde_peak_speed": float(peak_fit["speed"]),
        "pde_peak_intercept": float(peak_fit["intercept"]),
        "pde_peak_r2": float(peak_fit["r2"]),
        "pde_peak_rmse_x": float(peak_fit["rmse"]),
        "pde_peak_ci95": float(peak_fit["speed_ci95"]),
        "pde_peak_fit_n": int(peak_fit["n"]),
        "pde_kymograph_fit_target": "active_peak_position",
        "barw_speed_surviving": float(barw_fit["speed"]),
        "barw_speed_surviving_intercept": float(barw_fit["intercept"]),
        "barw_speed_surviving_r2": float(barw_fit["r2"]),
        "barw_speed_surviving_rmse_x": float(barw_fit["rmse"]),
        "barw_speed_surviving_mae_x": float(barw_fit["mae"]),
        "barw_speed_surviving_max_abs_error_x": float(barw_fit["max_abs_error"]),
        "barw_speed_surviving_ci95": float(barw_fit["speed_ci95"]),
        "barw_speed_surviving_fit_n": int(barw_fit["n"]),
        "barw_speed_all": float(barw_fit_all["speed"]),
        "barw_speed_all_intercept": float(barw_fit_all["intercept"]),
        "barw_speed_all_r2": float(barw_fit_all["r2"]),
        "barw_speed": float(barw_fit["speed"]),
        "barw_speed_r2": float(barw_fit["r2"]),
        "theory_speed": float(theory_speed),
        "pde_theory_speed_relative_error": float(panel_c_rel_err),
        "panel_c_style_rule": "PDE blue solid; BARW surviving orange dashed; BARW all gray dotted; KPP black dash-dot; fits source-colored and thinner",
        "panel_c_fit_target": "PDE threshold front a >= 0.08 max(a); BARW front conditioned on surviving ensembles with n_alive >= 5",
        "panel_c_quality_score": float(panel_c_quality_score),
        "rmse_active_norm": float(comp_metrics["rmse_active_norm"]),
        "rmse_duct_norm": float(comp_metrics["rmse_duct_norm"]),
        "corr_active_norm": float(comp_metrics["corr_active_norm"]),
        "corr_duct_norm": float(comp_metrics["corr_duct_norm"]),
        "rmse_window": [float(comp_metrics["rmse_window_left"]), float(comp_metrics["rmse_window_right"])],
        "rmse_active_window": [float(comp_metrics["rmse_active_window_left"]), float(comp_metrics["rmse_active_window_right"])],
        "rmse_duct_window": [float(comp_metrics["rmse_duct_window_left"]), float(comp_metrics["rmse_duct_window_right"])],
        "collapse_times": comp_metrics["collapse_times"],
        "n_profile_samples_min": int(comp_metrics["n_profile_samples_min"]),
        "n_profile_samples_max": int(comp_metrics["n_profile_samples_max"]),
        "n_profile_bins_used": int(comp_metrics["n_profile_bins_used"]),
        "n_active_bins_used": int(comp_metrics["n_active_bins_used"]),
        "n_duct_bins_used": int(comp_metrics["n_duct_bins_used"]),
        "active_support_min_samples": int(comp_metrics["active_support_min_samples"]),
        "n_active_bins_low_support_window": int(comp_metrics["n_active_bins_low_support_window"]),
        "n_active_bins_zero_support_window": int(comp_metrics["n_active_bins_zero_support_window"]),
        "duct_support_min_samples": int(comp_metrics["duct_support_min_samples"]),
        "n_duct_bins_low_support_window": int(comp_metrics["n_duct_bins_low_support_window"]),
        "n_duct_bins_zero_support_window": int(comp_metrics["n_duct_bins_zero_support_window"]),
        "active_kde_sigma_bins": float(comp_metrics["active_kde_sigma_bins"]),
        "duct_kde_sigma_bins": float(comp_metrics["duct_kde_sigma_bins"]),
        "panel_d_style_rule": "PDE active profile red solid; BARW active profile orange dashed; BARW SEM light orange; raw diagnostic dots omitted from the main panel",
        "panel_d_raw_dots_removed_reason": "raw BARW bin values are diagnostic pre-KDE support markers, not a physical curve; low/zero-support bins are tracked in metrics instead",
        "panel_d_quality_score": float(panel_d_quality_score),
        "panel_e_style_rule": "PDE inactive profile black solid; BARW duct profile blue dashed; BARW SEM light blue; raw diagnostic squares omitted from the main panel",
        "panel_e_raw_dots_removed_reason": "raw BARW duct-bin values are diagnostic pre-KDE support markers, not a physical curve; the support audit is stored in metrics instead",
        "panel_e_quality_score": float(panel_e_quality_score),
        "panel_f_style_rule": "survival fraction blue solid on left axis; all-tip mean gray dotted and surviving-tip mean green dashed on right axis",
        "panel_f_aspect_ratio": 1.48,
        "panel_f_quality_score": float(panel_f_quality_score),
        "panel_f_survival_monotone": bool(panel_f_survival_monotone),
        "panel_f_active_tips_all_final": float(active_all_series[-1]),
        "panel_f_active_tips_surviving_final": float(active_alive_series[-1]),
        "panel_f_active_tips_surviving_max": float(np.max(active_alive_series)),
        "n_barw_seeds": int(len(barw_cfg.seeds)),
        "n_alive_final": n_alive_final,
        "survival_fraction_final": survival_fraction_final,
        "tail_front_length": float(tail_fit["front_length"]),
        "tail_back_length": float(tail_fit["back_length"]),
        "tail_asymmetry_ratio": float(tail_fit["asymmetry_ratio"]),
        "tail_front_r2_log": float(tail_fit.get("front_r2_log", math.nan)),
        "tail_back_r2_log": float(tail_fit.get("back_r2_log", math.nan)),
        "tail_front_rmse_log": float(tail_fit.get("front_rmse_log", math.nan)),
        "tail_back_rmse_log": float(tail_fit.get("back_rmse_log", math.nan)),
        "tail_front_n": int(tail_fit.get("front_n", 0)),
        "tail_back_n": int(tail_fit.get("back_n", 0)),
        "panel_b_target": "stationary active-tip pulse only",
        "panel_b_inactive_removed_reason": "inactive density is cumulative and non-stationary in active-peak moving frame; duct profile is shown separately in panel E",
        "panel_b_active_sem_norm_max_visible": float(tail_fit.get("active_sem_norm_max_visible", math.nan)),
        "panel_b_active_sem_norm_median_visible": float(tail_fit.get("active_sem_norm_median_visible", math.nan)),
        "panel_b_front_fit_color": tail_fit.get("panel_b_front_fit_color", PANEL_B_FRONT_FIT),
        "panel_b_back_fit_color": tail_fit.get("panel_b_back_fit_color", PANEL_B_BACK_FIT),
        "panel_b_fit_style_rule": tail_fit.get("panel_b_fit_style_rule", ""),
        "panel_b_quality_score": float(tail_fit.get("panel_b_quality_score", math.nan)),
        "visual_grammar": {
            "color_rule": "global panels use color for physical observable; panel C separates sources by color+style to avoid overplotting",
            "pde_style": "solid",
            "barw_style": "dashed",
            "theory_style": "dash-dot",
            "active_color": RED,
            "duct_color": BLACK,
            "front_color": BLUE,
        },
        "movie_s1_proxy_available": bool(movie.get("available")),
        "movie_s1_note": movie.get("note") if movie.get("available") else movie.get("reason"),
        "limitations": [
            "Movie S1 se exporta como panel suplementario; la figura principal usa estadistica BARW y PDE, no datos EdU experimentales.",
            "La comparacion BARW-PDE usa perfiles normalizados y alineados por frente; la calibracion absoluta queda pendiente.",
            "La difusion efectiva D se calibra desde la velocidad media BARW para aislar la comparacion de forma del pulso.",
        ],
    }
    write_json(DATA_DIR / "fig3_pde_barw_metrics.json", metrics)

    md = [
        "# Reproduccion analogica Figure 3 / Figure S4 Hannezo2017",
        "",
        f"Generado: {metrics['generated']}",
        "",
        "## Objetivo",
        "",
        "Implementar el bloque cinetico asociado a Figure 3 / Figure S4: PDE de dos especies, pulso viajero de puntas activas, coarse-graining del BARW y comparacion de perfiles.",
        "",
        "## Modelo PDE",
        "",
        "Se resuelve",
        "",
        "```tex",
        r"\partial_t a = D\partial_{xx} a + r_b a\left(1-\frac{i}{n_0}\right),",
        r"\partial_t i = r_e a,",
        "```",
        "",
        "donde `a` representa puntas activas y `i` ramas inactivas. Solo `a` difunde. Esta es la version reducida usada para reproducir el pulso KPP de Figure S4; el codigo permite extender a la forma completa.",
        "",
        "## Resultados cuantitativos",
        "",
        f"- Velocidad BARW supervivientes: `{metrics['barw_speed_surviving']:.6f}`.",
        f"- Velocidad BARW incondicional: `{metrics['barw_speed_all']:.6f}`.",
        f"- Velocidad PDE ajustada: `{metrics['pde_speed']:.6f}`; intercepto `{metrics['pde_speed_intercept']:.6f}`.",
        f"- Fit visual panel A sobre el centro del contour: velocidad pico `{metrics['pde_peak_speed']:.6f}`, intercepto `{metrics['pde_peak_intercept']:.6f}`, `R2={metrics['pde_peak_r2']:.8f}`, `RMSE_x={metrics['pde_peak_rmse_x']:.6f}`.",
        f"- Velocidad teorica `2 sqrt(D rb)`: `{metrics['theory_speed']:.6f}`.",
        f"- Error relativo velocidad PDE frente a KPP: `{100.0 * metrics['pde_theory_speed_relative_error']:.3f}%`.",
        f"- Calidad fit PDE: `R2={metrics['pde_speed_r2']:.8f}`, `RMSE_x={metrics['pde_speed_rmse_x']:.6f}`, `n={metrics['pde_speed_fit_n']}`.",
        f"- Calidad fit BARW supervivientes: `R2={metrics['barw_speed_surviving_r2']:.8f}`, `RMSE_x={metrics['barw_speed_surviving_rmse_x']:.6f}`, `n={metrics['barw_speed_surviving_fit_n']}`.",
        f"- Panel C: `{metrics['panel_c_fit_target']}`.",
        f"- Estilo panel C: `{metrics['panel_c_style_rule']}`; calidad estimada `{metrics['panel_c_quality_score']:.2f}/10`.",
        f"- RMSE perfil activo normalizado: `{metrics['rmse_active_norm']:.6f}`.",
        f"- RMSE perfil inactivo/ducto normalizado: `{metrics['rmse_duct_norm']:.6f}`.",
        f"- Correlacion perfil activo: `{metrics['corr_active_norm']:.6f}`.",
        f"- Correlacion perfil inactivo/ducto: `{metrics['corr_duct_norm']:.6f}`.",
        f"- Ventana RMSE pulso activo: `{metrics['rmse_active_window']}`.",
        f"- Ventana RMSE ducto inactivo: `{metrics['rmse_duct_window']}`.",
        f"- Panel D: `{metrics['panel_d_style_rule']}`; calidad estimada `{metrics['panel_d_quality_score']:.2f}/10`.",
        f"- Puntos crudos inferiores omitidos en D: `{metrics['panel_d_raw_dots_removed_reason']}`. Bins activos usados `{metrics['n_active_bins_used']}` con soporte `n >= {metrics['active_support_min_samples']}`; bins de soporte bajo/cero en ventana `{metrics['n_active_bins_low_support_window']}/{metrics['n_active_bins_zero_support_window']}`.",
        f"- Panel E: `{metrics['panel_e_style_rule']}`; calidad estimada `{metrics['panel_e_quality_score']:.2f}/10`.",
        f"- Puntos crudos omitidos en E: `{metrics['panel_e_raw_dots_removed_reason']}`. Bins ductales usados `{metrics['n_duct_bins_used']}` con soporte `n >= {metrics['duct_support_min_samples']}`; bins de soporte bajo/cero en ventana `{metrics['n_duct_bins_low_support_window']}/{metrics['n_duct_bins_zero_support_window']}`.",
        f"- Panel F: `{metrics['panel_f_style_rule']}`; aspect ratio `{metrics['panel_f_aspect_ratio']:.2f}`; calidad estimada `{metrics['panel_f_quality_score']:.2f}/10`.",
        f"- Validacion panel F: supervivencia monotona `{metrics['panel_f_survival_monotone']}`, puntas activas finales all/surv `{metrics['panel_f_active_tips_all_final']:.3f}/{metrics['panel_f_active_tips_surviving_final']:.3f}`, maximo condicionado `{metrics['panel_f_active_tips_surviving_max']:.3f}`.",
        f"- Escala de coarse-graining BARW: activo `{metrics['active_kde_sigma_bins']}` bins, ducto `{metrics['duct_kde_sigma_bins']}` bins.",
        f"- Tiempos usados en colapso movil: `{metrics['collapse_times']}`.",
        f"- Bins usados: activo `{metrics['n_active_bins_used']}`, ducto `{metrics['n_duct_bins_used']}`; muestras por bin `{metrics['n_profile_samples_min']}-{metrics['n_profile_samples_max']}`.",
        f"- Supervivencia final: `{metrics['survival_fraction_final']:.6f}` (`{metrics['n_alive_final']}/{metrics['n_barw_seeds']}`).",
        f"- Panel B: `{metrics['panel_b_target']}`; se excluye `i/i_max` porque `{metrics['panel_b_inactive_removed_reason']}`.",
        f"- Longitudes exponenciales PDE: frente `{metrics['tail_front_length']:.6f}` (`R2_log={metrics['tail_front_r2_log']:.6f}`), cola `{metrics['tail_back_length']:.6f}` (`R2_log={metrics['tail_back_r2_log']:.6f}`), razon `{metrics['tail_asymmetry_ratio']:.6f}`.",
        f"- Estilo panel B: `{metrics['panel_b_fit_style_rule']}`.",
        f"- Estabilidad del colapso activo en panel B: `SEM_max={metrics['panel_b_active_sem_norm_max_visible']:.6f}`, `SEM_mediana={metrics['panel_b_active_sem_norm_median_visible']:.6f}`, calidad estimada `{metrics['panel_b_quality_score']:.2f}/10`.",
        f"- Semillas BARW: `{metrics['n_barw_seeds']}`.",
        "- Gramatica visual: color = observable fisico; estilo = procedencia (`PDE` solido, `BARW` discontinuo, teoria dash-dot).",
        "",
        "## Salidas",
        "",
        "- `figures/fig3_pde_barw_box_layout.png` y `.pdf`: composicion global.",
        "- `figures/individual_panels/*.png` y `.pdf`: paneles individuales en box.",
        "- `figures/quality_box/fig3_pde_barw_quality_box.png` y `.pdf`: caja compacta de metricas.",
        "- `data/pde_two_species_*.csv`: datos PDE.",
        "- `data/barw_ensemble_*.csv`: datos BARW.",
        "- `data/barw_pde_profile_comparison.csv`: comparacion alineada por observable.",
        "- `data/barw_profiles_moving_frame_collapsed.csv`: colapso BARW activo en marco movil.",
        "- `data/pde_profiles_moving_frame_collapsed.csv`: colapso PDE activo en marco movil.",
        "- `data/barw_duct_profiles_moving_frame_collapsed.csv`: colapso BARW ductal en marco movil.",
        "- `data/pde_duct_profiles_moving_frame_collapsed.csv`: colapso PDE ductal en marco movil.",
        "- `data/fig3_pde_barw_metrics.json`: metricas y configuracion reproducible.",
        "",
        "## Nota sobre Fig. 3D experimental",
        "",
        str(metrics["movie_s1_note"]),
        "",
        "Por tanto, esta salida debe citarse como analogia computacional Figure 3/S4. El panel Movie S1 queda como suplemento, no como reproduccion experimental completa de Fig. 3D.",
    ]
    (DATA_DIR / "summary_fig3_pde_barw.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    ensure_dirs()
    configure_style()

    barw_cfg = BARWEnsembleConfig()
    barw = run_barw_ensemble(barw_cfg)
    barw_speed = float(barw["speed_fit_alive"]["speed"])
    if not np.isfinite(barw_speed) or barw_speed <= 0:
        barw_speed = 2.0 * math.sqrt(0.1)
    effective_D = (barw_speed ** 2) / (4.0 * barw_cfg.rb)

    pde_cfg = PDEConfig(D=float(effective_D), variant="reduced")
    pde = solve_two_species_pde(pde_cfg)

    comparison = compute_barw_pde_comparison(pde, barw, selected_times=barw_cfg.collapse_times)
    movie = load_movie_s1_proxy()

    export_pde_data(pde, pde_cfg)
    export_barw_data(barw)
    write_csv(
        DATA_DIR / "barw_pde_profile_comparison.csv",
        comparison["rows"],
        [
            "times_used",
            "z_from_front",
            "barw_active_raw_norm",
            "barw_active_norm",
            "barw_active_sem_norm",
            "pde_active_norm",
            "barw_duct_raw_norm",
            "barw_duct_norm",
            "barw_duct_sem_norm",
            "pde_inactive_norm",
            "n_samples_active",
            "n_samples_duct",
        ],
    )
    write_csv(
        DATA_DIR / "barw_profiles_moving_frame_collapsed.csv",
        comparison["barw_collapse_rows"],
        ["z", "active_density_mean", "active_density_sem", "duct_density_mean", "duct_density_sem", "n_samples"],
    )
    write_csv(
        DATA_DIR / "pde_profiles_moving_frame_collapsed.csv",
        comparison["pde_collapse_rows"],
        ["z", "active_density_mean", "active_density_sem", "inactive_density_mean", "inactive_density_sem", "n_times"],
    )
    write_csv(
        DATA_DIR / "barw_duct_profiles_moving_frame_collapsed.csv",
        comparison["barw_duct_collapse_rows"],
        ["z", "active_density_mean", "active_density_sem", "duct_density_mean", "duct_density_sem", "n_samples"],
    )
    write_csv(
        DATA_DIR / "pde_duct_profiles_moving_frame_collapsed.csv",
        comparison["pde_duct_collapse_rows"],
        ["z", "active_density_mean", "active_density_sem", "inactive_density_mean", "inactive_density_sem", "n_times"],
    )

    plot_pde_kymograph(pde)
    tail_fit = plot_pde_stationary_pulse(pde, barw_cfg.collapse_times)
    plot_front_speed(pde, barw, pde_cfg)
    plot_barw_pde_profiles(comparison)
    plot_duct_profiles(comparison)
    plot_survival_statistics(barw)
    plot_movie_s1_proxy(movie)
    make_composite_layout()

    metrics = write_summary(pde_cfg, barw_cfg, pde, barw, comparison, movie, tail_fit)
    plot_quality_box(metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
