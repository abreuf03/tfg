
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"



from src.barw.busqueda_espacial import (
    ExhaustivaIndices,
    KDTreeIndices,
    QuadTreeIndices,
)
from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW

from paquetes_cerrados.benchmark_escalabilidad_barw.benchmark_escalabilidad_barw import (
    set_plot_style,
    median_iqr,
    canonical,
    count_mismatches,
    write_csv,
    rows_for_method,
    row_lookup,
    style_axes,
    panel_letter,
    reference_curve,
    fit_log_slope,
)

OUTPUT_DIR = PROJECT_ROOT / "resultados" / "benchmark_completo_barw"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures"
PANEL_DIR = FIG_DIR / "individual_panels"

METHODS = (
    "exhaustive",
    "kdtree",
    "quadtree",
)

METHOD_STYLE = {
    "exhaustive": {
        "label": "Exhaustiva",
        "color": "#2BA42F",
        "linestyle": "-",
        "marker": "o",
    },
    "kdtree": {
        "label": "cKDTree",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "s",
    },
    "quadtree": {
        "label": "Quadtree",
        "color": "#d62728",
        "linestyle": ":",
        "marker": "D",
    },
}


@dataclass(frozen=True)
class QueryRecord:
    """
    Consulta de vecinos realizada durante la simulación BARW.
    """

    x: float
    y: float
    branch_id: int
    n_points_before: int


@dataclass
class BenchmarkConfig:
    seed: int = 39

    sizes: tuple[int, ...] = (
        100,
        250,
        500,
        1000,
        2000,
    )

    repeats: int = 3
    max_queries_per_size: int = 2000

    linear_max_n: int = 2000

    quadtree_capacity: int = 8
    quadtree_max_depth: int = 15

    barw_tiempo_total: float = 1000.0


@dataclass
class Workload:
    points_xy: np.ndarray
    branch_ids: np.ndarray
    segments: np.ndarray

    Lx: float
    Ly: float
    radius: float

    n_branches: int
    n_terminations: int
    steps: int


#adaptación del código de los tutores

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)


def save_panel(name: str, plotter, *args) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.15))
    plotter(ax, *args)
    fig.tight_layout()

    fig.savefig(PANEL_DIR / f"{name}.png")
    fig.savefig(PANEL_DIR / f"{name}.pdf")

    plt.close(fig)


def plot_panel_C(ax: plt.Axes, rows: Sequence[dict[str, object]]) -> None:
    for method in METHODS:
        method_rows = rows_for_method(rows, method)
        if not method_rows:
            continue
        ns = np.asarray([int(row["N_points"]) for row in method_rows], dtype=float)
        ts = np.asarray([float(row["total_time_s_median"]) for row in method_rows])
        style = METHOD_STYLE[method]
        ax.loglog(
            ns,
            ts,
            marker=style["marker"],
            color=style["color"],
            ls=style["linestyle"],
            lw=1.8,
            ms=4.5,
            label=style["label"],
        )

    exhaustive_rows = rows_for_method(rows, "exhaustive")
    if exhaustive_rows:
        n0 = float(exhaustive_rows[0]["N_points"])
        t0 = float(exhaustive_rows[0]["total_time_s_median"])
        ns_ref = np.asarray([float(row["N_points"]) for row in rows_for_method(rows, "kdtree")])
        if ns_ref.size:
            reference_curve(ax, n0, t0, ns_ref, 2.0, r"$N^2$")
    kdtree_rows = rows_for_method(rows, "kdtree")
    if kdtree_rows:
        n0 = float(kdtree_rows[0]["N_points"])
        t0 = float(kdtree_rows[0]["total_time_s_median"])
        ns_ref = np.asarray([float(row["N_points"]) for row in kdtree_rows])
        if ns_ref.size:
            ax.plot(
                ns_ref,
                t0 * (ns_ref * np.log2(ns_ref)) / (n0 * np.log2(n0)),
                color="#bdbdbd",
                lw=1.0,
                ls="-.",
                label=r"$N\log N$",
            )
    ax.set_xlabel(r"Puntos de conducto $N$")
    ax.set_ylabel("Tiempo mediano (s)")
    ax.set_title("C. Escalado de consultas radiales")
    ax.grid(True, which="both", color="#dddddd", lw=0.5)
    ax.legend(loc="upper left", fontsize=8)
    panel_letter(ax, "C")
    style_axes(ax)

def plot_panel_D(ax: plt.Axes, rows: Sequence[dict[str, object]]) -> None:
    for method in METHODS:
        method_rows = [
            row
            for row in rows_for_method(rows, method)
            if math.isfinite(float(row["speedup_vs_exhaustive"]))
        ]
        if not method_rows:
            continue
        ns = np.asarray([int(row["N_points"]) for row in method_rows], dtype=float)
        sp = np.asarray([float(row["speedup_vs_exhaustive"]) for row in method_rows])
        style = METHOD_STYLE[method]
        ax.plot(
            ns,
            sp,
            marker=style["marker"],
            color=style["color"],
            ls=style["linestyle"],
            lw=1.8,
            ms=4.5,
            label=style["label"],
        )
    ax.axhline(1.0, color="#333333", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Puntos de conducto $N$")
    ax.set_ylabel("Speedup frente a exhaustiva")
    ax.set_title("D. Aceleracion sin cambiar vecinos")
    ax.grid(True, which="both", color="#dddddd", lw=0.5)
    ax.legend(fontsize=8)
    panel_letter(ax, "D")
    style_axes(ax)

def plot_panel_E(
    ax: plt.Axes,
    rows: Sequence[dict[str, object]],
) -> None:
    max_n = max(int(row["N_points"]) for row in rows)

    methods = [
        method
        for method in METHODS
        if row_lookup(rows, method, max_n) is not None
    ]

    build = np.asarray([
        float(row_lookup(rows, method, max_n)["build_time_s_median"])
        for method in methods
    ])

    query = np.asarray([
        float(row_lookup(rows, method, max_n)["query_time_s_median"])
        for method in methods
    ])

    labels = [METHOD_STYLE[method]["label"] for method in methods]
    x = np.arange(len(methods))
    width = 0.36

    ax.bar(
        x - width / 2,
        build,
        width=width,
        label="Construcción",
        edgecolor="#222222",
        linewidth=0.7,
    )

    ax.bar(
        x + width / 2,
        query,
        width=width,
        label="Consulta",
        edgecolor="#222222",
        linewidth=0.7,
    )

    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=18)
    ax.set_ylabel("Tiempo mediano (s)")
    ax.set_title(
        f"E. Descomposición del coste en $N={max_n:,}$"
    )
    ax.legend(fontsize=8)
    style_axes(ax)


def make_metrics(workload: Workload, rows: Sequence[dict[str, object]], cfg: BenchmarkConfig) -> dict[str, object]:
    methods = sorted({str(row["method"]) for row in rows})
    slopes = {method: fit_log_slope(rows, method) for method in methods}
    exact_all = all(int(row["mismatch_queries"]) == 0 for row in rows)
    max_n = max(int(row["N_points"]) for row in rows)
    max_checked_linear = max(
        int(row["N_points"]) for row in rows if row["method"] == "exhaustive"
    )
    speedups = {
        method: [
            float(row["speedup_vs_exhaustive"])
            for row in rows
            if row["method"] == method and math.isfinite(float(row["speedup_vs_exhaustive"]))
        ]
        for method in methods
    }
    best_speedup_at_linear_limit = {
        method: (
            float(row_lookup(rows, method, max_checked_linear)["speedup_vs_exhaustive"])
            if row_lookup(rows, method, max_checked_linear) is not None
            and math.isfinite(float(row_lookup(rows, method, max_checked_linear)["speedup_vs_exhaustive"]))
            else math.nan
        )
        for method in methods
        if method != "exhaustive"
    }
    kdtree_speedup = best_speedup_at_linear_limit.get("kdtree", math.nan)
    quality_checks = {
        "neighbor_equivalence_all_tested_queries": exact_all,
        "kdtree_available": "kdtree" in methods,
        "optimized_methods_reach_max_N": all(
            row_lookup(rows, method, max_n) is not None
            for method in ("kdtree", "grid", "quadtree")
        ),
        "exhaustive_measured_until_linear_limit": max_checked_linear == cfg.linear_max_n,
        "primary_kdtree_speedup_positive_at_linear_limit": (
            math.isfinite(kdtree_speedup) and kdtree_speedup > 1.0
        ),
        "grid_and_quadtree_reported_as_diagnostic_baselines": True,
    }
    score = 10.0
    if not exact_all:
        score -= 1.5
    if not quality_checks["optimized_methods_reach_max_N"]:
        score -= 0.5
    if not quality_checks["exhaustive_measured_until_linear_limit"]:
        score -= 0.4
    if not quality_checks["primary_kdtree_speedup_positive_at_linear_limit"]:
        score -= 0.5

    return {
        "config": asdict(cfg),
        "workload": {
            "points": int(workload.points_xy.shape[0]),
            "segments": int(workload.segments.shape[0]),
            "queries": int(len(workload.queries)),
            "branches": int(workload.n_branches),
            "terminations": int(workload.n_terminations),
            "roots": int(workload.n_roots),
            "steps": int(workload.steps),
        },
        "methods": methods,
        "slopes_loglog_total_time": slopes,
        "speedups_at_linear_limit": best_speedup_at_linear_limit,
        "max_N_benchmarked": max_n,
        "max_N_with_exhaustive_reference": max_checked_linear,
        "quality_checks": quality_checks,
        "estimated_quality_score_over_10": round(max(score, 0.0), 2),
    }


def generate_barw_workload(cfg: BenchmarkConfig) -> Workload:
    barw_config = replace(
        BARWConfig(),
        semilla=cfg.seed,
        tiempo_total=cfg.barw_tiempo_total,
    )

    simulacion = SimulacionBARW(
        config=barw_config,
        metodo_busqueda=0,
    )

    resultado = simulacion.ejecutar()

    points_xy = np.asarray(
        simulacion.busqueda_espacial.puntos,
        dtype=float,
    )

    branch_ids = np.asarray(
        simulacion.busqueda_espacial.ramas_ids,
        dtype=np.int64,
    )

    segments = np.asarray(
        resultado["conducto"],
        dtype=float,
    )

    if segments.size == 0:
        segments = np.empty((0, 5), dtype=float)

    return Workload(
        points_xy=points_xy,
        branch_ids=branch_ids,
        segments=segments,
        Lx=float(barw_config.Lx),
        Ly=float(barw_config.Ly),
        radius=float(barw_config.Ra),
        n_branches=int(simulacion.siguiente_id_rama),
        n_terminations=int(simulacion.contador_terminaciones),
        steps=int(simulacion.contador_pasos),
    )


def select_queries(
    workload: Workload,
    n_points: int,
    max_queries: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Selecciona posiciones de consulta entre los primeros N puntos
    de la geometría BARW.
    """
    n_queries = min(n_points, max_queries)

    rng = np.random.default_rng(seed + n_points)

    indices = rng.choice(
        n_points,
        size=n_queries,
        replace=False,
    )

    query_xy = workload.points_xy[indices]
    query_branch_ids = workload.branch_ids[indices]

    return query_xy, query_branch_ids

def create_index(
    method: str,
    workload: Workload,
    cfg: BenchmarkConfig,
):
    """
    Crea una estructura espacial vacía.
    """
    if method == "exhaustive":
        return ExhaustivaIndices()

    if method == "kdtree":
        return KDTreeIndices()

    if method == "quadtree":
        return QuadTreeIndices(
            x_min=0.0,
            x_max=workload.Lx,
            y_min=0.0,
            y_max=workload.Ly,
            capacidad=cfg.quadtree_capacity,
            profundidad_maxima=cfg.quadtree_max_depth,
        )

    raise ValueError(
        f"Método de búsqueda no reconocido: {method}"
    )

def build_index(
    method: str,
    index,
    points_xy: np.ndarray,
    branch_ids: np.ndarray,
) -> None:
    """
    Añade los puntos al índice espacial.

    En el caso del KDTree, construye el árbol después de añadir
    todos los puntos.
    """
    for (x, y), branch_id in zip(points_xy, branch_ids):
        index.agregar_punto(
            float(x),
            float(y),
            int(branch_id),
        )

    if method == "kdtree":
        index.construir_kdtree()


def query_index(
    index,
    query_xy: np.ndarray,
    query_branch_ids: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    """
    Ejecuta las consultas radiales sobre una estructura espacial.
    """
    results: list[np.ndarray] = []

    for (x, y), branch_id in zip(
        query_xy,
        query_branch_ids,
    ):
        neighbors = index.buscar_puntas_cercanas(
            x=float(x),
            y=float(y),
            Ra=radius,
            excluir_id_rama=int(branch_id),
        )

        results.append(
            np.asarray(neighbors, dtype=np.int64)
        )

    return results


def run_benchmark(
    workload: Workload,
    cfg: BenchmarkConfig,
) -> list[dict[str, object]]:
    """
    Compara exhaustiva, cKDTree y quadtree sobre los mismos puntos
    y las mismas consultas.

    Para cada método mide por separado:

    - construcción o inserción;
    - consultas;
    - coste total.

    También comprueba que los vecinos coincidan.
    """
    rows: list[dict[str, object]] = []

    total_points = workload.points_xy.shape[0]

    for n_points in cfg.sizes:
        if n_points > total_points:
            print(
                f"N={n_points} omitido: la geometría solo contiene "
                f"{total_points} puntos."
            )
            continue

        points_xy = workload.points_xy[:n_points]
        branch_ids = workload.branch_ids[:n_points]

        query_xy, query_branch_ids = select_queries(
            workload=workload,
            n_points=n_points,
            max_queries=cfg.max_queries_per_size,
            seed=cfg.seed,
        )

        active_methods = [
            method
            for method in METHODS
            if (
                method != "exhaustive"
                or n_points <= cfg.linear_max_n
            )
        ]

        timings: dict[str, dict[str, list[float]]] = {}
        results_by_method: dict[
            str,
            list[tuple[int, ...]],
        ] = {}

        for method in active_methods:
            timings[method] = {
                "build": [],
                "query": [],
                "total": [],
            }

            first_results: list[np.ndarray] | None = None

            for _ in range(cfg.repeats):
                index = create_index(
                    method=method,
                    workload=workload,
                    cfg=cfg,
                )

                start_build = time.perf_counter()

                build_index(
                    method=method,
                    index=index,
                    points_xy=points_xy,
                    branch_ids=branch_ids,
                )

                end_build = time.perf_counter()

                results = query_index(
                    index=index,
                    query_xy=query_xy,
                    query_branch_ids=query_branch_ids,
                    radius=workload.radius,
                )

                end_query = time.perf_counter()

                timings[method]["build"].append(
                    end_build - start_build
                )

                timings[method]["query"].append(
                    end_query - end_build
                )

                timings[method]["total"].append(
                    end_query - start_build
                )

                # Basta con conservar los resultados de una repetición,
                # porque las consultas son deterministas.
                if first_results is None:
                    first_results = results

            if first_results is None:
                raise RuntimeError(
                    f"No se obtuvieron resultados para {method}."
                )

            results_by_method[method] = canonical(
                first_results
            )

        # La exhaustiva es la referencia cuando se ha ejecutado.
        # Para tamaños mayores se usa cKDTree, previamente validado.
        reference_method = (
            "exhaustive"
            if "exhaustive" in active_methods
            else "kdtree"
        )

        reference_results = results_by_method[
            reference_method
        ]

        exhaustive_total = None

        if "exhaustive" in active_methods:
            exhaustive_total = median_iqr(
                timings["exhaustive"]["total"]
            )[0]

        for method in active_methods:
            build_median, build_q25, build_q75 = median_iqr(
                timings[method]["build"]
            )

            query_median, query_q25, query_q75 = median_iqr(
                timings[method]["query"]
            )

            total_median, total_q25, total_q75 = median_iqr(
                timings[method]["total"]
            )

            mismatches = count_mismatches(
                reference_results,
                results_by_method[method],
            )

            if method == "exhaustive":
                speedup = 1.0

            elif exhaustive_total is not None:
                speedup = exhaustive_total / total_median

            else:
                speedup = math.nan

            rows.append(
                {
                    "N_points": int(n_points),
                    "N_queries": int(query_xy.shape[0]),
                    "method": method,
                    "reference_method": reference_method,

                    "build_time_s_median": build_median,
                    "build_time_s_q25": build_q25,
                    "build_time_s_q75": build_q75,

                    "query_time_s_median": query_median,
                    "query_time_s_q25": query_q25,
                    "query_time_s_q75": query_q75,

                    "total_time_s_median": total_median,
                    "total_time_s_q25": total_q25,
                    "total_time_s_q75": total_q75,

                    "query_time_per_query_s": (
                        query_median / query_xy.shape[0]
                    ),

                    "mismatch_queries": int(mismatches),
                    "mismatch_rate": float(
                        mismatches / query_xy.shape[0]
                    ),

                    "speedup_vs_exhaustive": float(speedup),
                }
            )

        print(
            f"N={n_points}: "
            f"{query_xy.shape[0]} consultas, "
            f"métodos={', '.join(active_methods)}"
        )

    return rows

def make_figures(
    rows: Sequence[dict[str, object]],
) -> None:
    """
    Genera los paneles individuales y una figura compuesta.
    """
    save_panel(
        "panel_C_runtime_scaling",
        plot_panel_C,
        rows,
    )

    save_panel(
        "panel_D_speedup",
        plot_panel_D,
        rows,
    )

    save_panel(
        "panel_E_build_query",
        plot_panel_E,
        rows,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 3.8),
    )

    plot_panel_C(axes[0], rows)
    plot_panel_D(axes[1], rows)
    plot_panel_E(axes[2], rows)

    fig.suptitle(
        "Benchmark de búsqueda espacial en el modelo BARW",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "benchmark_completo_barw.png"
    )

    fig.savefig(
        FIG_DIR / "benchmark_completo_barw.pdf"
    )

    plt.close(fig)

def main() -> None:
    ensure_dirs()
    set_plot_style()

    cfg = BenchmarkConfig()

    print("[1/4] Generando geometría BARW...")
    workload = generate_barw_workload(cfg)

    print(
        f"Geometría generada: "
        f"{workload.points_xy.shape[0]} puntos, "
        f"{workload.segments.shape[0]} segmentos."
    )

    valid_sizes = [
        size
        for size in cfg.sizes
        if size <= workload.points_xy.shape[0]
    ]

    if not valid_sizes:
        raise RuntimeError(
            "La simulación no ha generado suficientes puntos "
            "para ninguno de los tamaños configurados."
        )

    print(
        "Tamaños que se medirán:",
        valid_sizes,
    )

    print("[2/4] Ejecutando benchmark...")
    rows = run_benchmark(
        workload=workload,
        cfg=cfg,
    )

    if not rows:
        raise RuntimeError(
            "El benchmark no ha generado resultados."
        )

    print("[3/4] Guardando resultados...")
    write_csv(
        DATA_DIR / "benchmark_busqueda.csv",
        rows,
    )

    print("[4/4] Generando figuras...")
    make_figures(rows)

    total_mismatches = sum(
        int(row["mismatch_queries"])
        for row in rows
    )

    print()
    print("Benchmark terminado.")
    print(
        f"Discrepancias totales: {total_mismatches}"
    )
    print(
        "CSV:",
        DATA_DIR / "benchmark_busqueda.csv",
    )
    print(
        "Figura:",
        FIG_DIR / "benchmark_completo_barw.png",
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()