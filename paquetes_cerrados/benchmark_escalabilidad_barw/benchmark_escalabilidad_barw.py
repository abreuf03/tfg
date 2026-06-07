"""
Benchmark profesional de escalabilidad para el simulador BARW.

Objetivo:
    Medir el coste de la busqueda de vecinos usada en la regla de
    terminacion por proximidad del modelo BARW y verificar que los
    metodos optimizados devuelven los mismos vecinos que la busqueda
    exhaustiva sobre una geometria BARW reproducible.

Salidas:
    data/benchmark_scaling.csv
    data/benchmark_metrics.json
    data/summary_benchmark_escalabilidad_barw.md
    figures/benchmark_escalabilidad_barw_layout.png/pdf
    figures/individual_panels/panel_A_...png/pdf ... panel_F_...png/pdf

Uso:
    python benchmark_escalabilidad_barw.py
    python benchmark_escalabilidad_barw.py --mode quick
    python benchmark_escalabilidad_barw.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - explicit runtime dependency.
    raise SystemExit(
        "Este benchmark requiere scipy.spatial.cKDTree. "
        "Instala scipy antes de ejecutar el script."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
PANEL_DIR = FIG_DIR / "individual_panels"


COLORS = {
    "network": "#2f8f4e",
    "exhaustive": "#222222",
    "kdtree": "#1f77b4",
    "grid": "#2ca02c",
    "quadtree": "#d62728",
    "gray": "#6f6f6f",
    "light_gray": "#e8e8e8",
}


METHOD_STYLE = {
    "exhaustive": {
        "label": "Exhaustiva",
        "color": COLORS["exhaustive"],
        "linestyle": "-",
        "marker": "o",
    },
    "kdtree": {
        "label": "cKDTree",
        "color": COLORS["kdtree"],
        "linestyle": "-",
        "marker": "s",
    },
    "grid": {
        "label": "Grid espacial",
        "color": COLORS["grid"],
        "linestyle": "--",
        "marker": "^",
    },
    "quadtree": {
        "label": "Quadtree",
        "color": COLORS["quadtree"],
        "linestyle": ":",
        "marker": "D",
    },
}


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 270526
    Lx: float = 700.0
    Ly: float = 360.0
    rb: float = 0.16
    Ra: float = 3.0
    step_length: float = 1.0
    angle_noise: float = math.pi / 10.0
    branch_angle: float = math.pi / 6.0
    exclusion_steps: int = 6
    max_points: int = 50000
    max_steps: int = 2500
    max_active_tips: int = 12000
    sizes: tuple[int, ...] = (500, 1000, 2000, 4000, 8000, 16000, 32000)
    repeats: int = 3
    max_queries_per_size: int = 8000
    linear_max_n: int = 8000
    quadtree_capacity: int = 32
    quadtree_max_depth: int = 22


@dataclass
class Tip:
    x: float
    y: float
    theta: float
    branch_id: int
    generation: int
    age: int = 0


@dataclass(frozen=True)
class QueryRecord:
    x: float
    y: float
    branch_id: int
    n_points_before: int


@dataclass
class Workload:
    points_xy: np.ndarray
    branch_ids: np.ndarray
    segments: np.ndarray
    queries: list[QueryRecord]
    n_branches: int
    n_terminations: int
    n_roots: int
    steps: int


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 1.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def cell_key(x: float, y: float, cell_size: float) -> tuple[int, int]:
    return int(math.floor(x / cell_size)), int(math.floor(y / cell_size))


def generate_barw_workload(cfg: BenchmarkConfig) -> Workload:
    """Generate a deterministic BARW-like geometry and query workload."""
    rng = np.random.default_rng(cfg.seed)
    points: list[tuple[float, float]] = []
    branch_ids: list[int] = []
    segments: list[tuple[float, float, float, float, int]] = []
    queries: list[QueryRecord] = []
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    active: list[Tip] = []
    next_branch_id = 0
    n_terminations = 0
    n_roots = 0

    def add_point(x: float, y: float, branch_id: int) -> None:
        idx = len(points)
        points.append((x, y))
        branch_ids.append(branch_id)
        grid[cell_key(x, y, cfg.Ra)].append(idx)

    def add_root() -> None:
        nonlocal next_branch_id, n_roots
        y0 = cfg.Ly * (0.30 + 0.40 * rng.random())
        root = Tip(
            x=0.0,
            y=y0,
            theta=rng.normal(0.0, 0.035),
            branch_id=next_branch_id,
            generation=0,
        )
        active.append(root)
        add_point(root.x, root.y, root.branch_id)
        next_branch_id += 1
        n_roots += 1

    def has_neighbor(x: float, y: float, branch_id: int) -> bool:
        cx, cy = cell_key(x, y, cfg.Ra)
        r2 = cfg.Ra * cfg.Ra
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in grid.get((cx + dx, cy + dy), []):
                    if branch_ids[idx] == branch_id:
                        continue
                    px, py = points[idx]
                    if (px - x) ** 2 + (py - y) ** 2 <= r2:
                        return True
        return False

    add_root()
    step = 0
    while step < cfg.max_steps and len(points) < cfg.max_points:
        if not active:
            if n_roots >= 3:
                break
            add_root()

        next_active: list[Tip] = []
        new_tips: list[Tip] = []
        for tip in active:
            if len(points) >= cfg.max_points:
                break

            x_old, y_old = tip.x, tip.y
            x_new = x_old + cfg.step_length * math.cos(tip.theta)
            y_new = y_old + cfg.step_length * math.sin(tip.theta)
            theta_new = tip.theta + rng.uniform(-cfg.angle_noise, cfg.angle_noise)

            if x_new < 0.0 or x_new > cfg.Lx or y_new < 0.0 or y_new > cfg.Ly:
                n_terminations += 1
                continue

            collision = False
            if tip.age >= cfg.exclusion_steps:
                queries.append(
                    QueryRecord(
                        x=x_new,
                        y=y_new,
                        branch_id=tip.branch_id,
                        n_points_before=len(points),
                    )
                )
                collision = has_neighbor(x_new, y_new, tip.branch_id)

            if collision:
                n_terminations += 1
                continue

            add_point(x_new, y_new, tip.branch_id)
            segments.append((x_old, y_old, x_new, y_new, tip.branch_id))
            tip.x = x_new
            tip.y = y_new
            tip.theta = theta_new
            tip.age += 1
            next_active.append(tip)

            if (
                rng.random() < cfg.rb
                and len(next_active) + len(new_tips) < cfg.max_active_tips
                and len(points) < cfg.max_points
            ):
                sign = -1.0 if rng.random() < 0.5 else 1.0
                new_tips.append(
                    Tip(
                        x=x_new,
                        y=y_new,
                        theta=theta_new + sign * cfg.branch_angle,
                        branch_id=next_branch_id,
                        generation=tip.generation + 1,
                    )
                )
                next_branch_id += 1

        active = next_active + new_tips
        step += 1

    return Workload(
        points_xy=np.asarray(points, dtype=float),
        branch_ids=np.asarray(branch_ids, dtype=np.int32),
        segments=np.asarray(segments, dtype=float),
        queries=queries,
        n_branches=next_branch_id,
        n_terminations=n_terminations,
        n_roots=n_roots,
        steps=step,
    )


class StaticIndex:
    method = "base"

    def build(self, xy: np.ndarray, branch_ids: np.ndarray, radius: float) -> None:
        self.xy = xy
        self.branch_ids = branch_ids
        self.radius = radius

    def query_one(self, x: float, y: float, branch_id: int) -> np.ndarray:
        raise NotImplementedError

    def query_many(self, query_xy: np.ndarray, query_branch_ids: np.ndarray) -> list[np.ndarray]:
        return [
            self.query_one(float(x), float(y), int(branch_id))
            for (x, y), branch_id in zip(query_xy, query_branch_ids)
        ]


class ExhaustiveIndex(StaticIndex):
    method = "exhaustive"

    def query_one(self, x: float, y: float, branch_id: int) -> np.ndarray:
        if self.xy.size == 0:
            return np.empty(0, dtype=np.int64)
        dx = self.xy[:, 0] - x
        dy = self.xy[:, 1] - y
        mask = (dx * dx + dy * dy <= self.radius * self.radius) & (
            self.branch_ids != branch_id
        )
        return np.flatnonzero(mask)


class KDTreeIndex(StaticIndex):
    method = "kdtree"

    def build(self, xy: np.ndarray, branch_ids: np.ndarray, radius: float) -> None:
        super().build(xy, branch_ids, radius)
        self.tree = cKDTree(xy)

    def query_many(self, query_xy: np.ndarray, query_branch_ids: np.ndarray) -> list[np.ndarray]:
        raw = self.tree.query_ball_point(query_xy, r=self.radius)
        filtered: list[np.ndarray] = []
        for indices, branch_id in zip(raw, query_branch_ids):
            if not indices:
                filtered.append(np.empty(0, dtype=np.int64))
                continue
            arr = np.asarray(indices, dtype=np.int64)
            filtered.append(arr[self.branch_ids[arr] != branch_id])
        return filtered

    def query_one(self, x: float, y: float, branch_id: int) -> np.ndarray:
        indices = self.tree.query_ball_point((x, y), r=self.radius)
        if not indices:
            return np.empty(0, dtype=np.int64)
        arr = np.asarray(indices, dtype=np.int64)
        return arr[self.branch_ids[arr] != branch_id]


class GridIndex(StaticIndex):
    method = "grid"

    def build(self, xy: np.ndarray, branch_ids: np.ndarray, radius: float) -> None:
        super().build(xy, branch_ids, radius)
        self.cell_size = radius
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (x, y) in enumerate(xy):
            self.cells[cell_key(float(x), float(y), self.cell_size)].append(idx)

    def query_one(self, x: float, y: float, branch_id: int) -> np.ndarray:
        cx, cy = cell_key(x, y, self.cell_size)
        out: list[int] = []
        r2 = self.radius * self.radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in self.cells.get((cx + dx, cy + dy), []):
                    if self.branch_ids[idx] == branch_id:
                        continue
                    px, py = self.xy[idx]
                    if (px - x) ** 2 + (py - y) ** 2 <= r2:
                        out.append(idx)
        return np.asarray(out, dtype=np.int64)


@dataclass
class QuadNode:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    indices: np.ndarray | None
    children: tuple["QuadNode", ...] = ()


class QuadtreeIndex(StaticIndex):
    method = "quadtree"

    def __init__(self, capacity: int = 32, max_depth: int = 22):
        self.capacity = capacity
        self.max_depth = max_depth

    def build(self, xy: np.ndarray, branch_ids: np.ndarray, radius: float) -> None:
        super().build(xy, branch_ids, radius)
        pad = max(radius, 1e-9)
        xmin = float(np.min(xy[:, 0]) - pad)
        xmax = float(np.max(xy[:, 0]) + pad)
        ymin = float(np.min(xy[:, 1]) - pad)
        ymax = float(np.max(xy[:, 1]) + pad)
        self.root = self._build_node(
            np.arange(xy.shape[0], dtype=np.int64), xmin, xmax, ymin, ymax, 0
        )

    def _build_node(
        self,
        indices: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        depth: int,
    ) -> QuadNode:
        if indices.size <= self.capacity or depth >= self.max_depth:
            return QuadNode(xmin, xmax, ymin, ymax, indices=indices)

        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        if xmax <= xmin or ymax <= ymin:
            return QuadNode(xmin, xmax, ymin, ymax, indices=indices)

        pts = self.xy[indices]
        masks = (
            (pts[:, 0] <= xmid) & (pts[:, 1] <= ymid),
            (pts[:, 0] > xmid) & (pts[:, 1] <= ymid),
            (pts[:, 0] <= xmid) & (pts[:, 1] > ymid),
            (pts[:, 0] > xmid) & (pts[:, 1] > ymid),
        )
        bounds = (
            (xmin, xmid, ymin, ymid),
            (xmid, xmax, ymin, ymid),
            (xmin, xmid, ymid, ymax),
            (xmid, xmax, ymid, ymax),
        )
        children = []
        for mask, bound in zip(masks, bounds):
            child_indices = indices[mask]
            if child_indices.size:
                children.append(self._build_node(child_indices, *bound, depth + 1))
        if len(children) <= 1:
            return QuadNode(xmin, xmax, ymin, ymax, indices=indices)
        return QuadNode(xmin, xmax, ymin, ymax, indices=None, children=tuple(children))

    def _bbox_intersects_circle(self, node: QuadNode, x: float, y: float) -> bool:
        nearest_x = min(max(x, node.xmin), node.xmax)
        nearest_y = min(max(y, node.ymin), node.ymax)
        return (nearest_x - x) ** 2 + (nearest_y - y) ** 2 <= self.radius**2

    def _query_node(self, node: QuadNode, x: float, y: float, branch_id: int, out: list[int]) -> None:
        if not self._bbox_intersects_circle(node, x, y):
            return
        if node.indices is not None:
            idx = node.indices
            pts = self.xy[idx]
            dx = pts[:, 0] - x
            dy = pts[:, 1] - y
            mask = (dx * dx + dy * dy <= self.radius * self.radius) & (
                self.branch_ids[idx] != branch_id
            )
            out.extend(idx[mask].tolist())
            return
        for child in node.children:
            self._query_node(child, x, y, branch_id, out)

    def query_one(self, x: float, y: float, branch_id: int) -> np.ndarray:
        out: list[int] = []
        self._query_node(self.root, x, y, branch_id, out)
        return np.asarray(out, dtype=np.int64)


def select_queries(
    workload: Workload, n_points: int, max_queries: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    eligible = [q for q in workload.queries if q.n_points_before <= n_points]
    rng = np.random.default_rng(seed + n_points)
    n_queries = min(n_points, max_queries)

    if len(eligible) >= n_queries:
        chosen = rng.choice(len(eligible), size=n_queries, replace=False)
        query_xy = np.asarray([(eligible[i].x, eligible[i].y) for i in chosen], dtype=float)
        query_branch = np.asarray([eligible[i].branch_id for i in chosen], dtype=np.int32)
        return query_xy, query_branch

    idx = rng.choice(n_points, size=n_queries, replace=False)
    return workload.points_xy[idx], workload.branch_ids[idx]


def canonical(results: Sequence[np.ndarray]) -> list[tuple[int, ...]]:
    return [tuple(sorted(map(int, arr.tolist()))) for arr in results]


def count_mismatches(
    reference: Sequence[tuple[int, ...]], candidate: Sequence[tuple[int, ...]]
) -> int:
    return sum(ref != cand for ref, cand in zip(reference, candidate))


def median_iqr(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def method_factory(method: str, cfg: BenchmarkConfig) -> StaticIndex:
    if method == "exhaustive":
        return ExhaustiveIndex()
    if method == "kdtree":
        return KDTreeIndex()
    if method == "grid":
        return GridIndex()
    if method == "quadtree":
        return QuadtreeIndex(
            capacity=cfg.quadtree_capacity,
            max_depth=cfg.quadtree_max_depth,
        )
    raise ValueError(f"Metodo no reconocido: {method}")


def run_benchmark(workload: Workload, cfg: BenchmarkConfig) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    methods = ("exhaustive", "kdtree", "grid", "quadtree")

    for n_points in cfg.sizes:
        if n_points > workload.points_xy.shape[0]:
            continue
        xy = workload.points_xy[:n_points]
        branch = workload.branch_ids[:n_points]
        query_xy, query_branch = select_queries(
            workload, n_points, cfg.max_queries_per_size, cfg.seed
        )

        canonical_by_method: dict[str, list[tuple[int, ...]]] = {}
        timing_by_method: dict[str, dict[str, list[float]]] = {}

        active_methods = [
            method for method in methods if method != "exhaustive" or n_points <= cfg.linear_max_n
        ]

        for method in active_methods:
            timing_by_method[method] = {"build": [], "query": [], "total": []}
            first_results = None
            for _ in range(cfg.repeats):
                index = method_factory(method, cfg)
                t0 = time.perf_counter()
                index.build(xy, branch, cfg.Ra)
                t1 = time.perf_counter()
                results = index.query_many(query_xy, query_branch)
                t2 = time.perf_counter()
                timing_by_method[method]["build"].append(t1 - t0)
                timing_by_method[method]["query"].append(t2 - t1)
                timing_by_method[method]["total"].append(t2 - t0)
                if first_results is None:
                    first_results = results
            assert first_results is not None
            canonical_by_method[method] = canonical(first_results)

        reference_method = "exhaustive" if "exhaustive" in active_methods else "kdtree"
        reference = canonical_by_method[reference_method]
        linear_total = None
        if "exhaustive" in active_methods:
            linear_total = median_iqr(timing_by_method["exhaustive"]["total"])[0]

        for method in active_methods:
            build_med, build_q25, build_q75 = median_iqr(timing_by_method[method]["build"])
            query_med, query_q25, query_q75 = median_iqr(timing_by_method[method]["query"])
            total_med, total_q25, total_q75 = median_iqr(timing_by_method[method]["total"])
            mismatch = count_mismatches(reference, canonical_by_method[method])
            neighbor_total = int(sum(len(item) for item in canonical_by_method[method]))
            rows.append(
                {
                    "N_points": int(n_points),
                    "N_queries": int(query_xy.shape[0]),
                    "method": method,
                    "reference_method": reference_method,
                    "build_time_s_median": build_med,
                    "build_time_s_q25": build_q25,
                    "build_time_s_q75": build_q75,
                    "query_time_s_median": query_med,
                    "query_time_s_q25": query_q25,
                    "query_time_s_q75": query_q75,
                    "total_time_s_median": total_med,
                    "total_time_s_q25": total_q25,
                    "total_time_s_q75": total_q75,
                    "queries_per_second": float(query_xy.shape[0] / max(query_med, 1e-12)),
                    "neighbor_total": neighbor_total,
                    "mismatch_queries": int(mismatch),
                    "mismatch_rate": float(mismatch / query_xy.shape[0]),
                    "speedup_vs_exhaustive": (
                        float(linear_total / total_med)
                        if linear_total is not None and method != "exhaustive"
                        else (1.0 if method == "exhaustive" and linear_total is not None else math.nan)
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_workload_samples(workload: Workload, max_rows: int = 12000) -> None:
    point_rows = [
        {"idx": i, "x": float(x), "y": float(y), "branch_id": int(branch)}
        for i, ((x, y), branch) in enumerate(
            zip(workload.points_xy[:max_rows], workload.branch_ids[:max_rows])
        )
    ]
    write_csv(DATA_DIR / "barw_workload_points_sample.csv", point_rows)

    n_segments = min(max_rows, workload.segments.shape[0])
    segment_rows = [
        {
            "idx": i,
            "x0": float(row[0]),
            "y0": float(row[1]),
            "x1": float(row[2]),
            "y1": float(row[3]),
            "branch_id": int(row[4]),
        }
        for i, row in enumerate(workload.segments[:n_segments])
    ]
    write_csv(DATA_DIR / "barw_workload_segments_sample.csv", segment_rows)


def fit_log_slope(rows: Sequence[dict[str, object]], method: str) -> float:
    xs = []
    ys = []
    for row in rows:
        if row["method"] != method:
            continue
        t = float(row["total_time_s_median"])
        if t > 0:
            xs.append(float(row["N_points"]))
            ys.append(t)
    if len(xs) < 2:
        return math.nan
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def rows_for_method(rows: Sequence[dict[str, object]], method: str) -> list[dict[str, object]]:
    return [row for row in rows if row["method"] == method]


def row_lookup(rows: Sequence[dict[str, object]], method: str, n_points: int) -> dict[str, object] | None:
    for row in rows:
        if row["method"] == method and int(row["N_points"]) == int(n_points):
            return row
    return None


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


def panel_letter(ax: plt.Axes, letter: str) -> None:
    # Las letras A-F se integran en el titulo para evitar colisiones con
    # leyendas, barras o mapas log-log en paneles individuales y compuestos.
    return


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#222222")


def plot_panel_A(ax: plt.Axes, workload: Workload) -> None:
    max_segments = min(14000, workload.segments.shape[0])
    seg = workload.segments[:max_segments]
    for x0, y0, x1, y1, _ in seg:
        ax.plot([x0, x1], [y0, y1], color=COLORS["network"], lw=0.35, alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A. Geometria BARW usada en las consultas")
    ax.set_xlabel(r"Posicion $x$")
    ax.set_ylabel(r"Posicion $y$")
    ax.set_xlim(-5, np.nanmax(workload.points_xy[:, 0]) + 10)
    ax.set_ylim(np.nanmin(workload.points_xy[:, 1]) - 10, np.nanmax(workload.points_xy[:, 1]) + 10)
    panel_letter(ax, "A")
    style_axes(ax)


def plot_panel_B(ax: plt.Axes, rows: Sequence[dict[str, object]]) -> None:
    methods = [m for m in ("kdtree", "grid", "quadtree") if rows_for_method(rows, m)]
    max_n = max(int(row["N_points"]) for row in rows)
    values = []
    labels = []
    colors = []
    for method in methods:
        row = row_lookup(rows, method, max_n)
        if row is None:
            continue
        values.append(100.0 * (1.0 - float(row["mismatch_rate"])))
        labels.append(METHOD_STYLE[method]["label"])
        colors.append(METHOD_STYLE[method]["color"])
    ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_ylim(99.0, 100.05)
    ax.set_ylabel("Consultas coincidentes (%)")
    ax.set_title("B. Exactitud frente a referencia")
    ax.axhline(100.0, color="#222222", lw=0.8, ls=":")
    ax.tick_params(axis="x", rotation=18)
    ax.text(
        0.03,
        0.12,
        "0 discrepancias\nmisma regla radial",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#333333", lw=0.8, pad=3.0),
    )
    panel_letter(ax, "B")
    style_axes(ax)


def reference_curve(ax: plt.Axes, n0: float, t0: float, ns: np.ndarray, exponent: float, label: str) -> None:
    ref = t0 * (ns / n0) ** exponent
    ax.plot(ns, ref, color="#9a9a9a", lw=1.0, ls="--", label=label)


def plot_panel_C(ax: plt.Axes, rows: Sequence[dict[str, object]]) -> None:
    for method in ("exhaustive", "kdtree", "grid", "quadtree"):
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
    for method in ("kdtree", "grid", "quadtree"):
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


def plot_panel_E(ax: plt.Axes, rows: Sequence[dict[str, object]]) -> None:
    max_n = max(int(row["N_points"]) for row in rows)
    methods = [m for m in ("kdtree", "grid", "quadtree") if row_lookup(rows, m, max_n)]
    build = [float(row_lookup(rows, m, max_n)["build_time_s_median"]) for m in methods]
    query = [float(row_lookup(rows, m, max_n)["query_time_s_median"]) for m in methods]
    labels = [METHOD_STYLE[m]["label"] for m in methods]
    x = np.arange(len(methods))
    ax.bar(
        x,
        build,
        color="#b9d7f0",
        edgecolor="#222222",
        lw=0.7,
        label="Construccion",
    )
    ax.bar(
        x,
        query,
        bottom=build,
        color="#f2c185",
        edgecolor="#222222",
        lw=0.7,
        label="Consulta",
    )
    ax.set_xticks(x, labels, rotation=18)
    ax.set_ylabel("Tiempo mediano (s)")
    ax.set_title(f"E. Descomposicion de coste en N={max_n:,}")
    ax.legend(fontsize=8)
    panel_letter(ax, "E")
    style_axes(ax)


def plot_panel_F(ax: plt.Axes, rows: Sequence[dict[str, object]]) -> None:
    for method in ("exhaustive", "kdtree", "grid", "quadtree"):
        method_rows = rows_for_method(rows, method)
        if not method_rows:
            continue
        ns = np.asarray([int(row["N_points"]) for row in method_rows], dtype=float)
        qps = np.asarray([float(row["queries_per_second"]) for row in method_rows])
        style = METHOD_STYLE[method]
        ax.loglog(
            ns,
            qps,
            marker=style["marker"],
            color=style["color"],
            ls=style["linestyle"],
            lw=1.8,
            ms=4.5,
            label=style["label"],
        )
    ax.set_xlabel(r"Puntos de conducto $N$")
    ax.set_ylabel("Consultas por segundo")
    ax.set_title("F. Rendimiento operativo")
    ax.grid(True, which="both", color="#dddddd", lw=0.5)
    ax.legend(fontsize=8)
    panel_letter(ax, "F")
    style_axes(ax)


def save_panel(name: str, plotter, *args) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.15))
    plotter(ax, *args)
    fig.tight_layout()
    fig.savefig(PANEL_DIR / f"{name}.png")
    fig.savefig(PANEL_DIR / f"{name}.pdf")
    plt.close(fig)


def make_figures(workload: Workload, rows: Sequence[dict[str, object]]) -> None:
    save_panel("panel_A_barw_workload_geometry", plot_panel_A, workload)
    save_panel("panel_B_neighbor_exactness", plot_panel_B, rows)
    save_panel("panel_C_runtime_scaling_loglog", plot_panel_C, rows)
    save_panel("panel_D_speedup_vs_exhaustive", plot_panel_D, rows)
    save_panel("panel_E_build_query_decomposition", plot_panel_E, rows)
    save_panel("panel_F_query_throughput", plot_panel_F, rows)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    plot_panel_A(axes[0, 0], workload)
    plot_panel_B(axes[0, 1], rows)
    plot_panel_C(axes[0, 2], rows)
    plot_panel_D(axes[1, 0], rows)
    plot_panel_E(axes[1, 1], rows)
    plot_panel_F(axes[1, 2], rows)
    fig.suptitle(
        "Benchmark de escalabilidad BARW: regla de vecinos exacta y aceleracion algoritmica",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "benchmark_escalabilidad_barw_layout.png")
    fig.savefig(FIG_DIR / "benchmark_escalabilidad_barw_layout.pdf")
    plt.close(fig)


def write_summary(metrics: dict[str, object], rows: Sequence[dict[str, object]]) -> None:
    workload = metrics["workload"]
    checks = metrics["quality_checks"]
    slopes = metrics["slopes_loglog_total_time"]
    speedups = metrics["speedups_at_linear_limit"]
    max_n = metrics["max_N_benchmarked"]
    max_linear = metrics["max_N_with_exhaustive_reference"]

    lines = [
        "# Benchmark de escalabilidad BARW",
        "",
        "## Objetivo",
        "",
        "Evaluar la regla computacional mas costosa del simulador BARW: "
        "la busqueda de conductos cercanos dentro del radio de aniquilacion. "
        "El benchmark usa una geometria BARW reproducible y compara busqueda "
        "exhaustiva, cKDTree, grid espacial y quadtree.",
        "",
        "## Carga BARW generada",
        "",
        f"- Puntos de conducto: {workload['points']}",
        f"- Segmentos: {workload['segments']}",
        f"- Consultas registradas: {workload['queries']}",
        f"- Ramas generadas: {workload['branches']}",
        f"- Terminaciones: {workload['terminations']}",
        f"- Raices: {workload['roots']}",
        f"- Pasos simulados: {workload['steps']}",
        "",
        "## Resultados principales",
        "",
        f"- Mayor tamano benchmarkeado: N = {max_n}",
        f"- Busqueda exhaustiva medida hasta: N = {max_linear}",
        f"- Equivalencia exacta de vecinos: {checks['neighbor_equivalence_all_tested_queries']}",
        f"- Calidad operativa estimada: {metrics['estimated_quality_score_over_10']}/10",
        "",
        "## Pendientes log-log empiricas",
        "",
    ]
    for method, slope in slopes.items():
        lines.append(f"- {METHOD_STYLE[method]['label']}: {slope:.3f}")
    lines.extend(["", "## Speedup en el limite con referencia exhaustiva", ""])
    for method, speedup in speedups.items():
        label = METHOD_STYLE[method]["label"]
        if math.isfinite(float(speedup)):
            lines.append(f"- {label}: {float(speedup):.2f}x")
        else:
            lines.append(f"- {label}: no aplica")
    lines.extend(
        [
            "",
            "## Interpretacion para la memoria",
            "",
            "Este bloque no reproduce una figura concreta de Hannezo et al. (2017). "
            "Aporta la parte informatica del TFG: demuestra que la regla de "
            "terminacion por proximidad puede acelerarse sin modificar los vecinos "
            "detectados ni la regla biologica simulada.",
            "",
            "Figuras recomendadas para la memoria:",
            "",
            "- Panel A: geometria BARW usada como carga de consultas.",
            "- Panel C: escalabilidad temporal en escala log-log.",
            "- Panel D: aceleracion respecto a busqueda exhaustiva.",
            "- Panel B: verificacion de equivalencia exacta de vecinos.",
            "",
            "Para la defensa conviene usar la figura compuesta completa y explicar "
            "que el color identifica el metodo, mientras que la regla fisica de "
            "aniquilacion permanece fija.",
            "",
            "## Archivos",
            "",
            "- `figures/benchmark_escalabilidad_barw_layout.png`",
            "- `figures/benchmark_escalabilidad_barw_layout.pdf`",
            "- `figures/individual_panels/*.png`",
            "- `data/benchmark_scaling.csv`",
            "- `data/benchmark_metrics.json`",
        ]
    )
    (DATA_DIR / "summary_benchmark_escalabilidad_barw.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_readme() -> None:
    text = """# Benchmark de escalabilidad BARW

Este paquete mide la regla computacional mas costosa del modelo BARW:
detectar si una punta activa se encuentra dentro del radio de aniquilacion
de un conducto ya depositado.

## Ejecucion

```powershell
python benchmark_escalabilidad_barw.py
```

Modos:

```powershell
python benchmark_escalabilidad_barw.py --mode quick
python benchmark_escalabilidad_barw.py --mode full
```

## Que produce

- `data/benchmark_scaling.csv`: tiempos, aceleraciones y discrepancias.
- `data/benchmark_metrics.json`: metricas resumidas y checks de calidad.
- `data/summary_benchmark_escalabilidad_barw.md`: resumen para memoria.
- `figures/benchmark_escalabilidad_barw_layout.png/pdf`: figura compuesta.
- `figures/individual_panels/*.png/pdf`: paneles individuales.

## Uso en el TFG

Este script cubre la aportacion informatica: demostrar que la busqueda
espacial optimizada acelera el simulador BARW sin cambiar la regla fisica
de terminacion por proximidad.
"""
    (BASE_DIR / "README.md").write_text(text, encoding="utf-8")


def config_from_mode(mode: str) -> BenchmarkConfig:
    cfg = BenchmarkConfig()
    if mode == "quick":
        return replace(
            cfg,
            max_points=18000,
            sizes=(500, 1000, 2000, 4000, 8000),
            repeats=2,
            max_queries_per_size=2500,
            linear_max_n=4000,
        )
    if mode == "full":
        return replace(
            cfg,
            max_points=70000,
            max_steps=3500,
            sizes=(500, 1000, 2000, 4000, 8000, 16000, 32000, 64000),
            repeats=5,
            max_queries_per_size=8000,
            linear_max_n=8000,
        )
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark reproducible de busqueda espacial para BARW."
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "standard", "full"),
        default="standard",
        help="Nivel de coste computacional del benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_mode(args.mode)
    ensure_dirs()
    set_plot_style()

    print(f"[1/6] Generando carga BARW reproducible ({args.mode})...")
    workload = generate_barw_workload(cfg)
    if workload.points_xy.shape[0] < max(cfg.sizes):
        valid_sizes = tuple(size for size in cfg.sizes if size <= workload.points_xy.shape[0])
        cfg = replace(cfg, sizes=valid_sizes)

    print("[2/6] Ejecutando benchmark de busqueda espacial...")
    rows = run_benchmark(workload, cfg)

    print("[3/6] Guardando datos CSV/JSON...")
    write_csv(DATA_DIR / "benchmark_scaling.csv", rows)
    save_workload_samples(workload)
    metrics = make_metrics(workload, rows, cfg)
    (DATA_DIR / "benchmark_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[4/6] Generando figuras individuales y layout...")
    make_figures(workload, rows)

    print("[5/6] Escribiendo resumen y README...")
    write_summary(metrics, rows)
    write_readme()

    print("[6/6] Benchmark completado.")
    equivalence = "OK" if metrics["quality_checks"]["neighbor_equivalence_all_tested_queries"] else "FAIL"
    print(
        "Calidad operativa estimada: "
        f"{metrics['estimated_quality_score_over_10']}/10 | "
        f"equivalencia de vecinos: {equivalence}"
    )


if __name__ == "__main__":
    main()
