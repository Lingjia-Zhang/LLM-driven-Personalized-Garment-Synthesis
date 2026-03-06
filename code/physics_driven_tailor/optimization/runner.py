"""
Optimization loop: simulate -> export cloth -> edge analysis -> pattern adjust.

The loop and adaptive learning-rate schedule live here. Scene setup, export, and
pattern moves are delegated to a RunnerAPI implemented by the MD/CLO host.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Protocol

from .config import AdjustmentConfig, EdgeAnalysisConfig, OptimizationConfig, ProjectPaths
from .edge_analysis import run_edge_analysis
from .pattern_adjust import PatternAdjustAPI, adjust_once_from_json


class RunnerAPI(Protocol):
    """Scene lifecycle and export; pattern adjustment is via PatternAdjustAPI."""

    def setup_scene(self, paths: ProjectPaths, adj_cfg: AdjustmentConfig) -> None: ...
    def export_current_cloth(self, paths: ProjectPaths) -> None: ...
    def export_final_results_and_log(
        self,
        paths: ProjectPaths,
        logs: list[tuple[Any, ...]],
        loss_logs: list[tuple[int, float | None]],
    ) -> None: ...


def compute_learning_rate(
    iter_id: int,
    loss_logs: list[tuple[int, float | None]],
    opt_cfg: OptimizationConfig,
) -> float:
    """
    Adaptive LR: first two iters use initial_learning_rate; then
    LR_k = initial_learning_rate * |L_{k-1} - L_{k-2}| / (|L_2 - L_1| + 1e-8).
    """
    if iter_id <= 2 or not opt_cfg.adaptive_learning_rate:
        return opt_cfg.initial_learning_rate
    if len(loss_logs) < 2:
        return opt_cfg.initial_learning_rate
    L_k_1 = loss_logs[-1][1]
    L_k_2 = loss_logs[-2][1]
    if L_k_1 is None or L_k_2 is None:
        return opt_cfg.initial_learning_rate
    base_diff: float | None = None
    if len(loss_logs) >= 2:
        L1, L2 = loss_logs[0][1], loss_logs[1][1]
        if L1 is not None and L2 is not None:
            base_diff = abs(L2 - L1)
    if base_diff is None or base_diff <= 0.0:
        return opt_cfg.initial_learning_rate
    diff_recent = abs(L_k_1 - L_k_2)
    return opt_cfg.initial_learning_rate * diff_recent / (base_diff + 1e-8)


def run_optimization_loop(
    paths: ProjectPaths,
    edge_cfg: EdgeAnalysisConfig,
    adj_cfg: AdjustmentConfig,
    opt_cfg: OptimizationConfig,
    scene_api: RunnerAPI,
    pattern_api: PatternAdjustAPI,
) -> tuple[list[tuple[Any, ...]], list[tuple[int, float | None]]]:
    """
    Run num_iters iterations: for each iter, simulate, export cloth OBJ,
    run edge analysis (writes edges JSON), compute LR, then adjust patterns.
    Returns (point-move logs, loss logs).
    """
    scene_api.setup_scene(paths, adj_cfg)
    logs: list[tuple[Any, ...]] = [
        (
            "pattern",
            "desired_edges_from_json",
            "edge",
            "type",
            "ratio",
            "raw_delta_mm",
            "actual_step_mm",
            "lines_moved",
            "moved_point_indices",
            "points_detail",
        )
    ]
    loss_logs: list[tuple[int, float | None]] = []

    for it in range(opt_cfg.num_iters):
        iter_id = it + 1
        lr = compute_learning_rate(iter_id, loss_logs, opt_cfg)

        if adj_cfg.sim_steps:
            pattern_api.simulate(int(adj_cfg.sim_steps))
        scene_api.export_current_cloth(paths)

        metrics = run_edge_analysis(paths=paths, cfg=edge_cfg, write_json=True)
        loss_logs.append((iter_id, metrics.normalized_loss))

        adjust_once_from_json(
            paths.edges_json,
            paths,
            adj_cfg,
            lr,
            logs,
            pattern_api,
        )

    scene_api.export_final_results_and_log(paths, logs, loss_logs)
    return logs, loss_logs


def write_loss_log_csv(
    loss_logs: list[tuple[int, float | None]],
    out_dir: str,
    filename: str = "log_normalized_loss_iter.csv",
) -> str:
    """Write iter, normalized_loss CSV under out_dir. Returns path to file."""
    path = os.path.join(out_dir, filename)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "normalized_loss"])
        for row in loss_logs:
            w.writerow(row)
    return path


def write_adjust_log_csv(
    logs: list[tuple[Any, ...]],
    out_dir: str,
    filename: str = "log_adjust_from_rsult_points_iter.csv",
) -> str:
    """Write point-adjust log CSV. Returns path to file."""
    path = os.path.join(out_dir, filename)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(logs)
    return path
