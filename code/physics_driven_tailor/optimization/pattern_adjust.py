"""
Pattern adjustment: one iteration of 2D pattern edits from edge-analysis JSON.

This module is decoupled from the MD/CLO scripting APIs. Callers must inject
an implementation of PatternAdjustAPI (e.g. from improved.py when running inside MD).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any, Protocol

from .config import AdjustmentConfig, ProjectPaths


def build_plan_from_edges(tasks: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    """
    Build a per-pattern, per-edge adjustment plan from the edges array in the
    analysis JSON. For each (pattern, edge) we keep the task with the highest
    edge_bad_ratio; type is 'tight' or 'loose'.
    """
    plan: dict[str, dict[str, Any]] = {}
    desired_edges_map: dict[str, set[str]] = {}
    valid_edges = frozenset(("bottom", "top", "left", "right"))

    for it in tasks:
        pname = str(it.get("pattern", "")).strip()
        edge = str(it.get("edge", "")).strip().lower()
        etype = str(it.get("type", "tight")).strip().lower()
        try:
            ratio = float(it.get("edge_bad_ratio", 0.0))
        except (TypeError, ValueError):
            ratio = 0.0
        if not pname or edge not in valid_edges:
            continue
        desired_edges_map.setdefault(pname, set()).add(edge)
        plan.setdefault(pname, {})
        cur = plan[pname].get(edge)
        if cur is None or ratio > cur["ratio"]:
            plan[pname][edge] = {"ratio": ratio, "type": etype}
    return plan, desired_edges_map


def _outward_dir_for_edge(edge_name: str) -> tuple[float, float]:
    tag = str(edge_name).lower()
    if tag == "bottom":
        return (0.0, -1.0)
    if tag == "top":
        return (0.0, 1.0)
    if tag == "left":
        return (-1.0, 0.0)
    if tag == "right":
        return (1.0, 0.0)
    return (0.0, 0.0)


def _unit(x: float, y: float) -> tuple[float, float]:
    L = (x * x + y * y) ** 0.5
    return (x / L, y / L) if L > 1e-8 else (0.0, 0.0)


def _sign_for_type(edge_type: str) -> float:
    return 1.0 if str(edge_type).lower() == "tight" else -1.0


class PatternAdjustAPI(Protocol):
    """Minimal interface for pattern moves and simulation; implemented by the MD host."""

    def resolve_pattern_index(self, target_name: str) -> int | None: ...
    def export_piece_lines(self, pattern_index: int, json_path: str) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]: ...
    def move_pattern_point(
        self,
        pattern_index: int,
        point_index: int,
        dx_mm: float,
        dy_mm: float,
        tmp_json_path: str,
        sim_steps: int,
        learning_rate: float,
    ) -> None: ...
    def simulate(self, steps: int) -> None: ...


def adjust_once_from_json(
    edges_json_path: str,
    paths: ProjectPaths,
    adj_cfg: AdjustmentConfig,
    learning_rate: float,
    logs: list[tuple[Any, ...]],
    api: PatternAdjustAPI,
) -> None:
    """
    Perform one iteration of pattern adjustments from the given edge-analysis JSON.

    Reads edges and params from edges_json_path, builds a plan, and for each
    (pattern, edge) above the ratio threshold computes weighted deltas and
    calls api.move_pattern_point / api.simulate. Appends log rows to `logs`.
    """
    if not os.path.isfile(edges_json_path):
        return
    with open(edges_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    tasks = cfg.get("edges", [])
    params = cfg.get("params", {}) or {}
    ratio_thr = float(params.get("min_edge_bad_ratio", adj_cfg.ratio_threshold))
    plan, desired_edges_map = build_plan_from_edges(tasks)

    tmpdir = tempfile.mkdtemp(prefix="patjson_")
    try:
        for pat_name, edge_dict in plan.items():
            pidx = api.resolve_pattern_index(pat_name)
            if pidx is None or pidx < 0:
                continue
            json_path = os.path.join(tmpdir, f"p_{pidx}.json")
            lines, start_pos_map = api.export_piece_lines(pidx, json_path)
            if not lines:
                continue
            xs = [start_pos_map[li][0] for li in start_pos_map]
            ys = [start_pos_map[li][1] for li in start_pos_map]
            if not xs or not ys:
                continue
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            span_x = max(maxx - minx, 1e-6)
            span_y = max(maxy - miny, 1e-6)
            desired_edges_str = "|".join(sorted(desired_edges_map.get(pat_name, set())))

            for edge_name, info in edge_dict.items():
                ratio = float(info.get("ratio", 0.0))
                etype = str(info.get("type", "tight")).lower()
                if ratio < ratio_thr:
                    continue

                edge_lines: list[dict[str, Any]] = []
                for e in lines:
                    li = e["line_index"]
                    px, py = start_pos_map.get(li, e["a"])
                    if edge_name == "bottom":
                        w = (maxy - py) / span_y
                    elif edge_name == "top":
                        w = (py - miny) / span_y
                    elif edge_name == "left":
                        w = (maxx - px) / span_x
                    elif edge_name == "right":
                        w = (px - minx) / span_x
                    else:
                        continue
                    if w <= 0.0:
                        continue
                    w = min(1.0, w)
                    e2 = dict(e)
                    e2["weight"] = w
                    edge_lines.append(e2)

                if not edge_lines:
                    continue

                over = max(0.0, ratio - ratio_thr)
                sgn = _sign_for_type(etype)
                raw_delta_mm = sgn * (over / 0.10) * adj_cfg.mm_per_ratio_01
                scale = (adj_cfg.micro_step_mm / abs(raw_delta_mm)) if abs(raw_delta_mm) > adj_cfg.micro_step_mm else 1.0
                delta_mm = raw_delta_mm * scale
                dirx, diry = _outward_dir_for_edge(edge_name)
                ux, uy = _unit(dirx, diry)
                base_dx, base_dy = ux * delta_mm, uy * delta_mm

                moved_indices: list[int] = []
                points_detail: list[str] = []
                for e in edge_lines:
                    li = int(e["line_index"])
                    w = float(e.get("weight", 1.0))
                    dx = base_dx * w
                    dy = base_dy * w
                    ax, ay = start_pos_map.get(li, e["a"])
                    newx, newy = ax + dx, ay + dy
                    tmp_path = os.path.join(tmpdir, f"move_p{pidx}_li{li}.json")
                    api.move_pattern_point(pidx, li, dx, dy, tmp_path, 0, learning_rate)
                    moved_indices.append(li)
                    points_detail.append(f"[{li},w={w:.2f}:({ax:.3f},{ay:.3f})->({newx:.3f},{newy:.3f})]")

                if adj_cfg.sim_steps:
                    api.simulate(int(adj_cfg.sim_steps))

                idx_str = "|".join(str(i) for i in moved_indices)
                pts_str = f"{edge_name}:" + "|".join(points_detail)
                logs.append(
                    (
                        pat_name,
                        desired_edges_str,
                        edge_name,
                        etype,
                        f"{ratio:.4f}",
                        f"{raw_delta_mm:.2f}",
                        f"{delta_mm:.2f}",
                        str(len(moved_indices)),
                        idx_str,
                        pts_str,
                    )
                )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
