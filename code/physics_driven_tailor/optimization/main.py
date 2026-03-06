"""
Single-script entrypoint for the garment fitting loop, to be run inside
Marvelous Designer / CLO scripting environment.

Loop (NUM_ITERS): simulate -> export cloth OBJ -> edge analysis (JSON) ->
pattern adjust. Adaptive learning rate after the first two iters.
All paths and hyperparameters can be overridden via environment variables
or by editing the defaults below. The optimization logic lives in
garment_synthesis.optimization.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from typing import Any

# Marvelous Designer / CLO API (available only inside the vendor host)
try:
    import import_api
    import export_api
    import utility_api
    import pattern_api
    try:
        import ApiTypes
    except Exception:
        ApiTypes = None
except ImportError as e:
    raise RuntimeError(
        "This script must run inside Marvelous Designer / CLO scripting context. "
        "Missing: " + str(e)
    ) from e

# Add project root so garment_synthesis can be imported
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from garment_synthesis.optimization.config import (
    AdjustmentConfig,
    EdgeAnalysisConfig,
    OptimizationConfig,
    ProjectPaths,
    normalize_pattern_band_overrides,
)
from garment_synthesis.optimization.runner import run_optimization_loop
from garment_synthesis.optimization.pattern_adjust import PatternAdjustAPI


# ---------------------------------------------------------------------------
# Default paths (override with GSP_BASE_DIR and optional GSP_* vars)
# ---------------------------------------------------------------------------
_DEFAULT_BASE_DIR = os.getenv("GSP_BASE_DIR", "").strip() or os.path.expanduser("~/MD_test")


def _default_paths() -> ProjectPaths:
    base = os.path.abspath(_DEFAULT_BASE_DIR)
    return ProjectPaths(
        base_dir=base,
        avatar_obj=os.path.join(base, "2.obj"),
        cloth_obj=os.path.join(base, "cloth_only_iter.obj"),
        edges_json=os.path.join(base, "rsult_iter.json"),
        avt_path=os.path.join(base, "2.avt"),
        zpac_path=os.path.join(base, "1.zpac"),
        out_obj_path=os.path.join(base, "cloth_auto_final.obj"),
        out_zpac_path=os.path.join(base, "garment_fit_auto_final.zpac"),
    )


# Default band overrides (pattern name -> (L, U) or dict with default/left/right/top/bottom)
PATTERN_BAND_OVERRIDES = {
    "Pattern_9125": {"default": (2.0, 30.0), "bottom": (2.0, 75.0)},
    "Pattern_7924": {"default": (2.0, 30.0), "bottom": (2.0, 75.0)},
    "Pattern_12911": {"default": (2.0, 30.0), "bottom": (2.0, 75.0)},
}


def _make_import_option():
    if ApiTypes is None:
        return None
    return ApiTypes.ImportExportOption()


def _make_export_obj_option():
    if ApiTypes is None:
        return None
    opt = ApiTypes.ImportExportOption()
    opt.bExportGarment = True
    opt.bExportAvatar = False
    opt.bSingleObject = False
    return opt


# ---------------------------------------------------------------------------
# RunnerAPI: scene setup and export (MD/CLO)
# ---------------------------------------------------------------------------
class _SceneRunner:
    def setup_scene(self, paths: ProjectPaths, adj_cfg: AdjustmentConfig) -> None:
        opt = _make_import_option()
        if hasattr(import_api, "ImportAvatar"):
            ok = import_api.ImportAvatar(paths.avt_path, opt) if opt else import_api.ImportAvatar(paths.avt_path)
        else:
            ok = import_api.ImportFile(paths.avt_path, opt) if opt else import_api.ImportFile(paths.avt_path)
        if not ok:
            raise RuntimeError("ImportAvatar/ImportFile failed for AVT: " + paths.avt_path)
        if hasattr(import_api, "ImportZpac"):
            ok = import_api.ImportZpac(paths.zpac_path, opt) if opt else import_api.ImportZpac(paths.zpac_path)
        else:
            ok = import_api.ImportFile(paths.zpac_path, opt) if opt else import_api.ImportFile(paths.zpac_path)
        if not ok:
            raise RuntimeError("ImportZpac/ImportFile failed for ZPAC: " + paths.zpac_path)
        try:
            n = pattern_api.GetPatternCount()
            for i in range(n):
                pattern_api.SetParticleDistanceOfPattern(i, float(adj_cfg.particle_distance_mm))
        except Exception as e:
            print("[pd-warn]", e)

    def export_current_cloth(self, paths: ProjectPaths) -> None:
        opt = _make_export_obj_option()
        if hasattr(export_api, "ExportOBJ"):
            try:
                if opt:
                    export_api.ExportOBJ(paths.cloth_obj, opt)
                else:
                    export_api.ExportOBJ(paths.cloth_obj)
            except Exception:
                print("[WARN] ExportOBJ failed:", traceback.format_exc())
        else:
            print("[WARN] ExportOBJ not available")

    def export_final_results_and_log(
        self,
        paths: ProjectPaths,
        logs: list[tuple[Any, ...]],
        loss_logs: list[tuple[int, float | None]],
    ) -> None:
        opt = _make_export_obj_option()
        try:
            if hasattr(export_api, "ExportOBJ"):
                if opt:
                    export_api.ExportOBJ(paths.out_obj_path, opt)
                else:
                    export_api.ExportOBJ(paths.out_obj_path)
                print("[OK] Exported final OBJ ->", paths.out_obj_path)
        except Exception:
            print("[WARN] ExportOBJ failed:", traceback.format_exc())
        try:
            if hasattr(export_api, "ExportZPac"):
                export_api.ExportZPac(paths.out_zpac_path)
                print("[OK] Exported final ZPAC ->", paths.out_zpac_path)
        except Exception:
            print("[WARN] ExportZPac failed:", traceback.format_exc())
        out_dir = os.path.dirname(paths.out_zpac_path)
        log_csv = os.path.join(out_dir, "log_adjust_from_rsult_points_iter.csv")
        try:
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(logs)
            print("[log]", log_csv)
        except Exception:
            print("[WARN] Failed to write adjust log CSV:", traceback.format_exc())
        loss_csv = os.path.join(out_dir, "log_normalized_loss_iter.csv")
        try:
            with open(loss_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["iter", "normalized_loss"])
                for row in loss_logs:
                    w.writerow(row)
            print("[loss-log]", loss_csv)
        except Exception:
            print("[WARN] Failed to write loss log CSV:", traceback.format_exc())


# ---------------------------------------------------------------------------
# PatternAdjustAPI: pattern resolution, export lines, move point, simulate
# ---------------------------------------------------------------------------
class _PatternAdapter:
    def resolve_pattern_index(self, target_name: str) -> int | None:
        if not target_name:
            return None
        target_name = str(target_name)
        try:
            n = pattern_api.GetPatternCount()
        except Exception as e:
            print("[resolve_pattern_index] GetPatternCount failed:", e)
            return None
        for i in range(n):
            try:
                nm = pattern_api.GetPatternPieceName(i)
            except Exception:
                continue
            if nm == target_name:
                return i
        return None

    def export_piece_lines(
        self, pattern_index: int, json_path: str
    ) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
        import json as _json
        pattern_api.SelectPatternViaIndex(int(pattern_index), True)
        ok = pattern_api.ExportPatternJSON(json_path)
        if not ok or not os.path.exists(json_path):
            raise RuntimeError("ExportPatternJSON failed for pattern " + str(pattern_index))
        with open(json_path, "r", encoding="utf-8") as f:
            d = _json.load(f)
        plist = d.get("PatternList", []) or []
        if not plist:
            return [], {}
        piece = plist[0]
        line_list = (piece.get("ShapeInfo") or {}).get("LineList") or []
        lines = []
        start_pos_map = {}
        for i, seg in enumerate(line_list):
            pts = seg.get("PointList", []) or []
            if not pts:
                continue
            ax = float(pts[0]["Position"]["x"])
            ay = float(pts[0]["Position"]["y"])
            if len(pts) >= 2:
                bx, by = float(pts[-1]["Position"]["x"]), float(pts[-1]["Position"]["y"])
            else:
                bx, by = ax, ay
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            lines.append({"line_index": i, "a": (ax, ay), "b": (bx, by), "mid": (mx, my)})
            start_pos_map[i] = (ax, ay)
        return lines, start_pos_map

    def move_pattern_point(
        self,
        pattern_index: int,
        point_index: int,
        dx_mm: float,
        dy_mm: float,
        tmp_json_path: str,
        sim_steps: int,
        learning_rate: float,
    ) -> None:
        ok = pattern_api.ExportPatternJSON(tmp_json_path)
        if not ok:
            raise RuntimeError("ExportPatternJSON failed in move_pattern_point")
        import json as _json
        with open(tmp_json_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        plist = data["PatternList"][pattern_index]["ShapeInfo"]["LineList"][point_index]["PointList"]
        pos = plist[0]["Position"]
        x0, y0 = float(pos["x"]), float(pos["y"])
        new_x = x0 + dx_mm * learning_rate
        new_y = y0 + dy_mm * learning_rate
        pattern_api.MovePatternPoint(int(pattern_index), int(point_index), float(new_x), float(new_y))
        if sim_steps and hasattr(utility_api, "Simulate"):
            utility_api.Simulate(int(sim_steps))

    def simulate(self, steps: int) -> None:
        if steps and hasattr(utility_api, "Simulate"):
            utility_api.Simulate(int(steps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    paths = _default_paths()
    if not os.path.isdir(paths.base_dir):
        print("[WARN] Base dir does not exist:", paths.base_dir)

    edge_cfg = EdgeAnalysisConfig(
        lband=2.0,
        uband=30.0,
        edge_band_thr=0.12,
        min_edge_ratio=0.06,
        min_edge_count=8,
        stride=5,
        pattern_band_overrides=normalize_pattern_band_overrides(PATTERN_BAND_OVERRIDES),
    )
    adj_cfg = AdjustmentConfig(
        particle_distance_mm=8.0,
        micro_step_mm=2.0,
        ratio_threshold=0.04,
        mm_per_ratio_01=6.0,
        sim_steps=50,
    )
    opt_cfg = OptimizationConfig(
        num_iters=10,
        initial_learning_rate=10.0,
        adaptive_learning_rate=True,
    )

    scene_api = _SceneRunner()
    pattern_api_adapter: PatternAdjustAPI = _PatternAdapter()
    run_optimization_loop(paths, edge_cfg, adj_cfg, opt_cfg, scene_api, pattern_api_adapter)
    print("[DONE] Optimization loop finished.")


if __name__ == "__main__":
    main()
