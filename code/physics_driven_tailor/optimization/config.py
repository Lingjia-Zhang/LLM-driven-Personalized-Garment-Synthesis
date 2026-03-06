from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any


Band = tuple[float, float]
BandOverride = Band | dict[str, Band]


def _getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return default if (v is None or v.strip() == "") else v


@dataclass(frozen=True)
class ProjectPaths:
    """
    File-system layout for a single optimization run.

    This project is expected to run inside a vendor-hosted interpreter with
    access to MD/CLO scripting APIs. Paths should point to assets that are not
    part of this repository (e.g., AVT/ZPAC, avatar OBJ, etc.).
    """

    base_dir: str
    avatar_obj: str
    cloth_obj: str
    edges_json: str
    avt_path: str
    zpac_path: str
    out_obj_path: str
    out_zpac_path: str

    @staticmethod
    def from_env() -> "ProjectPaths":
        """
        Build paths from environment variables.

        Required:
        - GSP_BASE_DIR: directory containing all runtime assets.

        Optional overrides:
        - GSP_AVATAR_OBJ, GSP_CLOTH_OBJ, GSP_EDGES_JSON
        - GSP_AVT_PATH, GSP_ZPAC_PATH
        - GSP_OUT_OBJ_PATH, GSP_OUT_ZPAC_PATH
        """

        base_dir = _getenv("GSP_BASE_DIR")
        if not base_dir:
            raise RuntimeError(
                "Missing required env var: GSP_BASE_DIR (runtime assets directory)."
            )
        base_dir = os.path.abspath(base_dir)

        def p(env: str, filename: str) -> str:
            return os.path.abspath(_getenv(env, os.path.join(base_dir, filename)) or "")

        return ProjectPaths(
            base_dir=base_dir,
            avatar_obj=p("GSP_AVATAR_OBJ", "avatar.obj"),
            cloth_obj=p("GSP_CLOTH_OBJ", "cloth_only_iter.obj"),
            edges_json=p("GSP_EDGES_JSON", "result_iter.json"),
            avt_path=p("GSP_AVT_PATH", "avatar.avt"),
            zpac_path=p("GSP_ZPAC_PATH", "garment.zpac"),
            out_obj_path=p("GSP_OUT_OBJ_PATH", "cloth_auto_final.obj"),
            out_zpac_path=p("GSP_OUT_ZPAC_PATH", "garment_fit_auto_final.zpac"),
        )


@dataclass(frozen=True)
class EdgeAnalysisConfig:
    lband: float = 2.0
    uband: float = 30.0
    edge_band_thr: float = 0.12
    min_edge_ratio: float = 0.06
    min_edge_count: int = 8
    stride: int = 5
    pattern_band_overrides: dict[str, BandOverride] = field(default_factory=dict)


@dataclass(frozen=True)
class AdjustmentConfig:
    particle_distance_mm: float = 8.0
    micro_step_mm: float = 2.0
    ratio_threshold: float = 0.04
    mm_per_ratio_01: float = 6.0
    sim_steps: int = 50
    strict_pattern_name_match: bool = True


@dataclass(frozen=True)
class OptimizationConfig:
    num_iters: int = 10
    initial_learning_rate: float = 10.0
    adaptive_learning_rate: bool = True


def normalize_pattern_band_overrides(
    raw: dict[str, Any] | None,
) -> dict[str, BandOverride]:
    """
    Normalize a loosely-typed overrides mapping.

    Supported formats:
    - {"PatternName": (L, U)}
    - {"PatternName": {"default": (L, U), "left": (L, U), ...}}
    """

    if not raw:
        return {}
    out: dict[str, BandOverride] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[str(k)] = (float(v[0]), float(v[1]))
            continue
        if isinstance(v, dict):
            vv: dict[str, Band] = {}
            for ek, ev in v.items():
                if isinstance(ev, (list, tuple)) and len(ev) == 2:
                    vv[str(ek).lower()] = (float(ev[0]), float(ev[1]))
            if vv:
                out[str(k)] = vv
    return out

