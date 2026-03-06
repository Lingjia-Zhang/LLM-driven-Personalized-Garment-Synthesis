from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any, Iterable

from .config import Band, BandOverride, EdgeAnalysisConfig, ProjectPaths


@dataclass(frozen=True)
class EdgeMetrics:
    total_loss: float
    normalized_loss: float
    samples: int
    ratio_below_l: float
    ratio_above_u: float


def _safe_makedirs(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _load_obj_full(path: str):
    vs: list[tuple[float, float, float]] = []
    vts: list[tuple[float, float]] = []
    vns: list[tuple[float, float, float]] = []
    faces: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []
    groups: list[tuple[int, int, str]] = []
    cur_group = ("default", 0)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            toks = line.strip().split()
            if not toks:
                continue

            tag = toks[0]
            if tag == "v" and len(toks) >= 4:
                vs.append((float(toks[1]), float(toks[2]), float(toks[3])))
            elif tag == "vt" and len(toks) >= 3:
                vts.append((float(toks[1]), float(toks[2])))
            elif tag == "vn" and len(toks) >= 4:
                vns.append((float(toks[1]), float(toks[2]), float(toks[3])))
            elif tag == "g":
                if faces and cur_group is not None:
                    groups.append((cur_group[1], len(faces), cur_group[0]))
                name = " ".join(toks[1:]) or "unnamed"
                cur_group = (name, len(faces))
            elif tag == "f":
                idx = []
                for part in toks[1:]:
                    sp = part.split("/")
                    vi = int(sp[0]) - 1 if sp[0] else -1
                    vti = int(sp[1]) - 1 if len(sp) > 1 and sp[1] else -1
                    vni = int(sp[2]) - 1 if len(sp) > 2 and sp[2] else -1
                    idx.append((vi, vti, vni))
                if len(idx) >= 3:
                    for k in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[k], idx[k + 1]))

    if faces and cur_group is not None:
        groups.append((cur_group[1], len(faces), cur_group[0]))
    return vs, vts, vns, faces, groups


def _load_obj_vertices_only(path: str) -> list[tuple[float, float, float]]:
    verts: list[tuple[float, float, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                verts.append((float(x), float(y), float(z)))
    return verts


def _squared_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _nearest_dist_bruteforce(
    p: tuple[float, float, float],
    cloud: list[tuple[float, float, float]],
    stride: int = 5,
    cap: int = 20_000,
) -> float:
    n = len(cloud)
    step = max(1, int(stride))
    limit = min(n, cap * step)
    best = 1e18
    for i in range(0, limit, step):
        d2 = _squared_distance(p, cloud[i])
        if d2 < best:
            best = d2
    return math.sqrt(best)


def _smooth_l1(x: float, beta: float) -> float:
    ax = abs(x)
    return 0.5 * (ax * ax) / beta if ax < beta else ax - 0.5 * beta


def _band_loss(dist: float, l: float, u: float, beta: float = 1.0) -> float:
    if dist < l:
        return _smooth_l1(l - dist, beta)
    if dist > u:
        return _smooth_l1(dist - u, beta)
    return 0.0


def _uv_bbox_for_group(vts, faces, s: int, e: int) -> tuple[float, float, float, float]:
    us: list[float] = []
    vs_: list[float] = []
    for fi in range(s, e):
        for (_, vti, _) in faces[fi]:
            if vti >= 0:
                u, v = vts[vti]
                us.append(u)
                vs_.append(v)
    if us:
        return (min(us), max(us), min(vs_), max(vs_))
    return (0.0, 1.0, 0.0, 1.0)


def _normalize_uv(
    u: float,
    v: float,
    umin: float,
    umax: float,
    vmin: float,
    vmax: float,
) -> tuple[float, float]:
    du = (u - umin) / (umax - umin + 1e-12)
    dv = (v - vmin) / (vmax - vmin + 1e-12)
    return du, dv


def _iter_unique_vertices_for_piece(faces, s: int, e: int) -> Iterable[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for fi in range(s, e):
        for (vi, vti, _) in faces[fi]:
            key = (vi, vti)
            if vi >= 0 and key not in seen:
                seen.add(key)
                yield (vi, vti)


def _get_default_band(pattern_name: str, cfg: EdgeAnalysisConfig) -> Band:
    if not pattern_name:
        return float(cfg.lband), float(cfg.uband)
    ov = cfg.pattern_band_overrides.get(str(pattern_name))
    if isinstance(ov, (list, tuple)) and len(ov) == 2:
        return float(ov[0]), float(ov[1])
    if isinstance(ov, dict) and "default" in ov:
        l, u = ov["default"]
        return float(l), float(u)
    return float(cfg.lband), float(cfg.uband)


def _get_edge_band(pattern_name: str, edge_name: str, cfg: EdgeAnalysisConfig) -> Band:
    if not pattern_name:
        return float(cfg.lband), float(cfg.uband)
    edge = (edge_name or "").lower()
    ov: BandOverride | None = cfg.pattern_band_overrides.get(str(pattern_name))

    if isinstance(ov, dict):
        if edge in ov:
            l, u = ov[edge]
            return float(l), float(u)
        if "default" in ov:
            l, u = ov["default"]
            return float(l), float(u)
    elif isinstance(ov, (list, tuple)) and len(ov) == 2:
        return float(ov[0]), float(ov[1])

    return float(cfg.lband), float(cfg.uband)


def run_edge_analysis(
    *,
    paths: ProjectPaths,
    cfg: EdgeAnalysisConfig,
    write_json: bool = True,
) -> EdgeMetrics:
    """
    Analyze garment-to-avatar distances near pattern edges and write a JSON report.

    Inputs:
    - paths.avatar_obj: avatar mesh (OBJ, vertices only)
    - paths.cloth_obj: garment mesh (OBJ with UVs + groups)
    - cfg: analysis hyperparameters and band overrides

    Output JSON schema matches the legacy `improved.py` implementation so that
    downstream adjustment code can be reused without changes.
    """

    if not os.path.isfile(paths.cloth_obj):
        raise FileNotFoundError(f"Missing cloth OBJ: {paths.cloth_obj}")
    if not os.path.isfile(paths.avatar_obj):
        raise FileNotFoundError(f"Missing avatar OBJ: {paths.avatar_obj}")

    vs, vts, _vns, faces, groups = _load_obj_full(paths.cloth_obj)
    if not vs or not faces:
        raise RuntimeError("Cloth OBJ has no vertices/faces.")
    if not vts:
        raise RuntimeError("Cloth OBJ has no UVs (vt); edge-banding requires UVs.")

    avatar_vs = _load_obj_vertices_only(paths.avatar_obj)
    if not avatar_vs:
        raise RuntimeError("Avatar OBJ has no vertices.")

    vts_norm_cache: dict[str, tuple[float, float, float, float]] = {}
    for (s, e, name) in groups:
        vts_norm_cache[name] = _uv_bbox_for_group(vts, faces, s, e)

    stride = max(1, int(cfg.stride))
    total_loss = 0.0
    cnt_all = 0
    cnt_below = 0
    cnt_above = 0

    per_group_samples: dict[str, dict[str, Any]] = {}

    # Pass 1: sample points and accumulate scalar metrics.
    for (s, e, name) in groups:
        l_def, u_def = _get_default_band(name, cfg)
        umin, umax, vmin, vmax = vts_norm_cache[name]
        records: list[dict[str, float]] = []
        piece_samples = 0

        uniq = list(_iter_unique_vertices_for_piece(faces, s, e))
        for idx, (vi, vti) in enumerate(uniq):
            if idx % stride != 0:
                continue
            p = vs[vi]
            dist = _nearest_dist_bruteforce(p, avatar_vs, stride=5, cap=20_000)

            total_loss += _band_loss(dist, l_def, u_def, beta=1.0)
            cnt_all += 1
            piece_samples += 1
            if dist < l_def:
                cnt_below += 1
            if dist > u_def:
                cnt_above += 1

            if vti >= 0:
                u, v = vts[vti]
                du, dv = _normalize_uv(u, v, umin, umax, vmin, vmax)
                records.append({"du": float(du), "dv": float(dv), "dist": float(dist)})

        per_group_samples[name] = {"records": records, "piece_samples": int(piece_samples)}

    # Pass 2: directional edge "badness" statistics using per-edge bands.
    edges: list[dict[str, Any]] = []
    per_piece_directions: dict[str, Any] = {}

    for (_s, _e, name) in groups:
        gdata = per_group_samples.get(name)
        if not gdata:
            per_piece_directions[name] = {
                "samples_in_piece": 0,
                "tight": {k: {"count": 0, "ratio": 0.0} for k in ("left", "right", "bottom", "top")},
                "loose": {k: {"count": 0, "ratio": 0.0} for k in ("left", "right", "bottom", "top")},
            }
            continue

        records = gdata["records"]
        piece_denom = max(1, int(gdata["piece_samples"]))

        dir_params = {
            "left": _get_edge_band(name, "left", cfg),
            "right": _get_edge_band(name, "right", cfg),
            "bottom": _get_edge_band(name, "bottom", cfg),
            "top": _get_edge_band(name, "top", cfg),
        }

        tight_counts = {k: 0 for k in ("left", "right", "bottom", "top")}
        loose_counts = {k: 0 for k in ("left", "right", "bottom", "top")}

        thr = float(cfg.edge_band_thr)
        for rec in records:
            du = float(rec["du"])
            dv = float(rec["dv"])
            dist = float(rec["dist"])

            # A point may contribute to multiple sides near corners (legacy behavior).
            if du < thr:
                l_edge, u_edge = dir_params["left"]
                if dist < l_edge:
                    tight_counts["left"] += 1
                if dist > u_edge:
                    loose_counts["left"] += 1

            if du > 1.0 - thr:
                l_edge, u_edge = dir_params["right"]
                if dist < l_edge:
                    tight_counts["right"] += 1
                if dist > u_edge:
                    loose_counts["right"] += 1

            if dv < thr:
                l_edge, u_edge = dir_params["bottom"]
                if dist < l_edge:
                    tight_counts["bottom"] += 1
                if dist > u_edge:
                    loose_counts["bottom"] += 1

            if dv > 1.0 - thr:
                l_edge, u_edge = dir_params["top"]
                if dist < l_edge:
                    tight_counts["top"] += 1
                if dist > u_edge:
                    loose_counts["top"] += 1

        tight_ratio = {k: (tight_counts[k] / float(piece_denom)) for k in tight_counts}
        loose_ratio = {k: (loose_counts[k] / float(piece_denom)) for k in loose_counts}

        per_piece_directions[name] = {
            "samples_in_piece": piece_denom,
            "tight": {k: {"count": tight_counts[k], "ratio": tight_ratio[k]} for k in tight_counts},
            "loose": {k: {"count": loose_counts[k], "ratio": loose_ratio[k]} for k in loose_counts},
        }

        for kind in ("tight", "loose"):
            counts = tight_counts if kind == "tight" else loose_counts
            ratios = tight_ratio if kind == "tight" else loose_ratio
            for edge_name in ("left", "right", "bottom", "top"):
                c = int(counts[edge_name])
                r = float(ratios[edge_name])
                if (c >= int(cfg.min_edge_count)) and (r >= float(cfg.min_edge_ratio)):
                    l_edge, u_edge = dir_params[edge_name]
                    edges.append(
                        {
                            "pattern": name,
                            "edge": edge_name,
                            "tight": (kind == "tight"),
                            "edge_bad_count": c,
                            "edge_bad_ratio": r,
                            "total_bad_points_in_piece": int(sum(counts.values())),
                            "type": kind,
                            "lband": float(l_edge),
                            "uband": float(u_edge),
                        }
                    )

    avg_loss = float(total_loss) / float(max(1, cnt_all))
    metrics = EdgeMetrics(
        total_loss=float(total_loss),
        normalized_loss=avg_loss,
        samples=int(cnt_all),
        ratio_below_l=float(cnt_below) / float(max(1, cnt_all)),
        ratio_above_u=float(cnt_above) / float(max(1, cnt_all)),
    )

    if write_json:
        payload = {
            "edges": edges,
            "metrics": {
                "total_loss": metrics.total_loss,
                "normalized_loss": metrics.normalized_loss,
                "samples": metrics.samples,
                "ratio_below_L": metrics.ratio_below_l,
                "ratio_above_U": metrics.ratio_above_u,
            },
            "params": {
                "lband": float(cfg.lband),
                "uband": float(cfg.uband),
                "edge_band_thr": float(cfg.edge_band_thr),
                "min_edge_bad_ratio": float(cfg.min_edge_ratio),
                "min_edge_bad_count": int(cfg.min_edge_count),
                "stride": stride,
                "trigger_logic": "count_and_ratio",
                "pattern_lband_uband_overrides": cfg.pattern_band_overrides,
            },
            "per_piece_directions": per_piece_directions,
        }

        _safe_makedirs(os.path.dirname(paths.edges_json))
        with open(paths.edges_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    return metrics

