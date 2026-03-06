# -*- coding: utf-8 -*-
"""
Create pattern pieces in Marvelous Designer / CLO from specification.json.

Reads panel definitions, inserts spline points along curved edges according to
curvature, creates patterns via pattern_api.CreatePatternWithPoints, and writes
a point log for each panel.

Notes:
- The trailing duplicate vertex (same as the first) is omitted; MD closes the shape.
- The first vertex type is set from whether the last edge is curved (spline vs line).
"""

import json
import math
import os

import pattern_api  # Available only inside MD/CLO scripting context.

# -----------------------------------------------------------------------------
# Paths and parameters (override as needed)
# -----------------------------------------------------------------------------
JSON_PATH = os.path.join(os.path.expanduser("~"), "MD_size_adjust", "specification.json")
LOG_PATH = os.path.join(os.path.expanduser("~"), "MD_size_adjust", "pattern_points_log.txt")

# Number of subdivisions for curved edges (spline interpolation).
CURVE_SUBDIVS = 6
# Horizontal gap between panels when laying out (mm), to avoid overlap.
PANEL_GAP_MM = 50.0
# Coefficient for curvature amplitude: amplitude = CUR_COE * c2 * edge_length.
CUR_COE = 0.45


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def polygon_signed_area(points_xy):
    """
    Compute signed area of a polygon in the xy-plane.
    Used to determine winding order (CCW vs CW) for inward normal.
    """
    area = 0.0
    n = len(points_xy)
    for i in range(n):
        x1, y1 = points_xy[i]
        x2, y2 = points_xy[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def make_curved_edge_points_spline(p0, p1, curvature, polygon_area, subdivs=CURVE_SUBDIVS):
    """
    Build points along a curved edge by inserting spline vertices between p0 and p1.

    Args:
        p0, p1: (x_mm, y_mm) endpoints.
        curvature: [c1, c2] curvature parameters; amplitude uses c2 * length.
        polygon_area: signed area of the panel (determines inward normal direction).
        subdivs: number of interior spline points.

    Returns:
        List of (x_mm, y_mm, vtype):
        - Does not include p0 (caller adds the start point).
        - Interior points use VertexType 2 (Spline).
        - Last point is p1 with VertexType 0 (Line).
    """
    x0, y0 = p0
    x1, y1 = p1

    vx, vy = x1 - x0, y1 - y0
    L = math.hypot(vx, vy)
    if L < 1e-6:
        return [(x1, y1, 0)]

    ux, uy = vx / L, vy / L

    # Inward normal from edge: CCW polygon -> left of edge; CW -> right.
    if polygon_area > 0:
        nx, ny = -uy, ux
    else:
        nx, ny = uy, -ux

    c1, c2 = curvature
    amplitude = CUR_COE * c2 * L

    pts = []
    for i in range(1, subdivs + 1):
        t = i / (subdivs + 1.0)
        bump = 4.0 * t * (1.0 - t)
        bx = x0 + vx * t + nx * amplitude * bump
        by = y0 + vy * t + ny * amplitude * bump
        pts.append((bx, by, 2))

    pts.append((x1, y1, 0))
    return pts


def build_panel_points(panel, unit_to_mm, offset_x=0.0, offset_y=0.0):
    """
    Build the point list for CreatePatternWithPoints for one panel.

    - Converts vertices to mm and applies offset.
    - For curved edges, inserts spline points; first vertex type is set from
      whether the last edge is curved (2 = Spline, 0 = Line).
    - Omits the duplicate closing vertex (MD closes the shape automatically).
    """
    raw_verts = panel["vertices"]
    edges = panel["edges"]

    verts_mm = [
        (vx * unit_to_mm + offset_x, vy * unit_to_mm + offset_y)
        for (vx, vy) in raw_verts
    ]

    poly_area = polygon_signed_area(verts_mm)
    all_points = []

    for eidx, edge in enumerate(edges):
        i0, i1 = edge["endpoints"]
        p0 = verts_mm[i0]
        p1 = verts_mm[i1]

        if eidx == 0:
            all_points.append((p0[0], p0[1], 0))

        if "curvature" in edge and edge["curvature"] is not None:
            curv_pts = make_curved_edge_points_spline(
                p0, p1, edge["curvature"], polygon_area=poly_area
            )
            all_points.extend(curv_pts)
        else:
            all_points.append((p1[0], p1[1], 0))

    if edges and all_points:
        last_edge = edges[-1]
        is_last_edge_curved = (
            "curvature" in last_edge and last_edge["curvature"] is not None
        )
        x0, y0, _ = all_points[0]
        first_type = 2 if is_last_edge_curved else 0
        all_points[0] = (x0, y0, first_type)

    if len(all_points) >= 2:
        all_points = all_points[:-1]

    return all_points


# -----------------------------------------------------------------------------
# Main: load JSON, build points, create patterns, write log
# -----------------------------------------------------------------------------

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    pattern = data["pattern"]
    panels = pattern["panels"]
    panel_order = pattern.get("panel_order", list(panels.keys()))

    units_in_meter = float(data.get("properties", {}).get("units_in_meter", 100.0))
    unit_to_mm = 1000.0 / units_in_meter

    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write("==== Pattern Point Log ====\n\n")
        current_x_offset = 0.0

        for idx, name in enumerate(panel_order):
            panel = panels[name]
            verts = panel["vertices"]
            xs = [v[0] for v in verts]
            width_mm = (max(xs) - min(xs)) * unit_to_mm

            pts = build_panel_points(
                panel,
                unit_to_mm,
                offset_x=current_x_offset,
                offset_y=0.0,
            )

            log.write(f"[Panel {name}]\n")
            for i, (x, y, vtype) in enumerate(pts):
                log.write(f"  Point {i}: x={x:.3f}, y={y:.3f}, type={vtype}\n")
            log.write("\n")

            pidx = pattern_api.CreatePatternWithPoints(pts)
            print(f"[OK] Created pattern {name}, index={pidx}")

            current_x_offset += width_mm + PANEL_GAP_MM

    print(f"[DONE] Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
