# Garment Synthesis Project

This repository implements core components of an **unpublished** integrated framework for AI-driven, personalized garment synthesis. The pipeline turns multi-modal user input (portrait + intent) into high-fidelity garment renders through four collaborative stages; the code here focuses on **pattern generation** and **physics-driven deformation**, and is intended to run **inside** Marvelous Designer / CLO scripting context.

---

## Framework overview (paper pipeline)

The full system comprises four stages:

| Stage | Role | What this repo provides |
|-------|------|-------------------------|
| **I. Requirement Analyst** | Multi-modal input (portrait, intent), biometric extraction (colorimetry, 3D avatar, body attributes), domain knowledge (aesthetic/anthropometric/TPO rules), Neuro-Symbolic RAG → high-level **Color & Texture** and **Shape** parameters. | — *(not included)* |
| **II. Generative Designer** | PBR texture generation (e.g. finetuned LDM + PBR decoder); sewing pattern generation (e.g. transformer → pattern token sequence → 2D panels). | **Pattern creation** from a specification (`utils/convert_cur.py`: vertices, edges, curvature → `CreatePatternWithPoints` + point log). |
| **III. Physics-Driven Deformer** | Virtual draping (avatar + pattern + texture → initial stress); **approximated optimization loop**: clearance-band loss → geometric topology / 2D pattern adjustment → physical simulation → convergence; output: final stress map + PBR-textured render. | **Optimization loop** (`improved.py`, `garment_synthesis.optimization` / `code/physics_driven_tailor.optimization`): simulate → export cloth OBJ → edge (clearance-band) analysis → pattern adjust; adaptive LR, per-pattern/per-edge band overrides. |
| **IV. Design Evaluator** | Dual-modal human feedback (user + designer, quantitative + qualitative); feedback reasoning → loopback to Requirement Analyst. | — *(not included)* |

Scripts in this repo assume **MD/CLO** APIs (`import_api`, `export_api`, `utility_api`, `pattern_api`).

---

## Repository layout

```
.
├── improved.py              # Entrypoint: run optimization loop inside MD/CLO
├── garment_synthesis/       # Optimization package (used by improved.py)
│   └── optimization/
│       ├── config.py        # Paths, edge analysis & adjustment config
│       ├── edge_analysis.py # OBJ load, band loss, edge JSON output
│       ├── pattern_adjust.py# One-iter pattern moves from edge JSON
│       └── runner.py       # Loop + adaptive LR
├── code/
│   └── physics_driven_tailor/
│       └── optimization/   # Standalone optimization module (config, edge_analysis, pattern_adjust, runner, main)
└── utils/
    └── convert_cur.py       # Create patterns from specification.json; write point log
```

---

## Requirements

- **Marvelous Designer** or **CLO** with scripting support (Python host).
- Python 3.9+ (as provided by the vendor host).
- No extra pip dependencies for the core scripts; the optimization package uses only the standard library and vendor APIs.

---

## Setup

1. Clone the repository.
2. Place your assets in a working directory (e.g. avatar OBJ, initial garment ZPAC/AVT, etc.).
3. Either set the environment variable **`GSP_BASE_DIR`** to that directory, or edit the default paths at the top of `improved.py` / inside `_default_paths()`.

---

## Usage

### 1. Optimization loop (fit garment to body)

Run **inside** Marvelous Designer / CLO:

1. Set `GSP_BASE_DIR` to the folder containing:
   - `2.obj` (avatar mesh)
   - `2.avt`, `1.zpac` (avatar + initial garment)
   - Outputs will be written there (e.g. `cloth_only_iter.obj`, `rsult_iter.json`, `cloth_auto_final.obj`, `garment_fit_auto_final.zpac`).
2. Execute `improved.py` from the project root (so that `garment_synthesis` is importable).

Optional environment overrides (all under `GSP_BASE_DIR` if not set):

| Variable           | Purpose                    |
|--------------------|----------------------------|
| `GSP_BASE_DIR`     | Base directory (required if not using script defaults) |
| `GSP_AVATAR_OBJ`   | Avatar OBJ path            |
| `GSP_CLOTH_OBJ`    | Current cloth OBJ (per iter) |
| `GSP_EDGES_JSON`   | Edge analysis result JSON  |
| `GSP_AVT_PATH`     | Avatar AVT file            |
| `GSP_ZPAC_PATH`    | Garment ZPAC file          |
| `GSP_OUT_OBJ_PATH` | Final cloth OBJ            |
| `GSP_OUT_ZPAC_PATH`| Final garment ZPAC         |

The loop: **simulate → export cloth OBJ → edge analysis (writes JSON) → pattern adjust**. Iteration count, learning rate, and band overrides are configured in `improved.py` (and in the optimization config types if you call the package directly).

### 2. Pattern creation from specification

Run **inside** MD/CLO:

1. Prepare a **`specification.json`** with a `pattern` object (e.g. `panels`, `panel_order`) and optional `properties.units_in_meter`.
2. Set **`JSON_PATH`** and **`LOG_PATH`** at the top of `utils/convert_cur.py` (defaults use `~/MD_size_adjust/`).
3. Run `utils/convert_cur.py`.

The script creates pattern pieces via `pattern_api.CreatePatternWithPoints` from panel vertices and curved edges (spline interpolation), and writes a point log for each panel.

---

## Configuration notes

- **Edge analysis** (in `garment_synthesis.optimization` or `code/physics_driven_tailor.optimization`): global `lband`/`uband`, `edge_band_thr`, `min_edge_ratio`, `min_edge_count`, `stride`, and optional **pattern band overrides** (per pattern, or per edge direction: left/right/top/bottom).
- **Adjustment**: particle distance, micro step (mm), ratio threshold, mm per 0.1 ratio, sim steps per move.
- **Optimization**: number of iterations, initial learning rate, and adaptive LR (first two iters fixed, then scaled by recent loss difference).

---

## Citation

The pipeline (Requirement Analyst → Generative Designer → Physics-Driven Deformer → Design Evaluator) and the optimization formulation correspond to an **unpublished** paper. A citation will be added upon publication.

---

## License

See the repository license file (if present). This project is for research and development use with Marvelous Designer / CLO.
