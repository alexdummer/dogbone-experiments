# AGENTS.md — Guidelines for AI Coding Assistants

This document outlines architectural conventions, physics constraints, numerical pitfalls, and development guidelines for AI agents working within `dogbone-experiments/`.

---

## 1. Directory Structure & File Map

```text
dogbone-experiments/
├── data/
│   ├── tensile-flat/         # Flat specimen characterization tests (A0V100 through A100V0)
│   │   └── {Material}/       # raw/ (git-untracked), cleaned/ (Git LFS), clean_data.py / sync_and_clean_data.py
│   ├── tensile-lattice/      # Lattice tensile validation experiments (raw/, cleaned/, clean_data.py)
│   └── dma/                  # Dynamic mechanical analysis (raw/, reference/, scripts/)
├── simulations/
│   ├── material-calibration/ # Constitutive parameter optimization (qlv-marmot/, bergstrom-boyce/, prf/, 1d-qlv/, parameters/)
│   ├── fe-tension/           # EdelweissFE tensile lattice validation (mesh/, generate_jobs*.py, jobs/, postprocess/)
│   └── fe-bending/           # EdelweissFE 3-point bending lattice validation (jobs/, postprocess/)
├── specimens/
│   ├── cad/                  # Specimen generators (create_specimens.py, dogbone.stl)
│   └── slices/               # Untracked OpenVCAD PNG print slices (tensile-flat/, tensile-lattice/, dma/, bending/)
└── analysis/                 # Publication figures & cross-material synthesis (plotstyle.py, compare_materials.py, plot_lattice_*.py)
```

---

## 2. Critical Physics & Data Conventions

### Data Sign Conventions
- **Cleaned Data (`data/**/cleaned/*.csv`)**: `force_N` is **negative-in-tension** across all materials.
- **Raw Tensile Data**: Raw `Ch:Load` is negative-in-tension for all materials **except** `A100V0`, where raw `Ch:Load` is positive-in-tension. Cleaning scripts invert `A100V0` so that `cleaned/` is consistent project-wide.
- **FE Reaction Outputs (`gripForce.csv`)**: Finite element reaction forces are **positive-in-tension**. Invert sign when plotting against cleaned experimental data.

### FE Symmetry Factors
EdelweissFE dogbone lattice meshes utilize three symmetry planes:
```python
X_SYMMETRY_FACTOR = 2.0              # Converts x=0 mirror displacement to full grip displacement
CROSS_SECTION_SYMMETRY_FACTOR = 4.0  # Scales quarter-section force (y=0, z=T/2) to full specimen force
```

### Raw Acquisition Multi-Chunk Files
The MTS machine acquisition splits large tests into base files plus continuation chunks (`_1.csv`, `_2.csv`, ...). Always concatenate all chunks contiguously by timestamp; missing chunks will silently truncate tests to ~250s instead of the full ~600-720s hold.

---

## 3. Constitutive Modeling (Marmot & Fitting)

### Volumetric-Isochoric Split
All hyperelastic bases in `Marmot` compose $\Psi = \Psi_{\text{iso}}(\bar{I}_1, \bar{I}_2) + \frac{\kappa}{8}(\ln I_3)^2$.
- Parameter layout in `elasticProperties` is always `[shape_params..., kappa]` (`kappa` last).
- Base potential mapping in QLV-Marmot: Mooney-Rivlin (`MooneyRivlin`, ID 2).
- Base potential in Bergström-Boyce: Arruda-Boyce 8-chain (`ArrudaBoyce`, ID 3).

### Fitting & Solver Robustness
- **Residual Weighting**: Always apply $1/\sqrt{n_{\text{phase}}}$ weighting between the ramp and hold phases; without it, dense ramp sampling silently dominates the objective.
- **Bergström-Boyce Return Mapping**: The non-linear Newton-Raphson return-mapping algorithm exhibits seed sensitivity near local optima. Always multi-start fits (`fit_bb_model.py`) using nudge clusters or random restarts.
- **Installed Library**: Python Marmot bindings link against `/home/alex/miniforge3/envs/marmot/lib/libMarmot.so.1`. If C++ code is modified, `make -j$(nproc) && make install` is mandatory.

---

## 4. Plotting Conventions (`analysis/plotstyle.py`)

- Always import styling:
  ```python
  from plotstyle import colors, figsize_double, figsize_single
  ```
- **Axis Labels**: Must follow `"Quantity in UNIT"` (e.g. `"Engineering stress in MPa"`), never `"Quantity (UNIT)"`.
- **Strain Annotations**: Always annotate the actual measured mean engineering strain rate over the ramp ($\text{s}^{-1}$), never the nominal commanded crosshead speed in mm/s.
- **Legends**: Put multi-curve legends outside the axis (`bbox_to_anchor=(1.02, 1.0)`) or in consistent top-right/top-left locations.

---

## 5. Git & Binary Storage Integrity

- **Never Track Raw Videos or Binary Dumps**:
  - `*.MOV`, `*.mp4`, `*.avi` (multi-gigabyte video files in `data/**/raw/`) must remain untracked.
  - `*_temp_frames/` (extracted frames from optical flow) must remain untracked.
  - `specimens/slices/` (OpenVCAD slice stacks, ~1+ GB) must remain untracked.
  - `ensightExport/` in simulation job folders must remain untracked.
- **Git LFS**:
  - Handled via `.gitattributes` for `**/cleaned/*.csv`, `*.msh`, and `*.inc`.
- **Never Run `git add -A` blindly**: Always check `git status` to verify no large binary dumps or temporary files are staged.
