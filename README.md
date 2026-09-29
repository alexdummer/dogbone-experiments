# Photopolymer Viscoelasticity & Lattice FE Validation

Characterization, constitutive modeling, and finite-element validation of 3D-printed multi-material photopolymers (Stratasys PolyJet VeroWhite / Agilus30 composites: A0V100 through A100V0).

This repository contains the complete experimental datasets (tensile relaxation, optical videoextensometry, dynamic mechanical analysis, lattice tension), calibrated constitutive model pipelines (QLV-Marmot, Bergström-Boyce, PRF, 1D QLV), EdelweissFE boundary-value simulation decks and results, specimen CAD generators, and publication figure scripts.

---

## Repository Structure

```text
dogbone-experiments/
├── data/                              # Experimental datasets
│   ├── tensile-flat/                  # Uniaxial tensile relaxation & cyclic test data
│   │   ├── A0V100/                    # 100% VeroWhite (raw/, cleaned/, scripts)
│   │   ├── A25V75/                    # 25% Agilus / 75% VeroWhite
│   │   ├── A50V50/                    # 50% Agilus / 50% VeroWhite (baseline & retest)
│   │   ├── A75V25/                    # 75% Agilus / 25% VeroWhite (optical video tracking)
│   │   ├── A100V0/                    # 100% Agilus30 (optical video tracking)
│   │   └── tests/                     # Preliminary/exploratory tests (git-untracked)
│   ├── tensile-lattice/               # Lattice-infill dogbone validation tests
│   │   ├── raw/                       # Raw machine dumps (git-untracked)
│   │   ├── cleaned/                   # Cleaned load-displacement CSVs (Git LFS)
│   │   └── clean_data.py              # Processing pipeline
│   └── dma/                           # Dynamic Mechanical Analysis (DMA)
│       ├── raw/                       # Frequency & temperature sweeps (git-untracked)
│       ├── reference/                 # Digitized literature benchmark data (Wirth et al. 2025)
│       └── scripts/                   # TTS, WLF shifts, master curve generators
│
├── simulations/                       # Constitutive calibrations & FE modeling
│   ├── material-calibration/          # Continuum constitutive parameter calibration
│   │   ├── qlv-marmot/                # 3D QLV Mooney-Rivlin + 3-term Prony (C++ Marmot backend)
│   │   ├── bergstrom-boyce/           # Bergström-Boyce 8-chain finite-strain viscoplasticity
│   │   ├── prf/                       # Parallel Rheological Framework
│   │   ├── 1d-qlv/                    # 1D QLV reference models
│   │   ├── abaqus-decks/              # Single-element Abaqus verification decks (.inp)
│   │   └── parameters/                # Calibrated parameter catalogs (CSV)
│   ├── fe-tension/                    # EdelweissFE tensile lattice validation jobs
│   │   ├── mesh/                      # Hex mesh generation (mesh_lattice.py, dogbone.stl)
│   │   ├── generate_jobs.py           # Deck generator for QLV-Marmot models
│   │   ├── generate_jobs_bb.py        # Deck generator for Bergström-Boyce models
│   │   ├── jobs/                      # Per-material simulation decks & reaction force/disp outputs
│   │   └── postprocess/               # Contour extraction & load-displacement plots
│   └── fe-bending/                    # EdelweissFE 3-point bending lattice simulations
│       ├── jobs/                      # Multi-crack size simulations (a5mm to a40mm)
│       └── postprocess/               # Bending load-displacement & size-effect postprocessing
│
├── specimens/                         # CAD specimen generation & print slices
│   ├── cad/                           # Unified generator (create_specimens.py, dogbone.stl)
│   └── slices/                        # OpenVCAD PNG print slices (git-untracked)
│
├── code/                              # Pinned code dependencies (Git submodules)
│   ├── EdelweissFE/                   # Nonlinear FE solver backend (branch: next_v26.11)
│   ├── Marmot/                        # Material Modeling Toolbox (branch: feature/bergstrom-boyce)
│   ├── Fastor/                        # SIMD tensor algebra library for C++
│   └── dogbone-videoextensometer/     # Optical videoextensometer & tracking CLI
│
└── analysis/                          # Publication figures & cross-cutting synthesis
    ├── plotstyle.py                   # Central styling module (colorblind palette, LaTeX labels)
    ├── compare_materials.py           # Multi-material tensile comparison figures
    ├── mixture_theory_comparison.py   # Rule-of-mixtures analysis
    ├── standup_slides.py              # Summary presentation figure generator
    ├── plot_calibration_*.py          # Calibration buildup & model comparisons
    └── plot_lattice_*.py              # Tensile lattice validation figures (FE vs Experiment)
```

---

## Key Conventions & Physics

### 1. Data Sign Conventions
- **Cleaned Tensile Data (`data/**/cleaned/*.csv`)**: `force_N` is **negative-in-tension** project-wide.
- **Raw Machine Data**: Raw `Ch:Load` is negative-in-tension for all materials **except** `A100V0` (which is positive-in-tension raw). The cleaning scripts automatically account for this inversion.
- **Finite Element Outputs (`gripForce.csv`)**: FE reaction force is **positive-in-tension**. Flip sign when comparing against cleaned experimental CSVs.

### 2. Finite Element Symmetry Scalings
The dogbone lattice FE model exploits three mirror symmetry planes:
- **Displacement**: `X_SYMMETRY_FACTOR = 2.0` (x=0 mirror plane converts half-gauge elongation to full grip displacement).
- **Force**: `CROSS_SECTION_SYMMETRY_FACTOR = 4.0` (y=0 and z=T/2 planes represent a quarter cross-section; multiply by 4 to recover total specimen force).

### 3. Constitutive Models
- **QLV-Marmot (`simulations/material-calibration/qlv-marmot/`)**: Mooney-Rivlin isochoric hyperelastic base + 3-term Prony series relaxation with volumetric penalty split $\Psi = \Psi_{\text{iso}}(\bar{I}_1, \bar{I}_2) + \frac{\kappa}{8}(\ln I_3)^2$.
- **Bergström-Boyce (`simulations/material-calibration/bergstrom-boyce/`)**: Arruda-Boyce 8-chain network + non-linear viscoplastic relaxation. Due to return-mapping sensitivity near optimum basins, multi-start optimization is always used.

### 4. Git LFS & Data Storage
- Cleaned CSV files (`data/**/cleaned/*.csv`) and FE mesh files (`*.msh`, `*.inc`) are tracked using **Git LFS**.
- Multi-gigabyte raw video files (`*.MOV`), raw machine exports (`*.tdms`), temporary optical-flow frame dumps (`*_temp_frames/`), and OpenVCAD print-slice stacks (`specimens/slices/`) are stored locally and ignored via `.gitignore`.

---

## Quickstart

### Prerequisites
- Python 3.10+
- Scientific Python stack: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pillow`
- Marmot constitutive modeling library with Python bindings (for Marmot simulation fits)
- LaTeX installation (for vector figure text rendering via `plotstyle.py`)

### Common Tasks

1. **Re-clean flat tensile data**:
   ```bash
   python data/tensile-flat/A0V100/clean_data.py
   python data/tensile-flat/A100V0/sync_and_clean_data.py
   ```

2. **Re-clean lattice validation data**:
   ```bash
   python data/tensile-lattice/clean_data.py
   ```

3. **Run cross-material comparison & generate figures**:
   ```bash
   python analysis/compare_materials.py
   ```

4. **Generate FE simulation job decks**:
   ```bash
   # QLV-Marmot models
   python simulations/fe-tension/generate_jobs.py
   # Bergström-Boyce models
   python simulations/fe-tension/generate_jobs_bb.py
   ```

5. **Generate lattice validation figures**:
   ```bash
   python analysis/plot_lattice_experiment_comparison.py
   python analysis/plot_lattice_experiment_sim_comparison.py
   python analysis/plot_lattice_stress_contours.py
   ```

---

## Code Submodules & Computational Dependencies

To reproduce the constitutive model fittings, finite element simulations, and optical tracking pipelines, initialize the pinned Git submodules in `code/`:

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:alexdummer/dogbone-experiments.git

# Or in an existing clone:
git submodule update --init --recursive
```

The pinned modules under `code/` provide:
- **`code/EdelweissFE`**: Finite-element solver backend used for all 3D boundary-value simulations (branch: `next_v26.11`).
- **`code/Marmot`**: C++ constitutive modeling library with Python bindings, implementing the 3D QLV Mooney-Rivlin + Prony series and Bergström-Boyce models (branch: `feature/bergstrom-boyce`).
- **`code/Fastor`**: SIMD tensor algebra library required for building Marmot.
- **`code/dogbone-videoextensometer`**: Optical-flow videoextensometer CLI tool used for tracking shoulder drift and true strain in soft photopolymers ($A75V25$ and $A100V0$).

### Setting Up the Simulation Environment (`edelweissfe`)

EdelweissFE runs on free-threaded Python 3.14 (`cp314t`) with full parallel element evaluation (`PYTHON_GIL=0`). Set up the environment and compile Marmot and EdelweissFE with:

```bash
# 1. Create dedicated conda environment from pinned official package
conda create -n edelweissfe -y -c https://repo.prefix.dev/matthiasneuner/edelweiss -c conda-forge edelweissfe-dev=2026.09.28.2
conda activate edelweissfe

# 2. Build and install Marmot into the conda environment
cmake -S code/Marmot -B code/Marmot/build \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
cmake --build code/Marmot/build -j$(nproc)
cmake --install code/Marmot/build

# 3. Install EdelweissFE in editable mode and additional simulation dependencies
pip install -v -e code/EdelweissFE
pip install pandas seaborn

# 4. Verify test suite (free-threaded execution)
PYTHON_GIL=0 run_tests_edelweissfe code/EdelweissFE/testfiles/edelweiss-only/
PYTHON_GIL=0 run_tests_edelweissfe code/EdelweissFE/testfiles/marmot/
```

### Running Simulations with EdelweissFE

```bash
conda activate edelweissfe
# Run single job deck
edelweissfe simulations/fe-tension/jobs/A0V100/A0V100_relax.inp
```


