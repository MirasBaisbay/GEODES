---
title: GEODES - Protein Geometry Descriptor Calculator
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 8501
---

# GEODES: Geometric Descriptors for Protein Structures

GEODES is a bioinformatics tool for computing geometric descriptors from protein 3D structures stored as PDB files. It is designed primarily for comparative analysis of the **Vitamin D Receptor (VDR)** ligand-binding domain across multiple species, but the underlying descriptor framework generalises to any alpha-helical protein.

---

## Table of Contents

1. [Background](#background)
   - [What is VDR?](#what-is-vdr)
   - [What are Geometric Descriptors?](#what-are-geometric-descriptors)
   - [What is PCA?](#what-is-pca)
2. [Features](#features)
3. [Descriptor Reference](#descriptor-reference)
4. [Species Configurations](#species-configurations)
5. [Installation](#installation)
6. [Usage](#usage)
   - [Web Application](#web-application)
   - [Python API](#python-api)
   - [Command-Line Batch Processing](#command-line-batch-processing)
7. [Configuration](#configuration)
8. [Project Structure](#project-structure)
9. [Requirements](#requirements)
10. [Running Tests](#running-tests)
11. [Docker](#docker)
12. [Citation](#citation)
13. [License](#license)

---

## Background

### What is VDR?

The **Vitamin D Receptor (VDR)** is a nuclear hormone receptor that regulates gene expression in response to the active form of vitamin D (calcitriol / 1α,25-dihydroxyvitamin D₃). Upon ligand binding, VDR undergoes conformational changes in its **ligand-binding domain (LBD)** that enable co-activator recruitment and transcription of target genes.

VDR is conserved across vertebrates but differs in helix count and residue numbering between species. GEODES provides pre-configured settings for:

| Species    | Organism            | Helices | Charge Clamp Residues  |
|------------|---------------------|---------|------------------------|
| Human      | *Homo sapiens*      | 14      | 246, 264, 420          |
| Rat        | *Rattus norvegicus* | 13      | 242, 260, 416          |
| Zebrafish  | *Danio rerio*       | 13      | 274, 292, 446          |

### What are Geometric Descriptors?

Geometric descriptors are numerical features derived from the 3D atomic coordinates of a protein structure. Instead of comparing raw coordinates (which are sensitive to rigid-body translations and rotations), geometric descriptors capture **internal structural relationships** that are invariant to overall position and orientation:

- **Center-of-mass (COM) distances** – how far each helix COM is from the protein COM or from other helices.
- **Helix lengths** – the end-to-end distance between the alpha-carbon (Cα) atoms of the first and last residues of a helix.
- **Pairwise helix angles** – the angle between the orientation vectors of two helices.
- **Charge-clamp geometry** – distances and angles formed by three key charged residues (Lys, Arg, Glu) that anchor co-activator peptides.
- **Secondary structure element (SSE) content** – percentage of residues in helix, strand, turn, bend, etc. as assigned by DSSP.
- **Solvent accessibility** – accessible surface area per helix from DSSP.

Together these descriptors encode the **shape, compactness, and flexibility** of the protein fold in a compact, interpretable numeric vector.

### What is PCA?

**Principal Component Analysis (PCA)** is an unsupervised linear dimensionality-reduction technique. Given a matrix of *n* structures × *p* descriptors, PCA finds orthogonal directions (principal components, PCs) in the *p*-dimensional space that capture the maximum variance in the data.

**How it is used in GEODES:**

1. All numeric descriptor columns are **standardised** (mean = 0, variance = 1) so that descriptors with different units and scales contribute equally.
2. PCA is applied and the top 2 or 3 PCs are extracted.
3. Structures are plotted in the reduced space, coloured by species (cross-species mode) or by file name (single-species mode).

PCA makes it easy to spot:
- **Outlier structures** (e.g. MD frames far from the cluster).
- **Species differences** – systematic separation of human, rat and zebrafish VDR.
- **Conformational heterogeneity** within a single ensemble.

**Cross-species PCA** uses only **species-invariant descriptors** (SSE content, charge-clamp distances/angles, COM-clamp distances) so that species with different helix counts can still be compared on a common feature set.

---

## Features

| Feature | Description |
|---------|-------------|
| Helix geometry | COM distances, pairwise separations, lengths, pairwise angles, COM-helix angles |
| Charge-clamp analysis | Distances and angles among three key charged residues |
| Secondary structure | DSSP-based SSE fractions, helix endpoint validation, extra-helical residue count |
| Solvent accessibility | Accessible surface area per helix |
| Batch processing | Process entire directories of PDB files; optional multiprocessing |
| 3D visualisation | Interactive `py3Dmol` viewer embedded in the web app |
| PCA | 2D and 3D scatter plots with explained-variance bar chart |
| CSV export | One-click download of all calculated descriptors |
| Multi-species | Simultaneous upload and comparison of Human, Rat, and Zebrafish VDR |
| Docker | Fully containerised deployment with DSSP and KPAX pre-installed |

---

## Descriptor Reference

All descriptors are toggled via `configs/desc_config.yml`. The table below lists every descriptor key and what it produces.

| Config Key | Description | Columns Produced |
|---|---|---|
| `prot_hel_dist` | Distance from the protein COM to each helix COM | `Dist prot-hel N` for each helix N |
| `pairwise_sep_dist` | Pairwise Euclidean distance between all helix COMs | `Dist hel N-M` for each pair |
| `com_calpha_angles` | Angle at each helix COM between the protein COM and the helix Cα endpoint vectors | `Angle COM-hel N` for each helix N |
| `len_of_hel` | Euclidean distance between first and last Cα of each helix | `Len hel N` for each helix N |
| `angles_between_hel` | Angle between orientation vectors of all helix pairs | `Angle hel N-M` for each pair |
| `com_clamp` | Distance from protein COM to each of the three charge-clamp Cα atoms | `Dist COM-clamp1/2/3` |
| `charge_clamp_dist` | Pairwise distances among the three charge-clamp residues | `Dist clamp1/2/3` |
| `charge_clamp_angles` | Angles of the triangle formed by the three charge-clamp Cα atoms | `Angle clamp1-2/1-3/2-3` |
| `acc_per_hel` | Solvent-accessible surface area per helix (Å²) | `Acc hel N` for each helix N |
| `dssp_hel` | DSSP-predicted start/end residue of each helix | `DSSP hel N start/end` |
| `sse_content` | Fraction of residues in each of 8 SSE types | `SSE Helix`, `SSE Strand`, `SSE Turn`, … |
| `dssp_extra` | Number of residues outside reference helix boundaries (from DSSP) | `N_res extra helical` |

---

## Species Configurations

Helix boundaries and charge-clamp residue IDs are defined in `app.py` under `SPECIES_CONFIG`. They reflect the canonical VDR LBD crystal structures:

```python
SPECIES_CONFIG = {
    "Human": {
        "href": [[127,142],[149,152],[218,222],[226,246],[257,265],
                 [268,278],[298,302],[308,322],[328,338],[350,369],
                 [379,396],[397,406],[411,413],[417,423]],
        "clamps": [246, 264, 420]
    },
    "Rat": {
        "href": [[127,142],[149,152],[222,242],[253,262],[263,273],
                 [294,298],[304,318],[324,334],[346,365],[376,392],
                 [393,402],[405,409],[413,418]],
        "clamps": [242, 260, 416]
    },
    "Zebrafish": {
        "href": [[159,174],[181,184],[254,274],[285,293],[295,303],
                 [326,330],[336,350],[356,366],[378,397],[405,422],
                 [423,432],[435,439],[443,447]],
        "clamps": [274, 292, 446]
    }
}
```

Residue numbering follows standard PDB chain numbering for each species.

---

## Installation

### Option A – Docker (recommended)

```bash
git clone https://github.com/rinnifox/GEODES.git
cd GEODES
docker-compose up --build
```

The web UI will be available at `http://localhost:8501`.

### Option B – Local Python environment

**Prerequisites:**
- Python ≥ 3.9
- `dssp` / `mkdssp` (version 4+ recommended; must be on `$PATH`)
- Boost libraries (required by DSSP)
- KPAX binary (included as `kpax.tar.gz`)

```bash
git clone https://github.com/rinnifox/GEODES.git
cd GEODES

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the geodes package in editable mode
pip install -e .

# Launch the web app
streamlit run app.py
```

---

## Usage

### Web Application

1. Open the app in your browser (`http://localhost:8501`).
2. Use the **sidebar** to upload one or more PDB files for each species.
3. Click **Run Analysis** to calculate all descriptors.
4. Inspect the results table and the **PCA plots**.
5. Click **Download Results as CSV** to save the descriptor matrix.

**Single-species mode** – all numeric descriptors are used for PCA. DSSP columns undergo a *delta transform* (deviation from the reference helix boundaries) before PCA.

**Cross-species mode** – only species-invariant descriptors (SSE content, charge-clamp geometry, COM-clamp distances) are used so that human (14 helices) and rat/zebrafish (13 helices) can be compared on equal footing.

### Python API

```python
from geodes import DescCalculator

# Define helix boundaries (residue number pairs) and charge-clamp residues
href = [[127,142],[149,152],[218,222],[226,246],[257,265],
        [268,278],[298,302],[308,322],[328,338],[350,369],
        [379,396],[397,406],[411,413],[417,423]]
clamps = [246, 264, 420]

calculator = DescCalculator(ref=href, clamp_resid=clamps)

# Single file
df = calculator.calc_single_file("structure.pdb")

# All PDB files in a directory
df = calculator.calc_all("path/to/pdbs/", save_to_csv=True, output_full_path="results.csv")

# Parallel processing (multiprocessing)
df = calculator.calc_all("path/to/pdbs/", parallel=True)
```

### Command-Line Batch Processing

Use the Python API from a script or notebook. See `examples/usage_example.ipynb` for a worked example and `examples/geodes-analysis/analysis_example.ipynb` for downstream analysis with RMSD/RMSF comparison.

---

## Configuration

`configs/desc_config.yml` controls which descriptor groups are calculated. Comment out any key to skip that descriptor group:

```yaml
descriptors:
  - prot_hel_dist          # protein COM → helix COM distances
  - pairwise_sep_dist      # pairwise helix COM distances
  - com_calpha_angles      # COM-helix angles
  - len_of_hel             # helix lengths
  - angles_between_hel     # pairwise helix angles
  - com_clamp              # COM → charge-clamp distances
  - charge_clamp_dist      # charge-clamp pairwise distances
  - charge_clamp_angles    # charge-clamp triangle angles
  - acc_per_hel            # solvent accessibility per helix
  - dssp_hel               # DSSP helix endpoints
  - sse_content            # secondary structure element fractions
  - dssp_extra             # extra-helical residue count
```

---

## Project Structure

```
GEODES/
├── src/geodes/                        # Core Python package
│   ├── __init__.py
│   ├── main.py                        # DescCalculator class
│   ├── utils.py                       # PDB parsing helpers
│   ├── constraints.py                 # Atomic weights, amino-acid max ASA
│   ├── COM_helix.py                   # Helix centre-of-mass
│   ├── COM_protein.py                 # Protein centre-of-mass
│   ├── dist_COMprot_COMhel.py        # Protein–helix COM distances
│   ├── dist_COMhel_pairwise.py       # Pairwise helix COM distances
│   ├── dist_hel_Ca_endpoints.py      # Helix lengths (Cα endpoints)
│   ├── dist_charge_clamp_Ca.py       # Charge-clamp distances
│   ├── dist_charge_clamp_Ca_COMprot.py  # COM → charge-clamp distances
│   ├── angle_hel_pairwise.py         # Pairwise helix angles
│   ├── angle_charge_clamp_Ca.py      # Charge-clamp triangle angles
│   ├── angle_COMprot_hel_Ca_endpoints.py  # COM–helix-endpoint angles
│   ├── dssp_sse_content.py           # SSE fractions
│   ├── dssp_hel_endpoints.py         # DSSP helix boundaries
│   └── dssp_acc_hel.py               # Solvent accessibility per helix
├── configs/
│   └── desc_config.yml               # Descriptor toggle configuration
├── tests/
│   ├── unit/
│   │   └── test_calc.py              # Unit tests
│   └── __init__.py
├── examples/
│   ├── usage_example.ipynb           # Basic API usage
│   └── geodes-analysis/
│       ├── analysis_example.ipynb    # Downstream analysis
│       ├── geodes_l3.csv             # Example output
│       ├── md_frames/                # Example PDB trajectories
│       └── rmsd_rmsf/                # RMSD/RMSF matrices
├── app.py                            # Streamlit web application
├── debug_dssp.py                     # DSSP diagnostic utility
├── robust_pdb_cleaner.py             # PDB cleaning utility
├── requirements.txt
├── setup.py
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── LICENSE
```

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.9 | Runtime |
| NumPy | 2.1.0 | Vector mathematics |
| Pandas | 2.2.2 | Tabular data |
| Biopython | ≥ 1.79 | PDB parsing, DSSP wrapper |
| scikit-learn | 1.5.1 | PCA, StandardScaler |
| Plotly | 5.24.0 | Interactive plots |
| PyYAML | 6.0 | Config file parsing |
| tqdm | 4.65.0 | Progress bars |
| Streamlit | latest | Web application framework |
| py3Dmol / stmol | latest | 3D structure viewer |
| DSSP (`mkdssp`) | ≥ 4.0 | Secondary structure assignment |

---

## Running Tests

```bash
pytest tests/ -v --cov=src/geodes
```

The test suite (`tests/unit/test_calc.py`) validates the five primary descriptor types against a reference PDB structure (`tests/data/1DB1.pdb`).

---

## Docker

The `Dockerfile` builds a self-contained image with:
- Python 3.10-slim-bookworm base
- DSSP installed via apt and wrapped for Biopython compatibility
- KPAX extracted and built from `kpax.tar.gz`
- All Python dependencies from `requirements.txt`
- Streamlit entry point on port 8501

```bash
# Build and run
docker build -t geodes .
docker run -p 8501:8501 geodes

# Or with docker-compose (includes volume mounts for live code editing)
docker-compose up
```

---

## Citation

If you use GEODES in your research, please cite:

> [Add your citation here]

---

## License

GEODES is released under the [MIT License](LICENSE).

**Author:** Karina Pats (karina.m.pats@gmail.com)
**Repository:** https://github.com/rinnifox/GEODES
