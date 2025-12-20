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

GEODES is a bioinformatics tool for calculating geometric descriptors from protein structures (PDB files).

## Features
- Calculate helix geometry descriptors
- Charge clamp analysis
- Interactive 3D structure visualization
- Batch processing support

## Usage
1. Upload your PDB file(s)
2. Configure helix boundaries and charge clamps
3. Run analysis
4. Download results as CSV

## Citation
If you use GEODES in your research, please cite:
[Add your citation here]

## Requirements
- Python 3.10
- DSSP (for secondary structure)
- Boost libraries
- KPAX

All dependencies are included in the Docker container.