import streamlit as st
import yaml
import os
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from geodes import DescCalculator

# Function to load default config
def load_default_config():
    config_path = Path("configs/desc_config.yml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {'descriptors': []}

def clean_pdb_for_dssp(input_path, output_path):
    """
    Clean PDB file for mkdssp v4+ compatibility.
    Uses ultra-minimal approach: HEADER + ATOM records + END only.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    has_atoms = False

    # Add minimal HEADER (required by mkdssp)
    cleaned_lines.append('HEADER    PROTEIN                                 01-JAN-00   XXXX\n')

    # Keep only ATOM and HETATM records (structure coordinates)
    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            cleaned_lines.append(line)
            has_atoms = True

    if not has_atoms:
        raise ValueError("PDB file contains no ATOM records!")

    # Add END record
    cleaned_lines.append('END\n')

    # Write cleaned PDB
    with open(output_path, 'w') as f:
        f.writelines(cleaned_lines)

    return output_path

def delta_transform(dssp_values, ref_flat):
    """
    Apply delta transform to DSSP columns.
    For even indices (helix starts): ref - value
    For odd indices (helix ends): value - ref
    """
    result = dssp_values.copy()
    for i in range(len(result)):
        if i % 2 == 0:
            result.iloc[i] = ref_flat[i] - result.iloc[i]
        else:
            result.iloc[i] = result.iloc[i] - ref_flat[i]
    return result

# Species-invariant descriptor columns (same count regardless of helix count)
INVARIANT_COLS = [
    'SSE Helix', 'SSE Beta bridge', 'SSE Strand', 'SSE Helix-3',
    'SSE Helix-5', 'SSE Turn', 'SSE Bend', 'SSE Other',
    'Dist COM-clamp1', 'Dist COM-clamp2', 'Dist COM-clamp3',
    'Dist clamp1', 'Dist clamp2', 'Dist clamp3',
    'Angle clamp1-2', 'Angle clamp1-3', 'Angle clamp2-3',
    'N_res extra helical'
]

# Define species-specific configurations
SPECIES_CONFIG = {
    "Human": {
        "href": [[127,142], [149,152], [218, 222], [226,246], [257,265], [268, 278], [298,302], [308,322], [328,338], [350,369], [379,396], [397,406], [411,413], [417,423]],
        "clamps": [246, 264, 420]
    },
    "Rat": {
        "href": [[127,142], [149,152], [222,242], [253,262], [263, 273], [294,298], [304,318], [324,334], [346,365], [376,392], [393,402], [405,409], [413,418]],
        "clamps": [242, 260, 416]
    },
    "Zebrafish": {
        "href": [[159,174], [181, 184], [254,274], [285, 293], [295,303], [326, 330], [336,350], [356,366], [378,397], [405,422], [423, 432], [435,439], [443, 447]],
        "clamps": [274, 292, 446]
    }
}

yaml_config = load_default_config()

# ---------------------------------------------------------------------------
# APP DESCRIPTION
# ---------------------------------------------------------------------------
#
# GEODES – Geometric Descriptors for Protein Structures
# ======================================================
# GEODES calculates a comprehensive set of geometric descriptors from protein
# 3D structure files (PDB format).  It is designed for comparative analysis of
# the Vitamin D Receptor (VDR) ligand-binding domain across multiple species,
# but the descriptor framework generalises to any alpha-helical protein.
#
# What is VDR?
# ------------
# The Vitamin D Receptor (VDR) is a nuclear hormone receptor that mediates the
# biological effects of calcitriol (1α,25-dihydroxyvitamin D₃).  Ligand
# binding triggers conformational rearrangements in the ligand-binding domain
# (LBD) that allow co-activator recruitment and gene activation.  Because VDR
# is conserved across vertebrates with slightly different helix topologies,
# comparing structures from multiple species requires descriptors that are
# independent of helix count.
#
# What is PCA?
# ------------
# Principal Component Analysis (PCA) is an unsupervised linear
# dimensionality-reduction technique.  GEODES standardises all numeric
# descriptors (zero mean, unit variance) and projects the structures onto the
# top 2-3 principal components so that conformational variation can be
# visualised in a 2D or 3D scatter plot.
#
# In single-species mode the full descriptor set is used (DSSP columns are
# delta-transformed: deviation from the reference helix boundaries).
# In cross-species mode only species-invariant descriptors are used (SSE
# content, charge-clamp geometry, COM-clamp distances) so that species with
# different helix counts can be compared on equal footing.
#
# Descriptors calculated
# ----------------------
# 1.  prot_hel_dist        – Euclidean distance from the protein centre-of-mass
#                            (COM) to each helix COM.
# 2.  pairwise_sep_dist    – Pairwise distances between all helix COMs.
# 3.  com_calpha_angles    – Angle at each helix COM between the protein COM
#                            and the helix Cα endpoint vectors.
# 4.  len_of_hel           – Helix length: distance between the first and last
#                            Cα atoms of each reference helix.
# 5.  angles_between_hel   – Angle between orientation vectors of all helix
#                            pairs.
# 6.  com_clamp            – Distance from the protein COM to each of the three
#                            charge-clamp Cα atoms.
# 7.  charge_clamp_dist    – Pairwise distances among the three charge-clamp
#                            residues.
# 8.  charge_clamp_angles  – Angles of the triangle formed by the three
#                            charge-clamp Cα atoms.
# 9.  acc_per_hel          – Solvent-accessible surface area per helix (Å²)
#                            from DSSP.
# 10. dssp_hel             – DSSP-predicted start and end residue for each
#                            helix.
# 11. sse_content          – Fraction of residues in each of 8 SSE classes
#                            (H helix, B beta-bridge, E strand, G 3-10 helix,
#                            I pi-helix, T turn, S bend, other).
# 12. dssp_extra           – Number of residues outside the reference helix
#                            boundaries as assigned by DSSP.
#
# Species pre-configurations
# --------------------------
# Human    (Homo sapiens)      – 14 helices, charge clamps 246 / 264 / 420
# Rat      (Rattus norvegicus) – 13 helices, charge clamps 242 / 260 / 416
# Zebrafish (Danio rerio)      – 13 helices, charge clamps 274 / 292 / 446
#
# Workflow
# --------
# 1. Upload one or more PDB files via the sidebar (per-species uploaders).
# 2. Each file is cleaned for mkdssp v4+ compatibility (HEADER + ATOM + END).
# 3. DescCalculator runs all enabled descriptor modules on every file.
# 4. Results from all species are concatenated into a single DataFrame.
# 5. PCA is applied and interactive 2D/3D scatter plots are displayed.
# 6. The full descriptor matrix can be downloaded as CSV.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GEODES: Protein Geometry", layout="wide", page_icon="🧬")

# --- Sidebar: Per-species file uploaders ---
with st.sidebar:
    st.header("Upload Structures")

    uploaded_human = st.file_uploader(
        "Human VDR", type=['pdb'], accept_multiple_files=True, key="human"
    )
    uploaded_rat = st.file_uploader(
        "Rat VDR", type=['pdb'], accept_multiple_files=True, key="rat"
    )
    uploaded_zebrafish = st.file_uploader(
        "Zebrafish VDR", type=['pdb'], accept_multiple_files=True, key="zebrafish"
    )

    species_uploads = {}
    if uploaded_human:
        species_uploads["Human"] = uploaded_human
    if uploaded_rat:
        species_uploads["Rat"] = uploaded_rat
    if uploaded_zebrafish:
        species_uploads["Zebrafish"] = uploaded_zebrafish

# --- Main UI ---
if species_uploads:

    # --- 3D Structure Preview ---
    st.subheader("3D Structure Preview")

    # Build list of all files with species labels
    all_files = []
    for sp, files in species_uploads.items():
        for f in files:
            all_files.append((sp, f))

    file_labels = [f"{sp}: {f.name}" for sp, f in all_files]
    selected_label = st.selectbox("Select file to visualize:", file_labels)
    selected_idx = file_labels.index(selected_label)
    selected_species, selected_file = all_files[selected_idx]

    col1, col2 = st.columns([3, 1])

    with col1:
        try:
            from stmol import showmol
            import py3Dmol

            pdb_content = selected_file.getvalue().decode("utf-8")

            view = py3Dmol.view(width=800, height=500)
            view.addModel(pdb_content, 'pdb')
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            view.addStyle({'hetflag': True}, {'stick': {}})
            view.zoomTo()
            showmol(view, height=500)

        except Exception as e:
            st.warning(f"Visualization unavailable: {e}")

    with col2:
        st.info(f"**{selected_species} VDR**\n\n{selected_file.name}")

    st.divider()

    # --- Analysis ---
    total_files = sum(len(files) for files in species_uploads.values())
    species_summary = ", ".join(f"{sp} ({len(files)})" for sp, files in species_uploads.items())

    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Processing {total_files} PDB files across {len(species_uploads)} species..."):
            try:
                base_path = Path("/app/data_temp") if os.path.exists("/app") else Path("data_temp")
                raw_dir = base_path / "raw"
                input_dir = base_path / "input"

                # Clean up temp dirs
                for dir_path in [raw_dir, input_dir]:
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True)

                run_config = {'descriptors': yaml_config.get('descriptors', [])}
                all_results = []
                progress_bar = st.progress(0)
                processed = 0

                for species_name, files in species_uploads.items():
                    config = SPECIES_CONFIG[species_name]
                    href = config["href"]
                    clamps = config["clamps"]

                    # Create species-specific input directory
                    species_input = input_dir / species_name
                    species_input.mkdir(parents=True, exist_ok=True)
                    species_raw = raw_dir / species_name
                    species_raw.mkdir(parents=True, exist_ok=True)

                    species_ok = 0
                    for f in files:
                        try:
                            raw_path = species_raw / f.name
                            with open(raw_path, "wb") as dest:
                                dest.write(f.getbuffer())

                            cleaned_path = species_input / f.name
                            clean_pdb_for_dssp(raw_path, cleaned_path)
                            species_ok += 1
                        except Exception as e:
                            st.error(f"Failed to process {f.name}: {e}")
                            continue

                        processed += 1
                        progress_bar.progress(processed / total_files)

                    if species_ok == 0:
                        st.warning(f"No valid PDB files for {species_name} VDR. Skipping.")
                        continue

                    st.info(f"Running GEODES for {species_name} VDR ({species_ok} files)...")
                    calculator = DescCalculator(href, clamps, config=run_config)
                    df = calculator.calc_all(str(species_input) + "/", save_to_csv=False, parallel=False)
                    df['Species'] = species_name
                    all_results.append(df)

                if not all_results:
                    st.error("No species could be processed. Please check your input files.")
                    st.stop()

                df_result = pd.concat(all_results, ignore_index=True)

                st.success(f"Analysis complete! {len(df_result)} structures across {len(all_results)} species.")
                st.dataframe(df_result)

                csv = df_result.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="geodes_results_all.csv",
                    mime="text/csv"
                )

                # --- PCA Analysis ---
                st.divider()
                st.subheader("PCA Analysis")

                multi_species = len(all_results) > 1

                if multi_species:
                    # Cross-species: use only invariant columns
                    available = [c for c in INVARIANT_COLS if c in df_result.columns]
                    if len(available) < 2:
                        st.warning("Not enough invariant descriptors for cross-species PCA.")
                        st.stop()
                    df_pca_input = df_result[available].copy()
                    valid_mask = df_pca_input.notna().all(axis=1)
                    df_pca_input = df_pca_input[valid_mask]
                    color_col = 'Species'
                    color_values = df_result.loc[df_pca_input.index, 'Species'].values
                    st.caption(f"Cross-species PCA using {len(available)} invariant descriptors (SSE content, charge clamps, COM clamps).")
                else:
                    # Single species: all numeric columns + delta transform
                    single_species = list(species_uploads.keys())[0]
                    href = SPECIES_CONFIG[single_species]["href"]
                    df_pca_input = df_result.drop(columns=['prot_name', 'Species']).select_dtypes(include=[np.number]).copy()
                    valid_mask = df_pca_input.notna().all(axis=1)
                    df_pca_input = df_pca_input[valid_mask]

                    # Delta transform on DSSP columns
                    dssp_cols = [c for c in df_pca_input.columns if c.startswith('DSSP')]
                    if dssp_cols:
                        ref_flat = np.array(href).flatten()
                        if len(ref_flat) == len(dssp_cols):
                            df_pca_input[dssp_cols] = df_pca_input[dssp_cols].apply(
                                lambda row: delta_transform(row, ref_flat), axis=1
                            )

                    color_col = 'Structure'
                    color_values = df_result.loc[df_pca_input.index, 'prot_name'].values

                if len(df_pca_input) < 2:
                    st.warning("PCA requires at least 2 structures with complete data.")
                else:
                    # Remove constant columns
                    df_pca_input = df_pca_input.loc[:, df_pca_input.std() > 0]

                    # Scale
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(df_pca_input)

                    # PCA
                    max_components = min(3, len(df_pca_input), len(df_pca_input.columns))
                    if max_components < 2:
                        st.warning("Not enough features or samples for PCA.")
                    else:
                        n_components = st.slider("Number of PCA components", 2, max_components, max_components)

                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(scaled)

                        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
                        pca_df[color_col] = color_values
                        pca_df['Structure'] = df_result.loc[df_pca_input.index, 'prot_name'].values

                        # Explained variance
                        var_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(n_components)],
                            'Explained Variance (%)': pca.explained_variance_ratio_ * 100
                        })
                        fig_var = px.bar(var_df, x='Component', y='Explained Variance (%)',
                                         title='Explained Variance per Component',
                                         text_auto='.1f')
                        fig_var.update_layout(template='simple_white')
                        st.plotly_chart(fig_var, use_container_width=True)

                        # 2D + 3D scatter
                        col_pca1, col_pca2 = st.columns(2)
                        with col_pca1:
                            fig_2d = px.scatter(pca_df, x='PC1', y='PC2',
                                                color=color_col, hover_name='Structure', opacity=0.8,
                                                title='PCA - 2D Projection')
                            fig_2d.update_traces(marker=dict(size=10))
                            fig_2d.update_layout(template='simple_white')
                            st.plotly_chart(fig_2d, use_container_width=True)

                        with col_pca2:
                            if n_components >= 3:
                                fig_3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                                                       color=color_col, hover_name='Structure', opacity=0.8,
                                                       title='PCA - 3D Projection')
                                fig_3d.update_traces(marker=dict(size=5))
                                fig_3d.update_layout(template='simple_white')
                                st.plotly_chart(fig_3d, use_container_width=True)

                        with st.expander("PCA Coordinates"):
                            st.dataframe(pca_df)

            except Exception as e:
                st.error(f"Analysis Error: {e}")
                with st.expander("Show Detailed Error"):
                    import traceback
                    st.code(traceback.format_exc())
else:
    st.title("GEODES: Geometric Descriptors for Protein Structures")
    st.markdown(
        "GEODES computes a comprehensive set of geometric and secondary-structure "
        "descriptors from protein PDB files for comparative structural analysis. "
        "It is pre-configured for the **Vitamin D Receptor (VDR)** ligand-binding "
        "domain across three vertebrate species."
    )
    st.info("Upload PDB file(s) using the sidebar to get started. You can upload structures for one or more species.")

    st.divider()

    col_about1, col_about2 = st.columns(2)

    with col_about1:
        st.subheader("What is VDR?")
        st.markdown("""
The **Vitamin D Receptor (VDR)** is a nuclear hormone receptor that mediates the
biological effects of calcitriol (1α,25-dihydroxyvitamin D₃).  Upon ligand binding,
the VDR ligand-binding domain (LBD) undergoes conformational changes that enable
co-activator recruitment and activation of vitamin D target genes.

Three key **charge-clamp residues** (one Lys, one Arg, one Glu) form a molecular
"clamp" that anchors the LXXLL helix of co-activator proteins.  GEODES tracks the
geometry of these residues as a primary readout of receptor activation state.
        """)

        st.subheader("What is PCA?")
        st.markdown("""
**Principal Component Analysis (PCA)** is an unsupervised dimensionality-reduction
method that projects high-dimensional descriptor data onto a small number of axes
(principal components, PCs) that explain the most variance.

GEODES standardises all descriptors before PCA so that features with different
physical units contribute equally.  The resulting 2D and 3D scatter plots reveal:

- **Conformational clusters** within a single MD ensemble
- **Species differences** between human, rat, and zebrafish VDR
- **Outlier structures** that deviate from the main population
        """)

    with col_about2:
        st.subheader("Descriptors Calculated")
        st.markdown("""
| # | Descriptor | What it measures |
|---|---|---|
| 1 | **Protein–helix COM distance** | Distance from protein centre-of-mass to each helix COM |
| 2 | **Pairwise helix separation** | Euclidean distance between all pairs of helix COMs |
| 3 | **COM–helix Cα angles** | Angle at each helix COM between the protein COM and the Cα endpoints |
| 4 | **Helix length** | End-to-end Cα distance for each reference helix |
| 5 | **Pairwise helix angles** | Angle between orientation vectors of all helix pairs |
| 6 | **COM–clamp distances** | Distance from protein COM to each charge-clamp Cα |
| 7 | **Charge-clamp distances** | Pairwise distances among the three charge-clamp residues |
| 8 | **Charge-clamp angles** | Triangle angles formed by the three charge-clamp Cα atoms |
| 9 | **Solvent accessibility / helix** | DSSP accessible surface area per helix (Å²) |
| 10 | **DSSP helix endpoints** | DSSP-predicted start/end residue for each helix |
| 11 | **SSE content** | Fraction of residues in 8 secondary-structure classes |
| 12 | **Extra-helical residues** | Residues outside reference helix boundaries (DSSP) |
        """)

    st.divider()

    with st.expander("How It Works – Step-by-Step Workflow"):
        st.markdown("""
**1. Upload PDB files**
Use the per-species uploaders in the sidebar.  You can upload one or many files
per species.  Files from multiple species can be analysed simultaneously.

**2. PDB cleaning**
Each file is stripped down to `HEADER + ATOM + END` records for full compatibility
with mkdssp v4+.  This step removes non-standard records that can cause DSSP to
fail.

**3. Descriptor calculation**
`DescCalculator` iterates over every PDB file and runs all descriptor modules
enabled in `configs/desc_config.yml`.  Results for each file are merged into a
single row; rows from all files are concatenated into a DataFrame.

**4. Results table**
The full descriptor matrix is displayed and can be downloaded as a CSV file.

**5. PCA analysis**
- *Single-species*: all numeric descriptors are used.  DSSP endpoint columns are
  delta-transformed (deviation from the reference helix boundaries) so that the
  absolute residue numbers do not dominate.
- *Cross-species*: only species-invariant descriptors are used (SSE content,
  charge-clamp distances/angles, COM-clamp distances) so that species with
  different helix counts can be compared on the same axes.
- Constant columns (zero variance) are removed before PCA.
- All remaining columns are standardised (mean = 0, variance = 1).
- The top 2–3 principal components are displayed as interactive 2D and 3D
  scatter plots with an explained-variance bar chart.
        """)

    with st.expander("Species Configuration"):
        st.markdown("""
**Pre-configured VDR settings for three species:**

| Species | Helices | Charge clamp residues |
|---|---|---|
| **Human** (*Homo sapiens*) | 14 | 246, 264, 420 |
| **Rat** (*Rattus norvegicus*) | 13 | 242, 260, 416 |
| **Zebrafish** (*Danio rerio*) | 13 | 274, 292, 446 |

Upload structures to the corresponding species uploader in the sidebar.
When multiple species are uploaded, PCA uses species-invariant descriptors
(SSE content, charge clamps) and colours points by species.
        """)
