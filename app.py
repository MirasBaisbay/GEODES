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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.graph_objects as go
import umap
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
    },
    "Sea_lamprey": {
       "href": [[5, 21], [27, 31], [33, 35], [82, 101], [111, 130], [131, 133], [146, 148], [162, 178], [182, 193], [204, 225], [231, 262], [271, 277]],
       "clamps": [101, 119, 275]  
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
# Sea Lamprey - 12 helices, charge clamps 101 / 119 / 275
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
    
    uploaded_sea_lamprey = st.file_uploader(
        "Sea_Lamprey VDR", type=['pdb'], accept_multiple_files=True, key="sea_lamprey"
    )

    species_uploads = {}
    if uploaded_human:
        species_uploads["Human"] = uploaded_human
    if uploaded_rat:
        species_uploads["Rat"] = uploaded_rat
    if uploaded_zebrafish:
        species_uploads["Zebrafish"] = uploaded_zebrafish
    if uploaded_sea_lamprey:
        species_uploads["Sea_lamprey"] = uploaded_sea_lamprey

# --- Main UI ---
if species_uploads:

    # Clear stale results if uploaded files change
    current_files_sig = str(sorted(
        (sp, f.name) for sp, files in species_uploads.items() for f in files
    ))
    if st.session_state.get('_files_sig') != current_files_sig:
        for key in ['df_result', 'all_results', 'species_uploads_keys']:
            st.session_state.pop(key, None)
        st.session_state['_files_sig'] = current_files_sig

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

                # Store results in session state so they persist across reruns
                st.session_state['df_result'] = df_result
                st.session_state['all_results'] = all_results
                st.session_state['species_uploads_keys'] = list(species_uploads.keys())
                st.success(f"Analysis complete! {len(df_result)} structures across {len(all_results)} species.")

            except Exception as e:
                st.error(f"Analysis Error: {e}")
                with st.expander("Show Detailed Error"):
                    import traceback
                    st.code(traceback.format_exc())

    # --- Display results and analysis tabs (persists across reruns) ---
    if 'df_result' in st.session_state:
        df_result = st.session_state['df_result']
        all_results = st.session_state['all_results']

        st.dataframe(df_result)

        csv = df_result.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="geodes_results_all.csv",
            mime="text/csv"
        )

        # --- Dimensionality Reduction & Clustering ---
        st.divider()
        st.subheader("Dimensionality Reduction & Clustering")

        multi_species = len(all_results) > 1

        if multi_species:
            # Cross-species: use only invariant columns
            available = [c for c in INVARIANT_COLS if c in df_result.columns]
            if len(available) < 2:
                st.warning("Not enough invariant descriptors for cross-species analysis.")
                st.stop()
            df_pca_input = df_result[available].copy()
            valid_mask = df_pca_input.notna().all(axis=1)
            df_pca_input = df_pca_input[valid_mask]
            color_col = 'Species'
            color_values = df_result.loc[df_pca_input.index, 'Species'].values
            st.caption(f"Cross-species analysis using {len(available)} invariant descriptors (SSE content, charge clamps, COM clamps).")
        else:
            # Single species: all numeric columns + delta transform
            single_species = st.session_state['species_uploads_keys'][0]
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
            st.warning("Analysis requires at least 2 structures with complete data.")
        else:
            # Remove constant columns
            df_pca_input = df_pca_input.loc[:, df_pca_input.std() > 0]

            # Scale
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df_pca_input)
            n_samples = len(df_pca_input)
            structure_names = df_result.loc[df_pca_input.index, 'prot_name'].values

            # Helper: compute PCA projection for cluster visualization
            def get_pca_projection(scaled_data, max_comp=3):
                nc = min(max_comp, scaled_data.shape[0], scaled_data.shape[1])
                if nc < 2:
                    return None, nc
                p = PCA(n_components=nc)
                return p.fit_transform(scaled_data), nc

            # Helper: display clustering metrics
            def show_cluster_metrics(scaled_data, labels):
                unique_labels = set(labels)
                unique_labels.discard(-1)
                n_clusters = len(unique_labels)
                if n_clusters < 2:
                    st.warning("Need at least 2 clusters for quality metrics.")
                    return
                # Filter out noise for metrics
                mask = np.array(labels) != -1
                if mask.sum() < 2:
                    st.warning("Not enough non-noise points for metrics.")
                    return
                filtered_data = scaled_data[mask]
                filtered_labels = np.array(labels)[mask]
                if len(set(filtered_labels)) < 2:
                    st.warning("All non-noise points are in the same cluster.")
                    return
                sil = silhouette_score(filtered_data, filtered_labels)
                ch = calinski_harabasz_score(filtered_data, filtered_labels)
                db = davies_bouldin_score(filtered_data, filtered_labels)
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Silhouette Score", f"{sil:.3f}",
                              help="Range [-1, 1]. Higher is better.")
                with mc2:
                    st.metric("Calinski-Harabasz", f"{ch:.1f}",
                              help="Higher is better. Between/within cluster dispersion ratio.")
                with mc3:
                    st.metric("Davies-Bouldin", f"{db:.3f}",
                              help="Lower is better. Avg similarity between clusters.")

            # Helper: show cluster scatter plots on PCA projection
            def show_cluster_scatter(labels, method_name, pca_coords, pca_nc):
                cluster_col_vals = [f'Cluster {c}' if c != -1 else 'Noise' for c in labels]
                viz_df = pd.DataFrame(
                    pca_coords, columns=[f'PC{i+1}' for i in range(pca_nc)]
                )
                viz_df['Cluster'] = cluster_col_vals
                viz_df['Structure'] = structure_names
                viz_df[color_col] = color_values

                c1, c2 = st.columns(2)
                with c1:
                    fig2d = px.scatter(viz_df, x='PC1', y='PC2',
                                       color='Cluster', hover_name='Structure',
                                       symbol=color_col, opacity=0.8,
                                       title=f'{method_name} on PCA 2D')
                    fig2d.update_traces(marker=dict(size=10))
                    fig2d.update_layout(template='simple_white')
                    st.plotly_chart(fig2d, use_container_width=True)
                with c2:
                    if pca_nc >= 3:
                        fig3d = px.scatter_3d(viz_df, x='PC1', y='PC2', z='PC3',
                                              color='Cluster', hover_name='Structure',
                                              symbol=color_col, opacity=0.8,
                                              title=f'{method_name} on PCA 3D')
                        fig3d.update_traces(marker=dict(size=5))
                        fig3d.update_layout(template='simple_white')
                        st.plotly_chart(fig3d, use_container_width=True)

                with st.expander(f"{method_name} Cluster Assignments"):
                    assign_df = pd.DataFrame({
                        'Structure': structure_names,
                        color_col: color_values,
                        'Cluster': labels
                    })
                    st.dataframe(assign_df)

            # --- Tabs ---
            tab_pca, tab_tsne, tab_umap, tab_km, tab_hier, tab_dbscan, tab_hdbscan = st.tabs(
                ["PCA", "t-SNE", "UMAP", "K-means", "Hierarchical", "DBSCAN", "HDBSCAN"]
            )

            # ===================== PCA Tab =====================
            with tab_pca:
                max_components = min(3, n_samples, len(df_pca_input.columns))
                if max_components < 2:
                    st.warning("Not enough features or samples for PCA.")
                else:
                    n_components = st.slider("Number of PCA components", 2, max_components, max_components, key="pca_slider")

                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(scaled)

                    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
                    pca_df[color_col] = color_values
                    pca_df['Structure'] = structure_names

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

            # ===================== t-SNE Tab =====================
            with tab_tsne:
                if n_samples < 3:
                    st.warning("t-SNE requires at least 3 samples.")
                else:
                    with st.expander("t-SNE Parameters", expanded=True):
                        tc1, tc2, tc3 = st.columns(3)
                        with tc1:
                            max_perp = min(50, n_samples - 1)
                            def_perp = min(30, max_perp)
                            tsne_perplexity = st.slider("Perplexity", 2, max_perp, def_perp, key="tsne_perp",
                                                        help="Balance between local/global structure. Typical: 5-50.")
                        with tc2:
                            tsne_lr = st.slider("Learning rate", 10, 1000, 200, key="tsne_lr")
                        with tc3:
                            tsne_iter = st.slider("Iterations", 250, 2000, 1000, key="tsne_iter")
                        tsne_comp = st.slider("Number of t-SNE components", 2, 3, 2, key="tsne_comp")

                    with st.spinner("Running t-SNE..."):
                        tsne = TSNE(n_components=tsne_comp, perplexity=tsne_perplexity,
                                    learning_rate=tsne_lr, n_iter=tsne_iter, random_state=42)
                        tsne_result = tsne.fit_transform(scaled)

                    tsne_df = pd.DataFrame(tsne_result, columns=[f'tSNE{i+1}' for i in range(tsne_comp)])
                    tsne_df[color_col] = color_values
                    tsne_df['Structure'] = structure_names

                    st.caption(f"t-SNE KL divergence: {tsne.kl_divergence_:.4f}")

                    ct1, ct2 = st.columns(2)
                    with ct1:
                        fig_tsne_2d = px.scatter(tsne_df, x='tSNE1', y='tSNE2',
                                                 color=color_col, hover_name='Structure', opacity=0.8,
                                                 title='t-SNE - 2D Projection')
                        fig_tsne_2d.update_traces(marker=dict(size=10))
                        fig_tsne_2d.update_layout(template='simple_white')
                        st.plotly_chart(fig_tsne_2d, use_container_width=True)
                    with ct2:
                        if tsne_comp >= 3:
                            fig_tsne_3d = px.scatter_3d(tsne_df, x='tSNE1', y='tSNE2', z='tSNE3',
                                                        color=color_col, hover_name='Structure', opacity=0.8,
                                                        title='t-SNE - 3D Projection')
                            fig_tsne_3d.update_traces(marker=dict(size=5))
                            fig_tsne_3d.update_layout(template='simple_white')
                            st.plotly_chart(fig_tsne_3d, use_container_width=True)

                    with st.expander("t-SNE Coordinates"):
                        st.dataframe(tsne_df)

            # ===================== UMAP Tab =====================
            with tab_umap:
                if n_samples < 3:
                    st.warning("UMAP requires at least 3 samples.")
                else:
                    with st.expander("UMAP Parameters", expanded=True):
                        uc1, uc2, uc3 = st.columns(3)
                        with uc1:
                            max_nn = min(50, n_samples - 1)
                            def_nn = min(15, max_nn)
                            umap_nn = st.slider("n_neighbors", 2, max_nn, def_nn, key="umap_nn",
                                                help="Larger = more global structure.")
                        with uc2:
                            umap_md = st.slider("min_dist", 0.0, 1.0, 0.1, step=0.05, key="umap_md",
                                                help="Smaller = tighter clusters.")
                        with uc3:
                            umap_metric = st.selectbox("Metric", ["euclidean", "cosine", "manhattan"], key="umap_met")
                        umap_comp = st.slider("Number of UMAP components", 2, 3, 2, key="umap_comp")

                    with st.spinner("Running UMAP..."):
                        reducer = umap.UMAP(n_components=umap_comp, n_neighbors=umap_nn,
                                            min_dist=umap_md, metric=umap_metric, random_state=42)
                        umap_result = reducer.fit_transform(scaled)

                    umap_df = pd.DataFrame(umap_result, columns=[f'UMAP{i+1}' for i in range(umap_comp)])
                    umap_df[color_col] = color_values
                    umap_df['Structure'] = structure_names

                    cu1, cu2 = st.columns(2)
                    with cu1:
                        fig_umap_2d = px.scatter(umap_df, x='UMAP1', y='UMAP2',
                                                 color=color_col, hover_name='Structure', opacity=0.8,
                                                 title='UMAP - 2D Projection')
                        fig_umap_2d.update_traces(marker=dict(size=10))
                        fig_umap_2d.update_layout(template='simple_white')
                        st.plotly_chart(fig_umap_2d, use_container_width=True)
                    with cu2:
                        if umap_comp >= 3:
                            fig_umap_3d = px.scatter_3d(umap_df, x='UMAP1', y='UMAP2', z='UMAP3',
                                                        color=color_col, hover_name='Structure', opacity=0.8,
                                                        title='UMAP - 3D Projection')
                            fig_umap_3d.update_traces(marker=dict(size=5))
                            fig_umap_3d.update_layout(template='simple_white')
                            st.plotly_chart(fig_umap_3d, use_container_width=True)

                    with st.expander("UMAP Coordinates"):
                        st.dataframe(umap_df)

            # ===================== K-means Tab =====================
            with tab_km:
                km_c1, km_c2 = st.columns(2)
                with km_c1:
                    k_val = st.slider("Number of clusters (k)", 2, min(10, n_samples - 1), 3, key="km_k")
                with km_c2:
                    km_ninit = st.slider("Number of initializations", 1, 20, 10, key="km_ninit")

                kmeans = KMeans(n_clusters=k_val, n_init=km_ninit, random_state=42)
                km_labels = kmeans.fit_predict(scaled)

                st.subheader("Clustering Quality Metrics")
                show_cluster_metrics(scaled, km_labels)
                st.caption(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

                # Elbow plot
                st.subheader("Elbow Plot")
                max_k_elbow = min(10, n_samples - 1)
                if max_k_elbow >= 2:
                    inertias = []
                    sil_scores = []
                    k_range = range(2, max_k_elbow + 1)
                    for k_test in k_range:
                        km_test = KMeans(n_clusters=k_test, n_init=km_ninit, random_state=42)
                        km_test.fit(scaled)
                        inertias.append(km_test.inertia_)
                        sil_scores.append(silhouette_score(scaled, km_test.labels_))

                    ec1, ec2 = st.columns(2)
                    with ec1:
                        elbow_df = pd.DataFrame({'k': list(k_range), 'Inertia': inertias})
                        fig_elbow = px.line(elbow_df, x='k', y='Inertia',
                                            title='Elbow Plot (Inertia vs. k)', markers=True)
                        fig_elbow.update_layout(template='simple_white',
                                                xaxis_title='Number of Clusters (k)')
                        st.plotly_chart(fig_elbow, use_container_width=True)
                    with ec2:
                        sil_df = pd.DataFrame({'k': list(k_range), 'Silhouette Score': sil_scores})
                        fig_sil = px.line(sil_df, x='k', y='Silhouette Score',
                                          title='Silhouette Score vs. k', markers=True)
                        fig_sil.update_layout(template='simple_white',
                                              xaxis_title='Number of Clusters (k)')
                        st.plotly_chart(fig_sil, use_container_width=True)

                st.subheader("Cluster Assignments on PCA Projection")
                pca_coords, pca_nc = get_pca_projection(scaled)
                if pca_coords is not None:
                    show_cluster_scatter(km_labels, f"K-means (k={k_val})", pca_coords, pca_nc)

            # ===================== Hierarchical Tab =====================
            with tab_hier:
                hc1, hc2, hc3 = st.columns(3)
                with hc1:
                    hier_linkage = st.selectbox("Linkage method",
                                                ["ward", "complete", "average", "single"],
                                                key="hier_link")
                with hc2:
                    hier_k = st.slider("Number of clusters", 2, min(10, n_samples - 1), 3, key="hier_k")
                with hc3:
                    if hier_linkage == "ward":
                        hier_metric = "euclidean"
                        st.info("Ward linkage requires Euclidean distance.")
                    else:
                        hier_metric = st.selectbox("Distance metric",
                                                   ["euclidean", "cosine", "manhattan"],
                                                   key="hier_met")

                # Compute linkage for dendrogram
                if hier_linkage == "ward":
                    Z = linkage(scaled, method='ward')
                else:
                    from scipy.spatial.distance import pdist
                    dist_matrix = pdist(scaled, metric=hier_metric)
                    Z = linkage(dist_matrix, method=hier_linkage)

                # Dendrogram
                st.subheader("Dendrogram")
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig_dend, ax = plt.subplots(figsize=(12, 5))
                # Compute color threshold from linkage distances
                if hier_k < len(Z) + 1:
                    sorted_distances = sorted(Z[:, 2], reverse=True)
                    color_threshold = sorted_distances[hier_k - 1] if hier_k - 1 < len(sorted_distances) else 0
                else:
                    color_threshold = 0
                dn = dendrogram(Z, labels=structure_names, ax=ax,
                                color_threshold=color_threshold,
                                leaf_rotation=90, leaf_font_size=8)
                ax.set_title(f"Hierarchical Clustering Dendrogram ({hier_linkage} linkage)")
                ax.set_ylabel("Distance")
                fig_dend.tight_layout()
                st.pyplot(fig_dend)
                plt.close(fig_dend)

                # Agglomerative clustering
                if hier_linkage == "ward":
                    agg = AgglomerativeClustering(n_clusters=hier_k, linkage=hier_linkage)
                else:
                    agg = AgglomerativeClustering(n_clusters=hier_k, linkage=hier_linkage,
                                                 metric=hier_metric)
                hier_labels = agg.fit_predict(scaled)

                st.subheader("Clustering Quality Metrics")
                show_cluster_metrics(scaled, hier_labels)

                st.subheader("Cluster Assignments on PCA Projection")
                pca_coords, pca_nc = get_pca_projection(scaled)
                if pca_coords is not None:
                    show_cluster_scatter(hier_labels, f"Hierarchical ({hier_linkage})", pca_coords, pca_nc)

            # ===================== DBSCAN Tab =====================
            with tab_dbscan:
                # k-distance plot to help estimate eps
                st.subheader("k-Distance Plot (for eps estimation)")
                db_ms = st.slider("min_samples", 2, min(20, n_samples - 1), min(5, n_samples - 1), key="db_ms")

                nn = NearestNeighbors(n_neighbors=db_ms)
                nn.fit(scaled)
                distances, _ = nn.kneighbors(scaled)
                k_distances = np.sort(distances[:, -1])[::-1]

                kdist_df = pd.DataFrame({'Point (sorted)': range(len(k_distances)),
                                         f'{db_ms}-th NN Distance': k_distances})
                fig_kdist = px.line(kdist_df, x='Point (sorted)', y=f'{db_ms}-th NN Distance',
                                   title=f'{db_ms}-Distance Plot (look for elbow)', markers=True)
                fig_kdist.update_layout(template='simple_white')
                st.plotly_chart(fig_kdist, use_container_width=True)

                # Suggest eps from the knee point
                default_eps = float(np.median(k_distances))

                dc1, dc2 = st.columns(2)
                with dc1:
                    db_eps = st.slider("eps (neighborhood radius)", 0.1, float(k_distances[0]) * 2 + 0.1,
                                       min(default_eps, float(k_distances[0]) * 2),
                                       step=0.1, key="db_eps")
                with dc2:
                    db_metric = st.selectbox("Distance metric", ["euclidean", "cosine", "manhattan"], key="db_met")

                dbscan = DBSCAN(eps=db_eps, min_samples=db_ms, metric=db_metric)
                db_labels = dbscan.fit_predict(scaled)

                n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                n_noise = list(db_labels).count(-1)
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.metric("Clusters found", n_clusters_db)
                with ic2:
                    st.metric("Noise points", n_noise)

                if n_clusters_db >= 2:
                    st.subheader("Clustering Quality Metrics")
                    show_cluster_metrics(scaled, db_labels)
                elif n_clusters_db < 2:
                    st.warning("DBSCAN found fewer than 2 clusters. Try adjusting eps or min_samples.")

                st.subheader("Cluster Assignments on PCA Projection")
                pca_coords, pca_nc = get_pca_projection(scaled)
                if pca_coords is not None:
                    show_cluster_scatter(db_labels, "DBSCAN", pca_coords, pca_nc)

            # ===================== HDBSCAN Tab =====================
            with tab_hdbscan:
                hdb_c1, hdb_c2, hdb_c3 = st.columns(3)
                with hdb_c1:
                    hdb_min_cluster = st.slider("min_cluster_size", 2, max(2, n_samples // 2),
                                                min(5, max(2, n_samples // 2)), key="hdb_mcs")
                with hdb_c2:
                    hdb_min_samples = st.slider("min_samples", 1, min(20, n_samples - 1),
                                                min(5, n_samples - 1), key="hdb_ms")
                with hdb_c3:
                    hdb_metric = st.selectbox("Distance metric", ["euclidean", "manhattan"], key="hdb_met")

                hdbscan = HDBSCAN(min_cluster_size=hdb_min_cluster, min_samples=hdb_min_samples,
                                  metric=hdb_metric)
                hdb_labels = hdbscan.fit_predict(scaled)

                n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
                n_noise_hdb = list(hdb_labels).count(-1)
                hi1, hi2 = st.columns(2)
                with hi1:
                    st.metric("Clusters found", n_clusters_hdb)
                with hi2:
                    st.metric("Noise points", n_noise_hdb)

                if n_clusters_hdb >= 2:
                    st.subheader("Clustering Quality Metrics")
                    show_cluster_metrics(scaled, hdb_labels)
                elif n_clusters_hdb < 2:
                    st.warning("HDBSCAN found fewer than 2 clusters. Try adjusting min_cluster_size or min_samples.")

                st.subheader("Cluster Assignments on PCA Projection")
                pca_coords, pca_nc = get_pca_projection(scaled)
                if pca_coords is not None:
                    show_cluster_scatter(hdb_labels, "HDBSCAN", pca_coords, pca_nc)
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

        st.subheader("Dimensionality Reduction")
        st.markdown("""
**PCA** (Principal Component Analysis) projects descriptors onto axes that explain
the most variance. It is linear and fast.

**t-SNE** (t-distributed Stochastic Neighbor Embedding) preserves local
neighbourhood structure. Good for revealing clusters but not distance-preserving.

**UMAP** (Uniform Manifold Approximation and Projection) preserves both local and
global structure. Faster than t-SNE and better at preserving distances.

All methods standardise descriptors first so that different physical units
contribute equally.
        """)

        st.subheader("Clustering Methods")
        st.markdown("""
**K-means** partitions structures into k clusters by minimising within-cluster
variance. Default k=3 with elbow plot for optimal k selection.

**Hierarchical** builds a tree (dendrogram) of structure relationships using
agglomerative linkage. Natural for comparing protein conformations.

**DBSCAN** finds density-based clusters and identifies outlier/noise structures.
Does not require specifying the number of clusters.

**HDBSCAN** extends DBSCAN to handle clusters of varying density. Robust for
heterogeneous datasets.

Quality is assessed via Silhouette Score, Calinski-Harabasz Index, and
Davies-Bouldin Index.
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

**5. Dimensionality reduction & clustering**
- *Single-species*: all numeric descriptors are used.  DSSP endpoint columns are
  delta-transformed (deviation from the reference helix boundaries) so that the
  absolute residue numbers do not dominate.
- *Cross-species*: only species-invariant descriptors are used (SSE content,
  charge-clamp distances/angles, COM-clamp distances) so that species with
  different helix counts can be compared on the same axes.
- Constant columns (zero variance) are removed; remaining columns are standardised.
- Seven analysis tabs: **PCA**, **t-SNE**, **UMAP**, **K-means**, **Hierarchical**,
  **DBSCAN**, **HDBSCAN**.
- Clustering methods include quality metrics (Silhouette, Calinski-Harabasz,
  Davies-Bouldin) and diagnostic plots (elbow plot, dendrogram, k-distance plot).
        """)

    with st.expander("Species Configuration"):
        st.markdown("""
**Pre-configured VDR settings for three species:**

| Species | Helices | Charge clamp residues |
|---|---|---|
| **Human** (*Homo sapiens*) | 14 | 246, 264, 420 |
| **Rat** (*Rattus norvegicus*) | 13 | 242, 260, 416 |
| **Zebrafish** (*Danio rerio*) | 13 | 274, 292, 446 |
| **Sea Lamprey** | 12 | 101, 119, 275 |

Upload structures to the corresponding species uploader in the sidebar.
When multiple species are uploaded, all analyses use species-invariant descriptors
(SSE content, charge clamps) and colour points by species.
        """)
