import streamlit as st
import ast
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

st.set_page_config(page_title="GEODES: Protein Geometry", layout="wide", page_icon="🧬")

with st.sidebar:
    st.header("1. Input Data")
    uploaded_files = st.file_uploader("Upload PDB Files", type=['pdb'], accept_multiple_files=True)
    
    st.divider()
    
    st.header("2. Species Selection")
    species = st.radio(
        "Select VDR Species:",
        options=["Human", "Rat", "Zebrafish"],
        index=0,
        help="Choose the species to automatically configure helix boundaries and charge clamps"
    )
    
    # Get configuration for selected species
    selected_config = SPECIES_CONFIG[species]
    
    st.divider()
    
    st.header("3. Protein Definition")
    st.info(f"Settings configured for **{species} VDR**. Adjust if needed.")
    
    # Convert to string for display
    default_href = str(selected_config["href"])
    default_clamps = str(selected_config["clamps"])
    
    st.markdown("**Helix Boundaries (`href`)**")
    href_input = st.text_area("Enter list of [start, end] residues:", value=default_href, height=150)

    st.markdown("**Charge Clamps**")
    enable_clamps = st.checkbox("Calculate Charge Clamp Descriptors?", value=True)
    clamps_input = st.text_input("Residue Indices:", value=default_clamps, disabled=not enable_clamps)

# Main UI Logic
if uploaded_files:
    
    # --- Visualization Section ---
    st.subheader("👁️ 3D Structure Preview")
    
    # Display current species selection
    st.caption(f"**Currently configured for: {species} VDR**")
    
    # Select specific file to view from the list of uploads
    file_names = [f.name for f in uploaded_files]
    selected_file_name = st.selectbox("Select file to visualize:", file_names)
    
    # Find the file object matching the selection
    selected_file = next(f for f in uploaded_files if f.name == selected_file_name)

    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            from stmol import showmol
            import py3Dmol
            
            # Read content
            pdb_content = selected_file.getvalue().decode("utf-8")
            
            # Setup Viewer
            # height=500 makes it larger
            view = py3Dmol.view(width=800, height=500)
            view.addModel(pdb_content, 'pdb')
            
            # Style: Spectrum colored cartoon
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            
            # Optional: Add sticks for ligands (HETATM) if present, to make them visible
            view.addStyle({'hetflag': True}, {'stick': {}})
            
            # Center the camera on the protein
            view.zoomTo()
            
            # Render
            showmol(view, height=500)
            
        except Exception as e:
            st.warning(f"Visualization unavailable: {e}")
            
    with col2:
        st.info(f"**Current File:** {selected_file_name}")
        st.caption("Use the dropdown above to switch between uploaded files.")

    st.divider()

    # --- Analysis Section ---
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Processing PDB files for {species} VDR analysis..."):
            try:
                # Setup Temp Dirs
                base_path = Path("/app/data_temp") if os.path.exists("/app") else Path("data_temp")
                input_dir = base_path / "input"
                raw_dir = base_path / "raw"  # Store original uploads
                output_dir = base_path / "output"
                
                # Clean up and recreate directories
                for dir_path in [input_dir, raw_dir, output_dir]:
                    if dir_path.exists(): 
                        shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True)

                # Save and clean PDB files
                st.info(f"📁 Processing {len(uploaded_files)} PDB file(s)...")
                processed_count = 0
                
                # Progress bar
                progress_bar = st.progress(0)
                
                for i, f in enumerate(uploaded_files):
                    try:
                        # Save raw file
                        raw_path = raw_dir / f.name
                        with open(raw_path, "wb") as dest:
                            dest.write(f.getbuffer())
                        
                        # Clean for DSSP compatibility
                        cleaned_path = input_dir / f.name
                        clean_pdb_for_dssp(raw_path, cleaned_path)
                        processed_count += 1
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.error(f"✗ Failed to process {f.name}: {e}")
                        continue

                if processed_count == 0:
                    st.error("No PDB files could be processed. Please check your input files.")
                    st.stop()

                # Parse Config
                href_parsed = ast.literal_eval(href_input)
                clamps_parsed = ast.literal_eval(clamps_input) if enable_clamps else []
                run_config = {'descriptors': yaml_config.get('descriptors', [])}

                # Run GEODES
                st.info(f"🧬 Running GEODES analysis for {species} VDR...")
                calculator = DescCalculator(href_parsed, clamps_parsed, config=run_config)
                
                # The calculation step that triggers the DSSP call
                # Note: We pass the 'input' folder where the CLEANED files are
                df_result = calculator.calc_all(str(input_dir) + "/", save_to_csv=False, parallel=False)
                
                st.success(f"✅ Analysis complete for {species} VDR!")
                st.dataframe(df_result)
                
                # Provide download option
                csv = df_result.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"geodes_results_{species.lower()}.csv",
                    mime="text/csv"
                )

                # --- PCA Analysis Section ---
                st.divider()
                st.subheader("📊 PCA Analysis")

                df_numeric = df_result.drop(columns=['prot_name']).select_dtypes(include=[np.number])
                df_clean = df_numeric.dropna(axis=0, how='any')

                if len(df_clean) < 2:
                    st.warning("PCA requires at least 2 structures with complete data. Some structures may have failed calculations.")
                else:
                    # Delta transform on DSSP columns
                    dssp_cols = [c for c in df_clean.columns if c.startswith('DSSP')]
                    if dssp_cols:
                        ref_flat = np.array(href_parsed).flatten()
                        if len(ref_flat) == len(dssp_cols):
                            df_clean[dssp_cols] = df_clean[dssp_cols].apply(
                                lambda row: delta_transform(row, ref_flat), axis=1
                            )

                    # Remove constant columns (zero variance)
                    df_clean = df_clean.loc[:, df_clean.std() > 0]

                    # Scale
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(df_clean)

                    # PCA
                    max_components = min(3, len(df_clean), len(df_clean.columns))
                    n_components = st.slider("Number of PCA components", 2, max_components, max_components)

                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(scaled)

                    # Build PCA DataFrame
                    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
                    valid_indices = df_clean.index
                    pca_df['Structure'] = df_result.loc[valid_indices, 'prot_name'].values

                    # Explained variance bar chart
                    var_df = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(n_components)],
                        'Explained Variance (%)': pca.explained_variance_ratio_ * 100
                    })
                    fig_var = px.bar(var_df, x='Component', y='Explained Variance (%)',
                                     title='Explained Variance per Component',
                                     text_auto='.1f')
                    fig_var.update_layout(template='simple_white')
                    st.plotly_chart(fig_var, use_container_width=True)

                    # 2D + 3D plots side by side
                    col_pca1, col_pca2 = st.columns(2)
                    with col_pca1:
                        fig_2d = px.scatter(pca_df, x='PC1', y='PC2',
                                            color='Structure', opacity=0.8,
                                            title='PCA — 2D Projection')
                        fig_2d.update_traces(marker=dict(size=10))
                        fig_2d.update_layout(template='simple_white')
                        st.plotly_chart(fig_2d, use_container_width=True)

                    with col_pca2:
                        if n_components >= 3:
                            fig_3d = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                                                   color='Structure', opacity=0.8,
                                                   title='PCA — 3D Projection')
                            fig_3d.update_traces(marker=dict(size=5))
                            fig_3d.update_layout(template='simple_white')
                            st.plotly_chart(fig_3d, use_container_width=True)

                    # Show PCA coordinates table
                    with st.expander("PCA Coordinates"):
                        st.dataframe(pca_df)

            except Exception as e:
                st.error(f"❌ Analysis Error: {e}")
                with st.expander("Show Detailed Error"):
                    import traceback
                    st.code(traceback.format_exc())
else:
    st.info("👈 Upload PDB file(s) using the sidebar to get started")
    
    # Show example
    with st.expander("ℹ️ Species Configuration"):
        st.markdown("""
        **Pre-configured settings for VDR across three species:**
        
        - **Human VDR**: Default helix boundaries and charge clamps (246, 264, 420)
        - **Rat VDR**: Adjusted boundaries and charge clamps (242, 260, 416)
        - **Zebrafish VDR**: Species-specific boundaries and charge clamps (274, 292, 446)
        
        Select the species in the sidebar, and the helix boundaries and charge clamps will automatically update. 
        You can still manually adjust these values if needed.
        """)