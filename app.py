import streamlit as st
import ast
import yaml
import os
import shutil
from pathlib import Path
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

yaml_config = load_default_config()

st.set_page_config(page_title="GEODES: Protein Geometry", layout="wide", page_icon="🧬")

with st.sidebar:
    st.header("1. Input Data")
    uploaded_files = st.file_uploader("Upload PDB Files", type=['pdb'], accept_multiple_files=True)
    
    st.divider()
    
    st.header("2. Protein Definition")
    st.info("Defaults are set for hVDR (per the GEODES paper). Adjust for other proteins.")
    
    # Defaults from usage_example.py
    default_href = "[[127,142], [149,152], [218,222], [226,246], [257,265], [268, 278], [298,302], [308,322], [328,338], [350,369], [379,396], [397,406], [411,413], [417,423]]"
    st.markdown("**Helix Boundaries (`href`)**")
    href_input = st.text_area("Enter list of [start, end] residues:", value=default_href, height=150)

    st.markdown("**Charge Clamps**")
    enable_clamps = st.checkbox("Calculate Charge Clamp Descriptors?", value=True)
    default_clamps = "[246, 264, 420]"
    clamps_input = st.text_input("Residue Indices:", value=default_clamps, disabled=not enable_clamps)

# Main UI Logic
if uploaded_files:
    
    # --- Visualization Section ---
    st.subheader("👁️ 3D Structure Preview")
    
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
        with st.spinner("Processing PDB files and running analysis..."):
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
                st.info("🧬 Running GEODES analysis...")
                calculator = DescCalculator(href_parsed, clamps_parsed, config=run_config)
                
                # The calculation step that triggers the DSSP call
                # Note: We pass the 'input' folder where the CLEANED files are
                df_result = calculator.calc_all(str(input_dir) + "/", save_to_csv=False, parallel=False)
                
                st.success("✅ Analysis complete!")
                st.dataframe(df_result)
                
                # Provide download option
                csv = df_result.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="geodes_results.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Analysis Error: {e}")
                with st.expander("Show Detailed Error"):
                    import traceback
                    st.code(traceback.format_exc())
else:
    st.info("👈 Upload PDB file(s) using the sidebar to get started")
    
    # Show example
    with st.expander("ℹ️ Example Configuration"):
        st.markdown("""
        **Default settings are configured for hVDR (Vitamin D Receptor)**
        
        - **Helix Boundaries**: Define the α-helix regions in your protein
        - **Charge Clamps**: Specific residues that form charge interactions
        
        Adjust these values based on your protein's structure and the residues you want to analyze.
        """)