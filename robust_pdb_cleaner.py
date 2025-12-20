"""
Robust PDB file cleaner for mkdssp compatibility.
Use this to preprocess PDB files before DSSP analysis.
"""

from Bio.PDB import PDBParser, PDBIO, Select
from pathlib import Path
import warnings

class CleanSelect(Select):
    """Select only ATOM records, skip HETATM"""
    def accept_residue(self, residue):
        # Accept standard amino acids only
        return residue.id[0] == ' '

def clean_pdb_for_mkdssp(input_path, output_path, keep_hetatm=False):
    """
    Clean PDB file using Biopython to ensure mkdssp compatibility.
    
    This approach:
    1. Parses the PDB structure
    2. Rewrites it in clean format
    3. Removes problematic records
    
    Args:
        input_path: Path to input PDB file
        output_path: Path to output cleaned PDB file
        keep_hetatm: Whether to keep HETATM records (default: False)
    
    Returns:
        Path to cleaned PDB file
    """
    # Suppress warnings during parsing
    warnings.filterwarnings('ignore')
    
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_path)
    
    # Write clean PDB
    io = PDBIO()
    io.set_structure(structure)
    
    # Use Select class to filter if needed
    if keep_hetatm:
        io.save(str(output_path))
    else:
        io.save(str(output_path), CleanSelect())
    
    # Now manually add a clean HEADER (mkdssp requires it)
    with open(output_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out any remaining problematic lines and ensure proper format
    cleaned_lines = []
    has_header = False
    
    for line in lines:
        if line.startswith('HEADER'):
            has_header = True
            cleaned_lines.append(line)
        elif line.startswith(('ATOM', 'HETATM', 'TER', 'END', 'MODEL', 'ENDMDL')):
            # Keep essential structure records
            cleaned_lines.append(line)
        elif line.startswith(('CRYST1', 'ORIGX', 'SCALE', 'MTRIX')):
            # Keep crystal/matrix records
            cleaned_lines.append(line)
        # Skip everything else (TITLE, REMARK, TURN, HELIX, SHEET, etc.)
    
    # Add minimal HEADER if missing
    if not has_header:
        header = 'HEADER    PROTEIN                                 01-JAN-00   XXXX\n'
        cleaned_lines.insert(0, header)
    
    # Ensure END record
    if not cleaned_lines or not cleaned_lines[-1].startswith('END'):
        cleaned_lines.append('END\n')
    
    # Write final cleaned version
    with open(output_path, 'w') as f:
        f.writelines(cleaned_lines)
    
    return output_path


def clean_pdb_minimal(input_path, output_path):
    """
    Ultra-minimal PDB cleaning - only HEADER + ATOM + END.
    Use this if the Biopython method fails.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    has_atoms = False
    
    # Add minimal header
    cleaned_lines.append('HEADER    PROTEIN                                 01-JAN-00   XXXX\n')
    
    # Keep only ATOM/HETATM records
    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            cleaned_lines.append(line)
            has_atoms = True
    
    if not has_atoms:
        raise ValueError("No ATOM records found in PDB file!")
    
    # Add END
    cleaned_lines.append('END\n')
    
    with open(output_path, 'w') as f:
        f.writelines(cleaned_lines)
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python robust_pdb_cleaner.py input.pdb output.pdb")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Cleaning {input_file}...")
    
    try:
        clean_pdb_for_mkdssp(input_file, output_file)
        print(f"✓ Cleaned PDB saved to {output_file}")
        
        # Test with mkdssp
        print("\nTesting with mkdssp...")
        import subprocess
        result = subprocess.run(
            ['mkdssp', '--output-format=dssp', output_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ mkdssp succeeded!")
        else:
            print(f"✗ mkdssp failed:")
            print(result.stderr)
            
            # Try ultra-minimal version
            print("\nTrying ultra-minimal cleaning...")
            clean_pdb_minimal(input_file, output_file)
            
            result = subprocess.run(
                ['mkdssp', '--output-format=dssp', output_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print("✓ mkdssp succeeded with minimal cleaning!")
            else:
                print(f"✗ mkdssp still fails:")
                print(result.stderr)
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()