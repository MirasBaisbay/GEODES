#!/usr/bin/env python3
"""
Debug script to test DSSP installation and PDB file compatibility
Run this inside your container to diagnose the issue
"""

import subprocess
import sys
from pathlib import Path
from Bio.PDB import PDBParser, DSSP

def test_mkdssp_version():
    """Check mkdssp version"""
    print("=" * 60)
    print("1. Testing mkdssp version:")
    print("=" * 60)
    try:
        result = subprocess.run(
            ['mkdssp', '--version'], 
            capture_output=True, 
            text=True
        )
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")

def test_dssp_command():
    """Check if dssp command exists"""
    print("\n" + "=" * 60)
    print("2. Testing dssp command:")
    print("=" * 60)
    try:
        result = subprocess.run(
            ['which', 'dssp'], 
            capture_output=True, 
            text=True
        )
        print(f"dssp location: {result.stdout.strip()}")
        
        result = subprocess.run(
            ['dssp', '--version'], 
            capture_output=True, 
            text=True
        )
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")

def test_pdb_file(pdb_path):
    """Test DSSP on a specific PDB file"""
    print("\n" + "=" * 60)
    print(f"3. Testing DSSP on: {pdb_path}")
    print("=" * 60)
    
    if not Path(pdb_path).exists():
        print(f"Error: File {pdb_path} not found!")
        return
    
    # First, check PDB file format
    print("\n--- PDB File Preview (first 10 lines) ---")
    with open(pdb_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"Line {i+1}: {line.rstrip()}")
    
    # Test mkdssp directly
    print("\n--- Testing mkdssp directly ---")
    try:
        result = subprocess.run(
            ['mkdssp', '--output-format=dssp', pdb_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ mkdssp succeeded!")
            print(f"Output length: {len(result.stdout)} characters")
        else:
            print(f"✗ mkdssp failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout[:500]}")
            print(f"STDERR: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("✗ mkdssp timed out!")
    except Exception as e:
        print(f"✗ Error running mkdssp: {e}")
    
    # Test Biopython DSSP
    print("\n--- Testing Biopython DSSP ---")
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", pdb_path)
        model = structure[0]
        
        # Try with 'dssp' command
        print("Trying with dssp='dssp'...")
        dssp = DSSP(model, pdb_path, dssp='dssp')
        print(f"✓ Success! Found {len(dssp)} residues")
        
    except Exception as e:
        print(f"✗ Failed with dssp='dssp': {e}")
        
        # Try with 'mkdssp' command
        try:
            print("\nTrying with dssp='mkdssp'...")
            dssp = DSSP(model, pdb_path, dssp='mkdssp')
            print(f"✓ Success! Found {len(dssp)} residues")
        except Exception as e2:
            print(f"✗ Failed with dssp='mkdssp': {e2}")

def main():
    test_mkdssp_version()
    test_dssp_command()
    
    # Test with a PDB file if provided
    if len(sys.argv) > 1:
        pdb_path = sys.argv[1]
        test_pdb_file(pdb_path)
    else:
        print("\n" + "=" * 60)
        print("To test a specific PDB file, run:")
        print(f"  python {sys.argv[0]} /path/to/your/file.pdb")
        print("=" * 60)

if __name__ == "__main__":
    main()