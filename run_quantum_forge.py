#!/usr/bin/env python3
"""
Quantum Forge Engine - Runner Script
-----------------------------------
This script launches the Quantum Forge Engine application.
Developed by Jasma Team.
"""

import os
import sys
import subprocess

def main():
    """Run the Quantum Forge Engine application."""
    print("Starting Quantum Forge Engine...")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run the application using streamlit
    try:
        subprocess.run(["streamlit", "run", "quantum_forge_engine.py"], check=True)
    except KeyboardInterrupt:
        print("\nQuantum Forge Engine stopped.")
    except Exception as e:
        print(f"Error running Quantum Forge Engine: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 