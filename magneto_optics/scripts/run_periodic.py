"""Run periodic metasurfaces directly from a source checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from magnetooptics.periodic_runner import main

if __name__ == '__main__':
    main()
