"""Run directly from the checkout, with no editable installation required."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from magnetooptics.runner import main

if __name__ == '__main__':
    main()
