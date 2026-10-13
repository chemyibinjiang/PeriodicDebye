#!/usr/bin/env python3
"""Run a schema-versioned finite/periodic/PXRD scattering configuration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mofdebye.runner import main

if __name__=='__main__':
    main()
