#!/usr/bin/env python3
"""Delegate to Mityane's verifier; never install another component SDK."""
import os
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[4]
arguments = sys.argv[1:] or ['--check']
os.execv(sys.executable, [sys.executable, '-B', str(project / 'code/scripts/install-libtorch.py'), *arguments])