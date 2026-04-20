"""
Root conftest.py — assists the import of test fixtures across the tests/ directory, and sets up sys.path for bare imports in test files.
"""
import sys
import os

ROOT = os.path.dirname(__file__)

sys.path.insert(0, ROOT)

for subdir in ("src/data_collection", "src/training", "src/app"):
    path = os.path.join(ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)