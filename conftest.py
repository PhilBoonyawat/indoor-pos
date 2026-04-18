"""
Root conftest — makes `from src.xxx import ...` work AND keeps the source
files' existing `from preprocess import ...` style working by adding each
src subdirectory to sys.path.
"""
import sys
import os

ROOT = os.path.dirname(__file__)

# Enables `from src.xxx.yyy import ...` in tests
sys.path.insert(0, ROOT)

# Enables the source files' bare imports (e.g. train.py does `from preprocess import ...`)
for subdir in ("src/data_collection", "src/training", "src/app"):
    path = os.path.join(ROOT, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)