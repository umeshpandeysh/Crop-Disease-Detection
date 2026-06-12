"""pytest shared fixtures and path configuration for Crop-Disease-Detection."""
from __future__ import annotations

import sys
import pathlib

# Ensure project root is in sys.path so that all src.* imports work
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
