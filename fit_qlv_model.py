"""Forwarding shim to simulations/material-calibration/1d-qlv/fit_qlv_model.py."""
import sys
import importlib.util
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "simulations" / "material-calibration" / "1d-qlv" / "fit_qlv_model.py"
spec = importlib.util.spec_from_file_location("_real_fit_qlv_model", _TARGET)
_mod = importlib.util.module_from_spec(spec)
sys.modules["_real_fit_qlv_model"] = _mod
spec.loader.exec_module(_mod)

# Export all public symbols
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
