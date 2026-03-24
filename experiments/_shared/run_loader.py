from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


@lru_cache(maxsize=None)
def load_run_module(key: str, path: str | Path):
    module_path = Path(path).resolve()
    module_name = f"shape_budget_{key}"

    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment module from {module_path}")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_symbols(key: str, path: str | Path, *names: str):
    module = load_run_module(key, path)
    return tuple(getattr(module, name) for name in names)
