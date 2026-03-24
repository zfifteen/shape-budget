from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_TARGET = Path(r"/Users/velocityworks/IdeaProjects/shape-budget/experiments/pose-anisotropy-diagnostics/oracle-alignment-ceiling/run.py")
_SPEC = spec_from_file_location(__name__ + "__impl", _TARGET)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
for _name, _value in vars(_MODULE).items():
    if _name.startswith("__") and _name not in {"__doc__"}:
        continue
    globals()[_name] = _value
