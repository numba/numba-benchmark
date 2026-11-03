"""Dynamic benchmark case registry.

A case lives in a module, takes one ``numpy.random.RandomState`` and returns
``(numpy_fn, numba_fn, args)``. ``@case`` registers it; discovery imports the
modules in this package and any module listed in ``NUMBA_BENCH_CASES``.

Stdlib only: importing this package must not require numpy or numba.
"""

import hashlib
import importlib
import importlib.util
import inspect
import os
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ._version import matches, parse_spec

_ENV_VAR = "NUMBA_BENCH_CASES"
_PACKAGE_DIR = Path(__file__).resolve().parent
_FIRST_ORDER = 10


@dataclass(frozen=True)
class Case:
    """One benchmark case as registered."""

    id: str
    build: object
    requires_numba: str | None
    order: int | None


_CASES = {}
_MODULES = {}
_AUTO_ORDER = _FIRST_ORDER
_LOADED = False


def _next_order():
    global _AUTO_ORDER
    order = _AUTO_ORDER
    _AUTO_ORDER += _FIRST_ORDER
    return order


def _validate_build(case_id, build):
    if not callable(build):
        raise TypeError(f"case {case_id!r} build must be callable")
    try:
        signature = inspect.signature(build)
    except (TypeError, ValueError):
        return
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 1:
        raise TypeError(
            f"case {case_id!r} build must take exactly one positional "
            f"parameter (the RandomState), got {len(positional)}"
        )


def register_case(case, module=None):
    """Validate and register *case*; return the case with a resolved order."""
    if not isinstance(case, Case):
        raise TypeError(f"register_case expects a Case, got {type(case).__name__}")
    if (
        not isinstance(case.id, str)
        or not case.id
        or any(character.isspace() for character in case.id)
        or "|" in case.id
    ):
        raise ValueError(f"invalid case id {case.id!r}")

    module = module or getattr(case.build, "__module__", "<unknown>")
    if case.id in _CASES:
        raise ValueError(
            f"duplicate case id {case.id!r} registered by "
            f"{_MODULES[case.id]!r} and {module!r}"
        )

    _validate_build(case.id, case.build)
    if case.requires_numba is not None:
        parse_spec(case.requires_numba)
    if case.order is not None and (
        not isinstance(case.order, int) or isinstance(case.order, bool)
    ):
        raise TypeError(f"case {case.id!r} order must be an int or None")

    resolved = Case(
        id=case.id,
        build=case.build,
        requires_numba=case.requires_numba,
        order=case.order if case.order is not None else _next_order(),
    )
    _CASES[case.id] = resolved
    _MODULES[case.id] = module
    return resolved


def case(case_id, *, requires_numba=None, order=None):
    """Register a case build function and return it unchanged."""

    def decorator(build):
        register_case(Case(case_id, build, requires_numba, order))
        return build

    return decorator


def build_case(case, rng):
    """Call ``case.build(rng)`` and validate the returned triple."""
    result = case.build(rng)
    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError(
            f"case {case.id!r} must return a 3-tuple "
            f"(numpy_fn, numba_fn, args), got {result!r}"
        )
    numpy_fn, numba_fn, args = result
    if not callable(numpy_fn) or not callable(numba_fn):
        raise TypeError(
            f"case {case.id!r} must return two callables, got "
            f"{type(numpy_fn).__name__} and {type(numba_fn).__name__}"
        )
    if not isinstance(args, tuple):
        raise TypeError(
            f"case {case.id!r} args must be a tuple, got {type(args).__name__}"
        )
    return result


def _load_file(path):
    # The module name is the path itself: functions defined by an out-of-tree
    # module then report the file they came from, so duplicate-id errors stay
    # actionable. Out-of-tree modules have no package context (absolute imports
    # or inline kernels only).
    spec = importlib.util.spec_from_file_location(str(path), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark case module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _load():
    global _AUTO_ORDER, _LOADED
    if _LOADED:
        return
    _AUTO_ORDER = _FIRST_ORDER

    modules = sorted(
        pkgutil.iter_modules([str(_PACKAGE_DIR)]), key=lambda info: info.name
    )
    for info in modules:
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")

    for entry in filter(None, os.environ.get(_ENV_VAR, "").split(os.pathsep)):
        path = Path(entry)
        if path.is_dir():
            for file in sorted(path.glob("*.py")):
                if not file.name.startswith("_"):
                    _load_file(file)
        elif path.is_file():
            _load_file(path)
        else:
            raise FileNotFoundError(
                f"{_ENV_VAR} entry does not exist as a file or directory: {entry}"
            )

    _LOADED = True


def iter_cases():
    """Return every registered case sorted by ``(order, id)``."""
    _load()
    return sorted(_CASES.values(), key=lambda case: (case.order, case.id))


def case_ids():
    """Return the sorted ids of every registered case."""
    return [case.id for case in iter_cases()]


def select_cases(numba_version):
    """Return ``(runnable, skipped)`` for *numba_version*, both sorted."""
    runnable = []
    skipped = []
    for case in iter_cases():
        if case.requires_numba is None or matches(case.requires_numba, numba_version):
            runnable.append(case)
        else:
            skipped.append(
                {
                    "id": case.id,
                    "order": case.order,
                    "requires_numba": case.requires_numba,
                    "reason": f"requires numba {case.requires_numba}; "
                    f"installed {numba_version}",
                }
            )
    return runnable, skipped


def seed_for(case_id):
    """Return a stable per-case RandomState seed derived from the id."""
    digest = hashlib.blake2b(case_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def _reload():
    """Discard discovery state; tests only."""
    global _AUTO_ORDER, _LOADED
    for name in [
        name
        for name in list(sys.modules)
        if name.startswith(f"{__name__}.") and name != __name__
    ]:
        del sys.modules[name]
    _CASES.clear()
    _MODULES.clear()
    _AUTO_ORDER = _FIRST_ORDER
    _LOADED = False
