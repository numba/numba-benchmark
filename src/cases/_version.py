"""Numba version parsing and specifier matching.

Stdlib only: the registry and the tests must import without numpy or numba.
"""

import re

_VERSION = re.compile(r"^\s*(\d+(?:\.\d+)*)")
_CLAUSE = re.compile(r"^(==|!=|<=|>=|<|>)\s*(\d+(?:\.\d+)*)$")


def parse_version(version):
    """Return the numeric prefix of *version* as an integer tuple.

    ``0.62.0dev0``, ``0.62.0rc1``, ``0.62.0.post1`` and ``0.62.0+local`` all
    parse as ``(0, 62, 0)``. A version without a numeric prefix raises.
    """
    if not isinstance(version, str):
        raise TypeError(f"numba version must be a string, got {version!r}")
    match = _VERSION.match(version)
    if match is None:
        raise ValueError(f"cannot parse a numeric prefix from numba {version!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def parse_spec(spec):
    """Parse ``">=0.55,<0.60"`` into ``[(op, version_tuple), ...]``."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(
            f"numba version specifier must be a non-empty string, got {spec!r}"
        )
    clauses = []
    for part in spec.split(","):
        match = _CLAUSE.match(part.strip())
        if match is None:
            raise ValueError(
                f"malformed numba version specifier {spec!r}: {part.strip()!r}"
            )
        version = tuple(int(piece) for piece in match.group(2).split("."))
        clauses.append((match.group(1), version))
    return clauses


def compare(left, right):
    """Compare two integer tuples, zero-padding the shorter one."""
    width = max(len(left), len(right))
    left = left + (0,) * (width - len(left))
    right = right + (0,) * (width - len(right))
    return (left > right) - (left < right)


def _holds(operator, installed, wanted):
    result = compare(installed, wanted)
    if operator == "==":
        return result == 0
    if operator == "!=":
        return result != 0
    if operator == "<=":
        return result <= 0
    if operator == ">=":
        return result >= 0
    if operator == "<":
        return result < 0
    return result > 0


def matches(spec, version):
    """Return whether *version* satisfies every clause in *spec*."""
    installed = parse_version(version)
    return all(_holds(op, installed, wanted) for op, wanted in parse_spec(spec))
