"""Bytes, as an operator reads them.

Two places report memory — the startup bound a capture measured and the resident memory a running fleet is holding — and a reader compares them directly: *at most 2.1 GiB per worker* against *4.3 GiB across 2 processes* is a sentence only if both were rounded the same way. One function, so they cannot come to disagree about what 1.05 GiB is called.

The other direction is `parse_size`, which reads a size an operator *wrote* — a disk floor in `_steward.yaml` — and like `_util.duration` it refuses a bare number, because `disk_low: 5` could be five bytes or five gibibytes and the misreading is silent in both directions.
"""

import re

_SIZE = re.compile(r"^(\d+)\s*([KMGT])i?B?$", re.IGNORECASE)

_SCALE = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}

_UNITS = ("KiB", "MiB", "GiB", "TiB")


class SizeError(ValueError):
    """A size could not be read. Carries the text that arrived, since the whole point is to tell an author what they typed. A `ValueError`, so a bad size in `_steward.yaml` surfaces as a field error naming the value, exactly as a bad duration does."""


def parse_size(text: str) -> int:
    """Read a byte size written with a unit.

    Args:
        text: A count and a binary unit, e.g. `5GiB`, `500MiB`, `2G`. The `i` and the trailing `B` are optional — `2G`, `2GB`, and `2GiB` all mean two gibibytes — and whitespace between the count and the unit is allowed. A bare number is not a size, because the unit is where the meaning is. Every unit is binary (1024-based), to match the way `format_bytes` renders one back out.

    Returns:
        Bytes.

    Raises:
        SizeError: Not a size, or zero.
    """
    match = _SIZE.match(text.strip())
    if match is None:
        raise SizeError(
            f"'{text}' is not a size — write a count and a unit, "
            f"one of {', '.join(_UNITS)} (e.g. '5GiB')"
        )
    size = int(match.group(1)) * _SCALE[match.group(2).lower()]
    if size == 0:
        raise SizeError(f"'{text}' is zero, which is not a size")
    return size


def format_bytes(value: int) -> str:
    """Render a byte count at the largest unit that still says something.

    Args:
        value: Bytes.

    Returns:
        GiB to one decimal, dropping to whole MiB below a tenth of a gibibyte — where `0.0 GiB` would read as *nothing* for something that is really 60 MiB.
    """
    gib = value / (1024**3)
    return f"{gib:.1f} GiB" if gib >= 0.1 else f"{value / (1024**2):.0f} MiB"
