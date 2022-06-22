from enum import Enum
from typing import NamedTuple


class Datatype(str, Enum):
    exposure = "exposure"
    particle = "particle"
    template = "template"
    volume = "volume"
    mask = "mask"


class Datafield(NamedTuple):
    """Definition of a prefix field within a CS file."""

    dtype: str
    """Datatype-specific string from common.py. e.g., movie_blob, ctf,
    alignments2D."""

    prefix: str
    """where to find that fild in a corresponding .cs file e.g.,
    alignments_class_1"""

    required: bool
    """whether this field must necessarily exist in acorresponding
    input/output"""
