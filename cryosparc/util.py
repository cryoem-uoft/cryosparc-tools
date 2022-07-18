from contextlib import contextmanager
from pathlib import PurePath
from typing import BinaryIO, Union
import numpy as n
from .column import Column


def isndarraylike(val):
    return isinstance(val, (n.ndarray, Column))


@contextmanager
def ioopen(file: Union[str, PurePath, BinaryIO], mode = "r"):
    if isinstance(file, (str, PurePath)):
        with open(file, mode) as f:
            yield f
    else:
        yield file
