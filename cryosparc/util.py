from contextlib import contextmanager
from pathlib import PurePath
from typing import IO, Union
from typing_extensions import Literal
import numpy as n
from .column import Column

OpenBinaryMode = Literal["rb", "wb", "xb", "ab", "r+b", "w+b", "x+b", "a+b"]


def isndarraylike(val):
    return isinstance(val, (n.ndarray, Column))


@contextmanager
def bopen(file: Union[str, PurePath, IO[bytes]], mode: OpenBinaryMode = "rb"):
    if isinstance(file, (str, PurePath)):
        with open(file, mode) as f:
            yield f
    else:
        yield file
