from contextlib import contextmanager
from pathlib import PurePath
from typing import IO, Union
from typing_extensions import Literal
import numpy as n

OpenBinaryMode = Literal["rb", "wb", "xb", "ab", "r+b", "w+b", "x+b", "a+b"]


def u32bytesle(x: int) -> bytes:
    """
    Get the uint32 bytes of for integer x in little endian
    """
    return n.array(x, dtype="<u4").tobytes()


def u32intle(buffer: bytes) -> int:
    """
    Get int from buffer representing a uint32 integer in little endian
    """
    return n.frombuffer(buffer, dtype="<u4")[0]


@contextmanager
def bopen(file: Union[str, PurePath, IO[bytes]], mode: OpenBinaryMode = "rb"):
    """
    open alias specific to finary files. If the given file is an open IO handle,
    will just yields directly.
    """
    if isinstance(file, (str, PurePath)):
        with open(file, mode) as f:
            yield f
    else:
        yield file
