from contextlib import contextmanager
from pathlib import PurePath
from typing import IO, Callable, Union
from typing_extensions import Literal
import numpy as n

OpenBinaryMode = Literal["rb", "wb", "xb", "ab", "r+b", "w+b", "x+b", "a+b"]


K = TypeVar("K")
V = TypeVar("V")


class hashcache(Dict[K, V], Generic[K, V]):
    """
    Utility class to cache string conversions and avoid excessive string
    allocation. Example usage for convering numpy "S" bytes to "str":

    ```
    a = n.array([b"Hello", b"Hello", b"Hello"], dtype="S")
    cache = hashcache.init(bytes.decode)
    strs = n.vectorize(cache.f, otypes="O")(a)
    ```

    This only allocates heap memory once for each unique string in the input
    array when converting to strings
    """

    __slots__ = ("factory", "f")

    def __new__(cls, _: Callable[[K], V]):
        return super().__new__(cls)

    def __init__(self, key_value_factory: Callable[[K], V]):
        super().__init__(self)
        self.factory = key_value_factory
        self.f = self.__getitem__

    def __missing__(self, key):
        new = self.factory(key)
        self.__setitem__(key, new)
        return new


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


def strbytelen(s: str) -> int:
    """
    Get the number of bytes in a string's UTF-8 representation
    """
    return len(str.encode(s))


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
