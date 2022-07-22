from contextlib import contextmanager
from pathlib import PurePath
from typing import IO, Callable, Dict, Generic, TypeVar, Union
from typing_extensions import Literal
import numpy as n

OpenBinaryMode = Literal["rb", "wb", "xb", "ab", "r+b", "w+b", "x+b", "a+b"]


K = TypeVar("K")
V = TypeVar("V")


class hashcache(Dict[K, V], Generic[K, V]):
    """
    Simple utility class to cache the result of a mapping and avoid excessive
    heap allocation. Initialize with `cache = hashcache(f)` and use `cache.f` as
    the mapping function.

    Here is an example of unoptimized code for convering `bytes` to `str`:

    ```
    a = [b"Hello", b"Hello", b"Hello"]
    strs = list(map(bytes.decode, a))
    ```

    The input array `a` has duplicate items. Using `bytes.decode` directly
    in the mapping causes Python to allocate a fresh string for each bytes.

    Here is the optimized version with `hashcache`:

    ```
    a = [b"Hello", b"Hello", b"Hello"]
    strs = list(map(hashcache(bytes.decode), a))
    ```

    This only allocates heap memory once for each unique item in the input list.
    After the first encounter, `cache.f` returns the previously-computed value
    of the same input, reducing heap usage and allocation by a factor of 3.

    For this to be most effective, ensure the given `f` function is pure and
    stable (i.e., always returns the same result for a given input).

    May also be used as a wrapper (must take a single hashable argument):

    ```
    @hashcache
    def f(x): ...
    ```
    """

    __slots__ = ("factory", "__call__")

    def __new__(cls, _f: Callable[[K], V]):
        return super().__new__(cls)

    def __init__(self, key_value_factory: Callable[[K], V]):
        super().__init__(self)
        self.factory = key_value_factory
        self.__call__ = self.__getitem__

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


def strencodenull(s: str) -> bytes:
    """
    Encode string into UTF-8 binary ending with a null-character terminator \0
    """
    return s.encode() + b"\0"


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
