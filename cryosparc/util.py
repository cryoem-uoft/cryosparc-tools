from contextlib import contextmanager
from pathlib import PurePath
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    BinaryIO,
    Callable,
    Dict,
    Generic,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
from typing_extensions import Literal
import numpy as n

if TYPE_CHECKING:
    from numpy.typing import NDArray  # type: ignore

from .dtype import Shape

OpenTextMode = Literal["r", "w", "x", "a", "r+", "w+", "x+", "a+"]
OpenBinaryMode = Literal["rb", "wb", "xb", "ab", "r+b", "w+b", "x+b", "a+b"]


K = TypeVar("K")
V = TypeVar("V")


class hashcache(Dict[K, V], Generic[K, V]):
    """
    Simple utility class to cache the result of a mapping and avoid excessive
    heap allocation. Initialize with `cache = hashcache(f)` and use `cache` the
    mapping function.

    Examples:

        Unoptimized code for convering `bytes` to `str`:

        >>> a = [b"Hello", b"Hello", b"Hello"]
        >>> strs = list(map(bytes.decode, a))

        The input array `a` has duplicate items. Using `bytes.decode` directly
        in the mapping causes Python to allocate a fresh string for each bytes.

        Here is the optimized version with `hashcache`:

        >>> a = [b"Hello", b"Hello", b"Hello"]
        >>> strs = list(map(hashcache(bytes.decode), a))

        This only allocates heap memory once for each unique item in the input
        list. After the first encounter, `cache.f` returns the
        previously-computed value of the same input, reducing heap usage and
        allocation by a factor of 3.

        For this to be most effective, ensure the given `f` function is pure and
        stable (i.e., always returns the same result for a given input).

        May also be used as a wrapper (must take a single hashable argument):

        >>> @hashcache
        >>> def f(x): ...
    """

    __slots__ = "factory"
    __call__ = dict.__getitem__

    def __new__(cls, _f: Callable[[K], V]):
        return super().__new__(cls)

    def __init__(self, key_value_factory: Callable[[K], V]):
        super().__init__(self)
        self.factory = key_value_factory

    def __missing__(self, key):
        new = self.factory(key)
        self.__setitem__(key, new)
        return new


class BinaryIteratorIO(BinaryIO):
    """
    Read through a iterator that yields bytes as if it was a file.

    Args:
        iter (Iterator[bytes]): Generator that yields bytes
    """

    def __init__(self, iter: Iterator[bytes]):
        self._iter = iter
        self._left = b""

    def readable(self):
        """
        Returns:
            bool: True
        """
        return True

    def seekable(self) -> bool:
        """
        Returns:
            bool: False
        """
        return False

    def _read1(self, n: Optional[int] = None):
        while not self._left:
            try:
                self._left = next(self._iter)
            except StopIteration:
                break
        ret = self._left[:n]
        self._left = self._left[len(ret) :]
        return ret

    def read(self, n: Optional[int] = None) -> bytes:
        """
        Read the given number of bytes from the iterator

        Args:
            n (int, optional): Number of bytes to read. Reads all bytes if not specified. Defaults to None.

        Returns:
            bytes: Zero or more bytes read
        """
        l = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                l.append(m)
        else:
            while n > 0:
                m = self._read1(n)
                if not m:
                    break
                n -= len(m)
                l.append(m)
        return b"".join(l)


class AsyncBinaryIteratorIO:
    """
    Similar to `BinaryIteratorIO` except the iterator yields bytes asynchronously.

    Args:
        iter (AsyncIterator[bytes]): Generator that asynchronously yields bytes

    """

    def __init__(self, iter: Union[AsyncIterator[bytes], AsyncGenerator[bytes, None]]):
        self._iter = iter
        self._left = b""

    def readable(self):
        """
        Returns:
            bool: True
        """

        return True

    def seekable(self) -> bool:
        """
        Returns:
            bool: False
        """
        return False

    async def _read1(self, n: Optional[int] = None):
        while not self._left:
            try:
                self._left = await self._iter.__anext__()
            except StopAsyncIteration:
                break
        ret = self._left[:n]
        self._left = self._left[len(ret) :]
        return ret

    async def read(self, n: Optional[int] = None):
        """
        Read the given number of bytes from the async iterator.

        Args:
            n (int, optional): Number of bytes to read. Reads all bytes if not specified. Defaults to None.

        Returns:
            bytes: Zero or more bytes read
        """
        l = []
        if n is None or n < 0:
            while True:
                m = await self._read1()
                if not m:
                    break
                l.append(m)
        else:
            while n > 0:
                m = await self._read1(n)
                if not m:
                    break
                n -= len(m)
                l.append(m)
        return b"".join(l)


def first(it: Union[Iterator[V], Sequence[V]]) -> Optional[V]:
    """
    Get the first item from the given iterator. Returns None if the iterator is
    empty
    """
    try:
        return it[0] if isinstance(it, Sequence) else next(it)
    except (StopIteration, IndexError):
        return None


def u32bytesle(x: int) -> bytes:
    """
    Get the uint32 bytes of for integer x in little endian
    """
    return n.array(x, dtype="<u4").tobytes()


def u32intle(buffer: bytes) -> int:
    """
    Get int from buffer representing a uint32 integer in little endian
    """
    return int(n.frombuffer(buffer, dtype="<u4")[0])


def strbytelen(s: str) -> int:
    """
    Get the number of bytes in a string's UTF-8 representation
    """
    return len(str.encode(s))


def strencodenull(s: Any) -> bytes:
    """
    Encode string-like value into UTF-8 binary ending with a null-character
    terminator \0
    """
    return ("" if s is None else str(s)).encode() + b"\0"


@contextmanager
def topen(file: Union[str, PurePath, IO[str]], mode: OpenTextMode = "r"):
    """
    "with open(...)" alias for text files that tranparently yields already-open
    files or file-like objects.
    """
    if isinstance(file, (str, PurePath)):
        with open(file, mode) as f:
            yield f
    else:
        yield file


@contextmanager
def bopen(file: Union[str, PurePath, IO[bytes]], mode: OpenBinaryMode = "rb"):
    """
    "with open(...)" alias for binary files that tranparently yields already-open
    files or file-like objects.
    """
    if isinstance(file, (str, PurePath)):
        with open(file, mode) as f:
            yield f
    else:
        yield file


def padarray(arr: "NDArray", dim: Optional[int] = None, val: n.number = n.float32(0)):
    """
    Pad the given 2D or 3D array so that the x and y dimensions are equal to the
    given dimension. If not dimension is given, will use the maximum of the
    width and height.
    """
    arr = n.reshape(arr, (-1,) + arr.shape[-2:])
    nz, ny, nx = arr.shape
    idim = max(ny, nx) if dim is None else dim
    res = n.full((nz, idim, idim), val, dtype=arr.dtype)
    nya, nxa = ny // 2, nx // 2
    nyb, nxb = ny - nya, nx - nxa
    ya, yb = (idim // 2) - nya, (idim // 2) + nyb
    xa, xb = (idim // 2) - nxa, (idim // 2) + nxb
    res[:, ya:yb, xa:xb] = arr

    return n.reshape(res, res.shape[-2:]) if nz == 1 else res


def trimarray(arr: "NDArray", shape: Shape):
    """
    Crop the given 2D or 3D array into the given shape
    """
    assert len(shape) == 2, f"Invalid trim shape {shape}; must be 2D"
    arr = n.reshape(arr, (-1,) + arr.shape[-2:])
    z, x, y = arr.shape
    ny, nx = shape
    nya, nxa = ny // 2, nx // 2
    nyb, nxb = ny - nya, nx - nxa
    ya, yb = (x // 2) - nya, (x // 2) + nyb
    xa, xb = (y // 2) - nxa, (y // 2) + nxb
    res = arr[:, ya:yb, xa:xb]
    return n.reshape(res, res.shape[-2:]) if z == 1 else res


def default_rng(seed=None) -> "n.random.Generator":
    """
    Create a numpy random number generator or RandomState (depending on numpy
    version)

    Args:
        seed (Any, optional): Seed to initialize generator with. Defaults to None.

    Returns:
        n.random.Generator: Random number generator
    """
    if hasattr(n.random, "default_rng"):
        return n.random.default_rng(seed)
    else:
        return n.random.RandomState(seed)  # type: ignore
