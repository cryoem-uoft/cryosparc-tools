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
    ContextManager,
    Dict,
    Generator,
    Generic,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Literal
import numpy as n

if TYPE_CHECKING:
    from numpy.typing import NDArray  # type: ignore

from .dtype import Shape

OpenTextMode = Literal["r", "w", "x", "a", "r+", "w+", "x+", "a+"]
"""
Text file read or write open modes.
"""

OpenBinaryMode = Literal["rb", "wb", "xb", "ab", "r+b", "w+b", "x+b", "a+b"]
"""
Binary file read or write open modes.
"""


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
E = TypeVar("E")  # error
INT = TypeVar("INT", bound=n.integer, covariant=True)

Result = Union[Tuple[T, Literal[None]], Tuple[Literal[None], E]]
"""
Use as the return type for functions that may return either a value or an
error.

Example:

    >>> def safe_divide(x: float, y: float) -> Result[int, str]:
    ...     if y == 0:
    ...         return None, "divide by zero detected"
    ...     else:
    ...         return x / y, None
"""


class hashcache(Dict[K, V], Generic[K, V]):
    """
    Simple utility class to cache the result of a mapping and avoid excessive
    heap allocation. Initialize with ``cache = hashcache(f)`` and use ``cache``
    as the mapping function.

    Examples:

        Unoptimized code for convering ``bytes`` to ``str``:

        >>> a = [b"Hello", b"Hello", b"Hello"]
        >>> strs = list(map(bytes.decode, a))

        The input array ``a`` has duplicate items. Using ``bytes.decode``
        directly in the mapping causes Python to allocate a fresh string for
        each bytes.

        Here is the optimized version with ``hashcache``:

        >>> a = [b"Hello", b"Hello", b"Hello"]
        >>> strs = list(map(hashcache(bytes.decode), a))

        This only allocates heap memory once for each unique item in the input
        list. After the first encounter, the map callable returns the
        previously-computed value of the same input, reducing heap usage and
        allocation calls by a factor of 3.

        For this to be most effective, ensure the given ``f`` function is pure
        and stable (i.e., always returns the same result for a given input).

        May also be used as a decorator (must take a single hashable argument):

        >>> @hashcache
        >>> def f(x): ...
    """

    __slots__ = ("factory",)
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
            n (int, optional): Number of bytes to read. Reads all bytes if not
                specified. Defaults to None.

        Returns:
            bytes: Zero or more bytes read
        """
        r = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                r.append(m)
        else:
            while n > 0:
                m = self._read1(n)
                if not m:
                    break
                n -= len(m)
                r.append(m)
        return b"".join(r)


class AsyncBinaryIteratorIO:
    """
    Similar to ``BinaryIteratorIO`` except the iterator yields bytes
    asynchronously.

    Args:
        iter (AsyncIterator[bytes]): Generator that asynchronously yields bytes.

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
            n (int, optional): Number of bytes to read. Reads all bytes if not
                specified. Defaults to None.

        Returns:
            bytes: Zero or more bytes read
        """
        r = []
        if n is None or n < 0:
            while True:
                m = await self._read1()
                if not m:
                    break
                r.append(m)
        else:
            while n > 0:
                m = await self._read1(n)
                if not m:
                    break
                n -= len(m)
                r.append(m)
        return b"".join(r)


def first(it: Union[Iterator[V], Sequence[V]]) -> Optional[V]:
    """
    Get the first item from the given iterator. Returns None if the iterator is
    empty.

    Args:
        it (Iterator[V] | Sequence[V]): Iterator or list-like accessor.

    Returns:
        V | None: First item in the list or iterator (consuming it if required).
    """
    try:
        return it[0] if isinstance(it, Sequence) else next(it)
    except (StopIteration, IndexError):
        return None


def u32bytesle(x: int) -> bytes:
    """
    Get the uint32 bytes of for integer x in little endian.

    Args:
        x (int): Integer to encode.

    Returns:
        bytes: Encoded integer bytes
    """
    return x.to_bytes(4, "little", signed=False)


def u32intle(buffer: bytes) -> int:
    """
    Get int from buffer representing a uint32 integer in little endian.

    Args:
        buffer (bytes): 4 bytes representing little-endian integer.

    Returns:
        int: decoded integer
    """
    return int.from_bytes(buffer, "little", signed=False)


def strbytelen(s: str) -> int:
    """
    Get the number of bytes in a string's UTF-8 representation.

    Args:
        s (str): string to test encoding for.

    Returns:
        int: number of bytes in encoded string.
    """
    return len(str.encode(s))


def strencodenull(s: Any) -> bytes:
    """
    Encode string-like value into UTF-8 binary ending with a null-character
    terminator \0.

    Args:
        s (any): string-like entity to encode.

    Returns:
        bytes: encoded string
    """
    return ("" if s is None else str(s)).encode() + b"\0"


@contextmanager
def topen(file: Union[str, PurePath, IO[str]], mode: OpenTextMode = "r"):
    """
    "with open(...)" alias for text files that tranparently yields open file or
    file-like object.

    Args:
        file (str | Path | IO): Readable text file path or handle.
        mode (OpenTextMode, optional): File open mode. Defaults to "r".

    Yields:
        IO: Text file handle.
    """
    if isinstance(file, (str, PurePath)):
        with open(file, mode) as f:
            yield f
    else:
        yield file


@contextmanager
def bopen(file: Union[str, PurePath, IO[bytes]], mode: OpenBinaryMode = "rb"):
    """
    "with open(...)" alias for binary files that tranparently yields an open
    file or file-like object.

    Args:
        file (str | Path | IO): binary file path or handle
        mode (OpenBinaryMode, optional): File open mode. Defaults to "rb".

    Yields:
        IO: Binary file handle
    """
    if isinstance(file, (str, PurePath)):
        with open(file, mode) as f:
            yield f
    else:
        yield file


@overload
def noopcontext() -> ContextManager[None]:
    ...


@overload
def noopcontext(x: T) -> ContextManager[T]:
    ...


@contextmanager
def noopcontext(x: Optional[T] = None) -> Generator[Optional[T], None, None]:
    """
    Context manager that yields the given argument without modification.

    Args:
        x (T, optional): Anything. Defaults to None.

    Yields:
        T: the given argument
    """
    yield x


def padarray(arr: "NDArray", dim: Optional[int] = None, val: n.number = n.float32(0)):
    """
    Pad the given 2D or 3D array so that the x and y dimensions are equal to the
    given dimension. If not dimension is given, will use the maximum of the
    width and height.

    Args:
        arr (NDArray): 2D or 3D Numpy array to pad
        dim (int, optional): Minimum desired dimension of micrograph. Defaults
            to None.
        val (float, optional): Value to pad with, should be a numpy number
            instance with the same dtype as the given array. Defaults to
            n.float32(0).

    Returns:
        NDArray: Padded copy of given array.
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
    Crop the given 2D or 3D array into the given shape. Will trim from the
    middle. May be used to undo padding from ``padarray()`` function.

    Args:
        arr (NDArray): 3D or 3D numpy array to crop
        shape (Shape): New desired shape.

    Returns:
        NDArray: Trimmed copy of the given array.
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
        numpy.random.Generator: Random number generator
    """
    try:
        return n.random.default_rng(seed)
    except AttributeError:
        return n.random.RandomState(seed)  # type: ignore


def random_integers(
    rng: "n.random.Generator",
    low: int,
    high: Optional[int] = None,
    size: Union[int, Shape, Literal[None]] = None,
    dtype: Type[INT] = n.uint64,
) -> "NDArray[INT]":
    """
    Generic way to get random integers from a numpy random generator (or
    RandomState for older numpy).

    Args:
        rng (numpy.random.Generator): random number generator
        low (int): Low integer value, inclusive
        high (int, optional): High integer value, exclusive. Defaults to None.
        size (Shape, optional): Size or shape of resulting array. Defaults to None.
        dtype (Type[numpy.integer], optional): Type of integer values to generate. Defaults to numpy.uint64.

    Returns:
        NDArray: Numpy array of randomly-generated integers.
    """
    try:
        f = rng.integers
    except AttributeError:
        f = rng.randint  # type: ignore
    return f(low=low, high=high, size=size, dtype=dtype)  # type: ignore
