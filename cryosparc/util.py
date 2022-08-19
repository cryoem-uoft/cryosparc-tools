from contextlib import contextmanager
from pathlib import PurePath
from typing import IO, Callable, Dict, Generic, Iterator, Optional, Sequence, TypeVar, Union
from typing_extensions import Literal
import numpy as n
import numpy.typing as nt

from .dtype import Shape

OpenTextMode = Literal["r", "w", "x", "a", "r+", "w+", "x+", "a+"]
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


def strencodenull(s: str) -> bytes:
    """
    Encode string into UTF-8 binary ending with a null-character terminator \0
    """
    return s.encode() + b"\0"


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


def downsample(arr: nt.NDArray, factor: int = 2):
    """
    Downsample a micrograph by the given factor
    """
    assert factor >= 1, "Must bin by a factor of 1 or greater"
    arr = n.reshape(arr, (-1,) + arr.shape[-2:])
    nz, ny, nx = arr.shape
    clipx = (nx // factor) * factor
    clipy = (ny // factor) * factor
    shape = (nz, (clipy // factor), (clipx // factor)) if nz > 1 else ((clipy // factor), (clipx // factor))
    out = arr[:, :clipy, :clipx].reshape(nz, clipy, (clipx // factor), factor)
    out = out.sum(axis=-1)
    out = out.reshape(nz, (clipy // factor), factor, -1).sum(axis=-2)
    return out.reshape(shape)


def padarray(arr: nt.NDArray, dim: Optional[int] = None, val: n.number = n.float32(0)):
    """
    Pad the given 2D or 3D array so that the x and y dimensions are equal to the
    given dimension. If not dimension is given, will use the maximum of the
    width and height.
    """
    arr = n.reshape(arr, (-1,) + arr.shape[-2:])
    nz, ny, nx = arr.shape
    dim = max(ny, nx) if dim is None else dim
    res = n.full((nz, dim, dim), val, dtype=arr.dtype)
    nya, nxa = ny // 2, nx // 2
    nyb, nxb = ny - nya, nx - nxa
    ya, yb = (dim // 2) - nya, (dim // 2) + nyb
    xa, xb = (dim // 2) - nxa, (dim // 2) + nxb
    res[:, ya:yb, xa:xb] = arr

    return n.reshape(res, res.shape[-2:]) if nz == 1 else res


def trimarray(arr: nt.NDArray, shape: Shape):
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


def lowpass(arr: nt.NDArray, psize: float, amount: float = 0.0, order: float = 1.0):
    """
    Apply butterworth lowpass filter to the 2D or 3D array data with the given
    pixel size. `amount` should be a non-negative integer specified in
    Angstroms.
    """
    assert amount > 0, "Lowpass filter amount must be non-negative"
    assert len(arr.shape) == 2 or (len(arr.shape) == 3 and arr.shape[0] == 1), (
        f"Cannot apply low-pass filter on data with shape {arr.shape}; " "must be two-dimensional"
    )

    arr = n.reshape(arr, arr.shape[-2:])
    shape = arr.shape
    if arr.shape[0] != arr.shape[1]:
        arr = padarray(arr, val=n.mean(arr))

    radwn = (psize * arr.shape[-1]) / amount
    inverse_cutoff_wn2 = 1.0 / radwn**2

    farr = n.fft.rfft2(arr)
    ny, nx = farr.shape
    yp = 0

    for y in range(ny // 2):
        yp = (ny // 2) if y == 0 else ny - y

        # y goes from DC to one before nyquist
        # x goes from DC to one before nyquist
        r2 = (n.arange(nx - 1) ** 2) + (y * y)
        f = 1.0 / (1.0 + (r2 * inverse_cutoff_wn2) ** order)
        farr[y][:-1] *= f
        farr[yp][:-1] *= f

    # zero nyquist at the end
    farr[ny // 2] = 0.0
    farr[:, nx - 1] = 0.0

    result = n.fft.irfft2(farr)
    return trimarray(result, shape) if result.shape != shape else result
