"""
Utilities and type definitions for working with dataset fields and column types.
"""
from typing import TYPE_CHECKING, List, Tuple, Union
import json
import numpy as n

if TYPE_CHECKING:
    from numpy.typing import NDArray, DTypeLike  # type: ignore

Shape = Tuple[int, ...]
"""A numpy shape tuple from ndarray.shape"""

DType = Union[str, Tuple[str, Shape]]
"""

    Can just be a single string such as "f4", "3u4" or "O".
    A datatype description of a ndarray entry.

    Can also be the a tuple with a string datatype name and its shape. For
    example, the following dtypes are equivalent.

    - "3u4"
    - "<u4", (3,))
"""

Field = Union[Tuple[str, str], Tuple[str, str, Shape]]
"""
    Description of a column in a numpy array with named fields

    Examples:
    - ("uid", "u8")
    - ("coords", "3f4")
    - ("coords", "<f4", (3,))
"""


def makefield(name: str, dtype: "DTypeLike") -> Field:
    dt = n.dtype(dtype)
    return (name, dt.base.str, dt.shape) if dt.shape else (name, dt.str)


def safe_makefield(name: str, dtype: "DTypeLike") -> Field:
    return ("uid", n.dtype(n.uint64).str) if name == "uid" else makefield(name, dtype)


def fielddtype(field: Field) -> DType:
    _, dt, *rest = field
    return (dt, rest[0]) if rest else dt


def arraydtype(a: "NDArray") -> DType:
    assert len(a.dtype.descr) == 1, "Cannot get dtype from record array"
    return (a.dtype.str, a.shape[1:]) if len(a.shape) > 1 else a.dtype.str


def dtypestr(dtype: "DTypeLike") -> str:
    dt = n.dtype(dtype)
    if dt.shape:
        shape = ",".join(map(str, dt.shape))
        return f"{dt.base.str[0]}{shape}{dt.base.str[1:]}"
    else:
        return dt.str


def encode_fields(fields: List[Field]) -> bytes:
    return json.dumps(fields).encode()


def decode_fields(data: Union[bytes, list]) -> List[Field]:
    try:
        fields = json.loads(data) if isinstance(data, bytes) else data
        return [(f, d, tuple(rest[0])) if rest else (f, d) for f, d, *rest in fields]
    except Exception:
        raise ValueError(f"Incorrect dataset field format: {data.decode() if isinstance(data, bytes) else data}")
