from typing import Tuple, Union
import numpy as n
import numpy.typing as nt

Shape = Tuple[int, ...]
"""A numpy shape tuple from ndarray.shape"""

DType = Union[str, Tuple[str, Shape]]
"""

    Can just be a single string such as `f4`, `3u4` or `O`.
    A datatype description of a ndarray entry.

    Can also be the a tuple with a string datatype name and its shape. For
    example, the following dtypes are equivalent.

    - '3u4'
    - '<u4', (3,))
"""

Field = Union[Tuple[str, str], Tuple[str, str, Shape]]
"""
    Description of a column in a numpy array with named fields

    Examples:
    - ('uid', 'u8')
    - ('shape', '3u4')
    - ('shape', 'u4', (3,))
"""


def field_dtype(field: Field) -> DType:
    _, dt, *rest = field
    return (dt, rest[0]) if rest else dt


def dtypestr(dtype: nt.DTypeLike) -> str:
    dt = n.dtype(dtype)
    if dt.shape:
        shape = ",".join(map(str, dt.shape))
        return f"{dt.base.str[0]}{shape}{dt.base.str[1:]}"
    else:
        return dt.str


def dtype_field(name: str, dtype: nt.DTypeLike) -> Field:
    if isinstance(dtype, str):
        return (name, dtype)
    dt = n.dtype(dtype)
    return (name, dt.base.str, dt.shape) if dt.shape else (name, dt.str)
