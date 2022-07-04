from typing import Tuple, Union

Shape = Tuple[int, ...]
"""A numpy shape tuple from ndarray.shape"""

Dtype = Union[str, Tuple[str, Shape]]
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


def field_dtype(field: Field) -> Dtype:
    _, dt, *rest = field
    return (dt, rest[0]) if rest else dt


def dtype_field(name: str, dtype: Dtype) -> Field:
    return (name, dtype) if isinstance(dtype, str) else (name, *dtype)
