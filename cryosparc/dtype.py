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
    - 'u4', (3,))
"""

Desc = Union[Tuple[str, str], Tuple[str, str, Shape]]
"""
    Description of a column in a numpy array with named fields

    Examples:
    - ('uid', 'u8')
    - ('shape', '3u4')
    - ('shape', 'u4', (3,))
"""


def desc_to_dtype(desc: Desc) -> Dtype:
    _, dt, *rest = desc
    return (dt, rest[0]) if rest else dt


def dtype_to_desc(name: str, dtype: Dtype) -> Desc:
    return (name, dtype) if isinstance(dtype, str) else (name, *dtype)
