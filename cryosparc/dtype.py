"""
Utilities and type definitions for working with dataset fields and column types.
"""

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union

import numpy as n
from typing_extensions import Literal, Sequence, TypedDict

from .core import Data, DsetType

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

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


class DatasetHeader(TypedDict):
    """
    Encoded dataset file description.
    """

    length: int
    """Number of rows in the dataset"""

    dtype: List[Field]
    """
    Column description
    """

    compression: Literal["lz4", None]
    """Compression library used for in the dataset"""

    compressed_fields: List[str]
    """Field names that require decompression."""


DSET_TO_TYPE_MAP: Dict[DsetType, Type] = {
    DsetType.T_F32: n.float32,
    DsetType.T_F64: n.float64,
    DsetType.T_C32: n.complex64,
    DsetType.T_C64: n.complex128,
    DsetType.T_I8: n.int8,
    DsetType.T_I16: n.int16,
    DsetType.T_I32: n.int32,
    DsetType.T_I64: n.int64,
    DsetType.T_U8: n.uint8,
    DsetType.T_U16: n.uint16,
    DsetType.T_U32: n.uint32,
    DsetType.T_STR: n.uint64,  # Note: Prefer T_OBJ when working in Python
    DsetType.T_U64: n.uint64,
    DsetType.T_OBJ: n.object_,
}

TYPE_TO_DSET_MAP = {
    **{v: k for k, v in DSET_TO_TYPE_MAP.items()},
    **{
        float: DsetType.T_F64,
        complex: DsetType.T_C64,
        int: DsetType.T_I64,
        str: DsetType.T_OBJ,
        object: DsetType.T_OBJ,
    },
}

# Set of dataset fields that should not be compressed when saving in
# NEWEST_FORMAT
NEVER_COMPRESS_FIELDS = {"uid"}


def normalize_field(name: str, dtype: "DTypeLike") -> Field:
    # Note: field name "uid" is always uint64, regardless of given dtype
    # Note: sd
    dt = n.dtype(dtype)
    if name == "uid":
        return name, n.dtype(n.uint64).str
    elif dt.char in {"O", "S", "U"}:  # all python string object types
        return name, n.dtype(object).str
    elif dt.shape:
        return name, dt.base.str, dt.shape
    else:
        return name, dt.str


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


def get_data_field(data: Data, field: str) -> Field:
    return normalize_field(field, get_data_field_dtype(data, field))


def get_data_field_dtype(data: Data, field: str) -> "DTypeLike":
    t = data.type(field)
    if t == 0 or t not in DSET_TO_TYPE_MAP:
        raise KeyError(f"Unknown dataset field {field} or field type {t}")
    dt = n.dtype(DSET_TO_TYPE_MAP[t])
    shape = data.getshp(field)
    return (dt.str, shape) if shape else dt.str


def filter_descr(
    descr: List[Field],
    *,
    keep_prefixes: Optional[Sequence[str]] = None,
    keep_fields: Optional[Sequence[str]] = None,
) -> List[Field]:
    # Get a filtered list of fields based on the user-specified prefixies
    # and/or fields. Returns all fields if no filter params are specified.
    # Always returns at least uid field, if it exists.
    filtered: List[Field] = []
    for field in descr:
        if (
            field[0] == "uid"
            or (keep_prefixes is None and keep_fields is None)
            or (keep_prefixes is not None and any(field[0].startswith(f"{p}/") for p in keep_prefixes))
            or (keep_fields is not None and field[0] in keep_fields)
        ):
            filtered.append(field)
    return filtered


def encode_dataset_header(fields: DatasetHeader) -> bytes:
    return json.dumps(fields).encode()


def decode_dataset_header(data: Union[bytes, dict]) -> DatasetHeader:
    try:
        header = json.loads(data) if isinstance(data, bytes) else data
        assert isinstance(header, dict), f"Incorrect decoded header type (expected dict, got {type(header)})"
        assert "length" in header and isinstance(
            header["length"], int
        ), 'Dataset header "length" key missing or has incorrect type'
        assert "dtype" in header and isinstance(
            header["dtype"], list
        ), 'Dataset header "dtype" key missing or has incorrect type'
        assert "compression" in header and header["compression"] in {
            None,
            "lz4",
        }, 'Dataset header "compression" key missing or has incorrect type'
        assert "compressed_fields" in header or isinstance(
            header["compressed_fields"], list
        ), 'Dataset header "compressed_fields" key missing or has incorrect type'

        length: int = header["length"]
        dtype: List[Field] = [(f, d, tuple(rest[0])) if rest else (f, d) for f, d, *rest in header["dtype"]]
        compression: Literal["lz4", None] = header["compression"]
        compressed_fields: List[str] = header["compressed_fields"]

        return DatasetHeader(
            length=length,
            dtype=dtype,
            compression=compression,
            compressed_fields=compressed_fields,
        )
    except Exception as e:
        raise ValueError(f"Incorrect dataset field format: {data.decode() if isinstance(data, bytes) else data}") from e
