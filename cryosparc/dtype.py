"""
Utilities and type definitions for working with dataset fields and column types.
"""
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple, Type, Union
import json
from typing_extensions import Literal, TypedDict
import numpy as n

if TYPE_CHECKING:
    from numpy.typing import NDArray, DTypeLike
    from .core import Data

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
    Dataset header description when saving in CSDAT format.
    """

    dtype: List[Field]
    compression: Literal["snap"]
    compressed_fields: List[str]


class DsetType(int, Enum):
    """
    Mirror of equivalent C-datatype enumeration
    """

    T_F32 = 1
    T_F64 = 2
    T_C32 = 3
    T_C64 = 4
    T_I8 = 5
    T_I16 = 6
    T_I32 = 7
    T_I64 = 8
    T_U8 = 9
    T_U16 = 10
    T_U32 = 11
    T_U64 = 12
    T_STR = 13
    T_OBJ = 14


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
    DsetType.T_U64: n.uint64,
    DsetType.T_STR: n.object0,  # Note: Prefer T_OBJ instead
    DsetType.T_OBJ: n.object0,
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


def get_data_field(data: "Data", field: str) -> Field:
    return makefield(field, get_data_field_dtype(data, field))


def get_data_field_dtype(data: "Data", field: str) -> "DTypeLike":
    t = data.type(field)
    if t not in DSET_TO_TYPE_MAP:
        raise KeyError(f"Unknown dset field type {t}")
    dt = n.dtype(DSET_TO_TYPE_MAP[DsetType(t)])
    shape = data.getshp(field)
    return (dt.str, shape) if shape else dt.str


def encode_dataset_header(fields: DatasetHeader) -> bytes:
    return json.dumps(fields).encode()


def decode_dataset_header(data: Union[bytes, dict]) -> DatasetHeader:
    try:
        header = json.loads(data) if isinstance(data, bytes) else data
        assert isinstance(header, dict), f"Incorrect decoded header type (expected dict, got {type(header)})"
        assert "dtype" in header and isinstance(
            header["dtype"], list
        ), 'Dataset header "dtype" key missing or has incorrect type'
        assert (
            "compression" in header and header["compression"] == "snap"
        ), 'Dataset header "compression" key missing or has incorrect type'
        assert (
            "compressed_fields" and header or isinstance(header["compressed_fields"], list)
        ), 'Dataset header "compressed_fields" key missing or has incorrect type'

        dtype: List[Field] = [(f, d, tuple(rest[0])) if rest else (f, d) for f, d, *rest in header["dtype"]]
        compression: Literal["snap"] = header["compression"]
        compressed_fields: List[str] = header["compressed_fields"]

        return DatasetHeader(dtype=dtype, compression=compression, compressed_fields=compressed_fields)
    except Exception as e:
        raise ValueError(f"Incorrect dataset field format: {data.decode() if isinstance(data, bytes) else data}") from e
