from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Type, Union, overload
import numpy as n

if TYPE_CHECKING:
    from numpy.typing import DTypeLike  # type: ignore

from .dtype import DType, Field, Shape, fielddtype
from . import core  # type: ignore


Int = Union[int, n.int8, n.int16, n.int32, n.int64]
Uint = Union[n.uint8, n.uint16, n.uint32, n.uint64]
Float = Union[float, n.float32, n.float64]
Complex = Union[complex, n.complex64, n.complex128]
Scalar = Union[Int, Uint, Float, Complex, n.object0]


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


class Data(core.Data, Mapping[str, DType]):
    """
    Accessor and memory management class for native dataset memory. Can be used
    as a mapping where values are field type descriptors.
    """

    __slots__ = tuple()

    @classmethod
    def allocate(cls, size: int = 0, fields: List[Field] = []):
        data = cls()
        data.addrows(size)
        for field in fields:
            data.addcol(field)
        return data

    def __len__(self) -> int:
        return self.ncol()

    def __getitem__(self, k: str) -> DType:
        t = self.type(k)
        if t == 0:
            raise KeyError(f"Unknown dset field {k}")
        assert t in DSET_TO_TYPE_MAP, f"Unknown dset field type {t}"
        dst = DsetType(t)
        dt = n.dtype(DSET_TO_TYPE_MAP[dst])
        shape = self.getshape(k)
        return (dt.str, shape) if shape else dt.str

    def __contains__(self, field: object):
        return self.type(str(field)) > 0

    def __iter__(self):
        for i in range(self.ncol()):
            yield self.key(i)

    def __copy__(self):
        return Data(self)

    def __deepcopy__(self):
        return Data(self)

    def copy(self) -> "Data":
        return Data(self)

    def getshape(self, field: str) -> Shape:
        val: int = self.getshp(field)
        shape = (val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF)
        return tuple(s for s in shape if s != 0)

    @overload
    def addcol(self, field: Field) -> DType:
        ...

    @overload
    def addcol(self, field: str, dtype: "DTypeLike") -> DType:
        ...

    def addcol(self, field: Union[str, Field], dtype: Optional["DTypeLike"] = None) -> DType:
        if isinstance(field, tuple):
            dt = n.dtype(fielddtype(field))
            field = field[0]
        else:
            dt = n.dtype(dtype)

        existing_type = self.type(field)
        assert existing_type == 0, f"Field {field} already defined in dataset"
        if dt.shape:
            assert dt.base.type in TYPE_TO_DSET_MAP, f"Unsupported column data type {dt.base}"
            shape = [0] * 3
            shape[0 : len(dt.shape)] = dt.shape
            assert self.addcol_array(
                field, TYPE_TO_DSET_MAP[dt.base.type], *shape
            ), f"Could not add {field} with dtype {dt}"
            return (dt.base.str, dt.shape)
        elif dt.char in {"O", "S", "U"}:  # all python string object types
            assert self.addcol_scalar(field, DsetType.T_OBJ), f"Could not add {field} with dtype {dt}"
            return dt.str
        else:
            assert dt.type in TYPE_TO_DSET_MAP, f"Unsupported column data type {dt}"
            assert self.addcol_scalar(field, TYPE_TO_DSET_MAP[dt.type]), f"Could not add {field} with dtype {dt}"
            return dt.str
