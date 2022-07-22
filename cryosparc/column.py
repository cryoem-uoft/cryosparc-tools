from typing import Any, Optional
import numpy as n
import numpy.typing as nt

from .data import Data
from .dtype import Field, field_shape
from .util import hashcache, strbytelen


class Column(nt.NDArray[Any]):
    """
    Dataset column, uses native numpy array interface. Used to keep _data
    instance from getting garbage collected if a single column is extracted from
    a dataset once that dataset has already been garbage collected.

    Note that if fields are added to the the dataset, a column instance may no
    longer be valid and must be retrieved again from `dataset[colname]`.
    """

    __slots__ = ("_data",)
    _data: Optional[Data]

    def __new__(cls, field: Field, data: Data):
        dtype = n.dtype(field[1])
        shape = (data.nrow(), *field_shape(field))
        buffer = data.getbuf(field[0])
        obj = super().__new__(cls, shape=shape, dtype=dtype, buffer=buffer)

        # Keep a reference to the data so that it only gets cleaned up when all
        # columns are cleaned up. No need to transfer this data during
        # `__array_finalize__` because this NDColumn instance will be kept as
        # the base
        obj._data = data

        return obj

    def to_fixed(self):
        """
        If this Column is composed of Python objects, convert to fixed-size
        strings. Otherwise the array is already in fixed form and may be
        returned as is
        """
        if self.dtype.char == "O":
            cache = hashcache(strbytelen)
            maxlen = n.vectorize(cache.f)(self).max() + 1
            return n.array(self, dtype=f"S{maxlen}")
        else:
            return self
