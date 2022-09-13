from typing import Optional
import numpy as n

from .data import Data
from .dtype import Field, fielddtype
from .util import hashcache, strencodenull


class Column(n.ndarray):
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
        dtype = n.dtype(fielddtype(field))
        shape = (data.nrow(), *dtype.shape)
        buffer = data.getbuf(field[0])
        obj = super().__new__(cls, shape=shape, dtype=dtype.base, buffer=buffer)

        # Keep a reference to the data so that it only gets cleaned up when all
        # columns are cleaned up. No need to transfer this data during
        # `__array_finalize__` because this NDColumn instance will be kept as
        # the base
        obj._data = data

        return obj

    def __array_wrap__(self, obj, **kwargs):
        # This prevents wrapping single results such as aggregations from n.sum
        # or n.median
        return obj[()] if obj.shape == () else super().__array_wrap__(obj, **kwargs)

    def to_fixed(self) -> "Column":
        """
        If this Column is composed of Python objects, convert to fixed-size
        strings. Otherwise the array is already in fixed form and may be
        returned as is
        """
        if self.dtype.char == "O":
            return n.vectorize(hashcache(strencodenull), otypes="S")(self)
        else:
            return self
