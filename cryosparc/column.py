from abc import ABC
from inspect import getmembers
from typing import (
    Any,
    Collection,
    Optional,
    Sequence,
    Union,
    overload,
)
import numpy as n
import numpy.typing as nt

from .data import Data
from .dtype import Field, field_shape


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
        if self.dtype.char == "O":
            # NOTE: This does not work correcly with strings containing
            # characters that take up more than 1 byte
            maxlen = n.vectorize(len)(self).max() + 1
            return n.array(self, dtype=f"S{maxlen}")
        else:
            return self
