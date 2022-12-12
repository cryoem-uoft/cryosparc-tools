from typing import Optional
import numpy as n

from .core import Data
from .dtype import Field, fielddtype
from .util import hashcache, strencodenull


class Column(n.ndarray):
    """
    Dataset column that inherits from the native numpy array interface.

    Note:
        Storing a column instance outside of a dataset prevents the whole
        dataset from getting garbage collected. Create a copy of its contents
        to prevent this, e.g., ``np.array(dset['ctf/phase_shift_rad']))``.

    Note:
        If new fields are added to the original dataset, a column instance may
        no longer be valid and must be retrieved again from
        ``dataset[fieldname]``.

    Examples:

        Storing column instances

        >>> dset = Dataset.allocate(1000, [('col1', 'f4'), ('col2', 'f4'), ('col3', 'f4')])
        Dataset(...)
        >>> col = dset['col1']
        >>> assert isinstance(col, Column)  # ok
        >>> del dset             # col still available but dset not garbage collected
        >>> col = np.array(col)  # dset now may be garbage collected

        Invalid column instances after adding columns

        >>> dset = Dataset.allocate(1000, [('col1', 'f4')])
        Dataset(...)
        >>> col = dset['col1']
        >>> dset.add_fields([('col2', 'f4'), ('col3', 'f4')])
        >>> np.sum(col)  # DANGER!! May result in invalid access
        >>> col = dset['col1']
        >>> np.sum(col)  # Retrieved from dataset, now valid
        0

    """

    __slots__ = ("_dataset",)
    _dataset: Optional[Data]

    def __new__(cls, field: Field, data: Data):
        dtype = n.dtype(fielddtype(field))
        nrow = data.nrow()
        shape = (nrow, *dtype.shape)
        buffer = data.getbuf(field[0]).memview if nrow else None
        obj = super().__new__(cls, shape=shape, dtype=dtype.base, buffer=buffer)  # type: ignore

        # Keep a reference to the data so that it only gets cleaned up when all
        # columns are cleaned up. No need to transfer this data during
        # __array_finalize__ because this NDColumn instance will be kept as
        # the base
        obj._dataset = data

        return obj

    def __array_wrap__(self, obj, **kwargs):
        # This prevents wrapping single results such as aggregations from n.sum
        # or n.median
        return obj[()] if obj.shape == () else super().__array_wrap__(obj, **kwargs)

    def to_fixed(self) -> "Column":
        """
        If this Column is composed of Python objects, convert to fixed-size
        strings. Otherwise the array is already in fixed form and may be
        returned as is.

        Returns:
            Column: Either same column with optionally converted data. If the
                data type is ``numpy.object_``, converts to ``numpy.bytes_``
                with a fixed element size based on the longest available string
                length.
        """
        if self.dtype.char == "O":
            return n.vectorize(hashcache(strencodenull), otypes="S")(self)
        else:
            return self
