from abc import ABC
from inspect import getmembers
from typing import (
    Any,
    Collection,
    Sequence,
    Union,
    overload,
)
import numpy as n
import numpy.typing as nt

from .data import Data
from .dtype import Field, field_itemsize, field_shape, field_strides


class Column(Sequence, n.lib.mixins.NDArrayOperatorsMixin, ABC):
    """
    Dataset Column entry. May be used anywhere in place of Numpy arrays
    """

    def __init__(self, data: Data, field: Field, subset: slice = slice(0, None)):
        start, stop, step = subset.indices(data.nrow())
        dtype = n.dtype(field[1])
        self.dtype = dtype.base if dtype.shape else dtype
        self.shape = (len(range(start, stop, step)), *field_shape(field))

        self._field = field
        self._data = data
        self._subset = subset

        # Copy over available numpy functions
        existing_attrs = set(dir(self))
        a = n.ndarray(0, dtype=self.dtype)
        for (attr, _) in getmembers(a, callable):
            if attr not in existing_attrs:
                setattr(self, attr, self.__get_callable__(attr))

    @property
    def field(self):
        return self._field

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        Interface for calling numpy universal function operations. Details here:
        https://numpy.org/doc/stable/user/basics.interoperability.html?highlight=__array_ufunc__#the-array-ufunc-protocol

        Calling ufuncs always creates instances of ndarray
        """
        out = kwargs.get("out", ())
        inputs = tuple(n.array(x, copy=False) if isinstance(x, Column) else x for x in args)

        if out:
            kwargs["out"] = tuple(n.array(x, copy=False) if isinstance(x, Column) else x for x in out)

        return getattr(ufunc, method)(*inputs, **kwargs)

    def __get_callable__(self, key):
        # Retrieve numpy ndarray method on this class with the given name
        def f(*args, **kwargs):
            return getattr(n.array(self, copy=False), key)(*args, **kwargs)

        return f

    def __len__(self) -> int:
        return len(n.array(self, copy=False))

    def __iter__(self):
        return n.array(self, copy=False).__iter__()

    @overload
    def __getitem__(self, key: slice) -> "Column":
        ...

    @overload
    def __getitem__(self, key: int) -> Any:
        ...

    @overload
    def __getitem__(self, key: Any) -> nt.NDArray:  # Returns a copy
        ...

    def __getitem__(self, key: Union[slice, int, Any]) -> Union["Column", nt.NDArray, Any]:
        if isinstance(key, slice):
            # Combine the given slices and current self._subset slice to create
            # a new subset (keeps underlying data)
            datalen = self._data.nrow()
            r = range(datalen)[self._subset][key]
            return type(self)(self._data, self._field, subset=slice(r.start, r.stop, r.step))
        else:
            # Indeces or mask
            return n.array(self, copy=False)[key]

    def __setitem__(self, key: Any, value: Any):
        n.array(self, copy=False)[key] = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}{repr(n.array(self, copy=False))[5:]}"

    def to_numpy(self, copy: bool = True, fixed: bool = False):
        """
        Get a numpy array of this data. Specify `fixed=True` to convert the
        underlying data to fixed-size (when applicable)
        """
        return n.array(self, copy=copy)


class NumericColumn(Column):
    def __init__(self, data: Data, field: Field, subset: slice = slice(0, None)):
        super().__init__(data, field, subset)
        self._strides = field_strides(field, self._subset.step or 1)
        self._offset = (self._subset.start or 0) * field_itemsize(field)

    @property
    def __array_interface__(self):
        """
        Allows Column instances to be used as numpy arrays. The abstract integer
        pointer value of the 'data' key may be dynamic because the underlying
        dataset gets re-computed
        """
        return {
            "data": (self._data.get(self.field[0]) + self._offset, False),
            "shape": self.shape,
            "typestr": self.dtype.str,
            "strides": self._strides,
            "version": 3,
        }


class StringColumn(Column):
    def __init__(self, data: Data, field: Field, subset: slice = slice(0, None)):
        assert len(field) == 2 and n.dtype(field[1]) == n.dtype(
            n.object0
        ), f"Cannot create String column with dtype {field[1]}"
        super().__init__(data, field, subset)
        # Available string indexes in this dataset
        self._idxs = n.array(list(range(*self._subset.indices(self._data.nrow()))))

    def __array__(self, dtype: nt.DTypeLike = n.object0):
        return n.array(list(iter(self)), dtype=dtype)

    def __len__(self) -> int:
        return len(self._idxs)

    def __iter__(self):
        for i in range(*self._subset.indices(self._data.nrow())):
            yield self._data.getstr(self.field[0], i)

    @overload
    def __getitem__(self, key: slice) -> "Column":
        ...

    @overload
    def __getitem__(self, key: int) -> str:
        ...

    @overload
    def __getitem__(self, key: Any) -> nt.NDArray:  # Returns a copy
        ...

    def __getitem__(self, key: Union[slice, int, Any]) -> Union["Column", nt.NDArray, str]:
        if isinstance(key, slice):
            return super().__getitem__(key)  # Return string column subset
        elif isinstance(key, (int, n.integer)):
            return self._data.getstr(self.field[0], self._idxs[key])
        elif isinstance(key, Collection):  # mask or index list
            idxs = self._idxs[n.array(key, copy=False)]
            return n.array([self._data.getstr(self.field[0], i) for i in idxs], dtype=n.object0)

        raise TypeError(f"Invalid index into StringColumn: {key}")

    def __setitem__(self, key: Any, value: Union[str, bytes, Collection[str], Collection[bytes]]):
        if isinstance(key, (int, n.integer)):
            idxs = [self._idxs[key]]
        else:
            idxs = self._idxs[key]

        f = self.field[0]
        if isinstance(value, (str, bytes)):
            for i in idxs:
                self._data.setstr(f, i, value)
        else:
            assert len(value) == len(idxs), f"Cannot assign [{key}] for {type(self).__name__} to {value}"
            for i, v in zip(idxs, value):
                self._data.setstr(f, i, v)

    def to_numpy(self, copy: bool = True, fixed: bool = False):
        if not fixed:
            return self.__array__()  # always copies
        maxstrlen = n.vectorize(len)(self).max() + 1
        dtype = "S{}".format(maxstrlen)
        return self.__array__(dtype)

    def __repr__(self) -> str:
        infix = '", "'
        if len(self) > 6:
            contents = f'{infix.join(self[:3])}", ... , "{infix.join(self[-3:])}'
        else:
            contents = infix.join(self)

        return f'{type(self).__name__}(["{contents}"])'
