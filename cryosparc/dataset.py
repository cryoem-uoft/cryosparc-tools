from inspect import getmembers
from pathlib import Path
import sys
from typing import (
    Any,
    BinaryIO,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
import numpy as n
import numpy.typing as nt

from .dtype import Dtype, Field

Int = Union[int, n.int8, n.int16, n.int32, n.int64]
Float = Union[float, n.float32, n.float64]
Complex = Union[complex, n.complex64, n.complex128]

BYTEORDER = "<" if sys.byteorder == "little" else ">"
VOID = b""


class Column(Sequence, n.lib.mixins.NDArrayOperatorsMixin):
    """
    Dataset Column entry. May be used anywhere in place of Numpy arrays
    """

    def __init__(self, data, field: str, subset: slice = slice(0, None, 1)):
        # dlen = len(data[field])  # FIXME: Get the data from the given field somehow
        dtype = n.dtype([(field, "<u8")])
        datalen = len(data) // dtype.itemsize
        start, stop, step = subset.indices(datalen)
        self.shape = (len(range(start, stop, step)),)
        self.dtype = dtype
        self._field = field
        self._data = data
        self._subset = subset

        # Copy over available numpy functions
        existing_attrs = set(dir(self))
        a = n.array(self, copy=False)
        for (attr, _) in getmembers(a, callable):
            if attr not in existing_attrs:
                setattr(self, attr, self.__get_callable__(attr))

    @property
    def __array_interface__(self):
        """
        Allows Column instances to be used as numpy arrays
        """
        print("CALLED ARRAY IFACE")
        datalen = len(self._data) // self.dtype.itemsize
        start, _, step = self._subset.indices(datalen)
        return {
            "data": self._data,
            "shape": self.shape,
            "typestr": self.dtype[0].str,
            "descr": self.dtype.descr,
            "strides": (self.dtype.itemsize * step,),
            "offset": start * self.dtype.itemsize,
            "version": 3,
        }

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        out = kwargs.get("out", ())
        inputs = tuple(n.array(x, copy=False) if isinstance(x, Column) else x for x in args)

        if out:
            kwargs["out"] = tuple(n.array(x, copy=False) if isinstance(x, Column) else x for x in out)

        return getattr(ufunc, method)(*inputs, **kwargs)

    def __get_callable__(self, key):
        # Retrieves numpy methods
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
    def __getitem__(self, key: n.integer) -> Any:
        ...

    @overload
    def __getitem__(self, key: Any) -> nt.NDArray:  # Returns a copy
        ...

    def __getitem__(self, key: Union[slice, int, n.integer, Any]) -> Union["Column", nt.NDArray, Any]:
        """
        FIXME: Should return ndarray instead (just have to be careful about
        mutating string columns in compute code)
        """
        if isinstance(key, (int, n.integer)):
            return n.array(self, copy=False)[key]
        elif isinstance(key, slice):
            # Combine the slices to create a new subset (keeps underlying data)
            datalen = len(self._data) // self.dtype.itemsize
            r = range(datalen)[self._subset][key]
            return type(self)(self._data, self._field, subset=slice(r.start, r.stop, r.step))
        else:
            return n.copy(n.array(self, copy=False)[key])

    def __setitem__(self, key: Any, value: Any):
        n.array(self, copy=False)[key] = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}{repr(n.array(self, copy=False))[5:]}"


class Row(Mapping):
    """
    Provides row-by-row access to the datasert
    """

    pass


R = TypeVar("R", bound=Row)


class Spool(List[R], Generic[R]):
    """
    List-like database row accessor class with support for splitting and
    randomizing based on row fields
    """

    pass


class Dataset(MutableMapping, Generic[R]):
    """
    Accessor class for working with cryoSPARC .cs files.

    Example usage

    ```
    dset = Dataset.load('/path/to/particles.cs')

    for particle in dset.rows:
        print(f"Particle located in file {particle['blob/path']} at index {particle['blob/idx']}")
    ```

    """

    @classmethod
    def load(cls, file: Union[str, Path, BinaryIO]) -> "Dataset":
        """
        Read a dataset from disk from a path or file handle
        """
        pass

    @classmethod
    def save(cls, file: Union[str, Path, BinaryIO]):
        """
        Save a dataset to the given path or file handle
        """
        pass

    def __init__(
        self,
        *args: Union[Iterable[Tuple[str, Collection]], Dict[str, Collection]],
        size: int = 0,
        fields: Union[List[Field], n.dtype] = [],
        row_class: Type[R] = Row,
    ) -> None:
        # Always initialize with at least a UID field
        super().__init__()
        self._data = None
        self._rows = None  # Uninitialized row-by-row iterator
        self._row_class = row_class

    def __len__(self):
        """
        Returns the number of rows in this dataset
        """
        return 0

    def __iter__(self):
        """
        Iterate over the fields in this dataset
        """
        pass

    def __getitem__(self, key: str) -> Column:
        """
        Get either a specific field in the dataset, a single row or a set of
        rows. Note that Datasets are internally organized by columns so
        row-based operations are always slower.
        """
        pass

    def __setitem__(self, key: str, val: Union[Any, list, nt.NDArray, Column]):
        """
        Set or add a field to the dataset. If the field already exists, enforces
        that the data type is the same.
        """
        pass

    def __delitem__(self, key: str):
        """
        Removes field from the dataset
        """
        pass

    def __eq__(self, other):
        """
        Check that two datasets share the same fields and that those fields have
        the same values.
        """
        pass

    @property
    def rows(self) -> Spool[R]:
        """
        A row-by-row accessor list for items in this dataset
        """
        pass

    def to_list(self, exclude_uid: bool = False) -> List[dict]:
        return [row.to_list(exclude_uid) for row in self.rows]

    def copy(self, keep_fields: Optional[Iterable[str]] = None) -> "Dataset":
        """
        Create a copy; optionally specifying which fields to keep (UIDs are always kept)
        """
        pass

    def fields(self, exclude_uid=False) -> List[str]:
        """
        Retrieve a list of field names available in this dataset
        """
        pass

    def descr(self, exclude_uid=False) -> List[Field]:
        """
        Retrive the numpy-compatible description for dataset fields
        """
        pass

    @overload
    def add_fields(self, fields: List[Field]) -> "Dataset":
        ...

    @overload
    def add_fields(self, fields: List[str], dtypes: Union[str, List[Dtype]]) -> "Dataset":
        ...

    def add_fields(
        self, fields: Union[List[str], List[Field]], dtypes: Optional[Union[str, List[Dtype]]] = None
    ) -> "Dataset":
        """
        Ensures the dataset has the given fields.
        """
        pass

    def drop_fields(self, names: Iterable[str]):
        """
        Remove the given field names from the dataset.
        """
        pass

    def rename_fields(self, field_map: Union[Dict[str, str], Callable[[str], str]]):
        """
        Specify a mapping dictionary or function that specifies how to rename
        each field.
        """
        if isinstance(field_map, dict):
            field_map = lambda x: field_map.get(x, x)
        # FIXME: allocate
        return self

    def filter_fields(self, name_test: Union[List[str], Callable[[str], bool]]):
        """
        Create a new dataset with all fields except for the desired ones.
        Specify either a list of fields to keep or a function that returns True
        of the given field should be kept.
        """
        test = lambda n: n in name_test if isinstance(name_test, list) else name_test
        return self.drop_fields(f for f in self.fields() if f != "uid" and not test(f))
        # Return a new dataset with the desired fields

    def query(self, query: Union[dict, Callable[[Any], bool]]) -> "Dataset":
        """
        Get a subset of data based on whether the fields match the values in the
        given query. Values can be either a single value or a set of possible
        values. If any field is not in the dataset, it is ignored and all data
        is returned.

        Example query:

            dset.query({
                'uid': [123456789, 987654321]
                'micrograph_blob/path': '/path/to/exposure.mrc',
            })

        """
        pass

    def subset(self, indexes: Union[Iterable[int], Iterable[Row]]) -> "Dataset":
        """
        Get a subset of dataset that only includes the given indexes or list of rows
        """
        pass

    def mask(self, mask: Collection[bool]) -> "Dataset":
        """
        Get a subset of the dataset that matches the given mask of rows
        """
        assert len(mask) == len(
            self
        ), f"Mask with length {len(mask)} does not match expected dataset length {len(self)}"
        return self  # FIXME

    def range(self, start: int = 0, stop: int = -1) -> "Dataset":
        """
        Get at subset of the dataset with rows in the given range
        """
        pass
