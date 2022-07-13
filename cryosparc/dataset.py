from abc import ABC
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
    OrderedDict,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
import numpy as n
import numpy.typing as nt

from .data import Data
from .dtype import Field, dtype_field, field_shape, field_strides

# Save format options
NUMPY_PROTOCOL = 0
CSDAT_PROTOCOL = 1
HIGHEST_PROTOCOL = CSDAT_PROTOCOL


def generate_uids(num: int = 0):
    """
    Generate the given number of random 64-bit unsigned integer uids
    """
    return n.random.randint(0, 2**64, size=(num,), dtype=n.uint64)


class Column(Sequence, n.lib.mixins.NDArrayOperatorsMixin, ABC):
    """
    Dataset Column entry. May be used anywhere in place of Numpy arrays
    """

    def __init__(self, data: Data, field: Field, subset: slice = slice(0, None)):
        # dlen = len(data[field])  # FIXME: Get the data from the given field somehow
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
            # Combine the given slices and current self._subset slice to create
            # a new subset (keeps underlying data)
            datalen = self._data.nrow()
            r = range(datalen)[self._subset][key]
            print(f"SLICE {key} RANGE {r} SUBSET {self._subset}")
            return type(self)(self._data, self._field, subset=slice(r.start, r.stop, r.step))
        else:
            # Indeces or mask, get a deep copy of underlying data subset
            return n.copy(n.array(self, copy=False)[key])

    def __setitem__(self, key: Any, value: Any):
        n.array(self, copy=False)[key] = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}{repr(n.array(self, copy=False))[5:]}"


class NumericColumn(Column):
    def __init__(self, data: Data, field: Field, subset: slice = slice(0, None)):
        super().__init__(data, field, subset)
        self._strides = field_strides(field, self._subset.step or 1)

    @property
    def __array_interface__(self):
        """
        Allows Column instances to be used as numpy arrays. The abstract integer
        pointer value of the 'data' key may be dynamic because the underlying
        dataset gets re-computed
        """
        return {
            "data": (self._data.get(self.field[0]), False),
            "shape": self.shape,
            "typestr": self.dtype.str,
            "strides": self._strides,
            "version": 3,
        }


class StringColumn(Column):
    def __init__(self, data: Data, field: Field, subset: slice = slice(0, None)):
        assert len(field) == 2 and n.dtype(field[1]) == n.dtype(
            n.object0
        ), f"Cannot create a String column with dtype {field[1]}"
        super().__init__(data, field, subset)
        # Available string indexes in this dataset
        self._idxs = n.array(list(range(*self._subset.indices(self._data.nrow()))))

    def __array__(self):
        return n.array(
            [self._data.getstr(self.field[0], i) for i in range(*self._subset.indices(self._data.nrow()))],
            dtype=n.object0,
        )

    def __getitem__(self, key: Union[slice, int, n.integer, Any]) -> Union["Column", nt.NDArray, Any]:
        if isinstance(key, slice):
            return super().__getitem__(key)  # Return string column subset
        elif isinstance(key, (int, n.integer)):
            return self._data.getstr(self.field[0], self._idxs[key])
        elif isinstance(key, Collection):  # mask or index list
            idxs = self._idxs[n.array(key, copy=False)]
            return n.array([self._data.getstr(self.field[0], i) for i in idxs], dtype=n.object0)

        raise TypeError(f"Invalid index into StringColumn: {key}")

    def __setitem__(self, key: Any, value: Union[str, Collection[str]]):
        if isinstance(key, (int, n.integer)):
            idxs = [self._idxs[key]]
        else:
            idxs = self._idxs[n.array(key, copy=False)]

        f = self.field[0]
        if isinstance(value, str):
            for i in idxs:
                self._data.setstr(f, i, value)
        else:
            assert len(value) == len(idxs), f"Cannot assign [{key}] for {type(self).__name__} to {value}"
            for i, v in zip(idxs, value):
                self._data.setstr(f, i, v)


class Row(Mapping):
    """
    Provides row-by-row access to the datasert
    """

    def __init__(self, dataset: "Dataset", idx: int):
        self.idx = idx
        self.dataset = dataset
        # note - don't keep around a ref to dataset.data because then when dataset.data changes (add field)
        # the already existing items will be referring to the old dataset.data!

    def __getitem__(self, key: str):
        return self.dataset[key][self.idx]

    def __setitem__(self, key: str, value):
        self.dataset[key][self.idx] = value

    def __contains__(self, key: str):
        return key in self.dataset.fields()

    def __iter__(self):
        return iter(self.dataset.fields())

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def get_item(self, key, default=None):
        return self.dataset[key][self.idx] if key in self else default

    def to_list(self, exclude_uid=False):
        """Convert into a list of native python types, ordered the same way as the fields"""
        return [self.get_item(key) for key in self.dataset.fields() if not exclude_uid or key != "uid"]

    def to_dict(self):
        return {key: self[key] for key in self.dataset.fields()}

    def to_item_dict(self):
        """Like to_dict, but all values are native python types"""
        return {key: self.get_item(key) for key in self.dataset.fields()}

    def from_dict(self, d):
        for k in self.dataset.fields():
            self[k] = d[k]


R = TypeVar("R", bound=Row)


class Spool(List[R], Generic[R]):
    """
    List-like database row accessor class with support for splitting and
    randomizing based on row fields
    """

    pass


class Dataset(MutableMapping[str, Column], Generic[R]):
    """
    Accessor class for working with cryoSPARC .cs files.

    Example usage

    ``` dset = Dataset.load('/path/to/particles.cs')

    for particle in dset.rows:
        print(f"Particle located in file {particle['blob/path']} at index
        {particle['blob/idx']}")
    ```

    A dataset may be initialized with `Dataset(allocate)` where `allocate` is
    one of the following:

    * A size of items to allocate (e.g., `42`)
    * A mapping from column names to their contents (dict or tuple list)
    * A numpy record array
    """

    @classmethod
    def load(cls, file: Union[str, Path, BinaryIO]) -> "Dataset":
        """
        Read a dataset from disk from a path or file handle
        """
        return NotImplemented

    @classmethod
    def save(cls, file: Union[str, Path, BinaryIO], protocol: int = NUMPY_PROTOCOL):
        """
        Save a dataset to the given path or file handle
        """
        return NotImplemented

    @classmethod
    def allocate(cls, size: int = 0, fields: List[Field] = []):
        dset = cls(size)
        dset.add_fields(fields)
        return dset

    def __init__(
        self,
        allocate: Union[
            Data,
            int,
            n.ndarray,
            Mapping[str, nt.ArrayLike],
            List[Tuple[str, nt.ArrayLike]],
        ] = 0,
        row_class: Type[R] = Row,
    ) -> None:
        # Always initialize with at least a UID field
        super().__init__()
        self._row_class = row_class
        self._cols = None
        self._rows = None

        if isinstance(allocate, Data):
            self._data = allocate
            return

        self._data = Data()
        populate: List[Tuple[Field, n.ndarray]] = []
        if isinstance(allocate, int):
            populate = [(("uid", "<u8"), generate_uids(allocate))]
        elif isinstance(allocate, n.ndarray):
            for field in allocate.dtype.descr:
                assert field[0], f"Cannot initialize with record array of dtype {allocate.dtype}"
                populate.append((field, allocate[field[0]]))
        elif isinstance(allocate, Mapping):
            for f, v in allocate.items():
                a = n.array(v, copy=False)
                populate.append((dtype_field(f, a.dtype), a))
        else:
            for f, v in allocate:
                a = n.array(v, copy=False)
                populate.append((dtype_field(f, a.dtype), a))

        # Check that all entries are the same length
        nrows = 0
        if populate:
            nrows = len(populate[0][1])
            assert all(
                len(entry[1]) == nrows for entry in populate
            ), "Target populate data does not all have the same length"

        # Add UID field at the beginning, if required
        if not any(entry[0][0] == "uid" for entry in populate):
            populate.insert(0, (("uid", "<u8"), generate_uids(nrows)))

        self.add_fields([entry[0] for entry in populate])
        self._data.addrows(nrows)
        for field, data in populate:
            self[field[0]] = data

    def __len__(self):
        """
        Returns the number of rows in this dataset
        """
        return self._data.nrow()

    def __iter__(self):
        """
        Iterate over the fields in this dataset
        """
        return self.cols.__iter__()

    def __getitem__(self, key: str) -> Column:
        """
        Get either a specific field in the dataset, a single row or a set of
        rows. Note that Datasets are internally organized by columns so
        row-based operations are always slower.
        """
        return self.cols[key]

    def __setitem__(self, key: str, val: nt.ArrayLike):
        """
        Set or add a field to the dataset.
        """
        if key not in self._data:
            val = n.array(val, copy=False)
            assert len(self) == len(val), (
                f"Cannot set '{key}' in {self} to {val}. " f"Expected length {len(self)}, actual {len(val)}"
            )
            self.add_fields([key], [val.dtype])
        self.cols[key][:] = val

    def __delitem__(self, key: str):
        """
        Removes field from the dataset
        """
        self.drop_fields([key])

    def __eq__(self, other):
        """
        Check that two datasets share the same fields and that those fields have
        the same values.
        """
        return self.cols == other.cols

    @property
    def cols(self) -> OrderedDict[str, Column]:
        if self._cols is None:
            self._cols = OrderedDict()
            for f, dt in self._data.items():
                Col = StringColumn if n.dtype(dt) == n.dtype(n.object0) else NumericColumn
                self._cols[f] = Col(self._data, dtype_field(f, dt))
        return self._cols

    @property
    def rows(self) -> Spool[R]:
        """
        A row-by-row accessor list for items in this dataset
        """
        if self._rows is None:
            self._rows = Spool(self._row_class(self, idx) for idx in range(len(self)))
        return self._rows

    @property
    def descr(self, exclude_uid=False) -> List[Field]:
        """
        Retrive the numpy-compatible description for dataset fields
        """
        return [dtype_field(f, dt) for f, dt in self._data.items()]

    def to_list(self, exclude_uid: bool = False) -> List[list]:
        return [row.to_list(exclude_uid) for row in self.rows]

    def copy(self):
        """
        Create a copy; optionally specifying which fields to keep (UIDs are always kept)
        """
        return type(self)(allocate=self._data.copy())

    def fields(self, exclude_uid=False) -> List[str]:
        """
        Retrieve a list of field names available in this dataset
        """
        return [k for k in self._data.keys() if not exclude_uid or k != "uid"]

    @overload
    def add_fields(self, fields: List[Field]) -> "Dataset":
        ...

    @overload
    def add_fields(self, fields: List[str], dtypes: Union[str, List[nt.DTypeLike]]) -> "Dataset":
        ...

    def add_fields(
        self,
        fields: Union[List[str], List[Field]],
        dtypes: Optional[Union[str, List[nt.DTypeLike]]] = None,
    ) -> "Dataset":
        """
        Ensures the dataset has the given fields.
        """
        if len(fields) == 0:
            return self  # noop

        desc: List[Field] = []
        if dtypes:
            dt = dtypes.split(",") if isinstance(dtypes, str) else dtypes
            assert len(fields) == len(dt), "Incorrect dtype spec"
            desc = [dtype_field(str(f), dt) for f, dt in zip(fields, dt)]
        else:
            desc = fields  # type: ignore

        for field in desc:
            if field[0] not in self._data:
                self._data.addcol(field)

        self._cols = None
        return self

    def drop_fields(self, names: Collection[str]):
        """
        Remove the given field names from the dataset.
        """
        new_fields = [dtype_field(f, d) for f, d in self._data.items() if f == "uid" or f not in names]
        newdata = Dataset.allocate(len(self), new_fields)
        for field in self.fields():
            if field not in names:
                newdata[field] = self[field]
        self._data = newdata._data
        self._cols = None
        return self

    def rename_fields(self, field_map: Union[Dict[str, str], Callable[[str], str]]):
        """
        Specify a mapping dictionary or function that specifies how to rename
        each field.
        """
        if isinstance(field_map, dict):
            field_map = lambda x: field_map.get(x, x)
        newdset = Dataset([(field_map(f), col) for f, col in self.cols.items()])
        self._data = newdset._data
        return self

    def filter_fields(self, name_test: Union[List[str], Callable[[str], bool]]):
        """
        Create a new dataset with all fields except for the desired ones.
        Specify either a list of fields to keep or a function that returns True
        of the given field should be kept.
        """
        test = lambda n: n in name_test if isinstance(name_test, list) else name_test
        return self.drop_fields([f for f in self.fields() if f != "uid" and not test(f)])

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
        return NotImplemented

    def subset(self, indexes: Union[Iterable[int], Iterable[Row]]) -> "Dataset":
        """
        Get a subset of dataset that only includes the given indexes or list of rows
        """
        return NotImplemented

    def mask(self, mask: Collection[bool]) -> "Dataset":
        """
        Get a subset of the dataset that matches the given mask of rows
        """
        assert len(mask) == len(self), f"Mask with size {len(mask)} does not match expected dataset size {len(self)}"
        return self  # FIXME

    def range(self, start: int = 0, stop: int = -1) -> "Dataset":
        """
        Get at subset of the dataset with rows in the given range
        """
        return NotImplemented
