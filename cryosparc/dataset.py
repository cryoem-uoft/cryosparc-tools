from pathlib import Path
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


class Data:
    def __init__(self, size: int = 0, fields: Union[List[Field], n.dtype] = []):
        """
        Allocated dataset memory in C based on given fields
        """
        # self.handle = allocate_dataset()  # FIXME
        pass

    def __del__(self):
        """
        Deallocate C-based memory when data is garbage collected
        """
        # deallocate_dataset(self.handle)  # FIXME
        pass


class Column(Sequence):
    """
    Dataset column entry
    """

    def __init__(self, data, field: str, start: int = 0, stop: int = -1):
        # dlen = len(data[field])  # FIXME: Get the data from the given field somehow
        dlen = data
        assert -dlen <= start and start < dlen, f"Invalid start index {start}"
        assert -dlen <= stop and stop <= dlen, f"Invalid stop index {stop}"
        self.data = data
        self.field = field
        self.start = start + dlen if start < 0 else start
        self.stop = stop + dlen + 1 if stop < 0 else stop
        assert self.start <= self.stop, "Cannot initialize a column in reverse order"

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self):
        for i in range(self.start, self.stop):
            yield i

    @overload
    def __getitem__(self, key: int) -> Any:
        ...

    @overload
    def __getitem__(self, key: Union[slice, Sequence[int], Sequence[bool]]) -> nt.NDArray:
        ...

    def __getitem__(self, key: Union[int, slice, Sequence[int], Sequence[bool]]) -> Union[nt.NDArray, Any]:
        """
        FIXME: Should return ndarray instead (just have to be careful about
        mutating string columns in compute code)
        """
        clen = len(self)
        if isinstance(key, slice):
            """ """
            assert key.step is None, "Column does not support slices with step"
            start = key.start or 0
            stop = key.stop or clen
            assert -clen <= start and start < clen, f"Invalid start index {start}"
            assert -clen <= stop and stop <= clen, f"Invalid stop index {stop}"
            start = start + clen if start < 0 else start
            stop = stop + clen + 1 if stop < 0 else stop
            return self.__class__(self.data, self.field, start=self.start + start, stop=self.start + stop)
        elif isinstance(key, Sequence):
            pass
            if len(key) == 0:
                return n.array([])  # FIXME: Include dtype
            elif isinstance(key[0], bool):
                assert len(key) == clen, f"Invalid mask length ({len(key)}) does not match "
        else:
            assert -clen <= key and key < clen, f"Invalid col index {key} for column with length {clen}"
            return self.start + key % clen

    def __setitem__(self, key: Union[int, slice, Sequence[bool]], value: Any):
        print(key, value)
        # assert isinstance(key, int) and key >= 0 and key < len(self), f"Invalid column index {key}, must be in range [0, {len(self)}))"

    def slice(self, start: int = 0, stop: int = -1) -> Column:
        """
        Get a view into this column with the given start/spot indeces
        """
        pass

    def to_numpy(self, copy=False):
        """
        Return the underlying data as a numpy ndarray. Generally faster than
        converting via iteration because it uses the underlying bytes directly
        (except for cases where column items are strings)

        Underlying data is read-only by default, unless copy is set to `True`.
        """
        pass


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

    def __getitem__(self, key: str) -> nt.NDArray:
        """
        Get either a specific field in the dataset, a single row or a set of
        rows. Note that Datasets are internally organized by columns so
        row-based operations are always slower.
        """
        pass

    def __setitem__(self, key: str, val: Union[Any, list, n.ndarray]):
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
        self,
        fields: Union[List[str], List[Field]],
        dtypes: Optional[Union[str, List[Dtype]]] = None
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

            dset.subset_query({
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
