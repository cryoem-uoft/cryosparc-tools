from pathlib import Path, PurePath
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
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
import numpy as n
import numpy.typing as nt
import numpy.core.records

from cryosparc.column import Column, NumericColumn, StringColumn
from cryosparc.util import ioopen

from .data import Data
from .dtype import Field, dtype_field

# Save format options
NUMPY_FORMAT = 1
CSDAT_FORMAT = 2
DEFAULT_FORMAT = NUMPY_FORMAT
NEWEST_FORMAT = CSDAT_FORMAT
FORMAT_MAGIC_PREFIXES = {
    NUMPY_FORMAT: b"\x93NUMPY",  # .npy file format
    CSDAT_FORMAT: b"\x94CSDAT",  # .csl binary format
}
MAGIC_PREFIX_FORMATS = {v: k for k, v in FORMAT_MAGIC_PREFIXES.items()}  # inverse dict


def generate_uids(num: int = 0):
    """
    Generate the given number of random 64-bit unsigned integer uids
    """
    return n.random.randint(0, 2**64, size=(num,), dtype=n.uint64)


class Row(Mapping):
    """
    Provides row-by-row access to the datasert
    """

    def __init__(self, dataset: "Dataset", idx: int):
        self.idx = idx
        self.dataset = dataset
        # note - don't keep around a ref to dataset.data because then when dataset.data changes (add field)
        # the already existing items will be referring to the old dataset.data!

    def __len__(self):
        return len(self.dataset.fields())

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

    A dataset may be initialized with `Dataset(data)` where `data` is
    one of the following:

    * A size of items to allocate (e.g., `42`)
    * A mapping from column names to their contents (dict or tuple list)
    * A numpy record array
    """

    @classmethod
    def allocate(cls, size: int = 0, fields: List[Field] = []):
        dset = cls(size)
        dset.add_fields(fields)
        return dset

    @classmethod
    def common_fields(cls, *datasets: "Dataset", assert_same_fields: bool = False) -> List[Field]:
        if not datasets:
            return []
        common_fields: Set[Field] = set.intersection(*(set(dset.descr) for dset in datasets))
        if assert_same_fields:
            for dset in datasets:
                assert len(dset.descr) == len(common_fields), (
                    "One or more datasets in this operation do not have the same fields. "
                    f"Common fields: {common_fields}. "
                    f"Excess fields: {set.difference(set(dset.descr), common_fields)}"
                )
        return [field for field in datasets[0].descr if field in common_fields]

    @classmethod
    def append_many(
        cls,
        *datasets: "Dataset",
        assert_same_fields: bool = False,
        assume_unique: bool = False,
        repeat_allowed: bool = False,
    ) -> "Dataset":
        if not datasets:
            return Dataset()  # empty

        first, *rest = datasets
        if not repeat_allowed:
            all_uids = first["uid"]
            for dset in rest:
                all_uids = n.concatenate((all_uids, dset["uid"]))

            assert len(all_uids) == len(n.unique(all_uids)), "Cannot append datasets that contain the same UIDs."

        size = sum(len(d) for d in datasets)
        keep_fields = cls.common_fields(*datasets, assert_same_fields=assert_same_fields)
        result = Dataset.allocate(size, keep_fields)
        startidx = 0
        for dset in datasets:
            num = len(dset)
            if num == 0:
                continue
            for key, *_ in keep_fields:
                result[key][startidx : startidx + num] = dset[key]
            startidx += num

        return result

    @classmethod
    def union_many(
        cls,
        *datasets: "Dataset",
        assert_same_fields: bool = False,
        assume_unique: bool = False,
    ) -> "Dataset":
        if not datasets:
            return Dataset()

        keep_fields = cls.common_fields(*datasets, assert_same_fields=assert_same_fields)
        first, *rest = datasets
        if assume_unique:
            keep_uids = first["uid"]
            keep_masks = [n.ones(len(first), dtype=bool)]
        else:
            keep_uids, first_idxs = n.unique(first["uid"], return_index=True)
            keep_masks = [n.zeros(len(first), dtype=bool)]
            keep_masks[0][first_idxs] = True

        for dset in rest:
            mask = n.in1d(dset["uid"], keep_uids, assume_unique=assume_unique, invert=True)
            keep_masks.append(mask)
            keep_uids = n.concatenate((keep_uids, dset["uid"][mask]))

        size = sum(mask.sum() for mask in keep_masks)
        result = Dataset.allocate(size, keep_fields)
        startidx = 0
        for mask, dset in zip(keep_masks, datasets):
            num = mask.sum()
            for field in keep_fields:
                key = field[0]
                result[key][startidx : startidx + num] = dset[key][mask]
            startidx += num
        return result

    @classmethod
    def interlace(cls, *datasets: "Dataset") -> "Dataset":
        return NotImplemented

    @classmethod
    def load(cls, file: Union[str, PurePath, BinaryIO]) -> "Dataset":
        """
        Read a dataset from disk from a path or file handle
        """
        with ioopen(file, "rb") as f:
            prefix = f.read(6)
            f.seek(0)
            if prefix == FORMAT_MAGIC_PREFIXES[NUMPY_FORMAT]:
                indata = n.load(f, allow_pickle=False)
                return Dataset(indata)
            else:
                return NotImplemented

    def save(self, file: Union[str, Path, BinaryIO], format: int = DEFAULT_FORMAT):
        """
        Save a dataset to the given path or file handle
        """
        if format == NUMPY_FORMAT:
            arrays = [col.to_numpy(copy=False, fixed=True) for col in self.cols.values()]
            dtype = [(f, a.dtype) for f, a in zip(self.cols, arrays)]
            outdata = numpy.core.records.fromarrays(arrays, dtype=dtype)
            with ioopen(file, "wb") as f:
                n.save(f, outdata, allow_pickle=False)
        else:
            return NotImplemented

    def __init__(
        self,
        allocate: Union[
            "Dataset",
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

        if isinstance(allocate, Dataset):
            # Create copy of underlying data
            self._data = allocate._data.copy()
            return

        if isinstance(allocate, Data):
            # Create from existing data (no copy)
            self._data = allocate
            return

        self._data = Data()
        populate: List[Tuple[Field, n.ndarray]] = []
        if isinstance(allocate, (int, n.integer)):
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
        Iterate over the fields in this dataset hello world
        """
        return self.cols.__iter__()

    def __getitem__(self, key: str) -> Column:
        """
        Get either a specific field in the dataset, a single row or a set of
        rows. Note that Datasets are internally organized by columns so
        row-based operations are always slower.
        """
        return self.cols[key]

    def __setitem__(self, key: str, val: Any):
        """
        Set or add a field to the dataset.
        """
        if key not in self._data:
            aval = n.array(val, copy=False)
            assert not aval.shape or aval.shape[0] == len(
                self
            ), f"Cannot broadcast '{key}' in {self} to {val} due to invalid shape"
            self.add_fields([key], [aval.dtype])
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

    def copy(self):
        return type(self)(allocate=self)

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

    def copy_fields(self, old_fields: List[str], new_fields: List[str]):
        assert len(old_fields) == len(new_fields), "Number of old and new fields must match"
        for old, new in zip(old_fields, new_fields):
            self._data.addcol(new, self._data[old])

        self._cols = None
        for old, new in zip(old_fields, new_fields):
            self[new] = self[old]

        return None

    def reassign_uids(self):
        self["uid"] = generate_uids(len(self))
        return self

    def to_list(self, exclude_uid: bool = False) -> List[list]:
        return [row.to_list(exclude_uid) for row in self.rows]

    def append(self, *others: "Dataset", repeat_allowed: bool = False):
        """Append the given dataset or datasets. Mutates the contents of this
        dataset"""
        if len(others) == 0:
            return self
        dset = type(self).append_many(self, *others, repeat_allowed=repeat_allowed)
        self._data = dset._data
        self._cols = None
        return self

    def union(self, *others: "Dataset"):
        """
        Unite this dataset with the given others (uses the `uid` field to
        determine uniqueness). Returns a new dataset.
        """
        return type(self).union_many(self, *others)

    def query(self, query: Union[dict, Callable[[Any], bool]]) -> "Dataset":
        """
        Get a subset of data based on whether the fields match the values in the
        given query. They query is either a test function that gets called on
        each row or a key/value map of allowed field values. Values can be
        either a single value or a set of possible values. If any field is not
        in the dataset, it is ignored and all data is returned.

        Example query:

            dset.query({
                'uid': [123456789, 987654321],
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

    def replace(self, query: dict, *others: "Dataset", assume_unique: bool = False):
        """
        Replaces values matching the given query with others. The query is a
        key/value map of allowed field values. The values can be either a single
        scalar value or a set of possible values. If nothing matches the query
        (e.g., {} specified), works the same way as append.
        """
        pass
