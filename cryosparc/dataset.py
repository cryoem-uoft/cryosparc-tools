from functools import reduce
from pathlib import Path, PurePath
from textwrap import wrap
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
from .dtype import Field, dtype_field, ndarray_dtype

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
        return key in self.dataset

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

    def __init__(self, items: Iterable[R], rng: n.random.Generator = n.random.default_rng()):
        self.indexes = None
        self.random = rng
        self.extend(items)

    def set_random(self, rng: n.random.Generator):
        self.random = rng

    # -------------------------------------------------- Spooling and Splitting
    def split(self, num: int, random=True, prefix=None):
        """Return two SpoolingLists with the split portions"""
        if random:
            idxs = self.random.permutation(len(self))
        else:
            idxs = n.arange(len(self))
        d1 = Spool(items=(self[i] for i in idxs[:num]), rng=self.random)
        d2 = Spool(items=(self[i] for i in idxs[num:]), rng=self.random)
        if prefix is not None:
            field = prefix + "/split"
            for img in d1:
                img[field] = 0
            for img in d2:
                img[field] = 1
        return d1, d2

    def split_half_in_order(self, prefix: str, random=True):
        if random:
            splitvals = n.random.randint(2, size=len(self))
        else:
            splitvals = n.arange(len(self)) % 2
        for idx, p in enumerate(self):
            p[prefix + "/split"] = splitvals[idx]
        d1 = Spool(items=(p for p in self if p[prefix + "/split"] == 0), rng=self.random)
        d2 = Spool(items=(p for p in self if p[prefix + "/split"] == 1), rng=self.random)
        return d1, d2

    def split_into_quarter(self, num: int, seed: int):
        """Return two Spools with the split portions"""
        idxs = n.random.default_rng(seed=seed).permutation(len(self))
        d1 = Spool(items=(self[i] for i in idxs[:num]), rng=n.random.default_rng(seed=seed))
        d2 = Spool(items=(self[i] for i in idxs[num:]), rng=n.random.default_rng(seed=seed))
        return d1, d2

    def split_with_split(self, num: int, random=True, prefix=None, split=0):
        """Return two Spools with the split portions"""
        if random:
            idxs = self.random.permutation(len(self))
        else:
            idxs = n.arange(len(self))
        d1 = Spool(items=(self[i] for i in idxs[:num]), rng=self.random)
        d2 = Spool(items=(self[i] for i in idxs[num:]), rng=self.random)
        if prefix is not None:
            field = prefix + "/split"
            for img in d1:
                img[field] = split
            for img in d2:
                img[field] = split
        return d1, d2

    def split_by_splits(self, prefix="alignments"):
        """Return two Spools with the split portions"""
        idxs = n.arange(len(self))
        field = "split"
        if prefix is not None:
            field = prefix + "/split"
        d1 = Spool(items=(self[i] for i in idxs if self[i][field] == 0), rng=self.random)
        d2 = Spool(items=(self[i] for i in idxs if self[i][field] == 1), rng=self.random)
        return d1, d2

    def split_from_field(self, field: str, vals: Tuple[Any, Any] = (0, 1)):
        """split into two from pre recorded split in field, between given vals"""
        d1 = Spool((img for img in self if img[field] == vals[0]), rng=self.random)
        d2 = Spool((img for img in self if img[field] == vals[1]), rng=self.random)
        return d1, d2

    def split_by(self, field: str) -> Dict[Any, List[R]]:
        items = {}
        for item in self:
            val = item[field]
            curr = items.get(val, [])
            curr.append(item)
            items[val] = curr
        return items

    def get_random_subset(self, num: int):
        "Just a randomly selected subset, without replacement. Returns a list."
        assert num <= len(self), "Not Enough Images!"
        idxs = self.random.choice(len(self), size=num, replace=False)
        return [self[i] for i in idxs]

    def setup_spooling(self, random=True):
        """Setup indices and minibatches for spooling"""
        if random:
            self.indexes = self.random.permutation(len(self))
        else:
            self.indexes = n.arange(len(self))
        self.spool_index = 0

    def spool(self, num: int, peek: bool = False):
        """Return a list consisting of num randomly selected elements.
        Advance the spool.
        Return self.minibatch_size elements if num is None.
        if peek is true, don't advance the spool, just return the first num elements.
        """
        if self.indexes is None:
            self.setup_spooling()
        assert self.indexes is not None
        if num >= len(self):  # asking for too many
            return [self[i] for i in range(len(self))]  # just return self, no random order.
        current_indices = n.arange(self.spool_index, self.spool_index + num) % len(self.indexes)
        if not peek:
            self.spool_index = (self.spool_index + num) % len(self.indexes)
        return [self[self.indexes[i]] for i in current_indices]

    def make_batches(self, num: int = 200):
        """
        Return a list of lists, each one being a consecutive set of num images.
        """
        return [self[idx : idx + num] for idx in n.arange(0, len(self), num)]

    def __str__(self):
        s = f"{type(self).__name__} object with {len(self)} items."
        return s


class Dataset(MutableMapping[str, Column], Generic[R]):
    """
    Accessor class for working with cryoSPARC .cs files.

    Example usage

    ```
    dset = Dataset.load('/path/to/particles.cs')

    for particle in dset.rows:
        print(
            f"Particle located in file {particle['blob/path']} "
            f"at index {particle['blob/idx']}")
    ```

    A dataset may be initialized with `Dataset(data)` where `data` is
    one of the following:

    * A size of items to allocate (e.g., `42`)
    * A mapping from column names to their contents (dict or tuple list)
    * A numpy record array
    """

    @classmethod
    def allocate(cls, size: int = 0, fields: List[Field] = []):
        """
        Allocate a dataset with the given number of rows and specified fields.
        """
        dset = cls(size)
        dset.add_fields(fields)
        return dset

    @classmethod
    def append_many(
        cls,
        *datasets: "Dataset",
        assert_same_fields: bool = False,
        repeat_allowed: bool = False,
    ) -> "Dataset":
        """
        Concatenate many datasets together into one new one.

        Set `assert_same_fields=True` to enforce that datasets have identical
        fields. Otherwise, only takes fields common to all datasets.

        Set `repeat_allowed=True` to skip duplicate uid checks.
        """
        if not repeat_allowed:
            all_uids = n.concatenate([dset["uid"] for dset in datasets])
            assert len(all_uids) == len(n.unique(all_uids)), "Cannot append datasets that contain the same UIDs."

        if len(datasets) == 1:
            return datasets[0].copy()

        size = sum(len(d) for d in datasets)
        keep_fields = cls.common_fields(*datasets, assert_same_fields=assert_same_fields)
        result = cls.allocate(size, keep_fields)
        startidx = 0
        for dset in datasets:
            num = len(dset)
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
        """
        Take the row union of all the given datasets, based on their uid fields.

        Set `assert_same_fields=True` to enforce that datasets have identical
        fields. Otherwise, only takes fields common to all datasets.

        Set `assume_unique=True` to assume that each dataset's UIDs are unique
        (though they may be shared between datasets)
        """
        keep_fields = cls.common_fields(*datasets, assert_same_fields=assert_same_fields)
        keep_masks = []
        keep_uids = n.array([], dtype=n.uint64)
        for dset in datasets:
            mask = n.isin(dset["uid"], keep_uids, assume_unique=assume_unique, invert=True)
            if assume_unique:
                unique_uids = dset["uid"][mask]
            else:
                unique_uids, first_idxs = n.unique(dset["uid"], return_index=True)
                unique_mask = n.zeros(len(dset), dtype=bool)
                unique_mask[first_idxs] = True
                mask &= unique_mask

            keep_masks.append(mask)
            keep_uids = n.concatenate((keep_uids, unique_uids))

        size = sum(mask.sum() for mask in keep_masks)
        result = cls.allocate(size, keep_fields)
        startidx = 0
        for mask, dset in zip(keep_masks, datasets):
            num = mask.sum()
            for key, *_ in keep_fields:
                result[key][startidx : startidx + num] = dset[key][mask]
            startidx += num
        return result

    @classmethod
    def interlace(cls, *datasets: "Dataset", assert_same_fields: bool = False) -> "Dataset":
        if not datasets:
            return cls()

        assert all(
            len(dset) == len(datasets[0]) for dset in datasets
        ), "All datasets must be the same length to interlace."
        keep_fields = cls.common_fields(*datasets, assert_same_fields=assert_same_fields)
        all_uids = n.concatenate([dset["uid"] for dset in datasets])
        assert len(all_uids) == len(n.unique(all_uids)), "Cannot append datasets that contain the same UIDs."

        step = len(datasets)
        stride = len(datasets[0])
        startidx = 0
        result = cls.allocate(len(all_uids), keep_fields)
        for dset in datasets:
            for key, *_ in keep_fields:
                result[key][startidx : startidx + (stride * step) : step] = dset[key]
            startidx += 1

        return result

    @classmethod
    def innerjoin_many(cls, *datasets: "Dataset", assume_unique=False) -> "Dataset":
        if not datasets:
            return Dataset()

        if len(datasets) == 1:
            return datasets[0].copy()  # Only one to join, noop

        # Gather common fields
        all_fields: List[Field] = []
        fields_by_dataset: List[List[Field]] = []
        for dset in datasets:
            group: List[Field] = []
            for field in dset.descr:
                if field not in all_fields:
                    all_fields.append(field)
                    group.append(field)
            fields_by_dataset.append(group)
        assert len({f[0] for f in all_fields}) == len(
            all_fields
        ), "Cannot innerjoin datasets with fields of the same name but different types"

        # Get common UIDs
        uids = map(lambda d: d["uid"], datasets)
        intersect = lambda a1, a2: n.intersect1d(a1, a2, assume_unique=assume_unique)
        common_uids = reduce(intersect, uids)

        # Create a new dataset with just the UIDs from both datasets
        result = cls.allocate(len(common_uids), fields=all_fields)
        for dset, group in zip(datasets, fields_by_dataset):
            mask = n.isin(dset["uid"], common_uids, assume_unique=assume_unique)
            for key, *_ in group:
                result[key] = dset[key][mask]

        return result

    @classmethod
    def common_fields(cls, *datasets: "Dataset", assert_same_fields: bool = False) -> List[Field]:
        """
        Get a list of fields common to all given datasets. Specify
        `assert_same_fields=True` to enforce that all datasets have the same
        fields.
        """
        if not datasets:
            return []
        fields: Set[Field] = set.intersection(*(set(dset.descr) for dset in datasets))
        if assert_same_fields:
            for dset in datasets:
                assert len(dset.descr) == len(fields), (
                    "One or more datasets in this operation do not have the same fields. "
                    f"Common fields: {fields}. "
                    f"Excess fields: {set.difference(set(dset.descr), fields)}"
                )
        return [f for f in datasets[0].descr if f in fields]

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
        elif isinstance(allocate, n.ndarray):  # record array
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
                populate.append((dtype_field(f, ndarray_dtype(a)), a))

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

    def __eq__(self, other: "Dataset"):
        """
        Check that two datasets share the same fields in the same order and that
        those fields have the same values.
        """
        return (
            type(self) == type(other)
            and len(self) == len(other)
            and self.descr == other.descr
            and all(n.array_equal(c1, c2) for c1, c2 in zip(self.values(), other.values()))
        )

    @property
    def cols(self) -> OrderedDict[str, Column]:
        if self._cols is None:
            self._cols = OrderedDict()
            for field in self.descr:
                Col = StringColumn if n.dtype(field[1]) == n.dtype(n.object0) else NumericColumn
                self._cols[field[0]] = Col(self._data, field)
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

    def drop_fields(self, names: Union[Collection[str], Callable[[str], bool]]):
        """
        Remove the given fields from the dataset. Provide a list of fields or
        function that returns `True` if a given field name should be removed.
        """
        test = lambda n: n in names if isinstance(names, Collection) else names
        new_fields = [f for f in self.descr if f[0] == "uid" or not test(f[0])]
        if len(new_fields) == len(self.fields()):
            return self

        result = self.allocate(len(self), new_fields)
        for key, *_ in new_fields:
            result[key] = self[key]
        self._data = result._data
        self._cols = result._cols
        return self

    def rename_fields(self, field_map: Union[Dict[str, str], Callable[[str], str]]):
        """
        Specify a mapping dictionary or function that specifies how to rename
        each field.
        """
        if isinstance(field_map, dict):
            field_map = lambda x: field_map.get(x, x)
        result = Dataset([(f if f == "uid" else field_map(f), col) for f, col in self.cols.items()])
        self._data = result._data
        self._cols = None
        return self

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
        """Append the given dataset or datasets. Return a new dataset"""
        if len(others) == 0:
            return self
        indent = "\n    "
        assert self.descr == self.common_fields(*others), (
            f"Cannot append datasets with mismatched types.\n"
            f"Self:\n{indent}{self.descr}"
            f"Others:\n{indent}{indent.join(str(d.descr) for d in others)}"
        )
        return type(self).append_many(self, *others, repeat_allowed=repeat_allowed)

    def union(self, *others: "Dataset"):
        """
        Unite this dataset with the given others (uses the `uid` field to
        determine uniqueness). Returns a new dataset.
        """
        return type(self).union_many(self, *others)

    def innerjoin(self, *others: "Dataset", assert_no_drop=False, assume_unique=False):
        result = type(self).innerjoin_many(self, *others, assume_unique=assume_unique)
        if assert_no_drop:
            assert len(result) == len(self), "innerjoined datasets that do not have all elements in common."
        return result

    def query(self, query: Union[Dict[str, nt.ArrayLike], Callable[[R], bool]]) -> "Dataset":
        """
        Get a subset of data based on whether the fields match the values in the
        given query. They query is either a test function that gets called on
        each row or a key/value map of allowed field values.

        If any field is not in the dataset, it is ignored and all data is kept.

        Example query:

            dset.query({
                'uid': [123456789, 987654321],
                'micrograph_blob/path': '/path/to/exposure.mrc'
            })

        """
        if isinstance(query, dict):
            return self.mask(self.query_mask(query))
        else:
            indexes = [row.idx for row in self.rows if query(row)]
            return self.indexes(indexes)

    def query_mask(self, query: Dict[str, nt.ArrayLike], invert: bool = False) -> nt.NDArray[n.bool_]:
        """
        Get a boolean array representing the items to keep in the dataset that
        match the given query filter. See `query` method for example query
        format.
        """
        query_fields = set(self.fields()).intersection(query.keys())
        mask = n.ones(len(self), dtype=bool)
        for field in query_fields:
            mask &= n.isin(self[field], query[field])

        return n.invert(mask, out=mask) if invert else mask

    def subset(self, rows: Collection[Row]) -> "Dataset":
        """
        Get a subset of dataset that only includes the given list of rows (from
        this dataset)
        """
        return self.indexes([row.idx for row in rows])

    def indexes(self, indexes: Collection[int]):
        return Dataset([(f, col[indexes]) for f, col in self.cols.items()])

    def mask(self, mask: Collection[bool]) -> "Dataset":
        """
        Get a subset of the dataset that matches the given mask of rows
        """
        assert len(mask) == len(self), f"Mask with size {len(mask)} does not match expected dataset size {len(self)}"
        return Dataset([(f, col[mask]) for f, col in self.cols.items()])

    def slice(self, start: int = 0, stop: Optional[int] = None, step: int = 1) -> "Dataset":
        """
        Get at subset of the dataset with rows in the given range
        """
        return Dataset([(f, col[slice(start, stop, step)]) for f, col in self.cols.items()])

    def split_by(self, field: str) -> Dict[Any, "Dataset"]:
        """
        Create a mapping from possible values of the given field and to a
        datasets filtered by rows of that value.

        Example:

        ```
        dset = Dataset([
            ('uid', [1, 2, 3, 4]),
            ('foo', ['hello', 'world', 'hello', 'world'])
        ])
        assert dset.split_by('foo') == {
            'hello': Dataset([('uid', [1, 3]), ('foo', ['hello', 'hello'])]),
            'world': Dataset([('uid', [2, 4]), ('foo', ['world', 'world'])])
        }
        ```
        """
        idxs = {}
        for idx, val in enumerate(self[field]):
            curr = idxs.get(val, [])
            curr.append(idx)
            idxs[val] = curr
        return {k: self.indexes(v) for k, v in idxs.items()}

    def replace(self, query: Dict[str, nt.ArrayLike], *others: "Dataset", assume_unique=False):
        """
        Replaces values matching the given query with others. The query is a
        key/value map of allowed field values. The values can be either a single
        scalar value or a set of possible values. If nothing matches the query
        (e.g., {} specified), works the same way as append.
        """
        keep_fields = self.common_fields(self, *others, assert_same_fields=True)
        others_len = sum(len(o) for o in others)
        keep_mask = n.ones(len(self), dtype=bool)
        for other in others:
            keep_mask &= n.isin(self["uid"], other["uid"], assume_unique=assume_unique, invert=True)
        if query:
            keep_mask &= self.query_mask(query, invert=True)

        offset = keep_mask.sum()
        result = Dataset.allocate(offset + others_len, keep_fields)
        for key, col in self.items():
            result[key][:offset] = col[keep_mask]

        for other in others:
            other_len = len(other)
            for field, value in result.items():
                value[offset : offset + other_len] = other[field]
            offset += other_len

        return result

    def __repr__(self) -> str:
        s = f"{type(self).__name__}(["
        size = len(self)
        infix = ", "

        for k, v in self.items():
            if size > 6:
                contents = f"{infix.join(map(repr, v[:3]))}, ... , {infix.join(map(repr, v[-3:]))}"
            else:
                contents = infix.join(map(repr, v))
            s += "\n" + "\n".join(
                wrap(
                    f"('{k}', array([" + contents + f"], dtype={v.dtype})),",
                    width=100,
                    initial_indent="    ",
                    subsequent_indent=" " * 8,
                )
            )
        s += "\n])"
        return s
