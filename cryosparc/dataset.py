from functools import reduce
from pathlib import PurePath
from typing import (
    IO,
    Any,
    Union,
    Callable,
    Collection,
    Dict,
    Generic,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)
import numpy as n
import numpy.typing as nt
import numpy.core.records
import snappy

from .data import Data
from .dtype import Field, dtype_field, field_dtype, array_dtype
from .column import Column
from .row import Row, Spool, R
from .util import bopen, hashcache, u32bytesle, u32intle

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


class Dataset(MutableMapping[str, Column], Generic[R]):
    """
    Accessor class for working with cryoSPARC .cs files.

    Example usage

    ```
    dset = Dataset.load('/path/to/particles.cs')

    for particle in dset.rows():
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
        assert_same_fields=False,
        repeat_allowed=False,
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
        assert_same_fields=False,
        assume_unique=False,
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
            uid = dset["uid"]
            mask = n.isin(uid, keep_uids, assume_unique=assume_unique, invert=True)
            if assume_unique:
                unique_uids = uid[mask]
            else:
                unique_uids, first_idxs = n.unique(uid, return_index=True)
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
    def interlace(cls, *datasets: "Dataset", assert_same_fields=False) -> "Dataset":
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
            for field in dset.descr():
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
    def common_fields(cls, *datasets: "Dataset", assert_same_fields=False) -> List[Field]:
        """
        Get a list of fields common to all given datasets. Specify
        `assert_same_fields=True` to enforce that all datasets have the same
        fields.
        """
        if not datasets:
            return []
        fields: Set[Field] = set.intersection(*(set(dset.descr()) for dset in datasets))
        if assert_same_fields:
            for dset in datasets:
                assert len(dset.descr()) == len(fields), (
                    "One or more datasets in this operation do not have the same fields. "
                    f"Common fields: {fields}. "
                    f"Excess fields: {set.difference(set(dset.descr()), fields)}"
                )
        return [f for f in datasets[0].descr() if f in fields]

    @classmethod
    def load(cls, file: Union[str, PurePath, IO[bytes]]) -> "Dataset":
        """
        Read a dataset from disk from a path or file handle
        """
        prefix = None
        with bopen(file, "rb") as f:
            prefix = f.read(6)
            if prefix == FORMAT_MAGIC_PREFIXES[NUMPY_FORMAT]:
                f.seek(0)
                indata = n.load(f, allow_pickle=False)
                return Dataset(indata)
            elif prefix == FORMAT_MAGIC_PREFIXES[CSDAT_FORMAT]:
                headersize = u32intle(f.read(4))
                header = f.read(headersize).decode()
                headerparts = [h.split(" ") for h in header.split("\n")]
                dtype = [(f, t, tuple(map(int, s[0].split(",")))) if s else (f, t) for f, t, *s in headerparts]
                cols = {}
                for field in dtype:
                    colsize = u32intle(f.read(4))
                    buffer = snappy.uncompress(f.read(colsize))
                    cols[field[0]] = n.frombuffer(buffer, dtype=field_dtype(field))
                return Dataset(cols)

        raise TypeError(f"Could not determine dataset format for file {file} (prefix is {prefix})")

    def save(self, file: Union[str, PurePath, IO[bytes]], format: int = DEFAULT_FORMAT):
        """
        Save a dataset to the given path or I/O buffer.

        By default, saves as a numpy record array in the .npy format. Specify
        `format=CSDAT_FORMAT` to save in the latest .cs file format which is
        faster and results in a smaller file size but is not numpy-compatible.
        """
        if format == NUMPY_FORMAT:
            cols = self.cols()
            arrays = [col.to_fixed() for col in cols.values()]
            dtype = [(f, array_dtype(a)) for f, a in zip(cols, arrays)]
            outdata = numpy.core.records.fromarrays(arrays, dtype=dtype)
            with bopen(file, "wb") as f:
                n.save(f, outdata, allow_pickle=False)
        elif format == CSDAT_FORMAT:
            with bopen(file, "wb") as f:
                for chunk in self.stream():
                    f.write(chunk)
        else:
            raise TypeError(f"Invalid dataset save format for {file}: {format}")

    def stream(self):
        """
        Generate a binary representation for this dataset. Results may be
        written to a file or buffer to be sent over the network.

        Buffer will have the same format as Dataset files saved with
        `format=CSDAT_FORMAT`. Call `Dataset.load` on the resulting file/buffer
        to retrieve the original data.
        """
        cols = self.cols()
        arrays = [col.to_fixed() for col in cols.values()]
        descr = [dtype_field(f, array_dtype(a)) for f, a in zip(cols, arrays)]

        yield FORMAT_MAGIC_PREFIXES[CSDAT_FORMAT]

        header = "\n".join(f'{f} {t} {",".join(map(str, s[0]))}' if s else f"{f} {t}" for f, t, *s in descr)
        header = header.encode()
        yield u32bytesle(len(header))
        yield header

        for arr in arrays:
            compressed: bytes = snappy.compress(arr.data)
            yield u32bytesle(len(compressed))
            yield compressed

    def __init__(
        self,
        allocate: Union[
            "Dataset",
            int,
            nt.NDArray,
            Mapping[str, nt.ArrayLike],
            List[Tuple[str, nt.ArrayLike]],
        ] = 0,
        row_class: Type[R] = Row,
    ) -> None:
        # Always initialize with at least a UID field
        super().__init__()
        self._row_class = row_class
        self._rows = None

        if isinstance(allocate, Dataset):
            # Create copy of underlying data
            self._data = allocate._data.copy()
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
                populate.append((dtype_field(f, array_dtype(a)), a))
        else:
            for f, v in allocate:
                a = n.array(v, copy=False)
                populate.append((dtype_field(f, array_dtype(a)), a))

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
        return self._data.__iter__()

    def __getitem__(self, key: str) -> Column:
        """
        Get either a specific field in the dataset.
        """
        return Column(dtype_field(key, self._data[key]), self._data)

    def __setitem__(self, key: str, val: Any):
        """
        Set or add a field to the dataset.
        """
        assert key in self._data, f"Cannot set non-existing dataset key {key}; use add_fields() first"
        if isinstance(val, n.ndarray):
            if val.dtype.char == "S":
                cache = hashcache.init(bytes.decode)
                val = n.vectorize(cache.f, otypes="O")(val)
            elif val.dtype.char == "U":
                cache = hashcache.init(str)
                val = n.vectorize(cache.f, otypes="O")(val)
        self[key][:] = val

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
            and self.descr() == other.descr()
            and all(n.array_equal(c1, c2) for c1, c2 in zip(self.values(), other.values()))
        )

    def cols(self) -> Dict[str, Column]:
        return {field[0]: Column(field, self._data) for field in self.descr()}

    def rows(self) -> Spool:
        """
        A row-by-row accessor list for items in this dataset. Note: Do not store
        this accessor outside of this instance for a long time, the values
        become invalid when fields are added or the dataset's contents change.

        e.g., do not do this:

        ```
        dset = Dataset.load('/path/to/dataset.cs')
        rows = dset.rows()
        dset.add_fields([('foo', 'f4')])
        rows[0].to_list()  # access may be invalid
        ```
        """
        if self._rows is None:
            cols = self.cols()
            self._rows = Spool([self._row_class(cols, idx) for idx in range(len(self))])
        return self._rows

    def descr(self, exclude_uid=False) -> List[Field]:
        """
        Retrive the numpy-compatible description for dataset fields
        """
        return [dtype_field(f, dt) for f, dt in self._data.items() if not exclude_uid or f != "uid"]

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

        self._rows = None
        return self

    def filter_fields(self, names: Union[Collection[str], Callable[[str], bool]]):
        """
        Remove the given fields from the dataset. Provide a list of fields or
        function that returns `True` if a given field name should be removed.
        """
        test = (lambda n: n in names) if isinstance(names, Collection) else names
        new_fields = [f for f in self.descr() if f[0] == "uid" or test(f[0])]
        if len(new_fields) == len(self.descr()):
            return self

        result = self.allocate(len(self), new_fields)
        for key, *_ in new_fields:
            result[key] = self[key]
        self._data = result._data
        self._rows = None
        return self

    def filter_prefixes(self, prefixes: Collection[str]):
        return self.filter_fields(lambda n: any(n.startswith(p + "/") for p in prefixes))

    def drop_fields(self, names: Union[Collection[str], Callable[[str], bool]]):
        test = (lambda n: n not in names) if isinstance(names, Collection) else (lambda n: not names(n))
        return self.filter_fields(test)

    def rename_fields(self, field_map: Union[Dict[str, str], Callable[[str], str]]):
        """
        Specify a mapping dictionary or function that specifies how to rename
        each field.
        """
        if isinstance(field_map, dict):
            field_map = lambda x: field_map.get(x, x)
        result = Dataset([(f if f == "uid" else field_map(f), col) for f, col in self.items()])
        self._data = result._data
        self._rows = None
        return self

    def copy_fields(self, old_fields: List[str], new_fields: List[str]):
        assert len(old_fields) == len(new_fields), "Number of old and new fields must match"
        current_fields = self.fields()
        missing_fields = [
            dtype_field(new, self._data[old]) for old, new in zip(old_fields, new_fields) if new not in current_fields
        ]
        if missing_fields:
            self.add_fields(missing_fields)
        for old, new in zip(old_fields, new_fields):
            self[new] = self[old]

        self._rows = None
        return self

    def reassign_uids(self):
        self["uid"] = generate_uids(len(self))
        return self

    def to_list(self, exclude_uid=False) -> List[list]:
        return [row.to_list(exclude_uid) for row in self.rows()]

    def append(self, *others: "Dataset", repeat_allowed=False):
        """Append the given dataset or datasets. Return a new dataset"""
        if len(others) == 0:
            return self
        indent = "\n    "
        assert self.descr() == self.common_fields(*others), (
            f"Cannot append datasets with mismatched types.\n"
            f"Self:\n{indent}{self.descr()}"
            f"Others:\n{indent}{indent.join(str(d.descr()) for d in others)}"
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
            assert len(result) == len(self), "innerjoin datasets that do not have all elements in common."
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
            mask = [query(row) for row in self.rows()]
            return self.mask(mask)

    def query_mask(self, query: Dict[str, nt.ArrayLike], invert=False) -> nt.NDArray[n.bool_]:
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

    def indexes(self, indexes: Union[List[int], nt.NDArray]):
        return Dataset([(f, col[indexes]) for f, col in self.items()])

    def mask(self, mask: Union[List[bool], nt.NDArray]) -> "Dataset":
        """
        Get a subset of the dataset that matches the given mask of rows
        """
        assert len(mask) == len(self), f"Mask with size {len(mask)} does not match expected dataset size {len(self)}"
        return Dataset([(f, col[mask]) for f, col in self.items()])

    def slice(self, start: int = 0, stop: Optional[int] = None, step: int = 1) -> "Dataset":
        """
        Get at subset of the dataset with rows in the given range
        """
        return Dataset([(f, col[slice(start, stop, step)]) for f, col in self.items()])

    def split_by(self, field: str):
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
        cols = self.cols()
        col = cols[field]
        idxs = {}
        for idx, val in enumerate(col):
            curr = idxs.get(val, [])
            curr.append(idx)
            idxs[val] = curr

        return {val: self.indexes(idx) for val, idx in idxs.items()}

    def replace(self, query: Dict[str, nt.ArrayLike], *others: "Dataset", assume_disjoint=False, assume_unique=False):
        """
        Replaces values matching the given query with others. The query is a
        key/value map of allowed field values. The values can be either a single
        scalar value or a set of possible values. If nothing matches the query
        (e.g., {} specified), works the same way as append.

        Specify `assume_disjoint=True` when all input datasets do not any UIDs
        in common.

        Specify `assume_unique=True` when all input datasets do not have any
        duplicate UIDs.
        """
        keep_fields = self.common_fields(self, *others, assert_same_fields=True)
        others_len = sum(len(o) for o in others)
        keep_mask = n.ones(len(self), dtype=bool)
        if not assume_disjoint:
            uids = self["uid"]
            for other in others:
                keep_mask &= n.isin(uids, other["uid"], assume_unique=assume_unique, invert=True)
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

        cols = self.cols()
        for k, v in cols.items():
            if size > 6:
                contents = f"{str(v[:3])[:-1]} ... {str(v[-3:])[1:]}"
            else:
                contents = str(v)
            contents = " ".join(contents.split("\n"))
            contents = " ".join([x for x in contents.split(" ") if len(x) > 0])
            s += "\n" + f"    ('{k}', {contents}),"

        s += f"\n])  # {size} items, {len(cols)} fields"
        return s

    def _ipython_key_completions_(self):
        return self.fields()


def load(file: Union[str, PurePath, IO[bytes]]) -> "Dataset":
    return Dataset.load(file)


def allocate(size: int = 0, fields: List[Field] = []):
    return Dataset.allocate(size, fields)


def append(*datasets: Dataset, assert_same_fields=False, repeat_allowed=False):
    return Dataset.append_many(*datasets, assert_same_fields=assert_same_fields, repeat_allowed=repeat_allowed)


def union(*datasets: Dataset, assert_same_fields=False, assume_unique=False):
    return Dataset.union_many(*datasets, assert_same_fields=assert_same_fields, assume_unique=assume_unique)


def interlace(*datasets: Dataset, assert_same_fields=False):
    return Dataset.interlace(*datasets, assert_same_fields=assert_same_fields)


def innerjoin(*datasets: Dataset, assume_unique=False):
    return Dataset.innerjoin_many(*datasets, assume_unique=assume_unique)


def generate_uids(num: int = 0):
    """
    Generate the given number of random 64-bit unsigned integer uids
    """
    return n.random.randint(0, 2**64, size=(num,), dtype=n.uint64)


allocate.__doc__ = Dataset.allocate.__doc__
append.__doc__ = Dataset.append_many.__doc__
union.__doc__ = Dataset.union_many.__doc__
interlace.__doc__ = Dataset.interlace.__doc__
innerjoin.__doc__ = Dataset.innerjoin_many.__doc__
