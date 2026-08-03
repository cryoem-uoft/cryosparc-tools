"""
Numpy file format support for CryoSPARC datasets.

:meta private:
"""

from pathlib import PurePath
from typing import IO, TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union

import numpy as n

from ..constants import EIGHT_MIB
from .dtype import DatasetHeader, fielddtype, filter_descr, normalize_field, rows_per_batch

if TYPE_CHECKING:
    from . import Dataset

_NUMPY_MAJOR_MINOR_VERSION = tuple(map(int, n.__version__.split(".")[:2]))  # e.g., "1.23.4" -> (1, 23)
_NUMPY_LOAD_KWARGS: Dict[str, Any] = {"max_header_size": 1024**3} if _NUMPY_MAJOR_MINOR_VERSION >= (1, 24) else {}
"""Numpy >= 1.24 load function require max_header_size, which is 10000 by default and too small for some datasets."""


# ==============================================================================
# Arrow writing/serialization
# ==============================================================================


def write_numpy_file(
    dset: "Dataset",
    file: Union[str, PurePath, IO[bytes]],
):
    """
    Write the dataset to ``file`` in the Numpy .npy format. Written as a single
    record array. Python strings are stored as byte arrays.
    """
    n.save(file, dset.to_records(fixed=True), allow_pickle=False)


# ==============================================================================
# Numpy loading/deserialization
# ==============================================================================


def load_numpy_file(
    cls: Type["Dataset"],
    file: Union[str, PurePath, IO[bytes]],
    prefixes: Optional[Sequence[str]] = None,
    fields: Optional[Sequence[str]] = None,
):
    """
    Load a .npy record-array directly into a CryoSPARC dataset.
    """
    # Use mmap to avoid loading full record array into memory
    # cast path to a string for older numpy/python
    mmap_mode, f = ("r", str(file)) if _use_mmap(file) else (None, file)
    indata = n.load(f, mmap_mode=mmap_mode, allow_pickle=False, **_NUMPY_LOAD_KWARGS)
    size = len(indata)
    chunk_size = rows_per_batch(indata.dtype.descr) if mmap_mode else size
    descr = filter_descr(indata.dtype.descr, keep_prefixes=prefixes, keep_fields=fields)
    dset = cls.allocate(size, descr)

    offset = 0
    while offset < size:
        end = min(offset + chunk_size, size)
        chunk = indata[offset:end]
        for field in descr:
            dset[field[0]][offset:end] = chunk[field[0]]
        offset += chunk_size
        if mmap_mode and offset < size:
            # reset mmap to avoid excessive memory usage
            del indata
            indata = n.load(f, mmap_mode=mmap_mode, allow_pickle=False, **_NUMPY_LOAD_KWARGS)

    return dset


def load_numpy_table(
    file: Union[str, PurePath, IO[bytes]],
    *,
    prefixes: Optional[Sequence[str]] = None,
    fields: Optional[Sequence[str]] = None,
):
    """
    Load a .npy record-array file as an Arrow table.
    """

    import pyarrow as pa

    from ._arrow import build_schema_from_descr, numpy_to_arrow_array

    # Use mmap to avoid loading full record array into memory
    mmap_mode, f = ("r", str(file)) if _use_mmap(file) else (None, file)
    indata = n.load(f, mmap_mode=mmap_mode, allow_pickle=False, **_NUMPY_LOAD_KWARGS)
    size = len(indata)
    chunk_size = rows_per_batch(indata.dtype.descr) if mmap_mode else size

    # descr_raw preserves the on-disk numpy dtypes (used to index columns out
    # of the record array), while ``descr`` normalizes them (S/U -> object,
    # shape metadata) to match the Arrow field types.
    descr_raw = filter_descr(indata.dtype.descr, keep_prefixes=prefixes, keep_fields=fields)
    descr = [normalize_field(field[0], fielddtype(field)) for field in descr_raw]
    schema = build_schema_from_descr(descr, size, compression=None)
    pa_types = [schema.field(i).type for i in range(len(descr))]

    batches = []
    offset = 0
    while offset < size:
        end = min(offset + chunk_size, size)
        chunk = indata[offset:end]
        # Column extraction from a record array is strided/non-contiguous, so
        # each conversion copies into a fresh Arrow array. Arrow then owns the
        # memory, allowing the mmap pages to be released between chunks.
        arrays = [numpy_to_arrow_array(chunk[field[0]], pa_types[i]) for i, field in enumerate(descr_raw)]
        batches.append(pa.record_batch(arrays, schema=schema))
        offset += chunk_size
        if mmap_mode and offset < size:
            # reset mmap to avoid excessive memory usage
            del indata
            indata = n.load(f, mmap_mode=mmap_mode, allow_pickle=False, **_NUMPY_LOAD_KWARGS)

    return pa.Table.from_batches(batches, schema=schema)


def inspect_numpy_file(file: Union[str, PurePath]) -> DatasetHeader:
    indata = n.load(str(file), mmap_mode="r", allow_pickle=False)
    fields = [normalize_field(f[0], fielddtype(f)) for f in indata.dtype.descr]
    return DatasetHeader(length=len(indata), dtype=fields, compression=None)


def _use_mmap(file: Union[str, PurePath, IO[bytes]]) -> bool:
    import os

    return (
        os.getenv("CRYOSPARC_DATASET_MMAP", "true").lower() == "true"
        and isinstance(file, (str, PurePath))
        and os.stat(file).st_size > EIGHT_MIB
    )
