"""
Default Numpy file format support for CryoSPARC datasets.
"""

from pathlib import PurePath
from typing import IO, TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union

import numpy as n

from ..constants import ONE_MIB
from .dtype import DatasetHeader, fielddtype, filter_descr, normalize_field

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
    import os

    # disable mmap by setting CRYOSPARC_DATASET_MMAP=false or dataset is small
    if (
        os.getenv("CRYOSPARC_DATASET_MMAP", "true").lower() == "true"
        and isinstance(file, (str, PurePath))
        and os.stat(file).st_size > ONE_MIB
    ):
        # Use mmap to avoid loading full record array into memory
        # cast path to a string for older numpy/python
        mmap_mode, f = "r", str(file)
        chunk_size = 2**14  # magic number optimizes memory and performance
    else:
        mmap_mode, f = None, file
        chunk_size = 2**60  # huge enough number so you don't use chunks

    indata = n.load(f, mmap_mode=mmap_mode, allow_pickle=False, **_NUMPY_LOAD_KWARGS)
    size = len(indata)
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


def inspect_numpy_file(file: Union[str, PurePath, IO[bytes]]) -> DatasetHeader:
    indata = n.load(str(file), mmap_mode="r", allow_pickle=False)
    fields = [normalize_field(f[0], fielddtype(f)) for f in indata.dtype.descr]
    return DatasetHeader(length=len(indata), dtype=fields, compression=None)
