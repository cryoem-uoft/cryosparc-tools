"""
Apache Arrow (IPC stream) serialization for Dataset.

Implements the streaming wire format (`Dataset.stream` / `Dataset.from_stream`)
using the Arrow IPC **stream** format, which can be produced and consumed
sequentially (e.g., over a network) without random access.

Also includes shared Arrow <-> Dataset conversion helpers, used by the on-disk
Parquet format in _parquet.py

The dataset's full field description and row count are stored in the Arrow
schema metadata so the exact dataset layout (including the order of fields
and the distinction between scalar and multi-dimensional columns) is
reconstructed on load.

:meta private:
"""

# Memory notes:
#
# - When saving/streaming, record batches are built as zero-copy views into the
#   dataset's own memory for numeric columns. Only string columns are copied
#   (Arrow requires contiguous offset/data buffers for strings).
# - When loading, the destination dataset is fully pre-allocated up front and
#   each record batch is written directly into dataset memory, so only a single
#   batch is ever materialized at a time.

import io
from pathlib import PurePath
from typing import IO, TYPE_CHECKING, Iterable, Iterator, List, Literal, Optional, Sequence, Type, Union

import numpy as n
import pyarrow as pa

from ..errors import DatasetLoadError
from .dtype import (
    DatasetHeader,
    Field,
    decode_dataset_header,
    encode_dataset_header,
    fielddtype,
    filter_descr,
    rows_per_batch,
)

if TYPE_CHECKING:
    from . import Dataset

_METADATA_KEY = b"cryosparc-dataset-header"
"""Arrow schema metadata key holding the JSON-encoded :class:`DatasetHeader`."""


# ==============================================================================
# Shared Arrow <-> Dataset conversion helpers
# ==============================================================================


def _field_arrow_type(field: Field) -> pa.DataType:
    """Translate a dataset :class:`Field` into the equivalent Arrow data type."""
    dt = n.dtype(fielddtype(field))
    base = dt.base
    if base.kind == "c":
        raise TypeError(
            f"Cannot serialize complex-typed dataset field {field[0]!r} to the Arrow format. "
            "Complex columns are not supported by Apache Arrow."
        )
    if base.kind == "O":
        # Python string columns are stored as large UTF-8 (64-bit offsets) to
        # support datasets with more than 2GB of string data per column.
        return pa.large_string()
    pa_type = pa.from_numpy_dtype(base)
    # Multi-dimensional columns become (possibly nested) fixed-size lists.
    for dim in reversed(dt.shape):
        pa_type = pa.list_(pa_type, dim)
    return pa_type


def build_schema_from_descr(descr: List[Field], length: int, compression: Optional[str]) -> pa.Schema:
    """
    Build an Arrow schema for the given field description and row count,
    embedding the dataset header (row count + field description + compression
    codec) as schema metadata. Raises ``TypeError`` for unsupported (e.g.,
    complex) columns.
    """
    pa_fields = [pa.field(f[0], _field_arrow_type(f)) for f in descr]
    header = DatasetHeader(length=length, dtype=descr, compression=compression)
    return pa.schema(pa_fields, metadata={_METADATA_KEY: encode_dataset_header(header)})


def build_schema(dset: "Dataset", descr: List[Field], compression: Optional[str]) -> pa.Schema:
    """
    Build an Arrow schema for the given dataset and field description. See
    :func:`build_schema_from_descr`.
    """
    return build_schema_from_descr(descr, len(dset), compression)


def numpy_to_arrow_array(values: n.ndarray, pa_type: pa.DataType) -> pa.Array:
    """
    Convert a single numpy column (extracted from a record array or dataset)
    into an Arrow array matching ``pa_type``.

    Handles fixed-width byte/unicode string columns (decoded to UTF-8), Python
    object string columns, and multi-dimensional columns (converted to nested
    fixed-size lists). Numeric columns are copied into a contiguous Arrow array.
    """
    from ..util import hashcache

    if values.dtype.char == "S":
        # Fixed-width byte strings (the on-disk .npy string representation).
        decoded = n.vectorize(hashcache(bytes.decode), otypes="O")(values) if values.size else values.astype(object)
        return pa.array(decoded, type=pa.large_string())
    if values.dtype.char == "U":
        decoded = n.vectorize(hashcache(str), otypes="O")(values) if values.size else values.astype(object)
        return pa.array(decoded, type=pa.large_string())
    if values.dtype.kind == "O":
        # Python string objects.
        return pa.array(values, type=pa.large_string())
    if values.ndim > 1:
        # Multi-dimensional: build (possibly nested) fixed-size lists from the
        # base values outward, one list level per trailing dimension.
        arr: pa.Array = pa.array(n.ascontiguousarray(values).reshape(-1))
        for dim in reversed(values.shape[1:]):
            arr = pa.FixedSizeListArray.from_arrays(arr, dim)
        return arr
    return pa.array(n.ascontiguousarray(values), type=pa_type)


def _column_to_array(dset: "Dataset", field: Field, offset: int, length: int, pa_type: pa.DataType) -> pa.Array:
    dt = n.dtype(fielddtype(field))
    sub = n.asarray(dset[field[0]][offset : offset + length])
    if dt.base.kind == "O":
        # Strings: not zero-copy, Arrow needs contiguous offset/data buffers.
        return pa.array(sub, type=pa.large_string())
    if dt.shape:
        # Multi-dimensional: build (possibly nested) fixed-size lists from the
        # base values outward. The flattened base array is a zero-copy view into
        # dataset memory.
        values: pa.Array = pa.array(n.ascontiguousarray(sub).reshape(-1))
        for dim in reversed(dt.shape):
            values = pa.FixedSizeListArray.from_arrays(values, dim)
        return values
    # Scalar numeric: zero-copy view into dataset memory.
    return pa.array(sub, type=pa_type)


def dataset_to_batches(dset: "Dataset", schema: pa.Schema, descr: List[Field]) -> Iterator[pa.RecordBatch]:
    """
    Yield the dataset as a sequence of Arrow record batches (near ~8MiB each)
    built as zero-copy views into dataset memory for numeric columns.
    """
    nrow = len(dset)
    if nrow == 0:
        return
    per_batch = rows_per_batch(descr)
    pa_types = [schema.field(i).type for i in range(len(descr))]
    for offset in range(0, nrow, per_batch):
        length = min(per_batch, nrow - offset)
        arrays = [_column_to_array(dset, field, offset, length, pa_types[i]) for i, field in enumerate(descr)]
        yield pa.record_batch(arrays, schema=schema)


def parse_header(schema: pa.Schema) -> DatasetHeader:
    """
    Decode the embedded :class:`DatasetHeader` from Arrow schema metadata.
    """
    metadata = schema.metadata or {}
    raw = metadata.get(_METADATA_KEY)
    if raw is None:
        raise DatasetLoadError("Arrow dataset is missing CryoSPARC schema metadata")
    return decode_dataset_header(raw)


def _write_batch_into_dataset(dset: "Dataset", field: Field, array: pa.Array, offset: int, length: int):
    dt = n.dtype(fielddtype(field))
    col = dset[field[0]]
    if dt.base.kind == "O":
        col[offset : offset + length] = array.to_numpy(zero_copy_only=False)
    elif dt.shape:
        values = array
        for _ in range(len(dt.shape)):
            values = values.flatten()
        col[offset : offset + length] = values.to_numpy(zero_copy_only=False).reshape((length, *dt.shape))
    else:
        col[offset : offset + length] = array.to_numpy(zero_copy_only=False)


def load_from_batches(
    cls: Type["Dataset"],
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    prefixes: Optional[Sequence[str]],
    fields: Optional[Sequence[str]],
) -> "Dataset":
    """
    Pre-allocate a dataset from the header embedded in ``schema``, then write
    each incoming record batch directly into dataset memory. Only a single batch
    is materialized at a time. Batches may contain a subset of columns (e.g.,
    when the caller has already applied column projection).
    """
    header = parse_header(schema)
    descr = filter_descr(header["dtype"], keep_prefixes=prefixes, keep_fields=fields)
    descr_by_name = {field[0]: field for field in descr}

    dset = cls.allocate(0, descr)
    dset._data.addrows(header["length"])

    offset = 0
    for batch in batches:
        names = batch.schema.names
        for index, name in enumerate(names):
            field = descr_by_name.get(name)
            if field is None:
                continue  # skip fields that were not selected
            _write_batch_into_dataset(dset, field, batch.column(index), offset, batch.num_rows)
        offset += batch.num_rows
    return dset


# ==============================================================================
# Arrow IPC stream format (streaming wire format)
# ==============================================================================


def _write_options(compression: Optional[str]) -> Optional["pa.ipc.IpcWriteOptions"]:
    return pa.ipc.IpcWriteOptions(compression=compression) if compression else None


class _ChunkSink(io.RawIOBase):
    """
    Writable file object that buffers writes so they can be yielded
    incrementally while an Arrow stream writer produces them.

    :meta private:
    """

    def __init__(self):
        super().__init__()
        self._chunks: List[bytes] = []
        self._pos = 0

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:  # type: ignore[override]
        data = bytes(b)
        self._chunks.append(data)
        self._pos += len(data)
        return len(data)

    def tell(self) -> int:
        return self._pos

    def drain(self) -> List[bytes]:
        chunks = self._chunks
        self._chunks = []
        return chunks


def stream_arrow(dset: "Dataset", *, compression: Literal["lz4", None] = None) -> Iterator[bytes]:
    """
    Yield the dataset encoded in the Arrow IPC stream format. Suitable for
    sequential consumers (e.g., sending over a network) that cannot seek.
    """
    descr = dset.descr()
    schema = build_schema(dset, descr, compression)  # raises early on complex columns
    sink = _ChunkSink()
    writer = pa.ipc.new_stream(sink, schema, options=_write_options(compression))
    try:
        yield from sink.drain()  # schema message
        for batch in dataset_to_batches(dset, schema, descr):
            writer.write_batch(batch)
            yield from sink.drain()
    finally:
        writer.close()
    yield from sink.drain()  # end-of-stream marker


def load_arrow_stream(
    cls: Type["Dataset"],
    source: Union[str, PurePath, IO[bytes]],
    *,
    prefixes: Optional[Sequence[str]] = None,
    fields: Optional[Sequence[str]] = None,
) -> "Dataset":
    """
    Load a dataset from the Arrow IPC stream format. Reads sequentially and does
    not require a seekable source.
    """
    reader = pa.ipc.open_stream(source)
    return load_from_batches(cls, reader.schema, reader, prefixes, fields)
