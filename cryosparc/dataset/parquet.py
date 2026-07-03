"""
Apache Parquet on-disk file format support for CryoSPARC datasets.

The dataset's field description and row count are stored in the Arrow schema
metadata, which PyArrow preserves inside the Parquet file's key/value metadata,
so the exact dataset layout is reconstructed on load.
"""

# Datasets are written to and read from Parquet via Arrow record batches. Batch
# writes keep peak memory usage low; numeric columns are zero-copy views into
# dataset memory.
#
# Reading iterates Parquet row groups/batches lazily  and writes each batch
# directly into the pre-allocated destination dataset, only a single batch is
# materialized at a time. Parquet column pushdown means only the requested
# columns are read from disk when ``prefixes``/``fields`` are specified.

from pathlib import PurePath
from typing import IO, TYPE_CHECKING, Optional, Sequence, Type, Union

import pyarrow.parquet as pq

from .arrow import build_schema, dataset_to_batches, load_from_batches, parse_header, rows_per_batch
from .dtype import DatasetHeader, filter_descr

if TYPE_CHECKING:
    from . import Dataset

PARQUET_MAGIC = b"PAR1"
"""First (and last) 4 bytes of an Apache Parquet file."""

_DEFAULT_COMPRESSION = "lz4"
"""Parquet compression codec. lz4 offers a good size/speed trade-off."""


# ==============================================================================
# Parquet writing/serialization
# ==============================================================================


def write_parquet_file(
    dset: "Dataset",
    sink: Union[str, PurePath, IO[bytes]],
    *,
    compression: str = _DEFAULT_COMPRESSION,
):
    """
    Write the dataset to ``sink`` in the Apache Parquet format.

    ``sink`` may be a file path or any writable binary file object. Each record
    batch is written as its own row group so that peak memory stays bounded.
    """
    descr = dset.descr()
    schema = build_schema(dset, descr, compression)  # raises early on complex columns
    # disabling statistics and dictionary improves write speed
    with pq.ParquetWriter(
        sink,
        schema,
        compression=compression,
        write_statistics=False,
        use_dictionary=False,
    ) as writer:
        for batch in dataset_to_batches(dset, schema, descr):
            writer.write_batch(batch, row_group_size=batch.num_rows)


# ==============================================================================
# Parquet loading/deserialization
# ==============================================================================


def _open_parquet_file(file: Union[str, PurePath, IO[bytes]]) -> pq.ParquetFile:
    if isinstance(file, (str, PurePath)):
        return pq.ParquetFile(str(file), memory_map=True)
    return pq.ParquetFile(file)


def load_parquet_file(
    cls: Type["Dataset"],
    file: Union[str, PurePath, IO[bytes]],
    *,
    prefixes: Optional[Sequence[str]] = None,
    fields: Optional[Sequence[str]] = None,
) -> "Dataset":
    """
    Load a dataset from a Parquet file. Requires a seekable source (file path or
    seekable file object). Only the requested columns are read from disk.
    """
    pf = _open_parquet_file(file)
    schema = pf.schema_arrow
    header = parse_header(schema)
    descr = filter_descr(header["dtype"], keep_prefixes=prefixes, keep_fields=fields)
    columns = [field[0] for field in descr]
    batches = pf.iter_batches(batch_size=rows_per_batch(descr), columns=columns)
    return load_from_batches(cls, schema, batches, prefixes, fields)


def inspect_parquet_file(file: Union[str, PurePath, IO[bytes]]) -> DatasetHeader:
    """Read just the schema metadata (length + field description) from a file."""
    return parse_header(_open_parquet_file(file).schema_arrow)
