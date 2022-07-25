from enum import Enum
from pathlib import PurePath
from typing import IO, Dict, NamedTuple, Tuple, Type, Union

import numpy as n
import numpy.typing as nt

from .util import bopen

DTType = Union[n.uint8, n.int16, n.float32, n.uint16, n.float16]


class DT(int, Enum):
    UINT8 = 0
    INT16 = 1
    FLOAT32 = 2
    UINT16 = 6
    FLOAT16 = 12


class Header(NamedTuple):
    """
    MRC file header
    """

    nx: int
    """
    Number of pixels in x dimension
    """
    ny: int
    """
    Number of pixels in y dimension
    """
    nz: int
    """
    Number of pixels in z dimension
    """
    datatype: DT
    """
    Integer-representation of MRC data type
    """
    xlen: float
    """
    Total size of X axis (e.g., in Angstroms)
    """
    ylen: float
    """
    Total size of Y axis (e.g., in Angstroms)
    """
    zlen: float
    """
    Total size of Z axis (e.g., in Angstroms)
    """
    origin: Tuple[float, float, float]
    """
    Location of image origin
    """
    nsymbt: int
    """
    Number of symbols in the symbol table following the header. Each symbol is 1024 bytes.
    """


DT_TO_DATATYPE: Dict[DT, Type[DTType]] = {
    DT.UINT8: n.uint8,
    DT.INT16: n.int16,
    DT.FLOAT32: n.float32,
    DT.UINT16: n.uint16,
    DT.FLOAT16: n.float16,
}

DATATYPE_TO_DT = {v: k for k, v in DT_TO_DATATYPE.items()}


def read(file: Union[str, PurePath, IO[bytes]]) -> Tuple[Header, nt.NDArray]:
    """
    Read a .mrc file at the given file into a numpy array. Returns the MRC
    header and the resulting array.
    """
    with bopen(file, "rb") as f:
        header = _read_header(f)
        dtype = DT_TO_DATATYPE[header.datatype]
        f.seek(1024 + header.nsymbt)  # seek to start of data

        data = n.fromfile(f, dtype=dtype, count=header.nz * header.ny * header.nx)
        data = data.reshape(header.nz, header.ny, header.nx)

        if dtype == n.float16:
            data = data.astype(n.float32)

        return header, data


def write(file: Union[str, PurePath, IO[bytes]], data: nt.NDArray, psize: float):
    """
    Write the given ndarray data to a file. Specify a pixel size for the mrc
    file as the last argument
    """
    while data.ndim < 3:
        data = n.array([data])
    assert data.ndim == 3, "Cannot write an array in MRC file"

    with bopen(file, "wb") as f:
        _write_header(f, data, psize)
        n.require(data, requirements="C").ravel().tofile(f)


def _read_header(file: IO) -> Header:
    header_int32 = n.fromfile(file, dtype=n.int32, count=256)
    assert len(header_int32) == 256, f"Could not read mrc header from {file}"

    header_float32 = header_int32.view(n.float32)
    nx, ny, nz, datatype = header_int32[:4]
    assert int(datatype) in DT_TO_DATATYPE, f"Unknown mrc datatype {datatype}"

    xlen, ylen, zlen = header_float32[10:13]
    origin = header_float32[49:52].tolist()
    nsymbt = header_int32[23:24][0]

    return Header(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        datatype=DT(datatype),
        xlen=float(xlen),
        ylen=float(ylen),
        zlen=float(zlen),
        origin=tuple(origin),
        nsymbt=int(nsymbt),
    )


def _write_header(file: IO, data: nt.NDArray, psize: float):
    assert data.dtype in DATATYPE_TO_DT, "Unsupported MRC dtype: {0}".format(data.dtype)

    header_int32 = n.zeros(256, dtype=n.int32)  # 1024 byte header
    header_float32 = header_int32.view(n.float32)

    # data is C order: nz, ny, nx
    header_int32[:3] = data.shape[::-1]  # nx, ny, nz
    header_int32[3] = DATATYPE_TO_DT[data.dtype]
    header_int32[7:10] = data.shape[::-1]  # mx, my, mz (grid size)
    header_float32[10:13] = [psize * i for i in data.shape[::-1]]  # xlen, ylen, zlen
    header_float32[13:16] = 90.0  # CELLB
    header_int32[16:19] = [1, 2, 3]  # axis order
    header_float32[19:22] = [data.min(), data.max(), data.mean()]  # data stats

    header_int32[52] = 542130509  # 'MAP ' chars
    header_int32[53] = 16708

    header_int32.tofile(file)
