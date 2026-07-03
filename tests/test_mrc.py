import tempfile

import numpy as n

from cryosparc import mrc


def test_read_write():
    data = n.random.rand(10, 20, 30).astype(n.float32)
    psize = 1.5

    with tempfile.NamedTemporaryFile() as f:
        mrc.write(f.name, data, psize)
        read_header, read_data = mrc.read(f.name)

        assert n.allclose(data, read_data), "Read data does not match written data"
        assert n.isclose(read_header.xlen, psize * data.shape[2]), "Incorrect x length in header"
        assert n.isclose(read_header.ylen, psize * data.shape[1]), "Incorrect y length in header"
        assert n.isclose(read_header.zlen, psize * data.shape[0]), "Incorrect z length in header"
