import os
from io import StringIO
from pathlib import Path

import pytest

from cryosparc import star


@pytest.fixture
def simple_star_file():
    return StringIO(
        """
data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
3410.0 5184.0
680.0 6525.0
816.0 7370.0
515.0 7118.0
1088.0 7254.0
3138.0 7195.0
3381.0 7040.0
3507.0 7127.0
3867.0 7147.0
4012.0 7020.0
4178.0 7079.0
4450.0 6952.0
4119.0 6729.0
3711.0 6321.0
3245.0 6127.0
4314.0 6088.0
4877.0 5738.0
4974.0 5631.0
4362.0 4504.0
4207.0 4495.0
4051.0 4495.0
4168.0 4864.0
3585.0 4320.0
3216.0 4281.0
3225.0 4125.0
3012.0 3533.0
3274.0 3591.0
2011.0 3377.0
1632.0 3377.0
1632.0 3484.0

"""
    )


@pytest.fixture
def big_star_file():
    return Path() / "tests" / "data" / "job096_run_data.star"


def test_read_simple(simple_star_file):
    data = star.read(simple_star_file)
    assert "" in data
    assert len(data[""]) == 30
    assert data[""]["rlnCoordinateX"][0] == 3410.0
    assert data[""]["rlnCoordinateY"][0] == 5184.0
    assert data[""]["rlnCoordinateX"][-1] == 1632.0
    assert data[""]["rlnCoordinateY"][-1] == 3484.0


@pytest.mark.skipif(os.getenv("CI") == "true", reason="To avoid GitHub LFS download on GitHub Actions")
def test_read_big(big_star_file):
    data = star.read(big_star_file)
    assert "optics" in data
    assert "particles" in data
    assert len(data["optics"]) == 1
    assert len(data["optics"][0]) == 13
    assert len(data["particles"]) == 29183
    assert len(data["particles"][0]) == 26
    assert data["optics"]["rlnOpticsGroupName"][0] == "mydata"
    assert data["optics"]["rlnOpticsGroup"][0] == 1
    assert data["particles"]["rlnCoordinateX"][0] == 2688.0
    assert data["particles"]["rlnCoordinateY"][0] == 1330.0
    assert data["particles"]["rlnImageName"][0] == "1@Polished/SARS2_HGKA54_ZK144_G2_K3_0000_shiny.mrcs"
    assert data["particles"]["rlnRandomSubset"][0] == 1
    assert data["particles"]["rlnCoordinateX"][-1] == 4221.0
    assert data["particles"]["rlnCoordinateY"][-1] == 3374.0
    assert data["particles"]["rlnImageName"][-1] == "12@Polished/SARS2_HGKA54_ZK144_G2_K3_7982_shiny.mrcs"
    assert data["particles"]["rlnRandomSubset"][-1] == 2


def test_write():
    result = StringIO()
    star.write(
        result,
        data=[
            (123.0, 456.0, "particles.mrc", "mic.mrc", 1),
            (789.0, 123.0, "particles.mrc", "mic.mrc", 0),
        ],
        labels=[
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnImageName",
            "rlnMicrographName",
            "rlnCtfDataAreCtfPremultiplied",
        ],
    )
    assert (
        result.getvalue()
        == """
data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnImageName #3
_rlnMicrographName #4
_rlnCtfDataAreCtfPremultiplied #5
123.0 456.0 particles.mrc mic.mrc 1
789.0 123.0 particles.mrc mic.mrc 0

"""
    )


def test_write_simple(simple_star_file):
    data = star.read(simple_star_file)
    result = StringIO()
    star.write_blocks(result, data)
    assert result.getvalue() == simple_star_file.getvalue()
