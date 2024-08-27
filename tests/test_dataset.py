from base64 import b64decode
from io import BytesIO

import numpy as n
import pytest

from cryosparc.dataset import CSDAT_FORMAT, Column
from cryosparc.row import Row

from .conftest import Dataset


@pytest.fixture
def io_data():
    data = b64decode(
        (
            b"k05VTVBZAQCmAHsnZGVzY3InOiBbKCd1aWQnLCAnPHU4JyksICgnZmllbGQxJywgJzx1NCcpLC"
            b"AoJ2ZpZWxkMicsICc8ZjQnKSwgKCdmaWVsZDMnLCAnfFM2JyksICgnZmllbGQ0JywgJzxmOCcp"
            b"LCAoJ2ZpZWxkNScsICc8aTgnKV0sICdmb3J0cmFuX29yZGVyJzogRmFsc2UsICdzaGFwZSc6IC"
            b"gyLCksIH0gIAp7AAAAAAAAACoAAADD9UhASGVsbG8AAAAAAAAAAAArAAAAAAAAAMgBAAAAAAAA"
            b"KgAAAFK4LkBXb3JsZAAAAAAAAADwPysAAAAAAAAA"
        )
    )
    return BytesIO(data)


@pytest.fixture
def small_dset():
    field3 = "long/fieldwithsuperduperlongcolumnnamethatislongandtestable"
    dset = Dataset.allocate(
        4,
        fields=[
            ("field/1", "u8", (2,)),
            ("field/2", "f4"),
            (field3, "O"),
        ],
    )
    dset["field/1"] = (42, 43)
    dset["field/2"] = n.array([3.14, 2.73, 1.62, 3.14], dtype="f8")
    dset[field3][:] = n.array(["Hello", "World", "!", "!"])
    return dset


@pytest.fixture
def small_dset_path(tmp_path, small_dset):
    path = tmp_path / "small_dset.cs"
    small_dset.save(path, format=CSDAT_FORMAT)
    return path


@pytest.fixture
def small_dset_stream(small_dset):
    stream = BytesIO()
    small_dset.save(stream, format=CSDAT_FORMAT)
    stream.seek(0)
    return stream


def test_allocate():
    storage = Dataset.allocate(size=2000000, fields=[("field1", "u8"), ("field2", "f4"), ("field3", "O")])
    assert storage is not None


def test_populate_new_0():
    storage = Dataset.allocate(size=0)
    assert len(storage) == 0


def test_populate_new_1():
    storage = Dataset.allocate(size=1)
    assert len(storage) == 1


def test_populate_new_many():
    storage = Dataset.allocate(size=3)
    assert len(storage) == 3


def test_storage_from_other():
    storage1 = Dataset(3)
    storage2 = storage1.copy()
    assert len(storage2) == 3


def test_basic_data_constructor():
    data = Dataset()
    assert len(data) == 0
    assert len(data.descr()) == 1


def test_empty_data_constructor():
    data = Dataset(0)
    assert len(data) == 0
    assert len(data.descr()) == 1


def test_fields():
    data = Dataset.allocate(3, [("test1", "<u4"), ("test2", "<f8", (2,))])
    assert data.fields() == ["uid", "test1", "test2"]
    assert data.fields(exclude_uid=True) == ["test1", "test2"]


def test_descr():
    data = Dataset.allocate(3, [("test1", "<u4"), ("test2", "<f8", (2,))])
    expected_descr = [("uid", "<u8"), ("test1", "<u4"), ("test2", "<f8", (2,))]
    assert data.descr() == expected_descr
    assert data.descr(exclude_uid=True) == expected_descr[1:]


def test_invalid_data_fields():
    # This is ok actually
    assert Dataset(
        [
            ("uid", n.array([1, 2, 3])),
            ("dat", ["Hello", "World", "!"]),
        ]
    )


def test_uneven_data_fields():
    with pytest.raises(AssertionError):
        Dataset(
            [
                ("uid", n.array([1, 2, 3])),
                ("dat", n.array(["Hello", "World"])),
            ]
        )


def test_invalid_key_assignment():
    storage = Dataset.allocate(size=3)
    with pytest.raises(AssertionError):
        storage["gain_ref_blob/path"] = ["Hello", "World!"]


def test_non_existent_key_assignment():
    storage = Dataset.allocate(size=3)
    with pytest.raises(AssertionError):
        storage["gain_ref_blob"] = n.zeros(3)


def test_valid_key_assignment():
    storage = Dataset.allocate(size=3, fields=[("gain_ref_blob/path", "O")])
    storage["gain_ref_blob/path"] = "Hello World!"
    assert isinstance(storage["gain_ref_blob/path"], Column)
    assert len(storage["gain_ref_blob/path"]) == 3


def test_valid_multi_dimensional_key_assignment():
    storage = Dataset.allocate(
        size=3,
        fields=[
            ("location/micrograph_shape", "<u4", (2,)),
            ("alignments3D/shift", "500,2f4"),
        ],
    )

    shapearr = n.array([42, 24])
    shiftarr = n.array([(3.14, 2.67)] * 500, dtype=n.float32)
    storage["location/micrograph_shape"] = shapearr
    storage["alignments3D/shift"] = shiftarr

    assert isinstance(storage["location/micrograph_shape"], Column)
    assert isinstance(storage["alignments3D/shift"], Column)
    assert len(storage["location/micrograph_shape"]) == 3
    assert len(storage["alignments3D/shift"]) == 3
    assert storage["location/micrograph_shape"].shape == (3, 2)
    assert storage["alignments3D/shift"].shape == (3, 500, 2)
    assert n.all(storage["location/micrograph_shape"][2] == shapearr)
    assert n.all(storage["alignments3D/shift"][2] == shiftarr)


def test_add_fields():
    storage = Dataset.allocate(size=2000000, fields=[("field1", "u8"), ("field2", "f4"), ("field3", "O")]).add_fields(
        [
            ("mscope_params/accel_kv", "f4"),
            ("mscope_params/cs_mm", "f4"),
            ("mscope_params/total_dose_e_per_A2", "f4"),
            ("mscope_params/phase_plate", "u4"),
            ("mscope_params/neg_stain", "u4"),
            ("mscope_params/exp_group_id", "u4"),
        ]
    )
    assert len(storage.fields()), 10


def test_add_fields_nonebug(t20s_dset):
    t20s_dset.add_fields([("micrograph_blob_non_dw/path", "O")])
    assert all(t20s_dset["gain_ref_blob/path"] == "J1/imported/norm-amibox05-0.mrc")


def test_to_list():
    storage = Dataset.allocate(size=1, fields=[("field1", "u8"), ("field2", "f4"), ("field3", "O")])
    lst = storage.to_list()
    assert len(lst) == 1
    assert len(lst[0]) == 4


def test_to_list_exclude_uid():
    storage = Dataset.allocate(size=1, fields=[("field1", "u8"), ("field2", "f4"), ("field3", "O")])
    storage["field3"][0] = "Hello"
    lst = storage.to_list(exclude_uid=True)
    assert len(lst) == 1
    assert len(lst[0]) == 3
    assert lst == [[0, 0.0, "Hello"]]


def test_save():
    dtype = [("uid", "u8"), ("field1", "u4"), ("field2", "f4"), ("field3", "S6"), ("field4", "f8")]
    expected = n.array([(1, 42, 3.14, "Hello", 1.0), (2, 42, 2.73, "World", 0.0)], dtype=dtype)
    dset = Dataset(expected)
    new_iodata = BytesIO()
    dset.save(new_iodata)
    new_iodata.seek(0)
    actual = n.load(new_iodata)
    assert expected.dtype.descr == actual.dtype.descr
    assert all(all(n.equal(expected[d[0]], actual[d[0]])) for d in dtype if d[0] != "field3")
    assert all(e.decode() == a.decode() for e, a in zip(expected["field3"], actual["field3"]))


def test_load(io_data):
    dtype = [("field1", "u4"), ("field2", "f4"), ("field3", "O"), ("field4", "f8"), ("field5", "i8")]
    expected = Dataset.allocate(size=2, fields=dtype)
    expected["field1"] = 42
    expected["field2"] = n.array([3.14, 2.73], dtype="f8")
    expected["field3"][:] = n.array(["Hello", "World"])
    expected["field4"][1] = 1.0
    expected["field5"][0:] = 43

    result = Dataset.load(io_data)
    assert expected.descr() == result.descr()
    assert all([n.equal(expected[d[0]], result[d[0]]).all() for d in dtype if d[0] != "uid"])


def test_load_fields(io_data):
    dtype = [("field2", "f4"), ("field3", "O"), ("field5", "i8")]
    expected = Dataset.allocate(size=2, fields=dtype)
    expected["field2"] = n.array([3.14, 2.73], dtype="f8")
    expected["field3"][:] = n.array(["Hello", "World"])
    expected["field5"][0:] = 43

    result = Dataset.load(io_data, fields=["field2", "field3", "field5"])
    assert expected.descr() == result.descr()
    assert all([n.equal(expected[d[0]], result[d[0]]).all() for d in dtype if d[0] != "uid"])


def test_subset_range_out_of_bounds():
    data = Dataset.allocate(size=3, fields=[("field1", "u8"), ("field2", "f4"), ("field3", "O")])
    subset = data.slice(2, 100)
    assert len(subset) == 1


def test_from_data_none():
    data = Dataset()  # FIXME: Not necessary, remove
    assert len(data) == 0


def test_load_stream(small_dset, small_dset_path):
    result = Dataset.load(small_dset_path)
    assert result == small_dset


def test_load_stream_prefixes(small_dset, small_dset_path):
    result = Dataset.load(small_dset_path, prefixes=["field"])
    assert result == small_dset.filter_prefixes(["field"], copy=True)


def test_load_stream_fields(small_dset, small_dset_stream):
    result = Dataset.load(small_dset_stream, fields=["field/2"])
    assert result == small_dset.filter_fields(["field/2"], copy=True)


def test_pickle_unpickle():
    import pickle

    dset = Dataset(
        [
            ("uid", n.array([1, 2, 3])),
            ("dat", n.array(["Hello", "World", "!"])),
        ]
    )
    pickled = pickle.dumps(dset, protocol=pickle.HIGHEST_PROTOCOL)
    del dset  # calls data destructor to clear dset memory

    dset = pickle.loads(pickled)
    assert n.array_equal(dset["uid"], [1, 2, 3])
    assert n.array_equal(dset["dat"], ["Hello", "World", "!"])


def test_column_aggregation(t20s_dset):
    assert type(t20s_dset["uid"]) is Column
    assert type(n.max(t20s_dset["uid"])) is n.uint64
    assert isinstance(n.mean(t20s_dset["uid"]), n.number)
    assert not isinstance(n.mean(t20s_dset["uid"]), n.ndarray)


@pytest.mark.skipif(n.__version__.startswith("1.15."), reason="works with newer numpy versions, use case is limited")
def test_row_array_type(t20s_dset):
    rowarr = n.array(t20s_dset.rows())
    assert isinstance(rowarr[0], Row)


def test_innerjoin_bigger():
    dat2 = "dat2dat2dat2dat2dat2dat2dat2dat2dat2dat2dat2dat2dat2"
    d1 = Dataset([("uid", [1, 2, 3]), ("dat1", ["Hello", "World", "!"])])
    d2 = Dataset([("uid", [0, 1, 2, 3, 4]), (dat2, ["(", "Hello", "World", "!", ")"])])
    assert d1.innerjoin(d2) == Dataset(
        [
            ("uid", [1, 2, 3]),
            ("dat1", ["Hello", "World", "!"]),
            (dat2, ["Hello", "World", "!"]),
        ]
    )


def test_innerjoin_smaller():
    dat2 = "dat2dat2dat2dat2dat2dat2dat2dat2dat2dat2dat2dat2dat2"
    d1 = Dataset([("uid", [1, 2, 3]), ("dat1", ["Hello", "World", "!"])])
    d2 = Dataset([("uid", [3, 1]), (dat2, ["Hello", "World"])])

    assert d1.innerjoin(d2) == Dataset(
        [
            ("uid", [1, 3]),
            ("dat1", ["Hello", "!"]),
            (dat2, ["World", "Hello"]),
        ]
    )


def test_append_many_empty():
    assert len(Dataset.append_many().rows()) == 0


def test_append_empty_fields():
    d1 = Dataset.allocate(0, [("field", "f4")])
    d2 = Dataset.allocate(0, [("field", "f4")])
    d3 = d1.append(d2)
    assert len(d3) == 0
    assert d3.fields() == ["uid", "field"]


def test_union_many_empty():
    assert len(Dataset.union_many().rows()) == 0


def test_union_empty_fields():
    d1 = Dataset.allocate(0, [("field", "f4")])
    d2 = Dataset.allocate(0, [("field", "f4")])
    d3 = d1.union(d2)
    assert len(d3) == 0
    assert d3.fields() == ["uid", "field"]


def test_allocate_many_separate():
    for _ in range(66_000):
        allocated = []
        for _ in range(3):
            allocated.append(Dataset(1))
        assert len(allocated) == 3
        del allocated


def test_allocate_many_together():
    # Checks for logic issues when allocating a lot of datasets
    for _ in range(3):
        allocated = []
        for _ in range(66_000):
            allocated.append(Dataset(1))
        assert len(allocated) == 66_000
        del allocated
