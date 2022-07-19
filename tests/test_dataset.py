from io import BytesIO
from base64 import b64decode
import pytest
import numpy as n
from cryosparc.dataset import Dataset, NumericColumn, StringColumn


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
    assert len(data.descr) == 1


def test_empty_data_constructor():
    data = Dataset(0)
    assert len(data) == 0
    assert len(data.descr) == 1


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
    storage["gain_ref_blob"] = n.zeros(3)
    assert all(storage["gain_ref_blob"] == n.zeros(3))


def test_valid_key_assignment():
    storage = Dataset.allocate(size=3, fields=[("gain_ref_blob/path", "O")])
    storage["gain_ref_blob/path"] = "Hello World!"
    assert isinstance(storage["gain_ref_blob/path"], StringColumn)
    assert len(storage["gain_ref_blob/path"]) == 3


def test_valid_multi_dimensional_key_assignment():
    storage = Dataset.allocate(size=3, fields=[("location/micrograph_shape", "<u4", (2,))])
    storage["location/micrograph_shape"] = n.array([42, 24])
    assert isinstance(storage["location/micrograph_shape"], NumericColumn)
    assert len(storage["location/micrograph_shape"]) == 3
    assert all(storage["location/micrograph_shape"][2] == n.array([42, 24]))


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


def test_to_list():
    storage = Dataset.allocate(size=1, fields=[("field1", "u8"), ("field2", "f4"), ("field3", "O")])
    l = storage.to_list()
    assert len(l) == 1
    assert len(l[0]) == 4


def test_to_list_exclude_uid():
    storage = Dataset.allocate(size=1, fields=[("field1", "u8"), ("field2", "f4"), ("field3", "O")])
    storage["field3"][0] = "Hello"
    l = storage.to_list(exclude_uid=True)
    assert len(l) == 1
    assert len(l[0]) == 3
    assert l == [[0, 0.0, "Hello"]]


def test_to_file():
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


def test_from_file(io_data):
    dtype = [("field1", "u4"), ("field2", "f4"), ("field3", "O"), ("field4", "f8"), ("field5", "i8")]

    result = Dataset.load(io_data)
    expected = Dataset.allocate(size=2, fields=dtype)

    expected["field1"] = 42
    expected["field2"] = n.array([3.14, 2.73], dtype="f8")
    expected["field3"][:] = n.array(["Hello", "World"])
    expected["field4"][1] = 1.0
    expected["field5"][0:] = 43

    assert expected.fields() == result.fields()
    assert expected.descr == result.descr
    assert all([n.equal(expected[d[0]], result[d[0]]).all() for d in dtype if d[0] != "uid"])


def test_subset_range_out_of_bounds():
    data = Dataset.allocate(size=3, fields=[("field1", "u8"), ("field2", "f4"), ("field3", "O")])
    subset = data.slice(2, 100)
    assert len(subset) == 1


def test_from_data_none():
    data = Dataset()  # FIXME: Not necessary, remove
    assert len(data) == 0


# FIXME: Is this required?
"""
def test_streaming_bytes():
    dset = Dataset(4, fields=[
        ('field1', 'u8'),
        ('field2', 'f4'),
        ('field3', 'O'),
    ])
    dset['field1'] = 42
    dset['field2'] = n.array([3.14, 2.73, 1.62, 3.14], dtype='f8')
    dset['field3'][:] = n.array(['Hello', 'World', '!', '!'])

    stream = BytesIO()
    for dat in dset.to_stream():
        stream.write(dat)
    stream.seek(0)
    result = dset.from_stream(stream)

    assert dset == result


def test_combine_queries():
    assert Dataset.combine_queries([
        {},
        {'uid': 123},
        {'uid': [456, 789]},
        {'uid': 987, 'location/micrograph_uid': n.array([654])},
        {'location/micrograph_path': '/path/to/mic0.mrc'},
        {'location/micrograph_path': n.array(['/path/to/mic1.mrc'], dtype='O')},
    ]) == {
        'uid': {123, 456, 789, 987},
        'location/micrograph_uid': {654},
        'location/micrograph_path': {'/path/to/mic0.mrc', '/path/to/mic1.mrc'}
    }
"""
