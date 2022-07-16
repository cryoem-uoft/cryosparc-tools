from io import BytesIO
import json
from time import sleep
import pytest
import numpy as n
from cryosparc.dataset import Dataset


@pytest.fixture
def dset(big_dset: Dataset):
    """Copy of big_set that can be mutated. big_dset should *NOT* but mutated
    because it's shared between tests!!!"""
    return big_dset.copy()


@pytest.fixture
def fields():
    return [
        ("uid", "<u8"),
        ("ctf/type", "O"),
        ("ctf/exp_group_id", "<u4"),
        ("ctf/accel_kv", "<f4"),
        ("ctf/cs_mm", "<f4"),
        ("ctf/amp_contrast", "<f4"),
        ("ctf/df1_A", "<f4"),
        ("ctf/df2_A", "<f4"),
        ("ctf/df_angle_rad", "<f4"),
        ("ctf/phase_shift_rad", "<f4"),
        ("ctf/scale", "<f4"),
        ("ctf/scale_const", "<f4"),
        ("ctf/shift_A", "<f4", (2,)),
        ("ctf/tilt_A", "<f4", (2,)),
        ("ctf/trefoil_A", "<f4", (2,)),
        ("ctf/tetra_A", "<f4", (4,)),
        ("ctf/anisomag", "<f4", (4,)),
        ("ctf/bfactor", "<f4"),
        ("location/micrograph_uid", "<u8"),
        ("location/exp_group_id", "<u4"),
        ("location/micrograph_path", "O"),
        ("location/micrograph_shape", "<u4", (2,)),
        ("location/center_x_frac", "<f4"),
        ("location/center_y_frac", "<f4"),
        ("pick_stats/ncc_score", "<f4"),
        ("pick_stats/power", "<f4"),
        ("pick_stats/template_idx", "<u4"),
        ("pick_stats/angle_rad", "<f4"),
    ]


def test_len(big_dset: Dataset):
    assert len(big_dset) == 1961726


def test_fields(big_dset: Dataset):
    fields = big_dset.fields()
    assert len(fields) == 28
    assert fields[0] == "uid"


def test_fields_no_uid(big_dset: Dataset):
    fields = big_dset.fields(exclude_uid=True)
    assert len(fields) == 27
    assert fields[0] != "uid"


def test_dtypes(big_dset: Dataset):
    assert len(big_dset.descr) == 28


def test_contains(big_dset: Dataset):
    assert "location/micrograph_uid" in big_dset


def test_add_fields(dset: Dataset):
    dset.add_fields(["foo", "baz"], ["u8", "f4"])
    assert len(dset.fields()) == 30


def test_rename_fields(dset: Dataset):
    renamed_fields = [f if f == "uid" else "{}__RENAMED".format(f) for f in dset.fields()]
    dset.rename_fields("{}__RENAMED".format)
    assert dset.fields() == renamed_fields


def test_get_items(benchmark, dset: Dataset):
    @benchmark
    def _():
        assert len(dset.rows) == 1961726


def test_get_items_to_list(benchmark, dset, fields):
    @benchmark
    def _():
        items = dset.rows
        first = items[0]
        l = first.to_list()
        assert len(l) == len(fields)
        assert any(l)


def test_get_items_to_dict(benchmark, dset, fields):
    @benchmark
    def _():
        items = dset.rows
        first = items[0]
        d = first.to_dict()
        assert list(d.keys()) == [f[0] for f in fields]
        assert any(d.values())


def test_get_items_to_item_dict(benchmark, dset, fields):
    @benchmark
    def _():
        first = dset.rows[0]
        item_d = first.to_item_dict()
        expected_keys = list({f[0]: None for f in fields}.keys())
        assert list(item_d.keys()) == expected_keys
        assert all(item_d.values())


def test_filter_fields_list(benchmark, dset: Dataset):
    @benchmark
    def _():
        d = dset.copy()
        fields = [
            "pick_stats/ncc_score",
            "pick_stats/power",
            "pick_stats/template_idx",
            "pick_stats/angle_rad",
        ]
        d.drop_fields(fields)
        assert d.fields() == ["uid"] + fields


def test_filter_prefixes(benchmark, dset: Dataset):
    @benchmark
    def _():
        prefix = "pick_stats/"
        filtered = dset.drop_fields(lambda field: field.startswith(prefix))
        assert len(filtered.fields()) == 24


def test_copy_fields(benchmark, dset: Dataset):
    @benchmark
    def _():
        dset.copy_fields(["ctf/type"], ["bar"])
        items = dset.rows
        ctf_types = [item["ctf/type"] for item in items]
        bars = [item["bar"] for item in items]
        assert ctf_types == bars


def test_append(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        new_dset = big_dset.copy().reassign_uids()
        new_dset.append(big_dset)
        assert len(new_dset) == len(big_dset) * 2


def test_append_union(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        other = big_dset.copy()
        other = other.union(big_dset)
        assert len(other) == len(big_dset)


def test_append_many(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        empty = Dataset()
        other = big_dset.copy().reassign_uids()

        new_dset = Dataset.append_many(dset, empty, other)
        assert len(new_dset) == len(big_dset) * 2


def test_append_many_union(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        empty = Dataset()
        other = dset.copy().reassign_uids()

        new_dset = Dataset.union_many(dset, dset, empty, other)
        assert len(new_dset) == len(big_dset) * 2


def test_append_many_union_repeat_allowed(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        empty = Dataset()
        other = dset.copy().reassign_uids()
        new_dset = Dataset.append_many(dset, dset, empty, other, repeat_allowed=True)
        assert len(new_dset) == len(big_dset) * 3


def test_append_many_simple(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        empty = Dataset.allocate(0, dset.descr)
        other = dset.copy().reassign_uids()

        new_dset = Dataset.append_many(dset, empty, other, assert_same_fields=True)
        assert len(new_dset) == len(big_dset) * 2


def test_append_many_simple_interlace(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        other = dset.copy()
        other.reassign_uids()

        new_dset = Dataset.interlace(dset, other)
        assert len(new_dset) == len(big_dset) * 2
        assert new_dset["uid"][0] == dset["uid"][0]
        assert new_dset["uid"][1] == other["uid"][0]


def test_append_replace(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other = Dataset.allocate(size=10000, fields=dset.descr)
        new_dset = dset.replace({}, other)
        assert len(new_dset) == len(big_dset) + len(other)
        assert new_dset["uid"][-1] == other["uid"][-1]


def test_append_replace_unique(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other = Dataset.allocate(size=10000, fields=dset.descr)
        new_dset = dset.replace({}, other, assume_unique=True)
        assert len(new_dset) == len(big_dset) + len(other)
        assert new_dset["uid"][-1] == other["uid"][-1]


def test_append_replace_empty(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        other = Dataset.allocate(0, fields=dset.descr)
        new_dset = dset.replace({}, other)
        assert len(new_dset) == len(big_dset)
        assert new_dset == big_dset


def test_append_replace_empty_query(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other = Dataset.allocate(0, fields=dset.descr)
        dset.replace({"location/micrograph_uid": 6655121610611186569}, other)
        assert len(dset) == len(big_dset) - 1191


def test_append_replace_query(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other = Dataset.allocate(size=10000, fields=dset.descr)
        dset.replace({"location/micrograph_uid": 6655121610611186569}, other)
        assert len(dset) == len(big_dset) + len(other) - 1191


def test_append_replace_query_unique(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other = Dataset.allocate(size=10000, fields=dset.descr)
        dset.replace({"location/micrograph_uid": 6655121610611186569}, other, assume_unique=True)
        assert len(dset) == len(big_dset) + len(other) - 1191


def test_append_replace_many(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other1 = Dataset.allocate(size=5000, fields=dset.descr)
        other2 = Dataset.allocate(size=5000, fields=dset.descr)
        new_dset = dset.replace({}, other1, other2)
        assert len(new_dset) == len(big_dset) + len(other1) + len(other2)
        assert new_dset["uid"][-1] == other2["uid"][-1]


def test_append_replace_many_unique(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other1 = Dataset.allocate(size=5000, fields=dset.descr)
        other2 = Dataset.allocate(size=5000, fields=dset.descr)
        new_dset = dset.replace({}, other1, other2, assume_unique=True)
        assert len(new_dset) == len(big_dset) + len(other1) + len(other2)
        assert new_dset["uid"][-1] == other2["uid"][-1]


def test_append_replace_many_query(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other1 = Dataset.allocate(size=5000, fields=dset.descr)
        other2 = Dataset.allocate(size=5000, fields=dset.descr)
        dset.replace(
            {"location/micrograph_uid": [2539634023577218663, 6655121610611186569]}, other1, other2, assume_unique=True
        )
        assert len(dset) == len(big_dset) + len(other1) + len(other2) - 1191 - 1210


def test_innerjoin(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        other = dset.slice(500000, 1500000)
        joined = dset.innerjoin(other)
        assert len(joined) == 1000000


def test_innerjoin_many(benchmark, dset: Dataset):
    @benchmark
    def _():
        other1 = dset.slice(500000, 1250000)
        other2 = dset.slice(750000, 1500000)
        new_dset = Dataset()
        new_dset = new_dset.innerjoin_many([dset, other1, other2])
        assert len(new_dset) == 500000


def test_filter(benchmark, big_dset: Dataset):
    # FIXME: This is redundant because of subset_idxs
    @benchmark
    def _():
        dset = big_dset.copy()
        dset = dset.indexes([i for i in range(1500000) if i % 2 == 0])  # Even entries up to 1.5 million
        assert len(dset) == 750000


def test_subset_idxs(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.indexes([i for i in range(1500000) if i % 2 == 0])  # Even entries up to 1.5 million
        assert len(dset) == len(big_dset), "Should not mutate original dset"
        assert len(new_dset) == 750000


def test_subset_mask(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():

        # Even entries up to 1.5 million
        mask = n.array([i < 1500000 and i % 2 == 0 for i in range(len(dset))])
        new_dset = dset.mask(mask)
        assert len(dset) == len(big_dset), "Should not mutate original dset"
        assert len(new_dset) == 750000


def test_subset_query(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.query(lambda item: item["location/micrograph_uid"] == 6655121610611186569)
        assert len(dset) == len(big_dset), "Should not mutate original dset"
        assert len(new_dset) == 1191


def test_subset_simple_query_1(benchmark, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.query({"location/micrograph_uid": 6655121610611186569})
        assert len(new_dset) == 1191


def test_subset_simple_query_2(benchmark, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.query({"location/micrograph_path": "J3/imported/18jam15a_0008_ali_DW.mrc"})
        assert len(dset) > 0
        assert len(new_dset) < len(dset)


def test_subset_simple_query_3(benchmark, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.query(
            {"uid": dset["uid"][1000:2000], "location/micrograph_path": "J3/imported/18jam15a_0008_ali_DW.mrc"}
        )
        assert len(new_dset) > 0
        assert len(new_dset) < len(dset)


def test_subset_simple_query_empty(benchmark, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.query({})
        assert len(new_dset) == len(dset)


def test_subset_simple_query_fake_field(benchmark, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.query({"fake_field": 42})
        assert new_dset == dset
        assert len(new_dset) == len(dset)


def test_subset_simple_query_nomatch(benchmark, big_dset, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.query({"uid": [42]})
        assert len(dset) == len(big_dset), "Should not mutate original dset"
        assert new_dset != dset
        assert len(new_dset) == 0


def test_subset_split_by(benchmark, dset: Dataset):
    @benchmark
    def _():
        dsets = dset.split_by("location/micrograph_uid")
        assert len(dsets) == 1644
        assert len(dsets[2539634023577218663]) == 1210


def test_items_split_by(benchmark, dset: Dataset):
    @benchmark
    def _():
        rows_split = dset.rows.split_by("location/micrograph_uid")
        assert len(rows_split) == 1644
        assert len(rows_split[2539634023577218663]) == 1210


def test_copy(benchmark, dset: Dataset):
    @benchmark
    def _():
        new_dset = dset.copy()
        assert id(new_dset) != id(dset)
        assert new_dset == dset
        assert len(new_dset) == len(dset)


# FIXME: Not required for this round of tests
"""
def test_to_dataframe(benchmark, dset: Dataset):
    @benchmark
    def _():
        # FIXME: Not required in this version
        dframe = dset.to_dataframe()
        assert dframe is not None


def test_streaming_bytes(benchmark, dset: Dataset):
    @benchmark
    def _():
        total_bytes = 0
        stream = BytesIO()
        for dat in dset.to_stream():
            stream.write(dat)
            total_bytes += len(dat)
        stream.seek(0)
        assert total_bytes > 0
        assert stream.read(6) == dataset.VERSION_MAGIC_PREFIXES[dataset.HIGHEST_VERSION]
        assert int(n.frombuffer(stream.read(4), dtype=n.uint32)[0]) == \
            len(json.dumps(dset.descr, separators=(',', ':')).encode()) + 1  # not sure why + 1??


def test_from_streaming_bytes(benchmark, big_dset, big_dset_stream: BytesIO):
    @benchmark
    def _():
        result = Dataset.from_stream(big_dset_stream)
        big_dset_stream.seek(0)
        assert len(result) == len(big_dset)
        assert result.fields() == big_dset.fields()
"""
