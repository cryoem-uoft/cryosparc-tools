from io import BytesIO
import pytest
import numpy as n
import cryosparc.dataset as ds
from cryosparc.dataset import Dataset, Row


@pytest.fixture
def dset(big_dset: Dataset):
    """Copy of big_set that can be mutated. big_dset should *NOT* but mutated
    because it's shared between tests!!!"""
    return big_dset.copy()


@pytest.fixture
def fields():
    return [
        ("uid", "<u8"),
        ("ctf/type", "|O"),
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
        ("location/micrograph_path", "|O"),
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
    # assert len(fields) == 26
    assert fields[0] == "uid"


def test_fields_no_uid(big_dset: Dataset):
    fields = big_dset.fields(exclude_uid=True)
    assert len(fields) == 27
    # assert len(fields) == 25
    assert fields[0] != "uid"


def test_dtypes(big_dset: Dataset):
    assert len(big_dset.descr()) == 28
    # assert len(big_dset.descr()) == 26


def test_contains(big_dset: Dataset):
    assert "location/micrograph_uid" in big_dset


def test_add_fields(dset: Dataset):
    dset.add_fields(["foo", "baz"], ["u8", "f4"])
    assert len(dset.fields()) == 30
    # assert len(dset.fields()) == 28


def test_rename_fields(dset: Dataset):
    renamed_fields = [f if f == "uid" else "{}__RENAMED".format(f) for f in dset.fields()]
    dset.rename_fields("{}__RENAMED".format)
    assert dset.fields() == renamed_fields


def test_get_items(benchmark, dset: Dataset):
    @benchmark
    def _():
        dset._rows = None
        assert len(dset.rows()) == 1961726


def test_get_items_to_list(benchmark, dset, fields):
    @benchmark
    def _():
        dset._rows = None
        items = dset.rows()
        first = items[0]
        l = first.to_list()
        assert len(l) == len(fields)
        assert any(l)


def test_get_items_to_dict(benchmark, dset: Dataset[Row], fields):
    @benchmark
    def _():
        dset._rows = None
        items = dset.rows()
        first = items[0]
        d = first.to_dict()
        assert set(d.keys()) == {f[0] for f in fields}
        assert any(d.values())


def test_get_items_to_item_dict(benchmark, dset: Dataset[Row], fields):
    @benchmark
    def _():
        dset._rows = None
        first = dset.rows()[0]
        item_d = first.to_item_dict()
        assert set(item_d.keys()) == {f[0] for f in fields}
        assert any(item_d.values())


def test_filter_fields_list(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        fields = [
            "pick_stats/ncc_score",
            "pick_stats/power",
            "pick_stats/template_idx",
            "pick_stats/angle_rad",
        ]
        dset.filter_fields(fields)
        assert dset.fields() == ["uid"] + fields


def test_filter_prefixes(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        filtered = dset.filter_fields(lambda n: not n.startswith("pick_stats"))
        assert len(filtered.fields()) == 24
        # assert len(filtered.fields()) == 22


def test_copy_fields(benchmark, big_dset: Dataset):
    @benchmark
    def _():
        dset = big_dset.copy()
        dset.copy_fields(["ctf/type", "ctf/exp_group_id"], ["foo", "bar"])
        # dset.copy_fields(["ctf/exp_group_id"], ["bar"])
        assert all(dset["ctf/type"] == dset["foo"])
        assert all(dset["ctf/exp_group_id"] == dset["bar"])


def test_append(benchmark, big_dset: Dataset, dset: Dataset):
    dset = dset.reassign_uids()
    new_dset = benchmark(dset.append, big_dset)
    assert len(new_dset) == len(big_dset) * 2


def test_append_union(benchmark, big_dset: Dataset, dset: Dataset):
    other = benchmark(dset.union, big_dset)
    assert len(other) == len(big_dset)


def test_append_many(benchmark, big_dset, dset: Dataset):
    empty = Dataset.allocate(0, dset.descr())
    other = dset.copy().reassign_uids()
    new_dset = benchmark(Dataset.append_many, dset, empty, other)
    assert len(new_dset) == len(big_dset) * 2


def test_append_many_union(benchmark, big_dset, dset: Dataset):
    empty = Dataset.allocate(0, dset.descr())
    other = dset.copy().reassign_uids()
    new_dset = benchmark(Dataset.union_many, dset, dset, empty, other)
    assert len(new_dset) == len(big_dset) * 2


def test_append_many_union_repeat_allowed(benchmark, big_dset, dset: Dataset):
    empty = Dataset.allocate(0, dset.descr())
    other = dset.copy().reassign_uids()
    new_dset = benchmark(Dataset.append_many, dset, dset, empty, other, repeat_allowed=True)
    assert len(new_dset) == len(big_dset) * 3


def test_append_many_simple(benchmark, big_dset, dset: Dataset):
    empty = Dataset.allocate(0, dset.descr())
    other = dset.copy().reassign_uids()
    new_dset = benchmark(Dataset.append_many, dset, empty, other, assert_same_fields=True)
    assert len(new_dset) == len(big_dset) * 2


def test_append_many_simple_interlace(benchmark, big_dset, dset: Dataset):
    other = dset.copy()
    other.reassign_uids()

    new_dset = benchmark(Dataset.interlace, dset, other)
    assert len(new_dset) == len(big_dset) * 2
    assert new_dset["uid"][0] == dset["uid"][0]
    assert new_dset["uid"][1] == other["uid"][0]


def test_append_replace(benchmark, big_dset: Dataset, dset: Dataset):
    other = Dataset.allocate(size=10000, fields=dset.descr())
    new_dset = benchmark(dset.replace, {}, other)
    assert len(new_dset) == len(big_dset) + len(other)
    assert new_dset["uid"][-1] == other["uid"][-1]


def test_append_replace_unique(benchmark, big_dset: Dataset, dset: Dataset):
    other = Dataset.allocate(size=10000, fields=dset.descr())
    new_dset = benchmark(dset.replace, {}, other, assume_disjoint=True, assume_unique=True)
    assert len(new_dset) == len(big_dset) + len(other)
    assert new_dset["uid"][-1] == other["uid"][-1]


def test_append_replace_empty(benchmark, big_dset, dset: Dataset):
    other = Dataset.allocate(0, fields=dset.descr())
    new_dset = benchmark(dset.replace, {}, other)
    assert len(new_dset) == len(big_dset)
    assert new_dset == big_dset


def test_append_replace_empty_query(benchmark, big_dset: Dataset, dset: Dataset):
    other = Dataset.allocate(0, fields=dset.descr())
    new_dset = benchmark(dset.replace, {"location/micrograph_uid": 6655121610611186569}, other)
    assert len(new_dset) == len(big_dset) - 1191


def test_append_replace_query(benchmark, big_dset: Dataset, dset: Dataset):
    other = Dataset.allocate(size=10000, fields=dset.descr())
    new_dset = benchmark(dset.replace, {"location/micrograph_uid": 6655121610611186569}, other)
    assert len(new_dset) == len(big_dset) + len(other) - 1191


def test_append_replace_query_unique(benchmark, big_dset: Dataset, dset: Dataset):
    other = Dataset.allocate(size=10000, fields=dset.descr())
    new_dset = benchmark(
        dset.replace, {"location/micrograph_uid": 6655121610611186569}, other, assume_disjoint=True, assume_unique=True
    )
    assert len(new_dset) == len(big_dset) + len(other) - 1191


def test_append_replace_many(benchmark, big_dset: Dataset, dset: Dataset):
    other1 = Dataset.allocate(size=5000, fields=dset.descr())
    other2 = Dataset.allocate(size=5000, fields=dset.descr())
    new_dset = benchmark(dset.replace, {}, other1, other2)
    assert len(new_dset) == len(big_dset) + len(other1) + len(other2)
    assert new_dset["uid"][-1] == other2["uid"][-1]


def test_append_replace_many_unique(benchmark, big_dset: Dataset, dset: Dataset):
    other1 = Dataset.allocate(size=5000, fields=dset.descr())
    other2 = Dataset.allocate(size=5000, fields=dset.descr())
    new_dset = benchmark(dset.replace, {}, other1, other2, assume_disjoint=True, assume_unique=True)
    assert len(new_dset) == len(big_dset) + len(other1) + len(other2)
    assert new_dset["uid"][-1] == other2["uid"][-1]


def test_append_replace_many_query(benchmark, big_dset: Dataset, dset: Dataset):
    other1 = Dataset.allocate(size=5000, fields=dset.descr())
    other2 = Dataset.allocate(size=5000, fields=dset.descr())
    new_dset = benchmark(
        dset.replace,
        {"location/micrograph_uid": [2539634023577218663, 6655121610611186569]},
        other1,
        other2,
        assume_disjoint=True,
        assume_unique=True,
    )
    assert len(new_dset) == len(big_dset) + len(other1) + len(other2) - 1191 - 1210


def test_innerjoin(benchmark, dset: Dataset):
    other = dset.slice(500000, 1500000)
    joined = benchmark(dset.innerjoin, other, assume_unique=True)
    assert len(joined) == 1000000


def test_innerjoin_many(benchmark, dset: Dataset):
    other1 = dset.slice(500_000, 1_250_000)
    other2 = dset.slice(750_000, 1_500_000)
    expected = dset.slice(750_000, 1_250_000)
    new_dset = benchmark(Dataset.innerjoin_many, dset, other1, other2, assume_unique=True)
    assert new_dset == expected


def test_filter(benchmark, dset: Dataset):
    # FIXME: This is redundant because of subset_idxs
    new_dset = benchmark(dset.indexes, list(range(0, 1_500_000, 2)))  # Even entries up to 1.5 million
    assert len(new_dset) == 750_000


def test_subset_idxs(benchmark, dset: Dataset):
    new_dset = benchmark(dset.indexes, list(range(0, 1_500_000, 2)))  # Even entries up to 1.5 million
    assert len(new_dset) == 750_000


def test_subset_mask(benchmark, big_dset, dset: Dataset):
    # Even entries up to 1.5 million
    mask = n.array([i < 1500000 and i % 2 == 0 for i in range(len(dset))])
    new_dset = benchmark(dset.mask, mask)
    assert len(dset) == len(big_dset), "Should not mutate original dset"
    assert len(new_dset) == 750_000


def test_subset_query(benchmark, big_dset, dset: Dataset):
    new_dset = benchmark(dset.query, lambda item: item["location/micrograph_uid"] == 6655121610611186569)
    assert len(dset) == len(big_dset), "Should not mutate original dset"
    assert len(new_dset) == 1191


def test_subset_simple_query_1(benchmark, dset: Dataset):
    new_dset = benchmark(dset.query, {"location/micrograph_uid": 6655121610611186569})
    assert len(new_dset) == 1191


def test_subset_simple_query_2(benchmark, dset: Dataset):
    new_dset = benchmark(dset.query, {"location/micrograph_path": "J3/imported/18jam15a_0008_ali_DW.mrc"})
    assert len(dset) > 0
    assert len(new_dset) < len(dset)


def test_subset_simple_query_3(benchmark, dset: Dataset):
    new_dset = benchmark(
        dset.query, {"uid": dset["uid"][1000:2000], "location/micrograph_path": "J3/imported/18jam15a_0008_ali_DW.mrc"}
    )
    assert len(new_dset) > 0
    assert len(new_dset) < len(dset)


def test_subset_simple_query_empty(benchmark, dset: Dataset):
    new_dset = benchmark(dset.query, {})
    assert len(new_dset) == len(dset)


def test_subset_simple_query_fake_field(benchmark, dset: Dataset):
    new_dset = benchmark(dset.query, {"fake_field": 42})
    assert new_dset == dset
    assert len(new_dset) == len(dset)


def test_subset_simple_query_nomatch(benchmark, big_dset, dset: Dataset):
    new_dset = benchmark(dset.query, {"uid": [42]})
    assert len(dset) == len(big_dset), "Should not mutate original dset"
    assert new_dset != dset
    assert len(new_dset) == 0


def test_subset_split_by(benchmark, dset: Dataset):
    dsets = benchmark(dset.split_by, "location/micrograph_uid")
    assert len(dsets) == 1644
    assert len(dsets[2539634023577218663]) == 1210


def test_items_split_by(benchmark, dset: Dataset):
    rows_split = benchmark(dset.rows().split_by, "location/micrograph_uid")
    assert len(rows_split) == 1644
    assert len(rows_split[2539634023577218663]) == 1210


def test_copy(benchmark, dset: Dataset):
    new_dset = benchmark(dset.copy)
    assert id(new_dset) != id(dset)
    assert new_dset == dset
    assert len(new_dset) == len(dset)


def test_streaming_bytes(benchmark, dset: Dataset):
    stream = BytesIO()

    @benchmark
    def _():
        total_bytes = 0
        for dat in dset.stream():
            stream.write(dat)
            total_bytes += len(dat)
        stream.seek(0)
        assert total_bytes > 0

    assert stream.read(6) == ds.FORMAT_MAGIC_PREFIXES[ds.NEWEST_FORMAT]


def test_from_streaming_bytes(benchmark, big_dset: Dataset):
    stream = BytesIO()
    for dat in big_dset.stream():
        stream.write(dat)
    stream.seek(0)

    def load():
        result = Dataset.load(stream)
        stream.seek(0)
        return result

    result = benchmark(load)
    assert len(result) == len(big_dset)