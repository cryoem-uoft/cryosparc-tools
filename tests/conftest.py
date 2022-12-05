import json
from pathlib import Path
import shutil
from time import time
from typing import Any, Dict
import urllib.request
import pytest
import httpretty
import numpy as n
from numpy.core.records import fromrecords

from cryosparc.tools import CryoSPARC
from cryosparc.dataset import Dataset as BaseDataset, Row
from cryosparc.util import default_rng

# Always use this class for testing to ensure Dataset#items property is never
# used internally. Downstream CryoSPARC relies on this.
class Dataset(BaseDataset[Row]):
    @property
    def items(self):
        return self.rows()

    def shuffle(self):
        idxs = n.arange(len(self))
        default_rng().shuffle(idxs)
        return self.take(idxs)


def request_callback_core(request, uri, response_headers):
    body = json.loads(request.body)
    procs = {
        "hello_world": {"hello": "world"},
        "get_id_by_email_password": "6372a35e821ed2b71d9fe4e3",
        "get_job": {"uid": "J1", "project_uid": "P1", "job_type": "homo_abinit"},
        "get_project_dir_abs": "/projects/my-project",
        "get_project": {"uid": "P1", "title": "My Project"},
        "job_send_streamlog": None,
    }
    procs["system.describe"] = {"procs": [{"name": m} for m in procs]}
    response_headers["content-type"] = "application/json"
    return [200, response_headers, json.dumps({"result": procs[body["method"]]})]


def request_callback_vis(request, uri, response_headers):
    body = json.loads(request.body)
    procs: Dict[str, Any] = {"hello_world": {"hello": "world"}}
    procs["system.describe"] = {"procs": [{"name": m} for m in procs]}
    response_headers["content-type"] = "application/json"
    return [200, response_headers, json.dumps({"result": procs[body["method"]]})]


@pytest.fixture(scope="session")
def big_dset():
    basename = "bench_big_dset"
    existing_path = Path.home() / f"{basename}.cs"

    print("")  # Keeps printing clean
    if existing_path.exists():
        print(f"Using local sample data {existing_path}; loading...", end="\r")
        tic = time()

        dset = Dataset.load(existing_path)
        print(f"Using local sample data {existing_path}; loaded in {time() - tic:.3f} seconds")
        yield dset
    else:
        import gzip
        import tempfile

        download_url = f"https://cryosparc-test-data-dist.s3.amazonaws.com/{basename}.cs.gz"
        with tempfile.TemporaryDirectory() as tmpdir:
            compressed_path = Path(tmpdir) / f"{basename}.cs.gz"
            cs_path = Path(tmpdir) / f"{basename}.cs"

            def download_report_hook(chunk, chunk_size, total_size):
                total_chunks = total_size / chunk_size
                progress = chunk / total_chunks * 100
                print(f"Downloading big dataset sample data ({total_size} bytes, {progress:.0f}%)", end="\r")

            urllib.request.urlretrieve(download_url, compressed_path, reporthook=download_report_hook)
            with gzip.open(compressed_path, "rb") as compressed_file:
                with open(cs_path, "wb") as cs_file:
                    shutil.copyfileobj(compressed_file, cs_file)

            print("")
            print("Downloaded big dataset sample data; loading...", end="\r")
            tic = time()
            dset = Dataset.load(cs_path)
            print(f"Downloaded big dataset sample data; loaded in {time() - tic:.3f} seconds")
            yield dset


@pytest.fixture
def t20s_dset():
    # fmt: off
    return Dataset(fromrecords([
        (15762140416835289736, "J1/imported/015762140416835289736_14sep05c_00024sq_00003hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 14011218726240589193, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 7260969837812210921, "J1/imported/007260969837812210921_14sep05c_00024sq_00003hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  5837173768828397691, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (10800491058062947733, "J1/imported/010800491058062947733_14sep05c_00024sq_00004hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  1196980374560176443, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (11991710339096669080, "J1/imported/011991710339096669080_14sep05c_00024sq_00006hl_00003es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  7129025008161402787, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (10066607318144328742, "J1/imported/010066607318144328742_14sep05c_c_00003gr_00014sq_00002hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  9773685899756902191, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (14059663819601300436, "J1/imported/014059663819601300436_14sep05c_c_00003gr_00014sq_00004hl_00004es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  3988647519168980973, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (  273322377584193832, "J1/imported/000273322377584193832_14sep05c_c_00003gr_00014sq_00005hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 16231222940328059768, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 3499866221971477386, "J1/imported/003499866221971477386_14sep05c_c_00003gr_00014sq_00005hl_00003es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  9286395140289536592, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 3848558957898561278, "J1/imported/003848558957898561278_14sep05c_c_00003gr_00014sq_00005hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  3782318515582039935, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 9829431748377878115, "J1/imported/009829431748377878115_14sep05c_c_00003gr_00014sq_00006hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 17466118442802771446, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 2867928639359948788, "J1/imported/002867928639359948788_14sep05c_c_00003gr_00014sq_00006hl_00003es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 17119601282827635162, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 8455357976034962133, "J1/imported/008455357976034962133_14sep05c_c_00003gr_00014sq_00006hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0, 10475424305309211320, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 4503027956576488873, "J1/imported/004503027956576488873_14sep05c_c_00003gr_00014sq_00007hl_00004es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  4382433041497215537, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 6657609670738933979, "J1/imported/006657609670738933979_14sep05c_c_00003gr_00014sq_00007hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  7889866034561464467, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 2455776040951224707, "J1/imported/002455776040951224707_14sep05c_c_00003gr_00014sq_00008hl_00005es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  5928873226622420600, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 8270327040298559173, "J1/imported/008270327040298559173_14sep05c_c_00003gr_00014sq_00009hl_00004es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  2857337051791529295, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        ( 2424724526127669191, "J1/imported/002424724526127669191_14sep05c_c_00003gr_00014sq_00010hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  1126123402376461244, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (15890047193456637277, "J1/imported/015890047193456637277_14sep05c_c_00003gr_00014sq_00011hl_00002es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  4134148380692987552, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (11981388473018700511, "J1/imported/011981388473018700511_14sep05c_c_00003gr_00014sq_00011hl_00003es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  8989915083856388927, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0),  # noqa
        (12098792569994765948, "J1/imported/012098792569994765948_14sep05c_c_00003gr_00014sq_00011hl_00004es.frames.tif", [  38, 7676, 7420], 0.6575, 0, "TIFF", 0,  7878917727182588741, "J1/imported/norm-amibox05-0.mrc", 0, [7676, 7420], 0, 0, 0, 300., 2.7, 53., 0, 0, 8, "", "", 0, [0, 0], 0., "", 0, 0., 0., 0)  # noqa
    ], dtype=[  # type: ignore
        ("uid", "<u8"),
        ("movie_blob/path", "O"),
        ("movie_blob/shape", "<u4", (3,)),
        ("movie_blob/psize_A", "<f4"),
        ("movie_blob/is_gain_corrected", "<u4"),
        ("movie_blob/format", "O"),
        ("movie_blob/has_defect_file", "<u4"),
        ("movie_blob/import_sig", "<u8"),
        ("gain_ref_blob/path", "O"),
        ("gain_ref_blob/idx", "<u4"),
        ("gain_ref_blob/shape", "<u4", (2,)),
        ("gain_ref_blob/flip_x", "<u4"),
        ("gain_ref_blob/flip_y", "<u4"),
        ("gain_ref_blob/rotate_num", "<u4"),
        ("mscope_params/accel_kv", "<f4"),
        ("mscope_params/cs_mm", "<f4"),
        ("mscope_params/total_dose_e_per_A2", "<f4"),
        ("mscope_params/phase_plate", "<u4"),
        ("mscope_params/neg_stain", "<u4"),
        ("mscope_params/exp_group_id", "<u4"),
        ("mscope_params/defect_path", "O"),
        ("micrograph_blob/path", "O"),
        ("micrograph_blob/idx", "<u4"),
        ("micrograph_blob/shape", "<u4", (2,)),
        ("micrograph_blob/psize_A", "<f4"),
        ("micrograph_blob/format", "O"),
        ("micrograph_blob/is_background_subtracted", "<u4"),
        ("micrograph_blob/vmin", "<f4"),
        ("micrograph_blob/vmax", "<f4"),
        ("micrograph_blob/import_sig", "<u8")
    ]))
    # fmt: on


@pytest.fixture
def cs():
    httpretty.enable(verbose=False, allow_net_connect=False)
    httpretty.register_uri(httpretty.POST, "http://localhost:39002/api", body=request_callback_core)  # type: ignore
    httpretty.register_uri(httpretty.POST, "http://localhost:39003/api", body=request_callback_vis)  # type: ignore
    yield CryoSPARC(license="00000000-0000-0000-0000-000000000000", email="test@structura.bio", password="password")
    httpretty.disable()
    httpretty.reset()


@pytest.fixture
def project(cs: CryoSPARC):
    return cs.find_project("P1")
