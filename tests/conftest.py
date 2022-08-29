import json
from pathlib import Path
import shutil
from time import time
import urllib.request
import pytest
import httpretty
from cryosparc.tools import CryoSPARC
from cryosparc.dataset import Dataset


def request_callback_core(request, uri, response_headers):
    procs = ["hello_world", "get_project", "get_job", "job_send_streamlog"]  # Possible methods list
    body = json.loads(request.body)

    res = None
    if body["method"] == "system.describe":
        procs = [{"name": m} for m in procs]
        res = {"procs": procs}
    elif body["method"] == "hello_world":
        res = {"hello": "world"}
    elif body["method"] == "get_project":
        res = {"uid": "P1", "title": "My Project"}
    elif body["method"] == "get_project_dir_abs":
        res = "/projects/my-project"
    elif body["method"] == "job_send_streamlog":
        res = None

    response_headers["content-type"] = "application/json"
    return [200, response_headers, json.dumps({"result": res})]


def request_callback_vis(request, uri, response_headers):
    procs = ["hello_world"]  # Possible methods list
    body = json.loads(request.body)

    res = None
    if body["method"] == "system.describe":
        procs = [{"name": m} for m in procs]
        res = {"procs": procs}
    elif body["method"] == "hello_world":
        res = {"hello": "world"}

    response_headers["content-type"] = "application/json"
    return [200, response_headers, json.dumps({"result": res})]


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
def cs():
    httpretty.enable(verbose=False, allow_net_connect=False)
    httpretty.register_uri(httpretty.POST, "http://localhost:39002/api", body=request_callback_core)  # type: ignore
    httpretty.register_uri(httpretty.POST, "http://localhost:39003/api", body=request_callback_vis)  # type: ignore
    yield CryoSPARC(license="00000000-0000-0000-0000-000000000000")
    httpretty.disable()
    httpretty.reset()


@pytest.fixture
def project(cs: CryoSPARC):
    return cs.find_project("P1")
