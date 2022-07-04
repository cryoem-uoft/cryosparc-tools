import json
from pathlib import Path
import shutil
import urllib.request
import pytest
import httpretty
from cryosparc.dataset import Dataset


def request_callback(request, uri, response_headers):
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
def big_dset(tmpdir):
    basename = "bench_big_dset"
    existing_path = Path.home() / f"{basename}.cs"

    if existing_path.exists():
        yield Dataset.load(existing_path)
    else:
        import gzip

        download_url = (
            f"https://cryosparc-test-data-dist.s3.amazonaws.com/{basename}.cs.gz"
        )
        compressed_path = Path(tmpdir) / f"{basename}.cs.gz"
        urllib.request.urlretrieve(download_url, compressed_path)
        cs_path = Path(tmpdir) / f"{basename}.cs"
        with gzip.open(compressed_path, "rb") as compressed_file:
            with open(cs_path, "wb") as cs_file:
                shutil.copyfileobj(compressed_file, cs_file)

        yield Dataset.load(cs_path)


@pytest.fixture(scope="session")
def command_core():
    httpretty.enable(allow_net_connect=False)
    httpretty.register_uri(httpretty.POST, "http://localhost:39002/api", body=request_callback)  # type: ignore
    yield
    httpretty.disable()
