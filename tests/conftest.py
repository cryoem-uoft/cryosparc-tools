from pathlib import Path
import pytest
from cryosparc.dataset import Dataset

@pytest.fixture(scope='session')
def big_dset():
    basename = 'bench_big_dset'
    existing_path = Path.home() / f'{basename}.cs'

    if existing_path.exists():
        yield Dataset.load(existing_path)
    else:
        # FIXME: Download test data from from S3
        import requests, tarfile, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            download_url = f'https://cryosparc-test-data-dist.s3.amazonaws.com/{basename}.tar.gz'
            download_checksum_url = f'{download_url}.sha256'
            tar_path = f'{tmpdir}/{basename}.tar.gz'
            cs_path = f'{tmpdir}/{basename}.cs'

            req = requests.get(download_checksum_url)
            req.raise_for_status()
            # checksum = req.text.strip()
            # util.download_and_verify_url(download_url, tar_path, checksum)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            yield Dataset.load(cs_path)
