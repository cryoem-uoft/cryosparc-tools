from io import BytesIO
from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Union
import os
import re
import tempfile

if TYPE_CHECKING:
    import numpy.typing as nt  # type: ignore

from . import mrc
from .command import CommandClient
from .dataset import Dataset
from .project import Project
from .job import Job
from .util import bopen


ONE_MB = 2**20
LICENSE_REGEX = re.compile(r"[a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12}")


class CryoSPARC:
    """
    High-level class for interfacing with a running cryoSPARC instance.

    Initialize with the host and base port of the running cryoSPARC instance
    accessible on the current network.

    Example usage:

    ```
    from cryosparc import CryoSPARC

    cs = CryoSPARC(port=39000)
    project = cs.find_project('P3')
    job = project.find_job('J42')
    micrographs = job.load_output('exposures')

    # Remove corrupt exposures
    filtered_micrographs = micrographs.query(is_mic_corrupt)
    job.save_output('micrographs', filtered_micrographs)
    ```

    """

    def __init__(
        self,
        license: str = os.getenv("CRYOSPARC_LICENSE_ID", ""),
        host: str = "localhost",
        port: int = 39000,
        timeout: int = 300,
    ):
        assert LICENSE_REGEX.fullmatch(license), f"Invalid or unspecified cryoSPARC license ID {license}"

        self.cli = CommandClient(host=host, port=port + 2, headers={"License-ID": license}, timeout=timeout)
        self.vis = CommandClient(host=host, port=port + 3, headers={"License-ID": license}, timeout=timeout)

    def test_connection(self):
        if self.cli.test_connection():  # type: ignore
            print(f"Connection succeeded to cryoSPARC command_core at {self.cli.url}")
        else:
            print(f"Connection FAILED to cryoSPARC command_core at {self.cli.url}")
            return False

        with self.vis._request() as response:
            if response.read():
                print(f"Connection succeeded to cryoSPARC command_vis at {self.vis.url}")
            else:
                print(f"Connection FAILED to cryoSPARC command_vis at {self.vis.url}")
                return False

        return True

    def find_project(self, project_uid: str) -> Project:
        project = Project(self, project_uid)
        project.refresh()
        return project

    def find_job(self, project_uid: str, job_uid: str) -> Job:
        job = Job(self, project_uid, job_uid)
        job.refresh()
        return job

    def download(self, project_uid: str, path: Union[str, PurePosixPath]):
        """
        Open a file in the current project for reading. Example usage:

        ```
        cs = CryoSPARC()
        with cs.download('P3', 'J42/particles.cs') as req:
            particles = Dataset.load(req)
        print(particles)
        ```
        """
        data = {"project_uid": project_uid, "path_rel": str(path)}
        return self.vis._json_request("/get_project_file", data=data)

    def download_file(self, project_uid: str, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        """
        Download
        """
        with self.download(project_uid, path) as response:
            with bopen(target, "wb") as f:
                f.write(response.read())
        return target

    def download_dataset(self, project_uid: str, path: Union[str, PurePosixPath]):
        with self.download(project_uid, path) as response:
            size = response.headers.get("Content-Length")
            mime = response.headers.get("Content-Type")
            if mime == "application/x-cryosparc-dataset":
                # Stream format; can load directly without seek
                return Dataset.load(response)

            # Numpy format, cannot load directly because requires seekable
            if size and int(size) < ONE_MB:
                # Smaller than 1MB, just read all into memory and load
                return Dataset.load(BytesIO(response.read()))

            # Read into temporary file in 1MB chunks. Load from that temporary file
            with tempfile.TemporaryFile("w+b", suffix=".cs") as f:
                data = response.read(ONE_MB)
                while data:
                    f.write(data)
                    data = response.read(ONE_MB)
                f.seek(0)
                return Dataset.load(f)

    def download_mrc(self, project_uid: str, path: Union[str, PurePosixPath]):
        with self.download(project_uid, path) as response:
            with tempfile.TemporaryFile("w+b", suffix=".cs") as f:
                data = response.read(ONE_MB)
                while data:
                    f.write(data)
                    data = response.read(ONE_MB)
                f.seek(0)
                return mrc.read(f)  # FIXME: Optimize file reading

    def upload(self, project_uid: str, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        """
        Open a file from the current project for reading. Note that this
        response is not seekable.
        """
        with bopen(file) as f:
            url = f"/upload_project_file/{project_uid}"
            query = {"path": path}
            with self.vis._request(url=url, query=query, data=f) as res:
                assert res.status >= 200 and res.status < 300, (
                    f"Could not upload project {project_uid} file {path}.\n"
                    f"Response from cryoSPARC: {res.read().decode()}"
                )

    def upload_dataset(self, project_uid: str, path: Union[str, PurePosixPath], dset: Dataset):
        """
        Similar to upload() method, but works with in-memory datasets
        """
        # FIXME: Get dataset memory buffer and send that to upload
        return NotImplemented

    def upload_mrc(self, project_uid: str, path: Union[str, PurePosixPath], data: "nt.NDArray", psize: float):
        """
        Similar to upload() method, but works with MRC numpy arrays
        """
        with tempfile.TemporaryFile("w+b") as f:
            mrc.write(f, data, psize)
            f.seek(0)
            return self.upload(project_uid, path, f)
