from pathlib import PurePath, PurePosixPath
import tempfile
from typing import IO, Optional, Union

import numpy.typing as nt

from . import mrc
from .command import CommandClient, RequestClient
from .dataset import Dataset
from .project import Project
from .util import bopen


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

    def __init__(self, host: str = "localhost", port: int = 39000, timeout: int = 300) -> None:
        self.cli = CommandClient(host=host, port=port + 2, timeout=timeout)
        self.vis = RequestClient(host=host, port=port + 3, timeout=timeout)
        self.timeout = timeout

    def find_project(self, project_uid: str) -> Optional[Project]:
        pass

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
        data = {"project_uid": project_uid, "file": str(path)}
        return self.vis.json_request("/get_project_file", data=data)

    def download_file(self, project_uid: str, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        """
        Download
        """
        with self.download(project_uid, path) as request:
            with bopen(target, "wb") as f:
                f.write(request.read())
        return target

    def download_dataset(self, project_uid: str, path: Union[str, PurePosixPath]):
        with self.download(project_uid, path) as request:
            return Dataset.load(request)

    def download_mrc(self, project_uid: str, path: Union[str, PurePosixPath]):
        with self.download(project_uid, path) as request:
            return mrc.read(request)

    def upload(self, project_uid: str, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        """
        Open a file from the current project for reading. Note that this
        response is not seekable.
        """
        with bopen(file) as f:
            url = f"/upload_project_file/{project_uid}"
            query = {"path": path}
            with self.vis.request(url=url, query=query, data=f) as res:
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

    def upload_mrc(self, project_uid: str, path: Union[str, PurePosixPath], data: nt.NDArray, psize: float):
        """
        Similar to upload() method, but works with MRC numpy arrays
        """
        with tempfile.TemporaryFile("w+b") as f:
            mrc.write(f, data, psize)
            f.seek(0)
            return self.upload(project_uid, path, f)


print("HELLO FROM CRYOSPARC")
