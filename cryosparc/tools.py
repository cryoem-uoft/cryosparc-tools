"""
Main module exporting the ``CryoSPARC`` class for interfaing with a CryoSPARC
instance from Python

Example:

    >>> from cryosparc.tools import CryoSPARC
    >>> cs = CryoSPARC()

"""
from io import BytesIO
from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Iterable, Optional, Union
import os
import re
import tempfile
import numpy as n

if TYPE_CHECKING:
    from numpy.typing import NDArray  # type: ignore

from . import mrc
from .command import CommandClient, make_json_request, make_request
from .dataset import DEFAULT_FORMAT, Dataset
from .project import Project
from .job import Job
from .util import bopen, padarray, trimarray


ONE_MIB = 2**20
LICENSE_REGEX = re.compile(r"[a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12}")
SUPPORTED_EXPOSURE_FORMATS = {
    "MRC",
    "MRCS",
    "TIFF",
    "CMRCBZ2",
    "MRCBZ2",
    "EER",
}
"""
Supported micrograph file formats.
"""


class CryoSPARC:
    """
    High-level class for interfacing with a CryoSPARC instance.

    Initialize with the host and base port of the running CryoSPARC instance.
    This hostname and (at minimum) ``port + 2`` and ``port + 3`` should be
    accessible on the network.

    Args:
        license (str, optional): CryoSPARC license key. Defaults to ``os.getenv("CRYOSPARC_LICENSE_ID")``.
        host (str, optional): Hostname or IP address running CryoSPARC master. Defaults to "localhost".
        port (int, optional): CryoSPARC base port number. Defaults to 39000.
        timeout (int, optional): Timeout error for HTTP requests to CryoSPARC command services. Defaults to 300.

    Attributes:
        cli (CommandClient): HTTP/JSONRPC client for ``command_core`` service (port + 2).
        vis (CommandClient): HTTP/JSONRPC client for ``command_vis`` service (port + 3).

    Examples:
        Load project job and micrographs

        >>> from cryosparc import CryoSPARC
        >>> cs = CryoSPARC(port=39000)
        >>> project = cs.find_project('P3')
        >>> job = project.find_job('J42')
        >>> micrographs = job.load_output('exposures')

        Remove corrupt exposures (assumes ``is_mic_corrupt`` function)

        >>> filtered_micrographs = micrographs.query(is_mic_corrupt)
        >>> job.save_output('micrographs', filtered_micrographs)
    """

    cli: CommandClient
    vis: CommandClient

    def __init__(
        self,
        license: str = os.getenv("CRYOSPARC_LICENSE_ID", ""),
        host: str = "localhost",
        port: int = 39000,
        timeout: int = 300,
    ):
        assert LICENSE_REGEX.fullmatch(license), f"Invalid or unspecified CryoSPARC license ID {license}"

        self.cli = CommandClient(
            service="command_core", host=host, port=port + 2, headers={"License-ID": license}, timeout=timeout
        )
        self.vis = CommandClient(
            service="command_vis", host=host, port=port + 3, headers={"License-ID": license}, timeout=timeout
        )

    def test_connection(self):
        """
        Verify connection to CryoSPARC command services

        Returns:
            bool: True if connection succeeded, False otherwise
        """
        if self.cli.test_connection():  # type: ignore
            print(f"Connection succeeded to CryoSPARC command_core at {self.cli._url}")
        else:
            print(f"Connection FAILED to CryoSPARC command_core at {self.cli._url}")
            return False

        with make_request(self.vis, method="get") as response:
            if response.read():
                print(f"Connection succeeded to CryoSPARC command_vis at {self.vis._url}")
            else:
                print(f"Connection FAILED to CryoSPARC command_vis at {self.vis._url}")
                return False

        return True

    def find_project(self, project_uid: str) -> Project:
        """
        Get a project by its unique ID.

        Args:
            project_uid (str): The project UID

        Returns:
            Project: project instance
        """
        project = Project(self, project_uid)
        project.refresh()
        return project

    def find_job(self, project_uid: str, job_uid: str) -> Job:
        """
        Get a job by its unique project and job ID.

        Args:
            project_uid (str): The project UID
            job_uid (str): The job UID

        Returns:
            Job: job instance
        """
        job = Job(self, project_uid, job_uid)
        job.refresh()
        return job

    def download(self, project_uid: str, path: Union[str, PurePosixPath]):
        """
        Open a file in the current project for reading. Use this method to get
        files from a remote CryoSPARC instance whose the project directories are
        not available on the client file system,

        Args:
            project_uid (str): Short unique ID of CryoSPARC project, e.g., "P3"
            path (str | PurePosixPath): Relative path to file in project directory

        Yields:
            HTTPResponse: Use a context manager to read the file from the
            request body

        Examples:

            Download a job's metadata

            >>> cs = CryoSPARC()
            >>> with cs.download('P3', 'J42/job.json') as res:
            >>>     job_data = json.loads(res.read())

        """
        data = {"project_uid": project_uid, "path_rel": str(path)}
        return make_json_request(self.vis, "/get_project_file", data=data)

    def download_file(self, project_uid: str, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        """
        Download a file from the project directory to the given writeable target.
        """
        with self.download(project_uid, path) as response:
            with bopen(target, "wb") as f:
                data = response.read(ONE_MIB)
                while data:
                    f.write(data)
                    data = response.read(ONE_MIB)
        return target

    def download_dataset(self, project_uid: str, path: Union[str, PurePosixPath]):
        with self.download(project_uid, path) as response:
            size = response.headers.get("Content-Length")
            mime = response.headers.get("Content-Type")
            if mime == "application/x-cryosparc-dataset":
                # Stream format; can load directly without seek
                return Dataset.load(response)

            # Numpy format, cannot load directly because requires seekable
            if size and int(size) < ONE_MIB:
                # Smaller than 1MiB, just read all into memory and load
                return Dataset.load(BytesIO(response.read()))

            # Read into temporary file in 1MiB chunks. Load from that temporary file
            with tempfile.TemporaryFile("w+b", suffix=".cs") as f:
                data = response.read(ONE_MIB)
                while data:
                    f.write(data)
                    data = response.read(ONE_MIB)
                f.seek(0)
                return Dataset.load(f)

    def download_mrc(self, project_uid: str, path: Union[str, PurePosixPath]):
        with self.download(project_uid, path) as response:
            with tempfile.TemporaryFile("w+b", suffix=".cs") as f:
                data = response.read(ONE_MIB)
                while data:
                    f.write(data)
                    data = response.read(ONE_MIB)
                f.seek(0)
                return mrc.read(f)  # FIXME: Optimize file reading

    def upload(self, project_uid: str, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        """
        Open a file from the current project for reading. Note that this
        response is not seekable.
        """
        with bopen(file) as f:
            url = f"/projects/{project_uid}/files"
            query = {"path": path}
            with make_request(self.vis, url=url, query=query, data=f) as res:
                assert res.status >= 200 and res.status < 300, (
                    f"Could not upload project {project_uid} file {path}.\n"
                    f"Response from CryoSPARC: {res.read().decode()}"
                )

    def upload_dataset(
        self, project_uid: str, path: Union[str, PurePosixPath], dset: Dataset, format: int = DEFAULT_FORMAT
    ):
        """
        Similar to upload() method, but works with in-memory datasets
        """
        if len(dset) < 100:
            # Probably small enough to upload from memory
            f = BytesIO()
            dset.save(f, format=format)
            f.seek(0)
            return self.upload(project_uid, path, f)

        # Write to temp file first
        with tempfile.TemporaryFile("w+b") as f:
            dset.save(f, format=format)
            f.seek(0)
            return self.upload(project_uid, path, f)

    def upload_mrc(self, project_uid: str, path: Union[str, PurePosixPath], data: "NDArray", psize: float):
        """
        Similar to upload() method, but works with MRC numpy arrays
        """
        with tempfile.TemporaryFile("w+b") as f:
            mrc.write(f, data, psize)
            f.seek(0)
            return self.upload(project_uid, path, f)


def get_import_signatures(abs_paths: Union[str, Iterable[str], "NDArray"]):
    """
    Get list of import signatures for the given path or paths.

    Args:
        abs_paths (str | Iterable[str]): Absolute path or list of file paths

    Returns:
        List[int]: List of import signatures as 64-bit numpy integers
    """
    from hashlib import sha1

    if isinstance(abs_paths, str):
        abs_paths = [abs_paths]

    return [n.frombuffer(sha1(path.encode()).digest()[:8], dtype=n.uint64)[0] for path in abs_paths]


def get_exposure_format(data_format: str, voxel_type: Optional[str] = None) -> str:
    """
    Get the format for an exposure type were

    Args:
        data_format (str): One of `SUPPORTED_EXPOSURE_FORMATS` such as `"TIFF"`
            or `"MRC"`. The value of the `<dataFormat>` tag in an EPU XML file.
        voxel_type (str, optional): The value of the `<voxelType>` tag in an EPU
            file such as `"32 BIT FLOAT"`. Required when `data_format` is `MRC`
            or `MRCS`. Defaults to None.

    Returns:
        str: The format string to save in a CryoSPARC exposure dataset. e.g.,
            `"TIFF"` or `"MRC/2"`
    """
    assert data_format in SUPPORTED_EXPOSURE_FORMATS, f"Unsupported exposure format {data_format}"
    if data_format not in {"MRC", "MRCS"}:
        return data_format

    assert (
        voxel_type and voxel_type in mrc.VOXEL_TYPES
    ), f'Unsupported voxel type "{voxel_type}" specified with MRC exposure format'
    return f"MRC/{mrc.VOXEL_TYPES[voxel_type]}"


def downsample(arr: "NDArray", factor: int = 2):
    """
    Downsample a micrograph by the given factor
    """
    assert factor >= 1, "Must bin by a factor of 1 or greater"
    arr = n.reshape(arr, (-1,) + arr.shape[-2:])
    nz, ny, nx = arr.shape
    clipx = (nx // factor) * factor
    clipy = (ny // factor) * factor
    shape = (nz, (clipy // factor), (clipx // factor)) if nz > 1 else ((clipy // factor), (clipx // factor))
    out = arr[:, :clipy, :clipx].reshape(nz, clipy, (clipx // factor), factor)
    out = out.sum(axis=-1)
    out = out.reshape(nz, (clipy // factor), factor, -1).sum(axis=-2)
    return out.reshape(shape)


def lowpass(arr: "NDArray", psize_A: float, cutoff_resolution_A: float = 0.0, order: float = 1.0):
    """
    Apply butterworth lowpass filter to the 2D or 3D array data with the given
    pixel size (`psize_A`). `cutoff_resolution_A` should be a non-negative
    number specified in Angstroms.
    """
    assert cutoff_resolution_A > 0, "Lowpass filter amount must be non-negative"
    assert len(arr.shape) == 2 or (len(arr.shape) == 3 and arr.shape[0] == 1), (
        f"Cannot apply low-pass filter on data with shape {arr.shape}; " "must be two-dimensional"
    )

    arr = n.reshape(arr, arr.shape[-2:])
    shape = arr.shape
    if arr.shape[0] != arr.shape[1]:
        arr = padarray(arr, val=n.mean(arr))

    radwn = (psize_A * arr.shape[-1]) / cutoff_resolution_A
    inverse_cutoff_wn2 = 1.0 / radwn**2

    farr = n.fft.rfft2(arr)
    ny, nx = farr.shape
    yp = 0

    for y in range(ny // 2):
        yp = (ny // 2) if y == 0 else ny - y

        # y goes from DC to one before nyquist
        # x goes from DC to one before nyquist
        r2 = (n.arange(nx - 1) ** 2) + (y * y)
        f = 1.0 / (1.0 + (r2 * inverse_cutoff_wn2) ** order)
        farr[y][:-1] *= f
        farr[yp][:-1] *= f

    # zero nyquist at the end
    farr[ny // 2] = 0.0
    farr[:, nx - 1] = 0.0

    result = n.fft.irfft2(farr)
    return trimarray(result, shape) if result.shape != shape else result
