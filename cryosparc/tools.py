"""
Main module exporting the ``CryoSPARC`` class for interfacing with a CryoSPARC
instance from Python

Examples:

    >>> from cryosparc.tools import CryoSPARC
    >>> license = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    >>> email = "ali@example.com"
    >>> password = "password123"
    >>> cs = CryoSPARC(
    ...     license=license,
    ...     email=email,
    ...     password=password,
    ...     host="localhost",
    ...     base_port=39000
    ... )
    >>> project = cs.find_project("P3")

"""
from io import BytesIO
from pathlib import Path, PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import os
import re
import tempfile
import numpy as n

if TYPE_CHECKING:
    from numpy.typing import NDArray  # type: ignore

from . import mrc
from .command import CommandClient, make_json_request, make_request
from .dataset import DEFAULT_FORMAT, Dataset
from .row import R
from .project import Project
from .workspace import Workspace
from .job import ExternalJob, Job
from .spec import (
    ASSET_EXTENSIONS,
    AssetDetails,
    Datafield,
    Datatype,
    JobSection,
    SchedulerLane,
    SchedulerTarget,
)
from .util import bopen, noopcontext, padarray, trimarray


ONE_MIB = 2**20  # bytes in one mebibyte

LICENSE_REGEX = re.compile(r"[a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12}")
"""Regular expression for matching CryoSPARC license IDs."""

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
    High-level session class for interfacing with a CryoSPARC instance.

    Initialize with the host and base port of the running CryoSPARC instance.
    This host and (at minimum) ``base_port + 2`` and ``base_port + 3`` should be
    accessible on the network.

    Args:
        license (str, optional): CryoSPARC license key. Defaults to
            ``os.getenv("CRYOSPARC_LICENSE_ID")``.
        host (str, optional): Hostname or IP address running CryoSPARC master.
            Defaults to ``os.getenv("CRYOSPARC_MASTER_HOSTNAME", "localhost")``.
        base_port (int, optional): CryoSPARC services base port number. Defaults
            to ``os.getenv("CRYOSPARC_MASTER_HOSTNAME", 39000)``.
        email (str, optional): CryoSPARC user account email address. Defaults
            to ``os.getenv("CRYOSPARC_EMAIL")``.
        password (str, optional): CryoSPARC user account password address.
            Defaults to ``os.getenv("CRYOSPARC_PASSWORD")``.
        timeout (int, optional): Timeout error for HTTP requests to CryoSPARC
            command services. Defaults to 300.

    Attributes:
        cli (CommandClient): HTTP/JSONRPC client for ``command_core`` service (port + 2).
        vis (CommandClient): HTTP/JSONRPC client for ``command_vis`` service (port + 3).
        user_id (str): Mongo object ID of user account performing operations for this session.

    Examples:

        Load project job and micrographs

        >>> from cryosparc.tools import CryoSPARC
        >>> cs = CryoSPARC(
        ...     license="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        ...     email="ali@example.com",
        ...     password="password123",
        ...     base_port=39000
        ... )
        >>> job = cs.find_job("P3", "J42")
        >>> micrographs = job.load_output('exposures')

        Remove corrupt exposures (assumes ``is_mic_corrupt`` function)

        >>> filtered_micrographs = micrographs.query(
        ...     lambda mic: is_mic_corrupt(mic["micrograph_blob/path"])
        ... )
        >>> cs.save_external_result(
        ...     project_uid="P3",
        ...     workspace_uid="W1",
        ...     dataset=filtered_micrographs,
        ...     type="exposure",
        ...     name="filtered_exposures",
        ...     passthrough=("J42", "exposures")
        ... )
        "J43"
    """

    cli: CommandClient
    vis: CommandClient
    user_id: str  # session user ID

    def __init__(
        self,
        license: str = os.getenv("CRYOSPARC_LICENSE_ID", ""),
        host: str = os.getenv("CRYOSPARC_MASTER_HOSTNAME", "localhost"),
        base_port: int = int(os.getenv("CRYOSPARC_BASE_PORT", 39000)),
        email: str = os.getenv("CRYOSPARC_EMAIL", ""),
        password: str = os.getenv("CRYOSPARC_PASSWORD", ""),
        timeout: int = 300,
    ):
        assert LICENSE_REGEX.fullmatch(license), f"Invalid or unspecified CryoSPARC license ID {license}"
        assert email, "Invalid or unspecified email"
        assert password, "Invalid or unspecified password"

        self.cli = CommandClient(
            service="command_core", host=host, port=base_port + 2, headers={"License-ID": license}, timeout=timeout
        )
        self.vis = CommandClient(
            service="command_vis", host=host, port=base_port + 3, headers={"License-ID": license}, timeout=timeout
        )
        try:
            self.user_id = self.cli.get_id_by_email_password(email, password)  # type: ignore
        except Exception as e:
            raise RuntimeError("Could not complete CryoSPARC authentication with given credentials") from e

    def test_connection(self):
        """
        Verify connection to CryoSPARC command services.

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

    def get_lanes(self) -> List[SchedulerLane]:
        """
        Get a list of available scheduler lanes.

        Returns:
            list[SchedulerLane]: Details about available lanes.
        """
        return self.cli.get_scheduler_lanes()  # type: ignore

    def get_targets(self, lane: Optional[str] = None) -> List[SchedulerTarget]:
        """
        Get a list of available scheduler targets.

        Args:
            lane (str, optional): Only get targets from this specific lane.
                Returns all targets if not specified. Defaults to None.

        Returns:
            list[SchedulerTarget]: Details about available targets.
        """
        targets: List[SchedulerTarget] = self.cli.get_scheduler_targets()  # type: ignore
        if lane is not None:
            targets = [t for t in targets if t["lane"] == lane]
        return targets

    def get_job_sections(self) -> List[JobSection]:
        """
        Get a summary of job types available for this instance, organized by
        category.

        Returns:
            list[JobSection]: List of job section dictionaries. Job types
                are listed in the ``"contains"`` key in each dictionary.
        """
        return self.cli.get_job_sections()  # type: ignore

    def find_project(self, project_uid: str) -> Project:
        """
        Get a project by its unique ID.

        Args:
            project_uid (str): project unique ID, e.g., "P3"

        Returns:
            Project: project instance
        """
        project = Project(self, project_uid)
        project.refresh()
        return project

    def find_workspace(self, project_uid: str, workspace_uid: str) -> Workspace:
        """
        Get a workspace accessor instance for the workspace in the given project
        with the given UID. Fails with an error if workspace does not exist.

        Args:
            project_uid (str): Project unique ID, e.g,. "P3"
            workspace_uid (str): Workspace unique ID, e.g., "W1"

        Returns:
            Workspace: accessor instance
        """
        workspace = Workspace(self, project_uid, workspace_uid)
        return workspace.refresh()

    def find_job(self, project_uid: str, job_uid: str) -> Job:
        """
        Get a job by its unique project and job ID.

        Args:
            project_uid (str): project unique ID, e.g., "P3"
            job_uid (str): job unique ID, e.g., "J42"

        Returns:
            Job: job instance
        """
        job = Job(self, project_uid, job_uid)
        job.refresh()
        return job

    def find_external_job(self, project_uid: str, job_uid: str) -> ExternalJob:
        """
        Get the External job accessor instance for an External job in this
        project with the given UID. Fails if the job does not exist or is not an
        external job.

        Args:
            project_uid (str): Project unique ID, e.g,. "P3"
            job_uid (str): Job unique ID, e.g,. "J42"

        Raises:
            TypeError: If job is not an external job

        Returns:
            ExternalJob: accessor instance
        """
        job = ExternalJob(self, project_uid, job_uid)
        job.refresh()
        if job.doc["job_type"] != "snowflake":
            raise TypeError(f"Job {project_uid}-{job_uid} is not an external job")
        return job

    def create_workspace(self, project_uid: str, title: str, desc: Optional[str] = None) -> Workspace:
        """
        Create a new empty workspace in the given project.

        Args:
            project_uid (str): Project UID to create in, e.g., "P3".
            title (str): Title of new workspace.
            desc (str, optional): Markdown text description. Defaults to None.

        Returns:
            Workspace: created workspace instance
        """
        workspace_uid: str = self.cli.create_empty_workspace(  # type: ignore
            project_uid=project_uid, created_by_user_id=self.user_id, title=title, desc=desc
        )
        return self.find_workspace(project_uid, workspace_uid)

    def create_job(
        self,
        project_uid: str,
        workspace_uid: str,
        type: str,
        connections: Dict[str, Tuple[str, str]] = {},
        params: Dict[str, Any] = {},
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> Job:
        """
        Create a new job with the given type. Use `CryoSPARC.get_job_sections`_
        to query available job types on the connected CryoSPARC instance.

        Args:
            project_uid (str): Project UID to create job in, e.g., "P3"
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            type (str): Job type identifier, e.g., "homo_abinit"
            connections (dict[str, tuple[str, str]]): Initial input connections.
                Each key is an input name and each value is a (job uid, output
                name) tuple. Defaults to {}
            params (dict[str, any], optional): Specify parameter values.
                Defaults to {}.
            title (str, optional): Job title. Defaults to None.
            desc (str, optional): Job markdown description. Defaults to None.

        Returns:
            Job: created job instance. Raises error if job cannot be created.

        Examples:

            Create an Import Movies job.

            >>> from cryosparc.tools import CryoSPARC
            >>> cs = CryoSPARC()
            >>> import_job = cs.create_job("P3", "W1", "import_movies")
            >>> import_job.set_param("blob_paths", "/bulk/data/t20s/*.tif")
            True

            Create a 3-class ab-initio job connected to existing particles.

            >>> abinit_job = cs.create_job("P3", "W1", "homo_abinit"
            ...     connections={"particles": ("J20", "particles_selected")}
            ...     params={"abinit_K": 3}
            ... )

        .. _CryoSPARC.get_job_sections:
            #cryosparc.tools.CryoSPARC.get_job_sections
        """
        job_uid: str = self.cli.create_new_job(  # type: ignore
            job_type=type, project_uid=project_uid, workspace_uid=workspace_uid, title=title, desc=desc
        )
        job = self.find_job(project_uid, job_uid)
        for input_name, (parent_job, output_name) in connections.items():
            job.connect(parent_job, output_name, input_name, refresh=False)
        for k, v in params.items():
            job.set_param(k, v, refresh=False)
        if connections or params:
            job.refresh()
        return job

    def create_external_job(
        self,
        project_uid: str,
        workspace_uid: str,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> ExternalJob:
        """
        Add a new External job to this project to save generated outputs to.

        Args:
            project_uid (str): Project UID to create in, e.g., "P3"
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            title (str, optional): Title for external job (recommended).
                Defaults to None.
            desc (str, optional): Markdown description for external job.
                Defaults to None.

        Returns:
            ExternalJob: created external job instance
        """
        job_uid: str = self.vis.create_external_job(  # type: ignore
            project_uid=project_uid, workspace_uid=workspace_uid, user=self.user_id, title=title, desc=desc
        )
        return self.find_external_job(project_uid, job_uid)

    def save_external_result(
        self,
        project_uid: str,
        workspace_uid: Optional[str],
        dataset: Dataset[R],
        type: Datatype,
        name: Optional[str] = None,
        slots: Optional[List[Union[str, Datafield]]] = None,
        passthrough: Optional[Tuple[str, str]] = None,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> str:
        """
        Save the given result dataset to the project. Specify at least the
        dataset to save and the type of data.

        Returns UID of the External job where the results were saved.

        Examples:

            Save all particle data

            >>> particles = Dataset()
            >>> cs.save_external_result("P1", "W1", particles, 'particle')
            "J43"

            Save new particle locations that inherit passthrough slots from a
            parent job.

            >>> particles = Dataset()
            >>> cs.save_external_result(
            ...     project_uid="P1",
            ...     workspace_uid="W1",
            ...     dataset=particles,
            ...     type='particle',
            ...     name='particles',
            ...     slots=['location'],
            ...     passthrough=('J42', 'selected_particles'),
            ...     title='Re-centered particles'
            ... )
            "J44"

            Save a result with multiple slots of the same type.

            >>> cs.save_external_result(
            ...     project_uid="P1",
            ...     workspace_uid="P1",
            ...     dataset=particles,
            ...     type="particle",
            ...     name="particle_alignments",
            ...     slots=[
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_0", "required": True},
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_1", "required": True},
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_2", "required": True},
            ...     ]
            ... )
            "J45"

        Args:
            project_uid (str): Project UID to save results into.
            workspace_uid (str | None): Workspace UID to save results into.
                Specify ``None`` to auto-select a workspace.
            dataset (Dataset): Result dataset.
            type (Datatype): Type of output dataset.
            name (str, optional): Name of output on created External job. Same
                as type if unspecified. Defaults to None.
            slots (list[str | Datafield], optional): List of slots expected to
                be created for this output such as ``location`` or ``blob``. Do
                not specify any slots that were passed through from an input
                unless those slots are modified in the output. Defaults to None.
            passthrough (tuple[str, str], optional): Indicates that this output
                inherits slots from the specified output. e.g.,
                ``("J1", "particles")``. Defaults to None.

            title (str, optional): Human-readable title for this output.
                Defaults to None.
            desc (str, optional): Markdown description for this output. Defaults
                to None.

        Returns:
            str: UID of created job where this output was saved
        """
        # Check slot names if present. If not provided, use all slots specified
        # in the dataset prefixes.
        prefixes = dataset.prefixes()
        if slots is None:
            slots = list(prefixes)
        slot_names = {s if isinstance(s, str) else s["prefix"] for s in slots}
        assert slot_names.intersection(prefixes) == slot_names, "Given dataset missing required slots"

        passthrough_str = ".".join(passthrough) if passthrough else None
        job_uid, output = self.vis.create_external_result(  # type: ignore
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            type=type,
            name=name,
            slots=slots,
            passthrough=passthrough_str,
            user=self.user_id,
            title=title,
            desc=desc,
        )

        job = self.find_external_job(project_uid, job_uid)
        with job.run():
            job.save_output(output, dataset)

        return job.uid

    def list_files(
        self, project_uid: str, prefix: Union[str, PurePosixPath] = "", recursive: bool = False
    ) -> List[str]:
        """
        Get a list of files inside the project directory.

        Args:
            project_uid (str): Project unique ID, e.g., "P3"
            prefix (str | Path, optional): Subdirectory inside project to list.
                Defaults to "".
            recursive (bool, optional): If True, lists files recursively.
                Defaults to False.

        Returns:
            list[str]: List of file paths relative to the project directory.
        """
        return self.vis.list_project_files(  # type: ignore
            project_uid=project_uid,
            prefix=str(prefix),
            recursive=recursive,
        )

    def download(self, project_uid: str, path_rel: Union[str, PurePosixPath]):
        """
        Open a file in the given project for reading. Use to get files from a
        remote CryoSPARC instance whose the project directories are not
        available on the client file system.

        Args:
            project_uid (str): Short unique ID of CryoSPARC project, e.g., "P3"
            path_rel (str | Path): Relative path to file in project directory

        Yields:
            HTTPResponse: Use a context manager to read the file from the
            request body

        Examples:
            Download a job's metadata

            >>> cs = CryoSPARC()
            >>> with cs.download('P3', 'J42/job.json') as res:
            >>>     job_data = json.loads(res.read())

        """
        data = {"project_uid": project_uid, "path_rel": str(path_rel)}
        return make_json_request(self.vis, "/get_project_file", data=data)

    def download_file(
        self, project_uid: str, path_rel: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]
    ):
        """
        Download a file from the project directory to the given writeable target.

        Args:
            project_uid (str): project unique ID, e.g., "P3"
            path_rel (str | Path): Relative path of file in project directory.
            target (str | Path | IO): Local file path, directory path or writeable
                file handle to write response data.

        Returns:
            Path | IO: resulting target path or file handle.
        """
        if isinstance(target, (str, PurePath)):
            target = Path(target)
            if target.is_dir():
                target /= PurePath(path_rel).name
        with bopen(target, "wb") as f:
            with self.download(project_uid, path_rel) as response:
                data = response.read(ONE_MIB)
                while data:
                    f.write(data)
                    data = response.read(ONE_MIB)
        return target

    def download_dataset(self, project_uid: str, path_rel: Union[str, PurePosixPath]):
        """
        Download a .cs dataset file from the given relative path in the project
        directory.

        Args:
            project_uid (str): project unique ID, e.g., "P3"
            path_rel (str | Path): Realtive path to .cs file in project
            directory.

        Returns:
            Dataset: Loaded dataset instance
        """
        with self.download(project_uid, path_rel) as response:
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

    def download_mrc(self, project_uid: str, path_rel: Union[str, PurePosixPath]):
        """
        Download a .mrc file from the given relative path in the project
        directory.

        Args:
            project_uid (str): project unique ID, e.g., "P3"
            path_rel (str | Path): Relative path to .mrc file in project
                directory.

        Returns:
            tuple[Header, NDArray]: MRC file header and data as a numpy array
        """
        with self.download(project_uid, path_rel) as response:
            with tempfile.TemporaryFile("w+b", suffix=".cs") as f:
                data = response.read(ONE_MIB)
                while data:
                    f.write(data)
                    data = response.read(ONE_MIB)
                f.seek(0)
                return mrc.read(f)  # FIXME: Optimize file reading

    def list_assets(self, project_uid: str, job_uid: str) -> List[AssetDetails]:
        """
        Get a list of files available in the database for given job. Returns a
        list with details about the assets. Each entry is a dict with a ``_id``
        key which may be used to download the file with the ``download_asset``
        method.

        Args:
            project_uid (str): project unique ID, e.g., "P3"
            job_uid (str): job unique ID, e.g., "J42"

        Returns:
            list[AssetDetails]: Asset details
        """
        return self.vis.list_job_files(project_uid=project_uid, job_uid=job_uid)  # type: ignore

    def download_asset(self, fileid: str, target: Union[str, PurePath, IO[bytes]]):
        """
        Download a file from CryoSPARC's MongoDB GridFS storage.

        Args:
            fileid (str): GridFS file object ID
            target (str | Path | IO): Local file path, directory path or
                writeable file handle to write response data.

        Returns:
            Path | IO: resulting target path or file handle.
        """
        with make_json_request(self.vis, url="/get_job_file", data={"fileid": fileid}) as response:
            if isinstance(target, (str, PurePath)):
                target = Path(target)
                if target.is_dir():
                    # Try to get download filename and content type from
                    # headers. If cannot be determined, defaults to "file.dat"
                    content_type: str = response.headers.get_content_type()
                    attachment_filename: Optional[str] = response.headers.get_filename()
                    target /= attachment_filename or f"file.{ASSET_EXTENSIONS.get(content_type, 'dat')}"
            with bopen(target, "wb") as f:
                data = response.read(ONE_MIB)
                while data:
                    f.write(data)
                    data = response.read(ONE_MIB)

            return target

    def upload(
        self, project_uid: str, target_path_rel: Union[str, PurePosixPath], source: Union[str, bytes, PurePath, IO]
    ):
        """
        Upload the given source file to the project directory at the given
        relative path.

        Args:
            project_uid (str): project unique ID, e.g., "P3"
            target_path_rel (str | Path): Relative target path in project
                directory.
            source (str | bytes | Path | IO): Local path or file handle to
                upload. May also specified as raw bytes.
        """
        with open(source, "rb") if isinstance(source, (str, PurePath)) else noopcontext(source) as f:
            url = f"/projects/{project_uid}/files"
            query = {"path": target_path_rel}
            with make_request(self.vis, url=url, query=query, data=f) as res:
                assert res.status >= 200 and res.status < 300, (
                    f"Could not upload project {project_uid} file {target_path_rel}.\n"
                    f"Response from CryoSPARC ({res.status}): {res.read().decode()}"
                )

    def upload_dataset(
        self, project_uid: str, target_path_rel: Union[str, PurePosixPath], dset: Dataset, format: int = DEFAULT_FORMAT
    ):
        """
        Upload a dataset as a CS file into the project directory.

        Args:
            project_uid (str): project unique ID, e.g., "P3"
            target_path_rel (str | Path): relative path to save dataset in
                project directory. Should have a ``.cs`` extension.
            dset (Dataset): dataset to save.
            format (int): format to save in from ``cryosparc.dataset.*_FORMAT``,
                defaults to NUMPY_FORMAT)
        """
        if len(dset) < 100:
            # Probably small enough to upload from memory
            f = BytesIO()
            dset.save(f, format=format)
            f.seek(0)
            return self.upload(project_uid, target_path_rel, f)

        # Write to temp file first
        with tempfile.TemporaryFile("w+b") as f:
            dset.save(f, format=format)
            f.seek(0)
            return self.upload(project_uid, target_path_rel, f)

    def upload_mrc(self, project_uid: str, target_path_rel: Union[str, PurePosixPath], data: "NDArray", psize: float):
        """
        Upload a numpy 2D or 3D array to the job directory as an MRC file.

        Args:
            project_uid (str): project unique ID, e.g., "P3"
            target_path_rel (str | Path): filename or relative path. Should have
                ``.mrc`` extension.
            data (NDArray): Numpy array with MRC file data.
            psize (float): Pixel size to include in MRC header.
        """
        with tempfile.TemporaryFile("w+b") as f:
            mrc.write(f, data, psize)
            f.seek(0)
            return self.upload(project_uid, target_path_rel, f)

    def mkdir(
        self,
        project_uid: str,
        target_path_rel: Union[str, PurePosixPath],
        parents: bool = False,
        exist_ok: bool = False,
    ):
        """
        Create a directory in the given project.

        Args:
            project_uid (str): Target project directory
            target_path_rel (str | Path): Relative path to create inside project
                directory.
            parents (bool, optional): If True, any missing parents are created
                as needed. Defaults to False.
            exist_ok (bool, optional): If True, does not raise an error for
                existing directories. Still raises if the target path is not a
                directory. Defaults to False.
        """
        self.vis.project_mkdir(  # type: ignore
            project_uid=project_uid,
            path_rel=str(target_path_rel),
            parents=parents,
            exist_ok=exist_ok,
        )

    def cp(
        self, project_uid: str, source_path_rel: Union[str, PurePosixPath], target_path_rel: Union[str, PurePosixPath]
    ):
        """
        Copy a file or folder within a project to another location within that
        same project. Note that argument order is reversed from
        equivalent ``cp`` command.

        Args:
            project_uid (str): Target project UID, e.g., "P3".
            source_path_rel (str | Path): Relative path in project of source
                file or folder to copy.
            target_path_rel (str | Path): Relative path in project to copy to.
        """
        self.vis.project_cp(  # type: ignore
            project_uid=project_uid,
            source_path_rel=str(source_path_rel),
            target_path_rel=str(target_path_rel),
        )

    def symlink(
        self, project_uid: str, source_path_rel: Union[str, PurePosixPath], target_path_rel: Union[str, PurePosixPath]
    ):
        """
        Create a symbolic link in the given project. May only create links for
        files within the project. Note that argument order is reversed from
        ``ln -s``.

        Args:
            project_uid (str): Target project UID, e.g., "P3".
            source_path_rel (str | Path): Relative path in project to file from
                which to create symlink.
            target_path_rel (str | Path): Relative path in project to new
                symlink.
        """
        self.vis.project_symlink(  # type: ignore
            project_uid=project_uid,
            source_path_rel=str(source_path_rel),
            target_path_rel=str(target_path_rel),
        )


def get_import_signatures(abs_paths: Union[str, Iterable[str], "NDArray"]):
    """
    Get list of import signatures for the given path or paths.

    Args:
        abs_paths (str | Iterable[str] | NDArray): Absolute path or list of file
            paths.

    Returns:
        list[int]: Import signatures as 64-bit numpy integers
    """
    from hashlib import sha1

    if isinstance(abs_paths, str):
        abs_paths = [abs_paths]

    return [n.frombuffer(sha1(path.encode()).digest()[:8], dtype=n.uint64)[0] for path in abs_paths]


def get_exposure_format(data_format: str, voxel_type: Optional[str] = None) -> str:
    """
    Get the ``movie_blob/format`` or ``micrograph_blob`` format value for an
    exposure type, where ``data_format`` is one of

    - "MRC"
    - "MRCS"
    - "TIFF"
    - "CMRCBZ2"
    - "MRCBZ2"
    - "EER"

    And ``voxel_type`` (if specified) is one of

    - "16 BIT FLOAT":
    - "32 BIT FLOAT"
    - "SIGNED 16 BIT INTEGER"
    - "UNSIGNED 8 BIT INTEGER"
    - "UNSIGNED 16 BIT INTEGER"

    Args:
        data_format (str): One of ``SUPPORTED_EXPOSURE_FORMATS`` such as
            ``"TIFF"`` or ``"MRC"``. The value of the ``<dataFormat>`` tag in an
            EPU XML file.
        voxel_type (str, optional): The value of the ``<voxelType>`` tag in an
            EPU file such as ``"32 BIT FLOAT"``. Required when ``data_format``
            is ``MRC`` or ``MRCS``. Defaults to None.

    Returns:
        str: The format string to save into the ``{prefix}/format`` field of a
            CryoSPARC exposure dataset. e.g., ``"TIFF"`` or ``"MRC/2"``
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
    Downsample a micrograph or movie by the given factor.

    Args:
        arr (NDArray): 2D or 3D numpy array factor (int, optional): How much to
            reduce size by. e.g., a factor of 2 would reduce a 1024px MRC to
            512px, and a factor of 3 would reduce it to 256px. Defaults to 2.

    Returns:
        NDArray: Downsampled MRC file
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


def lowpass2(arr: "NDArray", psize_A: float, cutoff_resolution_A: float = 0.0, order: float = 1.0):
    """
    Apply butterworth lowpass filter to the 2D array data with the given
    pixel size (``psize_A``). ``cutoff_resolution_A`` should be a non-negative
    number specified in Angstroms.

    Args:
        arr (NDArray): 2D numpy array to apply lowpass to.
        psize_A (float): Pixel size of array data.
        cutoff_resolution_A (float, optional): Cutoff resolution, in Angstroms.
            Defaults to 0.0.
        order (float, optional): Filter order. Defaults to 1.0.

    Returns:
        NDArray: Lowpass-filtered copy of given numpy array
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
