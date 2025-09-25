"""
Main module exporting the ``CryoSPARC`` class for interfacing with a CryoSPARC
instance from Python

Examples:

    >>> from cryosparc.tools import CryoSPARC
    >>> cs = CryoSPARC("http://localhost:39000")
    >>> project = cs.find_project("P3")

"""

import os
import re
import tempfile
import warnings
from contextlib import contextmanager
from functools import cached_property
from hashlib import sha256
from io import BytesIO, TextIOBase
from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Any, Container, Dict, Iterable, List, Literal, Optional, Tuple, Union, get_args

import numpy as n
from typing_extensions import Unpack

from . import __version__, model_registry, mrc, registry, stream_registry
from .api import APIClient
from .auth import InstanceAuthSessions
from .constants import API_SUFFIX
from .controllers import as_output_slot
from .controllers.job import ExternalJobController, JobController
from .controllers.project import ProjectController
from .controllers.workspace import WorkspaceController
from .dataset import CSDAT_FORMAT, DEFAULT_FORMAT, Dataset
from .dataset.row import R
from .models.asset import GridFSFile
from .models.external import ExternalOutputSpec
from .models.job_register import JobRegister
from .models.job_spec import Category, OutputRef, OutputSpec
from .models.scheduler_lane import SchedulerLane
from .models.scheduler_target import SchedulerTarget
from .models.user import User
from .search import In, JobSearch
from .spec import Datatype, JobSection, SlotSpec
from .stream import BinaryIteratorIO, Stream
from .util import clear_cached_property, padarray, print_table, trimarray

if TYPE_CHECKING:
    from numpy.typing import NDArray

assert stream_registry
assert model_registry
registry.finalize()  # no more models may be registered after this


LICENSE_REGEX = re.compile(r"[a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12}")
"""Regular expression for matching CryoSPARC license IDs."""

VERSION_REGEX = re.compile(r"^v\d+\.\d+\.\d+")
"""Regular expression for CryoSPARC minor version, e.g., 'v4.1.0'"""

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
    This host and (at minimum) ``base_port + 2`` should be accessible on the
    network.

    Args:
        base_url (str, optional): CryoSPARC instance URL, e.g.,
            "http://localhost:39000" or "https://cryosparc.example.com".
            Same URL used to access CryoSPARC from a web browser.
        host (str, optional): Hostname or IP address running CryoSPARC master.
            Cannot be specified with ``base_url``. Defaults to
            ``os.getenv("CRYOSPARC_MASTER_HOSTNAME", "localhost")``.
        base_port (int, optional): CryoSPARC services base port number.
            Cannot be specified with  ``base_url``. Defaults to
            ``os.getenv("CRYOSPARC_BASE_PORT", 39000)``.
        email (str, optional): CryoSPARC user account email address. Defaults
            to ``os.getenv("CRYOSPARC_EMAIL")``.
        password (str, optional): CryoSPARC user account password address.
            Defaults to ``os.getenv("CRYOSPARC_PASSWORD")``.
        license (str, optional): (Deprecated) CryoSPARC license key. Defaults to
            ``os.getenv("CRYOSPARC_LICENSE_ID")``.
        timeout (int, optional): Timeout error for HTTP requests to CryoSPARC
            command services. Defaults to 300.

    Examples:

        Load project job and micrographs

        >>> from cryosparc.tools import CryoSPARC
        >>> cs = CryoSPARC(
        ...     email="ali@example.com",
        ...     password="password123",
        ...     host="localhost",
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

    api: APIClient
    """
    HTTP REST API client for ``api`` service (port + 2).
    """

    base_url: str
    """
    URL used for communication CryoSPARC instance REST API.
    """

    def __init__(
        self,
        base_url: Optional[str] = os.getenv("CRYOSPARC_BASE_URL"),
        *,
        host: Optional[str] = os.getenv("CRYOSPARC_MASTER_HOSTNAME"),
        base_port: Union[int, str, None] = os.getenv("CRYOSPARC_BASE_PORT"),
        email: Optional[str] = os.getenv("CRYOSPARC_EMAIL"),
        password: Optional[str] = os.getenv("CRYOSPARC_PASSWORD"),
        license: Optional[str] = None,
        timeout: int = 300,
    ):
        if license:
            warnings.warn(
                "Support for license argument will be removed in a future release",
                DeprecationWarning,
                stacklevel=2,
            )

        if host or base_port:
            if base_url:
                raise TypeError("Cannot specify host or base_port when base_url is specified")
            host = host or "localhost"
            port = int(base_port or 39000)
            self.base_url = f"http://{host}:{port}"
        elif base_url:
            self.base_url = base_url
        else:
            raise TypeError("Must specify either base_url or host + base_port")

        auth = None
        if email and password:
            auth = (email, sha256(password.encode()).hexdigest())
        elif session := InstanceAuthSessions.load().find(self.base_url, email):
            auth = session.token.access_token
        elif license:
            auth = ("cryosparc", sha256(license.encode()).hexdigest())
        else:
            raise ValueError(
                f"CryoSPARC authentication not provided or expired for {self.base_url}. "
                "If required, create a new session with command\n\n"
                "    python3 -m cryosparc.tools login\n\n"
                "Please see documentation at https://tools.cryosparc.com for instructions."
            )

        tools_major_minor_version = ".".join(__version__.split(".")[:2])  # e.g., 4.1.0 -> 4.1
        try:
            self.api = APIClient(
                f"{self.base_url}{API_SUFFIX}",
                auth=auth,
                headers={"User-Agent": f"cryosparc-tools/{__version__}"},
                timeout=timeout,
            )
            assert self.user  # trigger user profile fetch
            cs_version = self.api.config.get_version()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to CryoSPARC at {self.base_url} due to error:\n{e}\n"
                "Please ensure your credentials are valid and that you are "
                "connecting to a CryoSPARC version compatible with "
                f"cryosparc-tools {tools_major_minor_version}.\n\n"
                "If required, create a new session with command\n\n"
                "    python -m cryosparc.tools login\n\n"
                "Please see the documentation at https://tools.cryosparc.com for details."
            ) from e

        if cs_version and VERSION_REGEX.match(cs_version):
            cs_major_minor_version = ".".join(cs_version[1:].split(".")[:2])  # e.g., v4.1.0 -> 4.1
            if cs_major_minor_version != tools_major_minor_version:
                tools_repo_url = "https://github.com/cryoem-uoft/cryosparc-tools.git"
                warnings.warn(
                    f"CryoSPARC at {self.base_url} with version {cs_version} "
                    f"may not be compatible with current cryosparc-tools version {__version__}.\n\n"
                    "To install a compatible version of cryosparc-tools:\n\n"
                    f"    pip install --force cryosparc-tools~={cs_major_minor_version}.0\n\n"
                    "Or, if running a CryoSPARC pre-release or private beta:\n\n"
                    f'    pip install --force "cryosparc-tools @ git+{tools_repo_url}@develop"\n',
                    stacklevel=2,
                )

    @cached_property
    def user(self) -> User:
        """
        User account performing operations for this session.
        """
        return self.api.users.me()

    @cached_property
    def job_register(self) -> JobRegister:
        """
        Information about jobs available on this instance.
        """
        return self.api.job_register()

    def refresh(self):
        """
        Reset cache and refresh instance details.

        Raises:
            APIError: cannot be refreshed.
        """
        clear_cached_property(self, "user")
        clear_cached_property(self, "job_register")
        assert self.user  # ensure we can still fetch a user

    def test_connection(self):
        """
        Verify connection to CryoSPARC command services.

        Returns:
            bool: True if connection succeeded, False otherwise
        """
        if self.api.health() == "OK":
            print(f"Success: Connected to CryoSPARC API at {self.base_url}")
            return True
        else:
            print(f"Error: Could not connect to CryoSPARC API at {self.base_url}")
            return False

    def get_lanes(self) -> List[SchedulerLane]:
        """
        Get a list of available scheduler lanes.

        Returns:
            list[SchedulerLane]: Details about available lanes.
        """
        return self.api.resources.find_lanes()

    def get_targets(self, lane: Optional[str] = None) -> List[SchedulerTarget]:
        """
        Get a list of available scheduler targets.

        Args:
            lane (str, optional): Only get targets from this specific lane.
                Returns all targets if not specified. Defaults to None.

        Returns:
            list[SchedulerTarget]: Details about available targets.
        """
        return self.api.resources.find_targets(lane=lane)

    def get_job_sections(self) -> List[JobSection]:
        """
        (Deprecated) Get a summary of job types available for this instance,
        organized by category.

        Returns:
            list[JobSection]: List of job section dictionaries. Job types
                are listed in the ``"contains"`` key in each dictionary.
        """
        warnings.warn("Use job_register property instead", DeprecationWarning, stacklevel=2)
        job_types_by_category = {
            category: [spec.type for spec in self.job_register.specs if spec.category == category]
            for category in get_args(Category)
        }
        return [
            {"name": category, "title": category.replace("_", " ").title(), "description": "", "contains": job_types}
            for category, job_types in job_types_by_category.items()
        ]

    def print_job_types(
        self,
        category: Union[Category, Container[Category], None] = None,
        *,
        show_legacy: bool = False,
    ):
        """
        Print a table of job types and their titles, organized by category.

        Args:
            category (Category | list[Category], optional): Only show jobs from
                the given category or list of categories. Defaults to None.
            show_legacy (bool, optional): If True, also show legacy jobs.
                Defaults to False.
        """
        allowed_categories = {category} if isinstance(category, str) else category
        register = self.job_register
        headings = ["Category", "Job", "Title", "Stability"]
        rows = []
        prev_category = None
        for job_spec in register.specs:
            if allowed_categories is not None and job_spec.category not in allowed_categories:
                continue
            if job_spec.hidden or job_spec.stability == "obsolete":
                continue
            if not show_legacy and job_spec.stability == "legacy":
                continue

            category = job_spec.category  # type: ignore
            display_category = "" if category == prev_category else category
            rows.append([display_category, job_spec.type, job_spec.title, job_spec.stability])
            prev_category = category

        print_table(headings, rows)

    def find_projects(self, *, order: Literal[1, -1] = 1) -> Iterable[ProjectController]:
        """
        Get all projects available to the current user.

        Args:
            order (int, optional): Sort order for projects, 1 for ascending, -1
                for descending. Defaults to 1.

        Returns:
            Iterable[ProjectController]: project accessor objects
        """
        after = None
        limit = 10
        while projects := self.api.projects.find(order=order, after=after, limit=limit):
            for project in projects:
                yield ProjectController(self, project)
            if len(projects) < limit:
                break
            after = projects[-1].id

    def find_project(self, project_uid: str) -> ProjectController:
        """
        Get a project by its unique ID.

        Args:
            project_uid (str): Project unique ID, e.g., "P3"

        Returns:
            ProjectController: project accessor object
        """
        return ProjectController(self, project_uid)

    def find_workspaces(
        self,
        project_uid: Optional[In[str]] = None,
        *,
        order: Literal[1, -1] = 1,
    ) -> Iterable[WorkspaceController]:
        """
        Get all workspaces available in the given project.

        Args:
            project_uid (str | list[str] | None): Project unique ID, e.g., "P3".
                If not specified, returns workspaces from all projects.
        Returns:
            Iterable[WorkspaceController]: workspace accessor objects
        """
        after = None
        limit = 100
        while workspaces := self.api.workspaces.find(
            project_uid=[project_uid] if isinstance(project_uid, str) else project_uid,
            after=after,
            limit=limit,
        ):
            for workspace in workspaces:
                yield WorkspaceController(self, workspace)
            if len(workspaces) < limit:
                break
            after = workspaces[-1].id

    def find_workspace(self, project_uid: str, workspace_uid: str) -> WorkspaceController:
        """
        Get a workspace accessor instance for the workspace in the given project
        with the given UID. Fails with an error if workspace does not exist.

        Args:
            project_uid (str): Project unique ID, e.g,. "P3"
            workspace_uid (str): Workspace unique ID, e.g., "W1"

        Returns:
            WorkspaceController: workspace accessor object
        """
        return WorkspaceController(self, (project_uid, workspace_uid))

    def find_jobs(
        self,
        project_uid: Optional[In[str]] = None,
        workspace_uid: Optional[In[str]] = None,
        *,
        order: Literal[1, -1] = 1,
        **search: Unpack[JobSearch],
    ) -> Iterable[JobController]:
        """
        Search available jobs.

        Example:
            >>> jobs = cs.find_jobs("P3", "W3")
            >>> jobs = cs.find_jobs(["P42", "P43"])
            >>> jobs = cs.find_jobs(
            ...     type="homo_reconstruct",
            ...     completed_at=(datetime(2025, 3, 1), datetime(2025, 3, 31)),
            ...     order=-1,
            ... )
            >>> for job in jobs:
            ...     print(job.uid)

        Args:
            project_uid (str | list[str] | None): Project unique ID, e.g., "P3".
                If not specified, returns jobs from all projects. Defaults to None.
            workspace_uid (str | list[str] | None): Workspace unique ID, e.g., "W1".
                If not specified, returns jobs from all workspaces. Defaults to None.
            **search (JobSearch): Additional search parameters to filter jobs,
                specified as keyword arguments.

        Returns:
            Iterable[JobController]: job accessor objects
        """
        after = None
        limit = 100
        while jobs := self.api.jobs.find(
            project_uid=[project_uid] if isinstance(project_uid, str) else project_uid,
            workspace_uid=[workspace_uid] if isinstance(workspace_uid, str) else workspace_uid,
            type=[type] if isinstance(type := search.get("type"), str) else type,
            status=[status] if isinstance(status := search.get("status"), str) else status,
            category=[category] if isinstance(category := search.get("category"), str) else category,
            created_at=search.get("created_at"),
            updated_at=search.get("updated_at"),
            queued_at=search.get("queued_at"),
            started_at=search.get("started_at"),
            completed_at=search.get("completed_at"),
            waiting_at=search.get("waiting_at"),
            killed_at=search.get("killed_at"),
            failed_at=search.get("failed_at"),
            exported_at=search.get("exported_at"),
            order=order,
            after=after,
            limit=limit,
        ):
            for job in jobs:
                yield JobController(self, job)
            if len(jobs) < limit:
                break
            after = jobs[-1].id

    def find_job(self, project_uid: str, job_uid: str) -> JobController:
        """
        Get a job by its unique project and job ID.

        Args:
            project_uid (str): Project unique ID, e.g., "P3"
            job_uid (str): job unique ID, e.g., "J42"

        Returns:
            JobController: job accessor object
        """
        return JobController(self, (project_uid, job_uid))

    def find_external_job(self, project_uid: str, job_uid: str) -> ExternalJobController:
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
            ExternalJobController: external job accessor object
        """
        return ExternalJobController(self, (project_uid, job_uid))

    def create_workspace(self, project_uid: str, title: str, desc: Optional[str] = None) -> WorkspaceController:
        """
        Create a new empty workspace in the given project.

        Args:
            project_uid (str): Project UID to create in, e.g., "P3".
            title (str): Title of new workspace.
            desc (str, optional): Markdown text description. Defaults to None.

        Returns:
            WorkspaceController: created workspace accessor object

        Raises:
            APIError: Workspace cannot be created.
        """
        workspace = self.api.workspaces.create(project_uid, title=title, description=desc)
        return WorkspaceController(self, workspace)

    def create_job(
        self,
        project_uid: str,
        workspace_uid: str,
        type: str,
        connections: Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]] = {},
        params: Dict[str, Any] = {},
        title: str = "",
        desc: str = "",
    ) -> JobController:
        """
        Create a new job with the given type. Use :py:attr:`job_register`
        to find available job types on the connected CryoSPARC instance.

        Args:
            project_uid (str): Project UID to create job in, e.g., "P3"
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            type (str): Job type identifier, e.g., "homo_abinit"
            connections (dict[str, tuple[str, str] | list[tuple[str, str]]]):
                Initial input connections. Each key is an input name and each
                value is a (job uid, output name) tuple. Defaults to {}
            params (dict[str, any], optional): Specify parameter values.
                Defaults to {}.
            title (str, optional): Job title. Defaults to "".
            desc (str, optional): Job markdown description. Defaults to "".

        Returns:
            JobController: created job accessor object.

        Raises:
            APIError: Job cannot be created.

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
        """
        job = self.api.jobs.create(project_uid, workspace_uid, params=params, type=type, title=title, description=desc)
        for input_name, connection in connections.items():
            connection = [connection] if isinstance(connection, tuple) else connection
            for source_job_uid, source_output_name in connection:
                job = self.api.jobs.connect(
                    job.project_uid,
                    job.uid,
                    input_name,
                    source_job_uid=source_job_uid,
                    source_output_name=source_output_name,
                )
        return JobController(self, job)

    def create_external_job(
        self,
        project_uid: str,
        workspace_uid: str,
        title: str = "",
        desc: str = "",
    ) -> ExternalJobController:
        """
        Add a new External job to this project to save generated outputs to.

        Args:
            project_uid (str): Project UID to create in, e.g., "P3"
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            title (str, optional): Title for external job (recommended).
                Defaults to "".
            desc (str, optional): Markdown description for external job.
                Defaults to "".

        Returns:
            ExternalJobController: created external job accessor object
        """
        job = self.api.jobs.create(project_uid, workspace_uid, type="snowflake", title=title, description=desc)
        return ExternalJobController(self, job)

    def save_external_result(
        self,
        project_uid: str,
        workspace_uid: Optional[str],
        dataset: Dataset[R],
        type: Datatype,
        name: Optional[str] = None,
        slots: Optional[List[SlotSpec]] = None,
        passthrough: Optional[Tuple[str, str]] = None,
        title: str = "",
        desc: str = "",
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
            slots (list[SlotSpec], optional): List of slots expected to
                be created for this output such as ``location`` or ``blob``. Do
                not specify any slots that were passed through from an input
                unless those slots are modified in the output. Defaults to None.
            passthrough (tuple[str, str], optional): Indicates that this output
                inherits slots from the specified output. e.g.,
                ``("J1", "particles")``. Defaults to None.
            title (str, optional): Human-readable title for this output.
                Defaults to "".
            desc (str, optional): Markdown description for this output. Defaults
                to "".

        Raises:
            APIError: General CryoSPARC network access error such as
                timeout, URL or HTTP

        Returns:
            str: UID of created job where this output was saved
        """
        # Check slot names if present. If not provided, use all slots specified
        # in the dataset prefixes.
        prefixes = dataset.prefixes()
        if slots is None:
            slots = list(prefixes)
        elif any(isinstance(s, dict) and "prefix" in s for s in slots):
            warnings.warn("'prefix' slot key is deprecated. Use 'name' instead.", DeprecationWarning, stacklevel=2)

        # Normalize slots to OutputSlot or strings
        output_slots = [s if isinstance(s, str) else as_output_slot(s) for s in slots]
        required_slot_names = {s if isinstance(s, str) else s.name for s in output_slots}
        missing_slot_names = required_slot_names.difference(prefixes)
        if missing_slot_names:
            raise ValueError(f"Given dataset missing required slots: {', '.join(missing_slot_names)}")

        if not name:
            name = type
        if not title:
            title = name.replace("_", " ").title()

        # Find the most recent workspace or create a new one if the project is empty
        if workspace_uid is None:
            workspaces = self.api.workspaces.find(project_uid=[project_uid], order=-1, limit=1)
            workspace = workspaces[0] if workspaces else self.api.workspaces.create(project_uid, title=title)
            workspace_uid = workspace.uid

        job = self.api.jobs.create_external_result(
            project_uid,
            workspace_uid,
            ExternalOutputSpec(
                name=name,
                spec=OutputSpec(type=type, title=title, description=desc, slots=output_slots),
                connection=OutputRef(job_uid=passthrough[0], output=passthrough[1]) if passthrough else None,
            ),
        )
        job = ExternalJobController(self, job)
        with job.run():
            job.save_output(name, dataset)
        return job.uid

    def list_files(
        self,
        project_uid: str,
        prefix: Union[str, PurePosixPath] = "",
        recursive: bool = False,
    ) -> List[str]:
        """
        Get a list of files inside the project directory.

        Args:
            project_uid (str): Project unique ID, e.g., "P3".
            prefix (str | Path, optional): Subdirectory inside project to list.
                Defaults to "".
            recursive (bool, optional): If True, lists files recursively.
                Defaults to False.

        Returns:
            list[str]: List of file paths relative to the project directory.
        """
        return self.api.projects.ls(project_uid, path=str(prefix), recursive=recursive)

    @contextmanager
    def download(self, project_uid: str, path: Union[str, PurePosixPath]):
        """
        Open a file in the given project for reading. Use to get files from a
        remote CryoSPARC instance whose the project directories are not
        available on the client file system.

        Args:
            project_uid (str): Short unique ID of CryoSPARC project, e.g., "P3"
            path (str | Path): Name or path of file in project directory.

        Yields:
            HTTPResponse: Use a context manager to read the file from the
            request body.

        Examples:
            Download a job's metadata

            >>> cs = CryoSPARC()
            >>> with cs.download('P3', 'J42/job.json') as res:
            >>>     job_data = json.loads(res.read())

        """
        if not path:
            raise ValueError("Download path cannot be empty")
        stream = self.api.projects.download_file(project_uid, path=str(path))
        iterator = BinaryIteratorIO(stream.stream())
        yield iterator

    def download_file(
        self,
        project_uid: str,
        path: Union[str, PurePosixPath],
        target: Union[str, PurePath, IO[bytes]] = "",
    ):
        """
        Download a file from the directory of the specified project to the given
        target path or writeable file handle.

        Args:
            project_uid (str): Project unique ID, e.g., "P3".
            path (str | Path): Name or path of file in project directory.
            target (str | Path | IO, optional): Local file path, directory path
                or writeable file handle to write response data. If not
                specified, downloads to current working directory with same file
                name. Defaults to "".

        Returns:
            Path | IO: resulting target path or file handle.
        """
        stream = self.api.projects.download_file(project_uid, path=str(path))
        stream.save(target)
        return target

    def download_dataset(self, project_uid: str, path: Union[str, PurePosixPath]):
        """
        Download a .cs dataset file from the given relative path in the project
        directory.

        Args:
            project_uid (str): Project unique ID, e.g., "P3".
            path (str | Path): Name or path to .cs file in project directory.

        Returns:
            Dataset: Loaded dataset instance
        """
        stream = self.api.projects.download_file(project_uid, path=str(path))
        if stream.media_type == "application/x-cryosparc-dataset":
            # Stream format; can load directly without seek
            return Dataset.from_iterator(stream.stream())

        # Numpy format, cannot load directly because requires seekable. Load from that temporary file
        with tempfile.TemporaryFile("w+b", suffix=".cs") as f:
            stream.save(f)
            f.seek(0)
            return Dataset.load(f)

    def download_mrc(self, project_uid: str, path: Union[str, PurePosixPath]):
        """
        Download a .mrc file from the given relative path in the project
        directory.

        Args:
            project_uid (str): Project unique ID, e.g., "P3"
            path (str | Path): Name or path to .mrc file in project directory.

        Returns:
            tuple[Header, NDArray]: MRC file header and data as a numpy array
        """
        stream = self.api.projects.download_file(project_uid, path=str(path))
        with tempfile.TemporaryFile("w+b", suffix=".mrc") as f:
            stream.save(f)
            f.seek(0)
            return mrc.read(f)  # FIXME: Optimize file reading

    def list_assets(self, project_uid: str, job_uid: str) -> List[GridFSFile]:
        """
        Get a list of files available in the database for given job. Returns a
        list with details about the assets. Each entry is a dict with a ``_id``
        key which may be used to download the file with the ``download_asset``
        method.

        Args:
            project_uid (str): Project unique ID, e.g., "P3"
            job_uid (str): job unique ID, e.g., "J42"

        Returns:
            list[GridFSFile]: Asset details
        """
        return self.api.assets.find(project_uid=project_uid, job_uid=job_uid)

    def download_asset(self, fileid: str, target: Union[str, PurePath, IO[bytes]]):
        """
        Download a file from CryoSPARC's MongoDB GridFS storage.

        Args:
            fileid (str): GridFS file object ID
            target (str | Path | IO): Local file path or writeable file handle
                to write response data.

        Returns:
            str | Path | IO: resulting target path or file handle.
        """
        stream = self.api.assets.download(fileid)
        stream.save(target)
        return target

    def upload(
        self,
        project_uid: str,
        target_path: Union[str, PurePosixPath],
        source: Union[str, bytes, PurePath, IO, Stream],
        *,
        overwrite: bool = False,
    ):
        """
        Upload the given source file to the project directory at the given
        relative path. Fails if target already exists.

        Args:
            project_uid (str): Project unique ID, e.g., "P3"
            target_path (str | Path): Name or path of file to write in project
                directory.
            source (str | bytes | Path | IO | Stream): Local path or file handle
                to upload. May also specified as raw bytes.
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        if isinstance(source, bytes):
            source = BytesIO(source)
        if isinstance(source, TextIOBase):  # e.g., open(p, "r") or StringIO()
            source = Stream.from_iterator(s.encode() for s in source)
        if not isinstance(source, Stream):
            source = Stream.load(source)
        self.api.projects.upload_file(project_uid, source, path=str(target_path), overwrite=overwrite)

    def upload_dataset(
        self,
        project_uid: str,
        target_path: Union[str, PurePosixPath],
        dset: Dataset,
        *,
        format: int = DEFAULT_FORMAT,
        overwrite: bool = False,
    ):
        """
        Upload a dataset as a CS file into the project directory. Fails if
        target already exists.

        Args:
            project_uid (str): Project unique ID, e.g., "P3"
            target_path (str | Path): Name or path of dataset to save in the
                project directory. Should have a ``.cs`` extension.
            dset (Dataset): dataset to save.
            format (int): Format to save in from ``cryosparc.dataset.*_FORMAT``,
                defaults to NUMPY_FORMAT)
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        if format == CSDAT_FORMAT:
            return self.upload(project_uid, target_path, Stream.from_iterator(dset.stream()), overwrite=overwrite)

        if len(dset) < 100:
            # Probably small enough to upload from memory
            f = BytesIO()
            dset.save(f, format=format)
            f.seek(0)
            return self.upload(project_uid, target_path, f, overwrite=overwrite)

        # Write to temp file first
        with tempfile.TemporaryFile("w+b") as f:
            dset.save(f, format=format)
            f.seek(0)
            return self.upload(project_uid, target_path, f, overwrite=overwrite)

    def upload_mrc(
        self,
        project_uid: str,
        target_path: Union[str, PurePosixPath],
        data: "NDArray",
        psize: float,
        *,
        overwrite: bool = False,
    ):
        """
        Upload a numpy 2D or 3D array to the job directory as an MRC file.
        Fails if target already exists.

        Args:
            project_uid (str): Project unique ID, e.g., "P3"
            target_path (str | Path): Name or path of MRC file to save in the
                project directory. Should have a ``.mrc`` extension.
            data (NDArray): Numpy array with MRC file data.
            psize (float): Pixel size to include in MRC header.
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        with tempfile.TemporaryFile("w+b") as f:
            mrc.write(f, data, psize)
            f.seek(0)
            return self.upload(project_uid, target_path, f, overwrite=overwrite)

    def mkdir(
        self,
        project_uid: str,
        target_path: Union[str, PurePosixPath],
        parents: bool = False,
        exist_ok: bool = False,
    ):
        """
        Create a directory in the given project.

        Args:
            project_uid (str): Target project directory
            target_path (str | Path): Name or path of folder to create inside
                the project directory.
            parents (bool, optional): If True, any missing parents are created
                as needed. Defaults to False.
            exist_ok (bool, optional): If True, does not raise an error for
                existing directories. Still raises if the target path is not a
                directory. Defaults to False.
        """
        self.api.projects.mkdir(
            project_uid,
            path=str(target_path),
            parents=parents,
            exist_ok=exist_ok,
        )

    def cp(self, project_uid: str, source_path: Union[str, PurePosixPath], target_path: Union[str, PurePosixPath] = ""):
        """
        Copy a file or folder within a project to another location within that
        same project.

        Args:
            project_uid (str): Target project UID, e.g., "P3".
            source_path (str | Path): Relative or absolute path of source file
                or folder to copy. If relative, assumed to be within the project
                directory.
            target_path (str | Path, optional): Name or path in the project
                directory to copy into. If not specified, uses the same file
                name as the source. Defaults to "".
        """
        self.api.projects.cp(project_uid, source=str(source_path), path=str(target_path))

    def symlink(
        self,
        project_uid: str,
        source_path: Union[str, PurePosixPath],
        target_path: Union[str, PurePosixPath] = "",
    ):
        """
        Create a symbolic link in the given project. May only create links for
        files within the project.

        Args:
            project_uid (str): Target project UID, e.g., "P3".
            source_path (str | Path): Relative or absolute path of source file
                or folder to create a link to. If relative, assumed to be within
                the project directory.
            target_path (str | Path): Name or path of new symlink in the project
                directory. If not specified, creates link with the same file
                name as the source. Defaults to "".
        """
        self.api.projects.symlink(project_uid, source=str(source_path), path=str(target_path))


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

    assert voxel_type and voxel_type in mrc.VOXEL_TYPES, (
        f'Unsupported voxel type "{voxel_type}" specified with MRC exposure format'
    )
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
        f"Cannot apply low-pass filter on data with shape {arr.shape}; must be two-dimensional"
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


if __name__ == "__main__":
    from .cli import run

    run()
