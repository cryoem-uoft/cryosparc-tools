# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_client.py
"""
Python functions for programmatically controlling CryoSPARC.
To execute functions described in this module directly, you may do one of the following:

1. Provide function call as a command-line argument to ``cryosparcm cli``, e.g.,

.. code:: bash

    cryosparcm cli "api.jobs.enqueue('P3', 'J42', lane='default')"

2. Call interactively in ``cryosparcm icli``, e.g.,

.. code:: bash

    $ cryosparcm icli

    Connected to cryoem0.sbi:61002
    api, db, gfs ready to use

    In [1]: api.jobs.enqueue('P3', 'J42', lane='default')

    In [2]:

3. Call from a cryosparc-tools script, e.g.,

.. code:: python

    from cryosparc.tools import CryoSPARC

    cs = CryoSPARC(...)
    cs.api.jobs.enqueue('P3', 'J42', lane='default')

The ``api`` object used in these examples is instance of the ``APIClient`` class
defined below. Each attribute of the ``api`` object, e.g., ``api.jobs``, is an
instance of one of the ``___API`` classes, e.g., ``JobsAPI``.

Positional v.s. Keyword Arguments
=================================

API functions have positional-only, positional-or-keyword, and keyword-only
`arguments <https://docs.python.org/3/glossary.html#term-argument>`__. These
are indicated in the function signature by the use of ``/`` and ``*`` markers:

- All arguments before the ``/`` are positional-only
- All arguments after the ``*`` are keyword-only
- All arguments in between can be specified either positionally or as keywords.

For example, in the following function signature:

.. code:: python

    upload(project_uid: str, job_uid: str, /, stream: Stream, *, filename: str | None = None, format: Literal['txt', 'pdf', 'png', ...] | None = None) -> Asset:

- ``project_uid`` and ``job_uid`` are positional-only
- ``stream`` is a positional-or-keyword
- ``filename`` and ``format`` are keyword-only

Correct usage when calling this function would be:

.. code:: python

    api.assets.upload('P3', 'J42', my_stream, filename='output.txt', format='txt')

Examples of incorrect usage:

.. code:: python

    # INCORRECT USAGE EXAMPLES - will raise TypeError:
    api.assets.upload('P3', 'J42', my_stream, 'output.txt', 'txt')
    api.assets.upload('P3', job_uid='J42', stream=my_stream, filename='output.txt', format='txt')
    api.assets.upload(project_uid='P3', job_uid='J42', stream=my_stream, filename='output.txt', format='txt')

"""

import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx

from .dataset import Dataset
from .models.api_request import AppSession, SHA256Password
from .models.api_response import (
    GetFinalResultsResponse,
    Hello,
    WorkspaceAncestorUidsResponse,
    WorkspaceDescendantUidsResponse,
)
from .models.asset import GridFSAsset, GridFSFile
from .models.auth import Token
from .models.benchmarks import PerformanceBenchmark, ReferencePerformanceBenchmark
from .models.config import SystemInfo
from .models.diagnostics import RuntimeDiagnostics
from .models.event import CheckpointEvent, Event, ImageEvent, InteractiveEvent, TextEvent
from .models.exposure import Exposure
from .models.external import ExternalOutputSpec
from .models.file_browser import BrowseFileResponse, FileBrowserPrefixes
from .models.job import Job, JobStatus
from .models.job_register import JobRegister
from .models.job_spec import Category, InputSpec, OutputResult, OutputSpec
from .models.license import LicenseInstance, UpdateTag
from .models.notification import Notification
from .models.preview import DeleteProjectPreview, DeleteWorkspacePreview
from .models.project import GenerateIntermediateResultsSettings, Project, ProjectSymlink
from .models.scheduler_lane import SchedulerLane
from .models.scheduler_target import SchedulerTarget, SchedulerTargetCluster, SchedulerTargetNode
from .models.services import LoggingService
from .models.session import (
    ExposureGroup,
    ExposureGroupUpdate,
    LiveComputeResources,
    Session,
    SessionStatus,
    TemplateSelectionThreshold,
)
from .models.session_config_profile import SessionConfigProfile, SessionConfigProfileBody
from .models.session_params import LiveAbinitParams, LiveClass2DParams, LivePreprocessingParams, LiveRefineParams
from .models.tag import Tag
from .models.user import User
from .models.workspace import Workspace
from .stream import Stream

Auth = Union[str, Tuple[str, str]]
"""
Auth token or email/password.
"""

class APINamespace:
    def __init__(self, http_client: httpx.Client) -> None:
        """
        Args:
            http_client (httpx.Client): HTTP client object for making requests

        """
        ...

class ConfigAPI(APINamespace):
    """
    Functions available in ``api.config``, e.g., ``api.config.get_file_browser_settings(...)``
    """
    def get_file_browser_settings(self) -> FileBrowserPrefixes:
        """
        Returns:
            FileBrowserPrefixes: Instance file browser settings

        """
        ...
    def set_file_browser_settings(self, body: FileBrowserPrefixes) -> Any:
        """
        Args:
            body (FileBrowserPrefixes):

        Returns:
            Any: Successful Response

        """
        ...
    def set_instance_banner(
        self, *, active: bool = False, title: Optional[str] = None, body: Optional[str] = None
    ) -> Any:
        """
        Args:
            active (bool, optional): Defaults to False
            title (str, optional): Defaults to None
            body (str, optional): Defaults to None

        Returns:
            Any: Successful Response

        """
        ...
    def set_login_message(
        self, *, active: bool = False, title: Optional[str] = None, body: Optional[str] = None
    ) -> Any:
        """
        Args:
            active (bool, optional): Defaults to False
            title (str, optional): Defaults to None
            body (str, optional): Defaults to None

        Returns:
            Any: Successful Response

        """
        ...
    def get_instance_uid(self) -> str:
        """
        Gets this CryoSPARC instance's unique UID.

        Returns:
            str: Successful Response

        """
        ...
    def set_default_job_priority(self, value: int) -> Any:
        """
        Job priority

        Args:
            value (int):

        Returns:
            Any: Successful Response

        """
        ...
    def get_version(self) -> str:
        """
        Gets the current CryoSPARC version (with patch suffix, if available)

        Returns:
            str: Successful Response

        """
        ...
    def get_system_info(self) -> SystemInfo:
        """
        System information related to the CryoSPARC application

        Returns:
            SystemInfo: Successful Response

        """
        ...
    def get(self, name: str, /, *, default: Any = "<<UNDEFINED>>") -> Any:
        """
        Gets config collection entry value for the given variable name.

        Args:
            name (str):
            default (Any, optional): Defaults to '<<UNDEFINED>>'

        Returns:
            Any: Successful Response

        """
        ...
    def write(self, name: str, /, value: Any = None, *, set_on_insert_only: bool = False) -> Any:
        """
        Sets config collection entry. Specify `set_on_insert_only` to prevent
        overwriting when the value already exists.

        Args:
            name (str):
            value (Any, optional): Defaults to None
            set_on_insert_only (bool, optional): Defaults to False

        Returns:
            Any: Value in the database

        """
        ...

class InstanceAPI(APINamespace):
    """
    Functions available in ``api.instance``, e.g., ``api.instance.get_update_tag(...)``
    """
    def get_update_tag(self) -> Optional[UpdateTag]:
        """
        Gets information about updating to the next CryoSPARC version, if one is available.

        Returns:
            UpdateTag | None: Successful Response

        """
        ...
    def live_enabled(self) -> bool:
        """
        Checks if CryoSPARC Live is enabled

        Returns:
            bool: Successful Response

        """
        ...
    def ecl_enabled(self) -> bool:
        """
        Checks if embedded CryoSPARC Live is enabled

        Returns:
            bool: Successful Response

        """
        ...
    def get_license_usage(self) -> List[LicenseInstance]:
        """
        Returns:
            List[LicenseInstance]: Successful Response

        """
        ...
    def browse_files(self, *, abs_path_glob: str) -> BrowseFileResponse:
        """
        Request details for a set of files or directories available to CryoSPARC,
        given an absolute path or glob expression. If the given path is a directory,
        returns a list all files in that directory.

        .. note::
            ``abs_path_glob`` may have shell variables in it (e.g., ``$HOME``,
            ``$SCRATCH``). Variables are expanded before globs.

        Args:
            abs_path_glob (str):

        Returns:
            BrowseFileResponse: List of available file details.

        """
        ...
    def get_service_log(
        self, service: LoggingService, /, *, days: int = 30, date: Optional[str] = None, max_lines: Optional[int] = None
    ) -> str:
        """
        Gets cryosparc service logs, filterable by date. Only lines with a date are counted for max_lines.

        Args:
            service (LoggingService):
            days (int, optional): Defaults to 30
            date (str, optional): Defaults to None
            max_lines (int, optional): Defaults to None

        Returns:
            str: Successful Response

        """
        ...
    def get_runtime_diagnostics(self) -> RuntimeDiagnostics:
        """
        Gets runtime diagnostics for the CryoSPARC instance

        Returns:
            RuntimeDiagnostics: Successful Response

        """
        ...
    def audit_dump(self, *, timestamp: Union[float, Literal["auto"], None] = None) -> Optional[str]:
        """
        Generate an audit dump file containing all audit logs since the given timestamp.

        Args:
            timestamp (float | Literal['auto'], optional): Leave unspecified to dump all audit logs, set to "auto" to dump new logs since the last dump, or set to a UNIX timestamp to dump every log that occurred after it.. Defaults to None

        Returns:
            str | None: Successful Response

        """
        ...
    def generate_new_uid(self, *, force_takeover_projects: bool = False) -> str:
        """
        Generates a new uid for the CryoSPARC instance
        If force_takeover_projects is True, overwrites existing lockfiles,
        otherwise, creates lockfiles in projects that don't already have one.

        Args:
            force_takeover_projects (bool, optional): Defaults to False

        Returns:
            str: New instance UID

        """
        ...

class CacheAPI(APINamespace):
    """
    Functions available in ``api.cache``, e.g., ``api.cache.get(...)``
    """
    def get(self, key: str, /, *, namespace: Optional[str] = None) -> Any:
        """
        Returns None if the value is not set or expired

        Args:
            key (str):
            namespace (str, optional): Defaults to None

        Returns:
            Any: Successful Response

        """
        ...
    def set(self, key: str, /, value: Any = None, *, namespace: Optional[str] = None, ttl: int = 60) -> None:
        """
        Sets key to the given value, with a ttl (Time-to-Live) in seconds

        Args:
            key (str):
            value (Any, optional): Defaults to None
            namespace (str, optional): Defaults to None
            ttl (int, optional): Defaults to 60

        """
        ...

class UsersAPI(APINamespace):
    """
    Functions available in ``api.users``, e.g., ``api.users.admin_exists(...)``
    """
    def admin_exists(self) -> bool:
        """
        Returns True if there exists at least one user with admin privileges, False
        otherwise

        Returns:
            bool: Successful Response

        """
        ...
    def count(self, *, role: Optional[Literal["user", "admin"]] = None) -> int:
        """
        Args:
            role (Literal['user', 'admin'], optional): Defaults to None

        Returns:
            int: Successful Response

        """
        ...
    def table(self) -> str:
        """
        Show a table of all CryoSPARC user accounts

        Returns:
            str: Successful Response

        """
        ...
    def me(self) -> User:
        """
        Returns the current user

        Returns:
            User: Successful Response

        """
        ...
    def find_one(self, user_id: str, /) -> User:
        """
        Finds a user with a matching user ID or email

        Args:
            user_id (str): User ID or Email Address

        Returns:
            User: Successful Response

        """
        ...
    def update(
        self,
        user_id: str,
        /,
        *,
        email: Optional[str] = None,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> User:
        """
        Updates a user's general details. other params will only be set if they are
        not empty.

        Args:
            user_id (str): User ID or Email Address
            email (str, optional): Defaults to None
            username (str, optional): Defaults to None
            first_name (str, optional): Defaults to None
            last_name (str, optional): Defaults to None

        Returns:
            User: Successful Response

        """
        ...
    def delete(self, user_id: str, /) -> None:
        """
        Removes a user from the CryoSPARC. Only authenticated admins may do this.

        Args:
            user_id (str): User ID or Email Address

        """
        ...
    def get_role(self, user_id: str, /) -> Literal["user", "admin"]:
        """
        Returns "admin" if the user has admin privileges, "user" otherwise.

        Args:
            user_id (str): User ID or Email Address

        Returns:
            Literal['user', 'admin']: Successful Response

        """
        ...
    def create(
        self,
        password: Optional[SHA256Password] = None,
        *,
        email: str,
        username: str,
        first_name: str,
        last_name: str,
        role: Literal["user", "admin"] = "user",
    ) -> User:
        """
        Creates a new CryoSPARC user account. Specify ``created_by_user_id`` as the
        id of user who is creating the new user.

        The password is expected as a SHA256 hash.

        Args:
            password (SHA256Password, optional): Defaults to None
            email (str):
            username (str):
            first_name (str):
            last_name (str):
            role (Literal['user', 'admin'], optional): Defaults to 'user'

        Returns:
            User: Successful Response

        """
        ...
    def request_reset_password(self, user_id: str, /) -> None:
        """
        Generates a password reset token for a user with the given email. The token
        will appear in the Admin > User Management interface.

        Args:
            user_id (str): User ID or Email Address

        """
        ...
    def register(self, user_id: str, /, body: SHA256Password, *, token: str) -> None:
        """
        Registers user with a token (unauthenticated).

        Args:
            user_id (str): User ID or Email Address
            body (SHA256Password):
            token (str):

        """
        ...
    def reset_password(self, user_id: str, /, body: SHA256Password, *, token: str) -> None:
        """
        Resets password function with a token (unauthenticated). password is expected
        as a sha256 hash.

        Args:
            user_id (str): User ID or Email Address
            body (SHA256Password):
            token (str):

        """
        ...
    def set_role(self, user_id: str, /, role: Literal["user", "admin"]) -> User:
        """
        Changes a user's from between "user" and "admin". Only admins may do this.
        This revokes all access tokens for the given used ID.

        Args:
            user_id (str): User ID or Email Address
            role (Literal['user', 'admin']):

        Returns:
            User: Successful Response

        """
        ...
    def get_my_state_var(self, key: str, /) -> Any:
        """
        Retrieves a user's state variable such as "licenseAccepted" or
        "recentProjects"

        Args:
            key (str):

        Returns:
            Any: Successful Response

        """
        ...
    def get_state_var(self, user_id: str, key: str, /) -> Any:
        """
        Retrieves a given user's state variable such as "licenseAccepted" or
        "recentProjects"

        Args:
            user_id (str): User ID or Email Address
            key (str):

        Returns:
            Any: Successful Response

        """
        ...
    def set_state_var(self, user_id: str, key: str, /, value: Any) -> User:
        """
        Sets a property of the user's state

        Args:
            user_id (str): User ID or Email Address
            key (str):
            value (Any):

        Returns:
            User: Successful Response

        """
        ...
    def unset_state_var(self, user_id: str, key: str, /) -> User:
        """
        Deletes a property of the user's state

        Args:
            user_id (str): User ID or Email Address
            key (str):

        Returns:
            User: Successful Response

        """
        ...
    def get_lanes(self, user_id: str, /) -> List[str]:
        """
        Gets the lane names a user has access to

        Args:
            user_id (str): User ID or Email Address

        Returns:
            List[str]: Successful Response

        """
        ...
    def set_lanes(self, user_id: str, /, lanes: List[str]) -> User:
        """
        Restrict lanes the given user ID may to queue to. Only admins and account
        owners may access this function.

        Args:
            user_id (str): User ID or Email Address
            lanes (List[str]):

        Returns:
            User: Updated user

        """
        ...
    def get_file_browser_settings(self, user_id: str, /) -> FileBrowserPrefixes:
        """
        Args:
            user_id (str): User ID or Email Address

        Returns:
            FileBrowserPrefixes: User file browser settings

        """
        ...
    def set_file_browser_settings(self, user_id: str, /, body: FileBrowserPrefixes) -> User:
        """
        Args:
            user_id (str): User ID or Email Address
            body (FileBrowserPrefixes):

        Returns:
            User: Successful Response

        """
        ...

class ResourcesAPI(APINamespace):
    """
    Functions available in ``api.resources``, e.g., ``api.resources.find_lanes(...)``
    """
    def find_lanes(self) -> List[SchedulerLane]:
        """
        Finds lanes that are registered with the master scheduler.

        Returns:
            List[SchedulerLane]: List of lanes

        """
        ...
    def add_lane(self, body: SchedulerLane) -> SchedulerLane:
        """
        Adds a new lane to the master scheduler.

        Args:
            body (SchedulerLane):

        Returns:
            SchedulerLane: Successful Response

        """
        ...
    def find_lane(self, name: str, /, *, type: Literal["node", "cluster", None] = None) -> SchedulerLane:
        """
        Finds a lane registered to the master scheduler with a given name and optional type.

        Args:
            name (str):
            type (Literal['node', 'cluster', None], optional): Defaults to None

        Returns:
            SchedulerLane: Successful Response

        """
        ...
    def remove_lane(self, name: str, /) -> None:
        """
        Removes the specified lane and any targets assigned under the lane in the
        master scheduler.

        Args:
            name (str):

        """
        ...
    def find_targets(self, *, lane: Optional[str] = None) -> List[SchedulerTarget]:
        """
        Finds a list of targets that are registered with the master scheduler.

        Args:
            lane (str, optional): Defaults to None

        Returns:
            List[SchedulerTarget]: List of targets

        """
        ...
    def find_nodes(self, *, lane: Optional[str] = None) -> List[SchedulerTargetNode]:
        """
        Finds a list of targets with type "node" that are registered with the master scheduler.
        These correspond to discrete worker hostname accessible over SSH.

        Args:
            lane (str, optional): Defaults to None

        Returns:
            List[SchedulerTargetNode]: List of targets with type 'node'

        """
        ...
    def add_node(self, body: SchedulerTargetNode, *, gpu: bool = True) -> SchedulerTargetNode:
        """
        Adds a node or updates an existing node. Updates existing node if they share
        share the same name. Attempts to connect to the node via SSH to run
        the ``cryosparcw connect`` command.

        Set ``gpu`` to False to skip GPU detection.

        Args:
            body (SchedulerTargetNode):
            gpu (bool, optional): Defaults to True

        Returns:
            SchedulerTargetNode: Successful Response

        """
        ...
    def find_clusters(self, *, lane: Optional[str] = None) -> List[SchedulerTargetCluster]:
        """
        Finds a list of targets with type "cluster" that are registered with the master scheduler.
        These are multi-node clusters managed by workflow managers like SLURM or PBS and are accessible via submission script.

        Args:
            lane (str, optional): Defaults to None

        Returns:
            List[SchedulerTargetCluster]: List of targets with type 'cluster'

        """
        ...
    def add_cluster(self, body: SchedulerTargetCluster) -> SchedulerTargetCluster:
        """
        Adds a cluster or updates an existing cluster. Updates existing cluster if
        they share share the same name.

        Args:
            body (SchedulerTargetCluster):

        Returns:
            SchedulerTargetCluster: Successful Response

        """
        ...
    def find_target_by_hostname(self, hostname: str, /) -> SchedulerTarget:
        """
        Finds a target with a given hostname.

        Args:
            hostname (str):

        Returns:
            SchedulerTarget: Successful Response

        """
        ...
    def find_target_by_name(self, name: str, /) -> SchedulerTarget:
        """
        Finds a target with a given name.

        Args:
            name (str):

        Returns:
            SchedulerTarget: Successful Response

        """
        ...
    def find_node(self, name: str, /) -> SchedulerTargetNode:
        """
        Finds a node with a given name.

        Args:
            name (str):

        Returns:
            SchedulerTargetNode: Successful Response

        """
        ...
    def remove_node(self, name: str, /) -> None:
        """
        Removes a target worker node from the master scheduler

        Args:
            name (str):

        """
        ...
    def find_cluster(self, name: str, /) -> SchedulerTargetCluster:
        """
        Finds a cluster with a given name.

        Args:
            name (str):

        Returns:
            SchedulerTargetCluster: Successful Response

        """
        ...
    def remove_cluster(self, name: str, /) -> None:
        """
        Removes the specified cluster/lane and any targets assigned under the lane
        in the master scheduler

        Note: This will remove any worker node associated with the specified cluster/lane.

        Args:
            name (str):

        """
        ...
    def find_cluster_script(self, name: str, /) -> str:
        """
        Finds the cluster script for a cluster with a given name.

        Args:
            name (str):

        Returns:
            str: Successful Response

        """
        ...
    def find_cluster_template_vars(self, name: str, /) -> List[str]:
        """
        Computes and retrieves all variable names defined in cluster templates.

        Args:
            name (str):

        Returns:
            List[str]: Successful Response

        """
        ...
    def find_cluster_template_custom_vars(self, name: str, /) -> List[str]:
        """
        Computes and retrieves all custom variables names defined in cluster templates
        (i.e., all variables not in the internal list of known variable names).

        Args:
            name (str):

        Returns:
            List[str]: Successful Response

        """
        ...
    def update_node_lane(self, name: str, /, lane: str) -> SchedulerTargetNode:
        """
        Changes the lane on the given target (assumed to exist). Target type must
        match lane type.

        Args:
            name (str):
            lane (str):

        Returns:
            SchedulerTargetNode: Successful Response

        """
        ...
    def refresh_nodes(self) -> None:
        """
        Asynchronously access target worker nodes. Load latest CPU, RAM and GPU
        info.

        """
        ...
    def verify_cluster(self, name: str, /) -> str:
        """
        Ensures cluster has been properly configured by executing a generic 'info'
        command

        Args:
            name (str):

        Returns:
            str: Successful Response

        """
        ...
    def update_cluster_custom_vars(self, name: str, /, value: Dict[str, str]) -> SchedulerTargetCluster:
        """
        Changes the custom cluster variables on the given target (assumed to exist)

        Args:
            name (str):
            value (Dict[str, str]):

        Returns:
            SchedulerTargetCluster: Successful Response

        """
        ...
    def update_target_cache_path(self, name: str, /, value: Optional[str]) -> SchedulerTarget:
        """
        Changes the cache path on the given target (assumed to exist)

        Args:
            name (str):
            value (str | None):

        Returns:
            SchedulerTarget: Successful Response

        """
        ...

class AssetsAPI(APINamespace):
    """
    Functions available in ``api.assets``, e.g., ``api.assets.find(...)``
    """
    def find(self, *, project_uid: Optional[str] = None, job_uid: Optional[str] = None) -> List[GridFSFile]:
        """
        List assets associated with projects or jobs on the given instance.
        Typically returns files creating during job runs, including plots and metadata.

        Args:
            project_uid (str, optional): Defaults to None
            job_uid (str, optional): Defaults to None

        Returns:
            List[GridFSFile]: Successful Response

        """
        ...
    def upload(
        self,
        project_uid: str,
        job_uid: str,
        /,
        stream: Stream,
        *,
        filename: Optional[str] = None,
        format: Union[
            Literal["txt", "csv", "html", "json", "xml", "bild", "bld", "log"],
            Literal["pdf", "gif", "jpg", "jpeg", "png", "svg"],
            None,
        ] = None,
    ) -> GridFSAsset:
        """
        Upload a new asset associated with the given project/job. When calling
        via HTTP, provide the contents of the file in the request body. At least
        one of filename or format must be provided.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str):
            stream (Stream): A binary stream representing a Stream class instance
            filename (str, optional): Defaults to None
            format (Literal['txt', 'csv', 'html', 'json', 'xml', 'bild', 'bld', 'log'] | Literal['pdf', 'gif', 'jpg', 'jpeg', 'png', 'svg'], optional): Defaults to None

        Returns:
            GridFSAsset: Successful Response

        """
        ...
    def download(self, id: str = "000000000000000000000000", /) -> Stream:
        """
        Download the asset with the given ID. When calling via HTTP, file contents
        will be in the response body.

        Args:
            id (str, optional): Defaults to '000000000000000000000000'

        Returns:
            Stream: A binary stream representing a Stream class instance

        """
        ...
    def find_one(self, id: str = "000000000000000000000000", /) -> GridFSFile:
        """
        Retrive the full details for an asset with the given ID.

        Args:
            id (str, optional): Defaults to '000000000000000000000000'

        Returns:
            GridFSFile: Successful Response

        """
        ...

class JobsAPI(APINamespace):
    """
    Functions available in ``api.jobs``, e.g., ``api.jobs.find(...)``
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        project_uid: Optional[List[str]] = None,
        workspace_uid: Optional[List[str]] = None,
        uid: Optional[List[str]] = None,
        type: Optional[List[str]] = None,
        status: Optional[List[JobStatus]] = None,
        category: Optional[List[Category]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        waiting_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        failed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        deleted: Optional[bool] = False,
    ) -> List[Job]:
        """
        Finds all jobs that match the supplied query

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            project_uid (List[str], optional): Defaults to None
            workspace_uid (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            type (List[str], optional): Defaults to None
            status (List[JobStatus], optional): Defaults to None
            category (List[Category], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            queued_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            started_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            waiting_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            completed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            killed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            failed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            exported_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            List[Job]: List of jobs matching supplied query

        """
        ...
    def delete_many(self, project_job_uids: List[Tuple[str, str]]) -> None:
        """
        Deletes the given jobs. Ignores protected jobs if `force` is `True`.

        Args:
            project_job_uids (List[Tuple[str, str]]):

        """
        ...
    def count(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        project_uid: Optional[List[str]] = None,
        workspace_uid: Optional[List[str]] = None,
        uid: Optional[List[str]] = None,
        type: Optional[List[str]] = None,
        status: Optional[List[JobStatus]] = None,
        category: Optional[List[Category]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        waiting_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        failed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        deleted: Optional[bool] = False,
    ) -> int:
        """
        Counts number of jobs that match the supplied query.

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            project_uid (List[str], optional): Defaults to None
            workspace_uid (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            type (List[str], optional): Defaults to None
            status (List[JobStatus], optional): Defaults to None
            category (List[Category], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            queued_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            started_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            waiting_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            completed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            killed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            failed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            exported_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            int: Successful Response

        """
        ...
    def get_active_count(self) -> int:
        """
        Counts number of active jobs.

        Returns:
            int: Successful Response

        """
        ...
    def clone_many(
        self,
        project_uid: str,
        /,
        job_uids: List[str],
        *,
        workspace_uid: Optional[str] = None,
        new_workspace_title: Optional[str] = None,
    ) -> List[Job]:
        """
        Clones the given list of jobs. If any jobs are related, it will try to
        re-create the input connections between the cloned jobs (but maintain the
        same connections to jobs that were not cloned)

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uids (List[str]):
            workspace_uid (str, optional): Defaults to None
            new_workspace_title (str, optional): Defaults to None

        Returns:
            List[Job]: List of cloned jobs

        """
        ...
    def get_chain(self, project_uid: str, /, *, start_job_uid: str, end_job_uid: str) -> List[str]:
        """
        Finds the chain of jobs between start job to end job.
        A job chain is the intersection of the start job's descendants and the end job's
        ancestors.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            start_job_uid (str):
            end_job_uid (str):

        Returns:
            List[str]: Successful Response

        """
        ...
    def clone_chain(
        self,
        project_uid: str,
        /,
        *,
        start_job_uid: str,
        end_job_uid: str,
        workspace_uid: Optional[str] = None,
        new_workspace_title: Optional[str] = None,
    ) -> List[Job]:
        """
        Clones jobs that directly descend from the start job UID up to the end job UID.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            start_job_uid (str):
            end_job_uid (str):
            workspace_uid (str, optional): Defaults to None
            new_workspace_title (str, optional): Defaults to None

        Returns:
            List[Job]: Newly created cloned jobs

        """
        ...
    def get_final_results(self, project_uid: str, /) -> GetFinalResultsResponse:
        """
        Gets all final results within a project, along with the ancestors and non-ancestors of those jobs.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            GetFinalResultsResponse: Successful Response

        """
        ...
    def find_one(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Finds the job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
        ...
    def delete(self, project_uid: str, job_uid: str, /) -> None:
        """
        Deletes a job. Will kill (if running) and clearing the job before deleting.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        """
        ...
    def get_directory(self, project_uid: str, job_uid: str, /) -> str:
        """
        Gets the job directory for a given job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            str: Successful Response

        """
        ...
    def get_log(self, project_uid: str, job_uid: str, /) -> str:
        """
        Returns contents of the job.log file. Returns empty string if job.log does not exist.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            str: Successful Response

        """
        ...
    def get_log_path(self, project_uid: str, job_uid: str, /) -> str:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            str: Successful Response

        """
        ...
    def get_output_fields(
        self, project_uid: str, job_uid: str, output_name: str, /, dtype_params: Dict[str, Any] = {}
    ) -> List[Tuple[str, str]]:
        """
        Expected dataset column definitions for given job output, excluding passthroughs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str):
            dtype_params (Dict[str, Any], optional): Defaults to {}

        Returns:
            List[Tuple[str, str]]: Successful Response

        """
        ...
    def create(
        self,
        project_uid: str,
        workspace_uid: str,
        /,
        params: Dict[str, Union[bool, int, float, str, str, None]] = {},
        *,
        type: str,
        title: str = "",
        description: str = "",
        created_by_job_uid: Optional[str] = None,
        enable_bench: bool = False,
    ) -> Job:
        """
        Creates a new job with the given type in the project/workspace

        To see all available job types and their parameters, see the `GET projects/{project_uid}:register` endpoint

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            params (Dict[str, Union[bool, int, float, str, str, None]], optional): Defaults to {}
            type (str): Type of job to create
            title (str, optional): Defaults to ''
            description (str, optional): Defaults to ''
            created_by_job_uid (str, optional): Defaults to None
            enable_bench (bool, optional): Defaults to False

        Returns:
            Job: Successful Response

        """
        ...
    def create_external_result(self, project_uid: str, workspace_uid: str, /, body: ExternalOutputSpec) -> Job:
        """
        Create an external result with the given specification. Returns an external
        job with the given output ready to be saved. Used with cryosparc-tools

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            body (ExternalOutputSpec):

        Returns:
            Job: Successful Response

        """
        ...
    def get_status(self, project_uid: str, job_uid: str, /) -> JobStatus:
        """
        Gets the status of a job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            JobStatus: Successful Response

        """
        ...
    def view(self, project_uid: str, workspace_uid: str, job_uid: str, /) -> Job:
        """
        Adds a project, workspace and job uid to a user's recently viewed jobs list

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
        ...
    def set_param(self, project_uid: str, job_uid: str, param: str, /, value: Any) -> Job:
        """
        Sets the given job parameter to the value

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            param (str):
            value (Any):

        Returns:
            Job: Successful Response

        """
        ...
    def clear_param(self, project_uid: str, job_uid: str, param: str, /) -> Job:
        """
        Resets the given parameter to its default value.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            param (str):

        Returns:
            Job: Successful Response

        """
        ...
    def load_input(
        self,
        project_uid: str,
        job_uid: str,
        input_name: str,
        /,
        *,
        force_join: Union[bool, Literal["auto"]] = "auto",
        slots: Union[Literal["default", "passthrough", "all"], List[str]] = "default",
    ) -> Dataset:
        """
        Load job input dataset. Raises exception if no inputs are connected.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str):
            force_join (bool | Literal['auto'], optional): Defaults to 'auto'
            slots (Literal['default', 'passthrough', 'all'] | List[str], optional): Defaults to 'default'

        Returns:
            Dataset: A binary stream representing a Dataset class instance

        """
        ...
    def load_output(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        /,
        *,
        version: Union[int, Literal["F"]] = "F",
        slots: Union[Literal["default", "passthrough", "all"], List[str]] = "default",
    ) -> Dataset:
        """
        Load job output dataset. Raises exception if output is empty or does not exists.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str):
            version (int | Literal['F'], optional): Set to F (default) to load the final version. Defaults to 'F'
            slots (Literal['default', 'passthrough', 'all'] | List[str], optional): Defaults to 'default'

        Returns:
            Dataset: A binary stream representing a Dataset class instance

        """
        ...
    def save_output(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        /,
        dataset: Dataset,
        *,
        filename: Optional[str] = None,
        version: int = 0,
    ) -> Job:
        """
        Save job output dataset. Job must be running or waiting.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str):
            dataset (Dataset): A binary stream representing a Dataset class instance
            filename (str, optional): Defaults to None
            version (int, optional): Defaults to 0

        Returns:
            Job: Successful Response

        """
        ...
    def connect(
        self, project_uid: str, job_uid: str, input_name: str, /, *, source_output_name: str, source_job_uid: str
    ) -> Job:
        """
        Connects the input slot on the child job to the output group on the
        parent job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str):
            source_output_name (str):
            source_job_uid (str):

        Returns:
            Job: Connected child job

        """
        ...
    def disconnect_all(self, project_uid: str, job_uid: str, input_name: str, /) -> Job:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str):

        Returns:
            Job: Successful Response

        """
        ...
    def reconnect(
        self,
        project_uid: str,
        job_uid: str,
        input_name: str,
        connection_index: int,
        /,
        *,
        source_output_name: str,
        source_job_uid: str,
    ) -> Job:
        """
        Replace the connection at the given index with a new connection.
        Specify index -1 to replace the last connection.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str):
            connection_index (int):
            source_output_name (str):
            source_job_uid (str):

        Returns:
            Job: Successful Response

        """
        ...
    def disconnect(self, project_uid: str, job_uid: str, input_name: str, connection_index: int, /) -> Job:
        """
        Removes connected inputs on the given input.

        Optionally specify an index to disconnect a specific connection.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str):
            connection_index (int):

        Returns:
            Job: Successful Response

        """
        ...
    def find_output_result(self, project_uid: str, job_uid: str, output_name: str, result_name: str, /) -> OutputResult:
        """
        Get a job's low-level output result.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str):
            result_name (str):

        Returns:
            OutputResult: Successful Response

        """
        ...
    def delete_output_result_files(self, project_uid: str, job_uid: str, output_name: str, result_name: str, /) -> None:
        """
        Remove all data files referenced in a job's output result. For example, for
        an Extract Particles job, specify the "particles" output and "blob" result
        to remove the particle stacks created by the job.

        Has no effect when clearing results from jobs where the result was
        passed through from an ancenstor job, e.g., cannot clear "blob" result
        from a 2D Classification job connected to the Extract job.

        This operation may affect downstream jobs that use these files as input.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str):
            result_name (str):

        """
        ...
    def connect_result(
        self,
        project_uid: str,
        job_uid: str,
        input_name: str,
        connection_index: int,
        result_name: str,
        /,
        *,
        source_output_name: str,
        source_result_name: str,
        source_result_version: Union[int, Literal["F"]] = "F",
        source_job_uid: str,
    ) -> Job:
        """
        Adds or replaces a result within an input connection with the given output result from a different job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str):
            connection_index (int):
            result_name (str):
            source_output_name (str):
            source_result_name (str):
            source_result_version (int | Literal['F'], optional): Defaults to 'F'
            source_job_uid (str):

        Returns:
            Job: Successful Response

        """
        ...
    def disconnect_result(
        self, project_uid: str, job_uid: str, input_name: str, connection_index: int, result_name: str, /
    ) -> Job:
        """
        Removes an output result connected within the given input connection.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str):
            connection_index (int):
            result_name (str):

        Returns:
            Job: Successful Response

        """
        ...
    def add_external_input(self, project_uid: str, job_uid: str, input_name: str, /, body: InputSpec) -> Job:
        """
        Add or replace an external job's input. This action is available while the
        job is building, running or waiting for results.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str): May only contain letters, numbers and underscores, and must start with a letter
            body (InputSpec):

        Returns:
            Job: Successful Response

        """
        ...
    def add_external_output(self, project_uid: str, job_uid: str, output_name: str, /, body: OutputSpec) -> Job:
        """
        Add or replace an external job's output. This action is available while the
        job is building, running or waiting for results.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): May only contain letters, numbers and underscores, and must start with a letter
            body (OutputSpec):

        Returns:
            Job: Successful Response

        """
        ...
    def set_output_image(self, project_uid: str, job_uid: str, output_name: str, /, body: GridFSAsset) -> Job:
        """
        Set a custom image for an External job output.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str):
            body (GridFSAsset):

        Returns:
            Job: Successful Response

        """
        ...
    def set_tile_image(self, project_uid: str, job_uid: str, /, body: GridFSAsset) -> Job:
        """
        Set a custom job tile image for an External job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            body (GridFSAsset):

        Returns:
            Job: Successful Response

        """
        ...
    def enqueue(
        self,
        project_uid: str,
        job_uid: str,
        /,
        *,
        lane: Optional[str] = None,
        hostname: Optional[str] = None,
        gpus: List[int] = [],
        no_check_inputs_ready: bool = False,
    ) -> Job:
        """
        Adds the job to the queue for the given worker lane (default lane if not specified)

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            lane (str, optional): Defaults to None
            hostname (str, optional): Defaults to None
            gpus (List[int], optional): Defaults to []
            no_check_inputs_ready (bool, optional): Defaults to False

        Returns:
            Job: Successful Response

        """
        ...
    def recalculate_size_async(self, project_uid: str, job_uid: str, /) -> None:
        """
        For a job, find intermediate results and recalculate their total size.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        """
        ...
    def recalculate_project_intermediate_results_size(self, project_uid: str, /) -> None:
        """
        Recaclulates intermediate result sizes for all jobs in a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
        ...
    def clear_intermediate_results(self, project_uid: str, job_uid: str, /, *, always_keep_final: bool = True) -> None:
        """
        Removes intermediate results from the job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            always_keep_final (bool, optional): Defaults to True

        """
        ...
    def export_output_results(
        self, project_uid: str, job_uid: str, output_name: str, /, result_names: Optional[List[str]] = None
    ) -> None:
        """
        Prepares a job's output for import to another project or instance.
        Creates a folder in the project directory → exports subfolder,
        then links the output's associated files there.

        Note that the returned .csg file's parent folder must be manually copied
        with symlinks resolved into the target project folder before importing.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str):
            result_names (List[str], optional): Defaults to None

        """
        ...
    def export_job(self, project_uid: str, job_uid: str, /) -> None:
        """
        Start export for the job into the project's exports directory

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        """
        ...
    def get_output_result_path(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        result_name: str,
        /,
        *,
        version: Union[int, Literal["F"]] = "F",
    ) -> str:
        """
        Get the absolute path for a job output's dataset or volume density.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str):
            result_name (str):
            version (int | Literal['F'], optional): Defaults to 'F'

        Returns:
            str: Successful Response

        """
        ...
    def interactive_post(
        self, project_uid: str, job_uid: str, /, body: Dict[str, Any], *, endpoint: str, timeout: int = 10
    ) -> Any:
        """
        Sends a message to an interactive job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            body (Dict[str, Any]):
            endpoint (str):
            timeout (int, optional): Defaults to 10

        Returns:
            Any: Successful Response

        """
        ...
    def mark_running(
        self, project_uid: str, job_uid: str, /, *, status: Literal["running", "waiting"] = "running"
    ) -> Job:
        """
        Indicate that an external job is running or waiting.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            status (Literal['running', 'waiting'], optional): Defaults to 'running'

        Returns:
            Job: Successful Response

        """
        ...
    def mark_completed(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Mark a killed or failed job, or an active external job, as completed.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
        ...
    def mark_failed(self, project_uid: str, job_uid: str, /, *, error: Optional[str] = None) -> Job:
        """
        Manually mark a job as failed.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            error (str, optional): Defaults to None

        Returns:
            Job: Successful Response

        """
        ...
    def add_event_log(
        self, project_uid: str, job_uid: str, /, text: str, *, type: Literal["text", "warning", "error"] = "text"
    ) -> TextEvent:
        """
        Add the message to the target job's event log.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            text (str):
            type (Literal['text', 'warning', 'error'], optional): Defaults to 'text'

        Returns:
            TextEvent: Successful Response

        """
        ...
    def get_event_logs(
        self, project_uid: str, job_uid: str, /, *, checkpoint: Optional[int] = None
    ) -> List[Union[TextEvent, ImageEvent, InteractiveEvent, CheckpointEvent, Event]]:
        """
        Gets all event logs for a job.

        Note: this may return a lot of documents.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            checkpoint (int, optional): Defaults to None

        Returns:
            List[Union[TextEvent, ImageEvent, InteractiveEvent, CheckpointEvent, Event]]: Successful Response

        """
        ...
    def add_image_log(
        self, project_uid: str, job_uid: str, /, images: List[GridFSAsset], *, text: str, flags: List[str] = ["plots"]
    ) -> ImageEvent:
        """
        Add an image or figure to the target job's event log.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            images (List[GridFSAsset]):
            text (str):
            flags (List[str], optional): Defaults to ['plots']

        Returns:
            ImageEvent: Successful Response

        """
        ...
    def add_checkpoint(self, project_uid: str, job_uid: str, /, meta: Dict[str, Any]) -> CheckpointEvent:
        """
        Add a checkpoint the target job's event log.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            meta (Dict[str, Any]):

        Returns:
            CheckpointEvent: Successful Response

        """
        ...
    def update_event_log(
        self,
        project_uid: str,
        job_uid: str,
        event_id: str = "000000000000000000000000",
        /,
        text: Optional[str] = None,
        *,
        type: Optional[Literal["text", "warning", "error"]] = None,
    ) -> TextEvent:
        """
        Update a text event log entry for a job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            event_id (str, optional): Defaults to '000000000000000000000000'
            text (str, optional): Defaults to None
            type (Literal['text', 'warning', 'error'], optional): Defaults to None

        Returns:
            TextEvent: Successful Response

        """
        ...
    def recalculate_size(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Recalculates the size of a given job's directory.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
        ...
    def clear(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Clears a job to get it back to building state (do not clear params or inputs).

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
        ...
    def clear_many(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        project_uid: Optional[List[str]] = None,
        workspace_uid: Optional[List[str]] = None,
        uid: Optional[List[str]] = None,
        type: Optional[List[str]] = None,
        status: Optional[List[JobStatus]] = None,
        category: Optional[List[Category]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        waiting_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        failed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        deleted: Optional[bool] = False,
    ) -> List[Job]:
        """
        Clears all jobs that matches the query.

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            project_uid (List[str], optional): Defaults to None
            workspace_uid (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            type (List[str], optional): Defaults to None
            status (List[JobStatus], optional): Defaults to None
            category (List[Category], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            queued_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            started_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            waiting_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            completed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            killed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            failed_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            exported_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            List[Job]: Successful Response

        """
        ...
    def clone(
        self,
        project_uid: str,
        job_uid: str,
        /,
        *,
        workspace_uid: Optional[str] = None,
        created_by_job_uid: Optional[str] = None,
    ) -> Job:
        """
        Creates a new job as a clone of the provided job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str, optional): Defaults to None
            created_by_job_uid (str, optional): Defaults to None

        Returns:
            Job: Cloned job

        """
        ...
    def kill(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Kills a running job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
        ...
    def set_final_result(self, project_uid: str, job_uid: str, /, *, is_final_result: bool) -> Job:
        """
        Marks a job as a final result. A job marked as a final result and its ancestor jobs are protected during data cleanup.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            is_final_result (bool):

        Returns:
            Job: Successful Response

        """
        ...
    def set_title(self, project_uid: str, job_uid: str, /, *, title: str) -> Job:
        """
        Sets job title.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            title (str):

        Returns:
            Job: Successful Response

        """
        ...
    def set_description(self, project_uid: str, job_uid: str, /, description: str) -> Job:
        """
        Sets job description.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            description (str):

        Returns:
            Job: Successful Response

        """
        ...
    def set_priority(self, project_uid: str, job_uid: str, /, *, priority: int) -> Job:
        """
        Sets job priority

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            priority (int):

        Returns:
            Job: Successful Response

        """
        ...
    def set_cluster_custom_vars(self, project_uid: str, job_uid: str, /, cluster_custom_vars: Dict[str, str]) -> Job:
        """
        Sets cluster custom variables for job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            cluster_custom_vars (Dict[str, str]):

        Returns:
            Job: Successful Response

        """
        ...
    def get_active_licenses_count(self) -> int:
        """
        Gets number of acquired licenses for running jobs

        Returns:
            int: Successful Response

        """
        ...
    def get_types(self) -> Any:
        """
        Gets list of available job types

        Returns:
            Any: Successful Response

        """
        ...
    def get_categories(self) -> Any:
        """
        Gets job types by category

        Returns:
            Any: Successful Response

        """
        ...
    def find_ancestor_uids(
        self, project_uid: str, job_uid: str, /, *, workspace_uid: Optional[str] = None
    ) -> List[str]:
        """
        Finds all ancestors of a single job and return a list of their UIDs

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str, optional): Defaults to None

        Returns:
            List[str]: Successful Response

        """
        ...
    def find_descendant_uids(
        self, project_uid: str, job_uid: str, /, *, workspace_uid: Optional[str] = None
    ) -> List[str]:
        """
        Find the list of all job UIDs that this job is an ancestor of based
        on its outputs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str, optional): Defaults to None

        Returns:
            List[str]: Successful Response

        """
        ...
    def link_to_workspace(self, project_uid: str, job_uid: str, workspace_uid: str, /) -> Job:
        """
        Adds a job to a workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Job: Successful Response

        """
        ...
    def unlink_from_workspace(self, project_uid: str, job_uid: str, workspace_uid: str, /) -> Job:
        """
        Removes a job from a workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Job: Successful Response

        """
        ...
    def move(self, project_uid: str, job_uid: str, /, *, from_workspace_uid: str, to_workspace_uid: str) -> Job:
        """
        Moves a job from one workspace to another.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            from_workspace_uid (str):
            to_workspace_uid (str):

        Returns:
            Job: Successful Response

        """
        ...
    def update_directory_symlinks(self, project_uid: str, job_uid: str, /, *, prefix_cut: str, prefix_new: str) -> int:
        """
        Rewrites all symbolic links in the job directory, modifying links prefixed with `prefix_cut` to instead be prefixed with `prefix_new`.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            prefix_cut (str):
            prefix_new (str):

        Returns:
            int: Number of symlinks updated

        """
        ...
    def add_tag(self, project_uid: str, job_uid: str, tag_uid: str, /) -> Job:
        """
        Tags a job with the given tag.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            tag_uid (str):

        Returns:
            Job: Successful Response

        """
        ...
    def remove_tag(self, project_uid: str, job_uid: str, tag_uid: str, /) -> Job:
        """
        Removes the given tag a job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            tag_uid (str):

        Returns:
            Job: Successful Response

        """
        ...
    def import_job(self, project_uid: str, workspace_uid: str, /, *, path: str = "") -> None:
        """
        Imports the exported job directory into the project. Exported job
        directory must be copied to the target project directory with all its symbolic links resolved.
        By convention, the exported job directory should be located in the project directory → exports subfolder

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        """
        ...
    def import_result_group(
        self, project_uid: str, workspace_uid: str, /, *, lane: Optional[str] = None, path: str = ""
    ) -> Job:
        """
        Creates and enqueues an Import Result Group job with the given path

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            lane (str, optional): Defaults to None
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            Job: Successful Response

        """
        ...
    def star_job(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Stars a job for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
        ...
    def unstar_job(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Unstars a job for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
        ...

class WorkspacesAPI(APINamespace):
    """
    Functions available in ``api.workspaces``, e.g., ``api.workspaces.find(...)``
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        uid: Optional[List[str]] = None,
        project_uid: Optional[List[str]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        deleted: Optional[bool] = False,
    ) -> List[Workspace]:
        """
        List all workspaces. Specify a filter to list all workspaces in a specific
        project.

        Examples:

            >>> api.workspaces.find(project_uid="P1")

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            List[Workspace]: Successful Response

        """
        ...
    def count(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        uid: Optional[List[str]] = None,
        project_uid: Optional[List[str]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        deleted: Optional[bool] = False,
    ) -> int:
        """
        Count all workspaces. Use a query to count workspaces in a specific project.

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            int: Successful Response

        """
        ...
    def preview_delete(self, project_uid: str, workspace_uid: str, /) -> DeleteWorkspacePreview:
        """
        Get a list of jobs that would be removed when the given workspace is deleted.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            DeleteWorkspacePreview: Successful Response

        """
        ...
    def find_one(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Find a specific workspace in a project

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Workspace: Successful Response

        """
        ...
    def delete(self, project_uid: str, workspace_uid: str, /) -> None:
        """
        Marks the workspace as "deleted". Deletes jobs that are only linked to this workspace
        and no other workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        """
        ...
    def create(
        self,
        project_uid: str,
        /,
        *,
        title: str,
        description: Optional[str] = None,
        created_by_job_uid: Optional[str] = None,
    ) -> Workspace:
        """
        Create a new workspace

        Args:
            project_uid (str): Project UID, e.g., "P3"
            title (str):
            description (str, optional): Defaults to None
            created_by_job_uid (str, optional): Defaults to None

        Returns:
            Workspace: Successful Response

        """
        ...
    def set_title(self, project_uid: str, workspace_uid: str, /, *, title: str) -> Workspace:
        """
        Set title of a workspace

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            title (str):

        Returns:
            Workspace: Successful Response

        """
        ...
    def set_description(self, project_uid: str, workspace_uid: str, /, description: str) -> Workspace:
        """
        Set description of a workspace

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            description (str):

        Returns:
            Workspace: Successful Response

        """
        ...
    def view(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Adds a workspace uid to a user's recently viewed workspaces list.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Workspace: Successful Response

        """
        ...
    def add_tag(self, project_uid: str, workspace_uid: str, tag_uid: str, /) -> Workspace:
        """
        Tag the given workspace with the given tag.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            tag_uid (str):

        Returns:
            Workspace: Successful Response

        """
        ...
    def remove_tag(self, project_uid: str, workspace_uid: str, tag_uid: str, /) -> Workspace:
        """
        Removes a tag from a workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            tag_uid (str):

        Returns:
            Workspace: Successful Response

        """
        ...
    def clear_intermediate_results(
        self, project_uid: str, workspace_uid: str, /, *, always_keep_final: bool = False
    ) -> None:
        """
        Remove intermediate results from a workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            always_keep_final (bool, optional): Defaults to False

        """
        ...
    def find_workspace_ancestor_uids(
        self, project_uid: str, workspace_uid: str, /, *, job_uids: List[str]
    ) -> WorkspaceAncestorUidsResponse:
        """
        Finds ancestors of the given jobs in the workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            job_uids (List[str]):

        Returns:
            WorkspaceAncestorUidsResponse: Successful Response

        """
        ...
    def find_workspace_descendant_uids(
        self, project_uid: str, workspace_uid: str, /, *, job_uids: List[str]
    ) -> WorkspaceDescendantUidsResponse:
        """
        Finds descendants of jobs in the workspace

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            job_uids (List[str]):

        Returns:
            WorkspaceDescendantUidsResponse: Successful Response

        """
        ...
    def star_workspace(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Stars a workspace for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Workspace: Successful Response

        """
        ...
    def unstar_workspace(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Unstars a workspace for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Workspace: Successful Response

        """
        ...

class SessionsAPI(APINamespace):
    """
    Functions available in ``api.sessions``, e.g., ``api.sessions.find(...)``
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        uid: Optional[List[str]] = None,
        session_uid: Optional[List[str]] = None,
        project_uid: Optional[List[str]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        cleared_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        status: Optional[List[SessionStatus]] = None,
        deleted: Optional[bool] = False,
    ) -> List[Session]:
        """
        Lists all sessions (optionally, in a project)

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            uid (List[str], optional): Defaults to None
            session_uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            cleared_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            status (List[SessionStatus], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            List[Session]: Successful Response

        """
        ...
    def count(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        uid: Optional[List[str]] = None,
        session_uid: Optional[List[str]] = None,
        project_uid: Optional[List[str]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        cleared_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        status: Optional[List[SessionStatus]] = None,
        deleted: Optional[bool] = False,
    ) -> int:
        """
        Counts all sessions in a project

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            uid (List[str], optional): Defaults to None
            session_uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            cleared_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            status (List[SessionStatus], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            int: Successful Response

        """
        ...
    def find_one(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Finds a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def delete(self, project_uid: str, session_uid: str, /) -> None:
        """
        Sets the session document as "deleted"
        Will throw an error if any undeleted jobs exist within the session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
        ...
    def create(
        self,
        project_uid: str,
        /,
        *,
        title: str,
        description: Optional[str] = None,
        created_by_job_uid: Optional[str] = None,
    ) -> Session:
        """
        Creates a new session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            title (str):
            description (str, optional): Defaults to None
            created_by_job_uid (str, optional): Defaults to None

        Returns:
            Session: Successful Response

        """
        ...
    def clone(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        created_by_job_uid: Optional[str] = None,
    ) -> Session:
        """
        Clones an existing session, copying session configuration, parameters, and exposure groups.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            title (str, optional): Defaults to None
            description (str, optional): Defaults to None
            created_by_job_uid (str, optional): Defaults to None

        Returns:
            Session: Successful Response

        """
        ...
    def find_exposure_groups(self, project_uid: str, session_uid: str, /) -> List[ExposureGroup]:
        """
        Finds all exposure groups in a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            List[ExposureGroup]: Successful Response

        """
        ...
    def create_exposure_group(self, project_uid: str, session_uid: str, /) -> ExposureGroup:
        """
        Creates an exposure group for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            ExposureGroup: Successful Response

        """
        ...
    def find_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> ExposureGroup:
        """
        Finds an exposure group with a specific id for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_group_id (int):

        Returns:
            ExposureGroup: Successful Response

        """
        ...
    def update_exposure_group(
        self, project_uid: str, session_uid: str, exposure_group_id: int, /, body: ExposureGroupUpdate
    ) -> ExposureGroup:
        """
        Updates properties of an exposure group.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_group_id (int):
            body (ExposureGroupUpdate):

        Returns:
            ExposureGroup: Successful Response

        """
        ...
    def delete_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> Session:
        """
        Deletes an exposure group from a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_group_id (int):

        Returns:
            Session: Successful Response

        """
        ...
    def finalize_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> ExposureGroup:
        """
        Finalizes an exposure group.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_group_id (int):

        Returns:
            ExposureGroup: Successful Response

        """
        ...
    def start(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Builds and starts a CryoSPARC Live Session. Builds file engines based on file
        engine parameters in the session doc, builds phase one workers based on lane
        parameters in the session doc.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def pause(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Pauses a CryoSPARC Live Session. Gracefully stops and kills all phase one workers, file engines and phase two jobs

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def configure_auto_pause(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        auto_pause: Literal["disabled", "graceful", "immediate"],
        auto_pause_after_idle_minutes: int = 10,
    ) -> Session:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            auto_pause (Literal['disabled', 'graceful', 'immediate']):
            auto_pause_after_idle_minutes (int, optional): Defaults to 10

        Returns:
            Session: Successful Response

        """
        ...
    def update_compute_configuration(
        self, project_uid: str, session_uid: str, /, body: LiveComputeResources
    ) -> LiveComputeResources:
        """
        Updates compute configuration for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveComputeResources):

        Returns:
            LiveComputeResources: Successful Response

        """
        ...
    def add_tag(self, project_uid: str, session_uid: str, tag_uid: str, /) -> Session:
        """
        Tags a session with the given tag.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            tag_uid (str):

        Returns:
            Session: Successful Response

        """
        ...
    def remove_tag(self, project_uid: str, session_uid: str, tag_uid: str, /) -> Session:
        """
        Removes the given tag from a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            tag_uid (str):

        Returns:
            Session: Successful Response

        """
        ...
    def update_session_params(
        self,
        project_uid: str,
        session_uid: str,
        /,
        body: LivePreprocessingParams,
        *,
        reprocess: bool = True,
        priority: int = 1,
    ) -> Session:
        """
        Updates a session's params. Updates each exposure inside the session with the new stage to start processing at (if there is one).

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LivePreprocessingParams):
            reprocess (bool, optional): Defaults to True
            priority (int, optional): Defaults to 1

        Returns:
            Session: Successful Response

        """
        ...
    def update_attribute_threshold(
        self,
        project_uid: str,
        session_uid: str,
        attribute: str,
        /,
        *,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Session:
        """
        Updates thresholds for a given attribute.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            attribute (str):
            min_val (float, optional): Defaults to None
            max_val (float, optional): Defaults to None

        Returns:
            Session: Successful Response

        """
        ...
    def clear_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Deletes all file engine documents (removing all previously known files and
        max timestamps), all Phase 1 Worker jobs and all associated
        exposure documents.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def view(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Adds a project, workspace and job uid to a user's recently viewed sessions list

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def setup_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Setup streaming 2D classification job for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
        ...
    def reinitialize_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Setup streaming 2D classification job for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
        ...
    def enqueue_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueues streaming 2D Classification job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
        ...
    def stop_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stops streaming 2D Classification job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def clear_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clears streaming 2D Classification job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def update_phase2_class2D_params(self, project_uid: str, session_uid: str, /, body: LiveClass2DParams) -> Session:
        """
        Updates streaming 2D Classification job params for session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveClass2DParams):

        Returns:
            Session: Successful Response

        """
        ...
    def toggle_class2d_template(self, project_uid: str, session_uid: str, template_idx: int, /) -> Session:
        """
        Inverts selected template for the streaming 2D Classification job of a job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_idx (int):

        Returns:
            Session: Successful Response

        """
        ...
    def toggle_all_class2d_templates(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Inverts all templates for a session's streaming 2D classification job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def select_all_class2d_templates(
        self, project_uid: str, session_uid: str, direction: Literal["select", "deselect"], /
    ) -> Session:
        """
        Sets all templates in the session's streaming 2D Classification job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            direction (Literal['select', 'deselect']):

        Returns:
            Session: Successful Response

        """
        ...
    def select_class2d_templates_with_threshold_index(
        self,
        project_uid: str,
        session_uid: str,
        template_idx: int,
        /,
        *,
        dimension: Literal["num_particles_total", "res_A", "class_ess"],
        direction: Literal["above", "below"] = "above",
    ) -> Session:
        """
        Sets all templates above or below an index for a session's streaming 2D Classification

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_idx (int):
            dimension (Literal['num_particles_total', 'res_A', 'class_ess']):
            direction (Literal['above', 'below'], optional): Defaults to 'above'

        Returns:
            Session: Successful Response

        """
        ...
    def select_class2d_templates_with_thresholds(
        self, project_uid: str, session_uid: str, /, template_selection_thresholds: List[TemplateSelectionThreshold]
    ) -> Session:
        """
        Selects all templates above or below an index in a template creation job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_selection_thresholds (List[TemplateSelectionThreshold]):

        Returns:
            Session: Successful Response

        """
        ...
    def start_extract_manual(self, project_uid: str, session_uid: str, /) -> None:
        """
        Extracts manual picks from a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
        ...
    def set_session_exposure_processing_priority(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        exposure_processing_priority: Literal["normal", "oldest", "latest", "alternate"],
    ) -> Session:
        """
        Sets session exposure processing priority

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_processing_priority (Literal['normal', 'oldest', 'latest', 'alternate']):

        Returns:
            Session: Successful Response

        """
        ...
    def set_session_phase_one_wait_for_exposures(
        self, project_uid: str, session_uid: str, /, *, phase_one_wait_for_exposures: bool
    ) -> Session:
        """
        Sets whether to wait until exposures are available before queuing the session's preprocessing worker jobs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            phase_one_wait_for_exposures (bool):

        Returns:
            Session: Successful Response

        """
        ...
    def update_picking_threshold_values(
        self,
        project_uid: str,
        session_uid: str,
        picker_type: Literal["blob", "template", "deep"],
        /,
        *,
        ncc_value: float,
        power_min_value: float,
        power_max_value: float,
    ) -> Session:
        """
        Updates picking threshold values for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            picker_type (Literal['blob', 'template', 'deep']):
            ncc_value (float):
            power_min_value (float):
            power_max_value (float):

        Returns:
            Session: Successful Response

        """
        ...
    def reset_attribute_threshold(self, project_uid: str, session_uid: str, attribute: str, /) -> Session:
        """
        Resets attribute threshold for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            attribute (str):

        Returns:
            Session: Successful Response

        """
        ...
    def reset_all_attribute_thresholds(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Resets all attribute thresholds for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def setup_template_creation_class2D(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        num_classes: int,
        picker_type: Literal["blob", "template", "manual"],
        num_mics: int,
        override_particle_diameter_A: Optional[float] = None,
        uid_lte: Optional[int] = None,
    ) -> Job:
        """
        Setup template creation class2D job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            num_classes (int):
            picker_type (Literal['blob', 'template', 'manual']):
            num_mics (int):
            override_particle_diameter_A (float, optional): Defaults to None
            uid_lte (int, optional): Defaults to None

        Returns:
            Job: Successful Response

        """
        ...
    def set_template_creation_job(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        job_uid: str,
        template_creation_project_uid: Optional[str] = None,
    ) -> Session:
        """
        Set template creation class2D job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            job_uid (str):
            template_creation_project_uid (str, optional): Defaults to None

        Returns:
            Session: Successful Response

        """
        ...
    def clear_template_creation_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clears template creation class2D job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def toggle_picking_template(self, project_uid: str, session_uid: str, template_idx: int, /) -> Session:
        """
        Toggles template for template creation job at a specific index for a session's template creation job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_idx (int):

        Returns:
            Session: Successful Response

        """
        ...
    def toggle_all_picking_templates(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Toggles templates for all templates for a session's template creation job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def select_all_picking_templates(
        self, project_uid: str, session_uid: str, direction: Literal["select", "deselect"], /
    ) -> Session:
        """
        Selects or deselects all templates for a template creation job in a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            direction (Literal['select', 'deselect']):

        Returns:
            Session: Successful Response

        """
        ...
    def select_picking_templates_with_threshold_index(
        self,
        project_uid: str,
        session_uid: str,
        template_idx: int,
        direction: Literal["above", "below"],
        /,
        *,
        dimension: Literal["num_particles_total", "res_A", "class_ess"],
    ) -> Session:
        """
        Selects all templates above or below an index in a template creation job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_idx (int):
            direction (Literal['above', 'below']):
            dimension (Literal['num_particles_total', 'res_A', 'class_ess']):

        Returns:
            Session: Successful Response

        """
        ...
    def select_picking_templates_with_thresholds(
        self, project_uid: str, session_uid: str, /, template_selection_thresholds: List[TemplateSelectionThreshold]
    ) -> Session:
        """
        Selects all templates above or below an index in a template creation job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_selection_thresholds (List[TemplateSelectionThreshold]):

        Returns:
            Session: Successful Response

        """
        ...
    def setup_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Setup Ab-Initio Reconstruction job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
        ...
    def set_phase2_abinit_job(self, project_uid: str, session_uid: str, /, *, job_uid: str) -> Session:
        """
        Sets a Live Ab-Initio Reconstruction job for the session. May specify any job with volume outputs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            job_uid (str):

        Returns:
            Session: Successful Response

        """
        ...
    def enqueue_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueues Ab-Initio Reconstruction job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
        ...
    def clear_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clears Ab-Initio Reconstruction job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def update_phase2_abinit_params(self, project_uid: str, session_uid: str, /, body: LiveAbinitParams) -> Session:
        """
        Updates Ab-Initio Reconstruction parameters for the session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveAbinitParams):

        Returns:
            Session: Successful Response

        """
        ...
    def select_phase2_abinit_volume(self, project_uid: str, session_uid: str, /, *, volume_name: str) -> Session:
        """
        Selects a volume for an Ab-Initio Reconstruction job in a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            volume_name (str):

        Returns:
            Session: Successful Response

        """
        ...
    def stop_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stops an Ab-Initio Reconstruction job for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def clear_phase2_refine(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clears streaming Homogenous Refinement job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def setup_phase2_refine(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
        ...
    def enqueue_phase2_refine(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueues a streaming Homogenous Refinement job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
        ...
    def stop_phase2_refine(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stops a streaming Homogenous Refinement job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def update_phase2_refine_params(self, project_uid: str, session_uid: str, /, body: LiveRefineParams) -> Session:
        """
        Updates parameters for a streaming Homogenous Refinement job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveRefineParams):

        Returns:
            Session: Successful Response

        """
        ...
    def create_and_enqueue_export_particles(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        picker_type: Optional[Literal["blob", "template", "manual"]] = None,
        num_mics: Optional[int] = None,
        uid_lte: Optional[int] = None,
        test_only: bool = False,
    ) -> Job:
        """
        Creates and enqueues a Live Particle Export job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            picker_type (Literal['blob', 'template', 'manual'], optional): Defaults to None
            num_mics (int, optional): Defaults to None
            uid_lte (int, optional): Defaults to None
            test_only (bool, optional): Defaults to False

        Returns:
            Job: Enqueued export particles job

        """
        ...
    def create_and_enqueue_export_exposures(
        self, project_uid: str, session_uid: str, /, *, export_ignored: bool = False
    ) -> Job:
        """
        Creates and enqueues a Live Exposure Export job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            export_ignored (bool, optional): Defaults to False

        Returns:
            Job: Enqueued Live Exposure Export job

        """
        ...
    def mark_session_completed(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Marks the session as completed

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def get_configuration_profiles(self) -> List[SessionConfigProfile]:
        """
        Gets all session configuration profiles

        Returns:
            List[SessionConfigProfile]: Successful Response

        """
        ...
    def create_configuration_profile(self, body: SessionConfigProfileBody) -> SessionConfigProfile:
        """
        Creates a session configuration profile

        Args:
            body (SessionConfigProfileBody):

        Returns:
            SessionConfigProfile: Successful Response

        """
        ...
    def apply_configuration_profile(self, project_uid: str, session_uid: str, /, *, configuration_id: str) -> Session:
        """
        Applies a configuration profile to a session, overwriting its resources, parameters, and exposure group

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            configuration_id (str):

        Returns:
            Session: Successful Response

        """
        ...
    def update_configuration_profile(
        self, configuration_id: str, /, body: SessionConfigProfileBody
    ) -> SessionConfigProfile:
        """
        Updates a configuration profile

        Args:
            configuration_id (str):
            body (SessionConfigProfileBody):

        Returns:
            SessionConfigProfile: Successful Response

        """
        ...
    def delete_configuration_profile(self, configuration_id: str, /) -> None:
        """
        Deletes a configuration profile

        Args:
            configuration_id (str):

        """
        ...
    def compact_session(self, project_uid: str, session_uid: str, /) -> None:
        """
        Compacts a session by clearing each exposure group and its related files for each exposure in the session.
        Also clears gridfs data.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
        ...
    def restore_session(self, project_uid: str, session_uid: str, /, body: LiveComputeResources) -> None:
        """
        Restores exposures of a compacted session. Starts phase 1 worker(s) on the specified lane to restore each exposure by re-processing starting from motion correction, skipping the
        picking stage.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveComputeResources):

        """
        ...
    def get_session_base_params(self) -> Any:
        """
        Gets base session parameters

        Returns:
            Any: Successful Response

        """
        ...
    def star_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stars a session for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def unstar_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stars a session for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
        ...
    def export_session(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        export_movies: bool = False,
        export_ignored_exposures: bool = False,
        picker_type: Optional[Literal["blob", "template", "manual"]] = None,
    ) -> None:
        """
        Writes session results, including all exposures and particles, to the
        project exports directory. The resulting directory contains .csg files.
        Copy the directory to another CryoSPARC project (while resolving symlinks)
        and import individual .csg files with the Import Result Group job.

        Example copy command:

            mkdir -p /path/to/projects/cs-project-b/imports/
            cp -rL /path/to/projects/cs-project-a/exports/S1 /path/to/projects/cs-project-b/imports/

        Note: the original movies are not added to the export directory by default.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            export_movies (bool, optional): Defaults to False
            export_ignored_exposures (bool, optional): Defaults to False
            picker_type (Literal['blob', 'template', 'manual'], optional): Defaults to None

        """
        ...

class ExposuresAPI(APINamespace):
    """
    Functions available in ``api.exposures``, e.g., ``api.exposures.find(...)``
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        uid: Optional[List[str]] = None,
        session_uid: Optional[List[str]] = None,
        project_uid: Optional[List[str]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        stage: Optional[
            List[
                Literal[
                    "go_to_found",
                    "found",
                    "check",
                    "motion",
                    "ctf",
                    "thumbs",
                    "pick",
                    "extract",
                    "extract_manual",
                    "ready",
                    "restoring",
                    "restoring_motion",
                    "restoring_thumbs",
                    "restoring_ctf",
                    "restoring_extract",
                    "restoring_extract_manual",
                    "compacted",
                ]
            ]
        ] = None,
        deleted: Optional[bool] = False,
    ) -> List[Exposure]:
        """
        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            uid (List[str], optional): Defaults to None
            session_uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            stage (List[Literal['go_to_found', 'found', 'check', 'motion', 'ctf', 'thumbs', 'pick', 'extract', 'extract_manual', 'ready', 'restoring', 'restoring_motion', 'restoring_thumbs', 'restoring_ctf', 'restoring_extract', 'restoring_extract_manual', 'compacted']], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            List[Exposure]: Successful Response

        """
        ...
    def count(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        uid: Optional[List[str]] = None,
        session_uid: Optional[List[str]] = None,
        project_uid: Optional[List[str]] = None,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = None,
        stage: Optional[
            List[
                Literal[
                    "go_to_found",
                    "found",
                    "check",
                    "motion",
                    "ctf",
                    "thumbs",
                    "pick",
                    "extract",
                    "extract_manual",
                    "ready",
                    "restoring",
                    "restoring_motion",
                    "restoring_thumbs",
                    "restoring_ctf",
                    "restoring_extract",
                    "restoring_extract_manual",
                    "compacted",
                ]
            ]
        ] = None,
        deleted: Optional[bool] = False,
    ) -> int:
        """
        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            uid (List[str], optional): Defaults to None
            session_uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            stage (List[Literal['go_to_found', 'found', 'check', 'motion', 'ctf', 'thumbs', 'pick', 'extract', 'extract_manual', 'ready', 'restoring', 'restoring_motion', 'restoring_thumbs', 'restoring_ctf', 'restoring_extract', 'restoring_extract_manual', 'compacted']], optional): Defaults to None
            deleted (bool, optional): Defaults to False

        Returns:
            int: Successful Response

        """
        ...
    def find_one(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Successful Response

        """
        ...
    def reset_manual_reject_exposures(self, project_uid: str, session_uid: str, /) -> List[Exposure]:
        """
        Resets manual rejection status on all exposures in a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            List[Exposure]: List of exposures reset

        """
        ...
    def reset_all_exposures(self, project_uid: str, session_uid: str, /) -> None:
        """
        Resets all exposures in a session to initial state.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
        ...
    def reset_failed_exposures(self, project_uid: str, session_uid: str, /) -> None:
        """
        Resets all failed exposures in a session to initial state.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
        ...
    def reset_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Resets exposure to intial state.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Exposure that has been reset

        """
        ...
    def manual_reject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Manually rejects exposure.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Rejected exposure

        """
        ...
    def manual_unreject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Manually unrejects exposure.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Unrejected exposure

        """
        ...
    def mark_failed(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Mark an exposure as failed.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Successful Response

        """
        ...
    def toggle_manual_reject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Toggles manual rejection state on exposure.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Toggled exposure

        """
        ...
    def reprocess_single_exposure(
        self,
        project_uid: str,
        session_uid: str,
        exposure_uid: int,
        /,
        body: LivePreprocessingParams,
        *,
        picker_type: Literal["blob", "template"],
    ) -> Exposure:
        """
        Reprocesses a single micrograph with the passed parameters. If there is a test micrograph
        in the session already that is not the same micrograph that the user is currently trying to test, it will be reset
        back to the "ctf" stage without the test flag.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):
            body (LivePreprocessingParams):
            picker_type (Literal['blob', 'template']): Automatic picker types available in Live.

        Returns:
            Exposure: Reprocessed exposure

        """
        ...
    def add_manual_pick(
        self, project_uid: str, session_uid: str, exposure_uid: int, /, *, x_frac: float, y_frac: float
    ) -> Exposure:
        """
        Adds a manual pick to an exposure.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):
            x_frac (float):
            y_frac (float):

        Returns:
            Exposure: Exposure with added manual picks

        """
        ...
    def remove_manual_pick(
        self,
        project_uid: str,
        session_uid: str,
        exposure_uid: int,
        /,
        *,
        x_frac: float,
        y_frac: float,
        dist_frac: float = 0.02,
    ) -> Exposure:
        """
        Removes manual pick from an exposure

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):
            x_frac (float):
            y_frac (float):
            dist_frac (float, optional): Defaults to 0.02

        Returns:
            Exposure: Exposure with removed manual picks

        """
        ...
    def get_individual_picks(
        self,
        project_uid: str,
        session_uid: str,
        exposure_uid: int,
        picker_type: Literal["blob", "template", "manual"],
        /,
    ) -> List[List[float]]:
        """
        Gets list of picks from an exposure

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):
            picker_type (Literal['blob', 'template', 'manual']):

        Returns:
            List[List[float]]: Successful Response

        """
        ...

class ProjectsAPI(APINamespace):
    """
    Functions available in ``api.projects``, e.g., ``api.projects.check_directory(...)``
    """
    def check_directory(self, *, path: str) -> str:
        """
        Checks if a candidate project directory exists, and if it is readable and writeable.

        Args:
            path (str):

        Returns:
            str: Successful Response

        """
        ...
    def get_title_slug(self, *, title: str) -> str:
        """
        Returns a slugified version of a project title.

        Args:
            title (str):

        Returns:
            str: Successful Response

        """
        ...
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        uid: Optional[List[str]] = None,
        project_dir: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        users_with_access: Optional[List[str]] = None,
        deleted: Optional[bool] = False,
        archived: Optional[bool] = None,
        detached: Optional[bool] = None,
    ) -> List[Project]:
        """
        Finds projects matching the filter.

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            uid (List[str], optional): Defaults to None
            project_dir (str, optional): Defaults to None
            owner_user_id (str, optional): Defaults to None
            users_with_access (List[str], optional): Defaults to None
            deleted (bool, optional): Defaults to False
            archived (bool, optional): Defaults to None
            detached (bool, optional): Defaults to None

        Returns:
            List[Project]: Successful Response

        """
        ...
    def create(self, *, title: str, description: Optional[str] = None, parent_dir: str) -> Project:
        """
        Creates a new project, project directory and creates a new document in
        the project collection

        Args:
            title (str):
            description (str, optional): Defaults to None
            parent_dir (str):

        Returns:
            Project: Successful Response

        """
        ...
    def count(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        uid: Optional[List[str]] = None,
        project_dir: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        users_with_access: Optional[List[str]] = None,
        deleted: Optional[bool] = False,
        archived: Optional[bool] = None,
        detached: Optional[bool] = None,
    ) -> int:
        """
        Counts the number of projects matching the filter.

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            uid (List[str], optional): Defaults to None
            project_dir (str, optional): Defaults to None
            owner_user_id (str, optional): Defaults to None
            users_with_access (List[str], optional): Defaults to None
            deleted (bool, optional): Defaults to False
            archived (bool, optional): Defaults to None
            detached (bool, optional): Defaults to None

        Returns:
            int: Successful Response

        """
        ...
    def set_title(self, project_uid: str, /, *, title: str) -> Project:
        """
        Sets the title of a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            title (str):

        Returns:
            Project: Successful Response

        """
        ...
    def set_description(self, project_uid: str, /, description: str) -> Project:
        """
        Sets the description of a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            description (str):

        Returns:
            Project: Successful Response

        """
        ...
    def view(self, project_uid: str, /) -> Project:
        """
        Adds a project uid to a user's recently viewed projects list.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
        ...
    def mkdir(self, project_uid: str, /, *, parents: bool = False, exist_ok: bool = False, path: str = "") -> str:
        """
        Create a directory in the project directory at the given path.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            parents (bool, optional): Defaults to False
            exist_ok (bool, optional): Defaults to False
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            str: Successful Response

        """
        ...
    def cp(self, project_uid: str, /, *, source: str, path: str = "") -> str:
        """
        Copy the source file or directory to the project directory at the given
        path. Returns the absolute path of the copied file.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            source (str):
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            str: Successful Response

        """
        ...
    def symlink(self, project_uid: str, /, *, source: str, path: str = "") -> str:
        """
        Create a symlink from the source path in the project directory at the given path.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            source (str):
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            str: Successful Response

        """
        ...
    def upload_file(self, project_uid: str, /, stream: Stream, *, overwrite: bool = False, path: str = "") -> str:
        """
        Upload a file to the project directory at the given path. Returns absolute
        path of the uploaded file.

        Path may be specified as a filename, a relative path inside the project
        directory, or a full absolute path.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            stream (Stream): A binary stream representing a Stream class instance
            overwrite (bool, optional): Defaults to False
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            str: Successful Response

        """
        ...
    def download_file(self, project_uid: str, /, *, path: str = "") -> Stream:
        """
        Download a file from the project directory at the given path.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            Stream: A binary stream representing a Stream class instance

        """
        ...
    def ls(self, project_uid: str, /, *, recursive: bool = False, path: str = "") -> List[str]:
        """
        List files in the project directory. Note that enabling recursive will
        include parent directories in the result.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            recursive (bool, optional): Defaults to False
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            List[str]: Successful Response

        """
        ...
    def get_job_register(self, project_uid: str, /) -> JobRegister:
        """
        Gets the job register model for the project. The same for every project.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            JobRegister: Successful Response

        """
        ...
    def preview_delete(self, project_uid: str, /) -> DeleteProjectPreview:
        """
        Retrieves the workspaces and jobs that would be affected when the project is deleted.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            DeleteProjectPreview: Successful Response

        """
        ...
    def find_one(self, project_uid: str, /) -> Project:
        """
        Finds a project by its UID

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
        ...
    def delete(self, project_uid: str, /) -> None:
        """
        Starts project deletion task. Will delete the project, its full directory, and all associated workspaces, sessions, jobs and results.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
        ...
    def get_directory(self, project_uid: str, /) -> str:
        """
        Gets the project's absolute directory with all environment variables in the
        path resolved

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            str: Successful Response

        """
        ...
    def get_owner_id(self, project_uid: str, /) -> str:
        """
        Get user account ID for the owner of a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            str: Successful Response

        """
        ...
    def set_owner(self, project_uid: str, user_id: str, /) -> Project:
        """
        Sets owner of the project to the user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            user_id (str): User ID or email

        Returns:
            Project: Successful Response

        """
        ...
    def add_user_access(self, project_uid: str, user_id: str, /) -> Project:
        """
        Grants access to another user to view and edit the project.
        May only be called by project owners and administrators.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            user_id (str): User ID or email

        Returns:
            Project: Successful Response

        """
        ...
    def remove_user_access(self, project_uid: str, user_id: str, /) -> Project:
        """
        Removes a user's access from a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            user_id (str): User ID or email

        Returns:
            Project: Successful Response

        """
        ...
    def refresh_size(self, project_uid: str, /) -> None:
        """
        Starts project size recalculation asynchronously.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
        ...
    def get_symlinks(self, project_uid: str, /) -> List[ProjectSymlink]:
        """
        Gets all symbolic links in the project directory

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            List[ProjectSymlink]: Successful Response

        """
        ...
    def set_default_param(self, project_uid: str, name: str, /, value: Union[bool, int, float, str]) -> Project:
        """
        Sets a default value for a parameter name globally for the project

        Args:
            project_uid (str): Project UID, e.g., "P3"
            name (str):
            value (bool | int | float | str):

        Returns:
            Project: Successful Response

        """
        ...
    def clear_default_param(self, project_uid: str, name: str, /) -> Project:
        """
        Clears the per-project default value for a parameter name.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            name (str):

        Returns:
            Project: Successful Response

        """
        ...
    def claim_instance_ownership(self, project_uid: str, /, *, force: bool = False) -> None:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            force (bool, optional): Defaults to False

        """
        ...
    def claim_all_instance_ownership(self, *, force: bool = False) -> None:
        """
        Claims ownership of all projects in instance. Call when upgrading from an older CryoSPARC version that did not support project locks.

        Args:
            force (bool, optional): Defaults to False

        """
        ...
    def archive(self, project_uid: str, /) -> None:
        """
        Archives a project. This means that the project can no longer be modified
        and jobs cannot be created or run. Once archived, the project directory may
        be safely moved to long-term storage.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
        ...
    def unarchive(self, project_uid: str, /, *, path: str) -> Project:
        """
        Reverses archive operation.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            path (str):

        Returns:
            Project: Successful Response

        """
        ...
    def detach(self, project_uid: str, /) -> None:
        """
        Detaches a project, removing its lockfile. This hides the project from the interface and allows other
        instances to attach and run this project.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
        ...
    def attach(self, *, path: str) -> Project:
        """
        Attaches a project directory at a specified path and writes a new
        lockfile. Must be run on a project directory without a lockfile.

        Args:
            path (str):

        Returns:
            Project: Successful Response

        """
        ...
    def move(self, project_uid: str, /, *, path: str) -> Project:
        """
        Renames the project directory for a project. Provide either the new
        directory name or the full new directory path.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            path (str):

        Returns:
            Project: Successful Response

        """
        ...
    def get_next_exposure_group_id(self, project_uid: str, /) -> int:
        """
        Gets next exposure group ID

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            int: Successful Response

        """
        ...
    def cleanup_data(
        self,
        project_uid: str,
        /,
        *,
        workspace_uid: Optional[str] = None,
        delete_non_final: bool = False,
        delete_statuses: List[JobStatus] = [],
        clear_non_final: bool = False,
        clear_categories: List[Category] = [],
        clear_types: List[str] = [],
        clear_statuses: List[JobStatus] = [],
        clear_preprocessing: bool = False,
    ) -> None:
        """
        Cleanup project or workspace data, clearing/deleting jobs based on final result status, sections, types, or job status

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str, optional): Defaults to None
            delete_non_final (bool, optional): Defaults to False
            delete_statuses (List[JobStatus], optional): Defaults to []
            clear_non_final (bool, optional): Defaults to False
            clear_categories (List[Category], optional): Defaults to []
            clear_types (List[str], optional): Defaults to []
            clear_statuses (List[JobStatus], optional): Defaults to []
            clear_preprocessing (bool, optional): Defaults to False

        """
        ...
    def add_tag(self, project_uid: str, tag_uid: str, /) -> Project:
        """
        Tags a project with the given tag.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            tag_uid (str):

        Returns:
            Project: Successful Response

        """
        ...
    def remove_tag(self, project_uid: str, tag_uid: str, /) -> Project:
        """
        Removes the given tag from a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            tag_uid (str):

        Returns:
            Project: Successful Response

        """
        ...
    def get_generate_intermediate_results_settings(self, project_uid: str, /) -> GenerateIntermediateResultsSettings:
        """
        Gets generate intermediate result settings.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            GenerateIntermediateResultsSettings: Successful Response

        """
        ...
    def set_generate_intermediate_results_settings(
        self, project_uid: str, /, body: GenerateIntermediateResultsSettings
    ) -> Project:
        """
        Sets settings for intermediate result generation.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            body (GenerateIntermediateResultsSettings):

        Returns:
            Project: Successful Response

        """
        ...
    def clear_intermediate_results(self, project_uid: str, /, *, always_keep_final: bool = True) -> None:
        """
        Removes intermediate results from the project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            always_keep_final (bool, optional): Defaults to True

        """
        ...
    def get_generate_intermediate_results_job_types(self) -> List[str]:
        """
        Gets intermediate result job types

        Returns:
            List[str]: Successful Response

        """
        ...
    def star_project(self, project_uid: str, /) -> Project:
        """
        Stars a project for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
        ...
    def unstar_project(self, project_uid: str, /) -> Project:
        """
        Unstars a project for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
        ...
    def reset_autodump(self, project_uid: str, /) -> Project:
        """
        Clear project directory write failures. After calling this endpoint,
        CryoSPARC's scheduler will attempt to write modified jobs and workspaces to
        the project directory that previously could not be saved.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
        ...

class TagsAPI(APINamespace):
    """
    Functions available in ``api.tags``, e.g., ``api.tags.find(...)``
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = None,
        limit: int = 100,
        created_by_user_id: Optional[str] = None,
        type: Optional[List[Literal["general", "project", "workspace", "session", "job"]]] = None,
        uid: Optional[str] = None,
    ) -> List[Tag]:
        """
        Finds tags that match the given query.

        Args:
            order (int, optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            created_by_user_id (str, optional): Defaults to None
            type (List[Literal['general', 'project', 'workspace', 'session', 'job']], optional): Defaults to None
            uid (str, optional): Defaults to None

        Returns:
            List[Tag]: Successful Response

        """
        ...
    def create(
        self,
        *,
        type: Literal["general", "project", "workspace", "session", "job"],
        colour: Optional[
            Literal[
                "black",
                "gray",
                "red",
                "orange",
                "yellow",
                "green",
                "teal",
                "cyan",
                "sky",
                "blue",
                "indigo",
                "purple",
                "pink",
            ]
        ] = None,
        description: Optional[str] = None,
        created_by_workflow: Optional[str] = None,
        title: Optional[str],
    ) -> Tag:
        """
        Creates a new tag

        Args:
            type (Literal['general', 'project', 'workspace', 'session', 'job']):
            colour (Literal['black', 'gray', 'red', 'orange', 'yellow', 'green', 'teal', 'cyan', 'sky', 'blue', 'indigo', 'purple', 'pink'], optional): Defaults to None
            description (str, optional): Defaults to None
            created_by_workflow (str, optional): Defaults to None
            title (str | None):

        Returns:
            Tag: Successful Response

        """
        ...
    def update(
        self,
        tag_uid: str,
        /,
        *,
        colour: Optional[
            Literal[
                "black",
                "gray",
                "red",
                "orange",
                "yellow",
                "green",
                "teal",
                "cyan",
                "sky",
                "blue",
                "indigo",
                "purple",
                "pink",
            ]
        ] = None,
        description: Optional[str] = None,
        title: Optional[str],
    ) -> Tag:
        """
        Updates the title, colour and/or description of the given tag UID

        Args:
            tag_uid (str):
            colour (Literal['black', 'gray', 'red', 'orange', 'yellow', 'green', 'teal', 'cyan', 'sky', 'blue', 'indigo', 'purple', 'pink'], optional): Defaults to None
            description (str, optional): Defaults to None
            title (str | None):

        Returns:
            Tag: Successful Response

        """
        ...
    def delete(self, tag_uid: str, /) -> None:
        """
        Deletes a given tag

        Args:
            tag_uid (str):

        """
        ...
    def get_tags_by_type(self) -> Dict[str, List[Tag]]:
        """
        Gets all tags as a dictionary, where the types are the keys

        Returns:
            Dict[str, List[Tag]]: Successful Response

        """
        ...
    def get_tag_count_by_type(self) -> Dict[str, int]:
        """
        Gets a count of all tags by type

        Returns:
            Dict[str, int]: Successful Response

        """
        ...

class NotificationsAPI(APINamespace):
    """
    Functions available in ``api.notifications``, e.g., ``api.notifications.deactivate_notification(...)``
    """
    def deactivate_notification(self, notification_id: str, /) -> Notification:
        """
        Deactivates a notification

        Args:
            notification_id (str):

        Returns:
            Notification: Successful Response

        """
        ...

class BlueprintsAPI(APINamespace):
    """
    Functions available in ``api.blueprints``, e.g., ``api.blueprints.create_blueprint(...)``
    """
    def create_blueprint(
        self,
        schema: Dict[str, Any],
        *,
        blueprint_id: str,
        imported: bool,
        project_uid: str,
        job_uid: str,
        job_type: str,
    ) -> None:
        """
        For cryosparc app only

        Args:
            schema (Dict[str, Any]):
            blueprint_id (str):
            imported (bool):
            project_uid (str):
            job_uid (str):
            job_type (str):

        """
        ...
    def edit_blueprint(
        self, blueprint_id: str, /, schema: Dict[str, Any], *, project_uid: str, job_uid: str, job_type: str
    ) -> None:
        """
        For cryosparc app only

        Args:
            blueprint_id (str):
            schema (Dict[str, Any]):
            project_uid (str):
            job_uid (str):
            job_type (str):

        """
        ...
    def delete_blueprint(self, blueprint_id: str, /, *, job_type: str) -> None:
        """
        For cryosparc app only

        Args:
            blueprint_id (str):
            job_type (str):

        """
        ...
    def apply_blueprint(
        self, blueprint_id: str, /, schema: Dict[str, Any], *, project_uid: str, job_uid: str, job_type: str
    ) -> None:
        """
        For cryosparc app only

        Args:
            blueprint_id (str):
            schema (Dict[str, Any]):
            project_uid (str):
            job_uid (str):
            job_type (str):

        """
        ...

class WorkflowsAPI(APINamespace):
    """
    Functions available in ``api.workflows``, e.g., ``api.workflows.create_workflow(...)``
    """
    def create_workflow(
        self,
        schema: Dict[str, Any],
        *,
        workflow_id: str,
        forked: bool = False,
        imported: bool = False,
        rebuilt: bool = False,
    ) -> None:
        """
        For cryosparc app only

        Args:
            schema (Dict[str, Any]):
            workflow_id (str):
            forked (bool, optional): Defaults to False
            imported (bool, optional): Defaults to False
            rebuilt (bool, optional): Defaults to False

        """
        ...
    def edit_workflow(self, workflow_id: str, /, schema: Dict[str, Any]) -> None:
        """
        For cryosparc app only

        Args:
            workflow_id (str):
            schema (Dict[str, Any]):

        """
        ...
    def delete_workflow(self, workflow_id: str, /) -> None:
        """
        For cryosparc app only

        Args:
            workflow_id (str):

        """
        ...
    def apply_workflow(self, workflow_id: str, /, schema: Dict[str, Any]) -> None:
        """
        For cryosparc app only

        Args:
            workflow_id (str):
            schema (Dict[str, Any]):

        """
        ...

class ExternalAPI(APINamespace):
    """
    Functions available in ``api.external``, e.g., ``api.external.get_empiar_latest_entries(...)``
    """
    def get_empiar_latest_entries(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: Successful Response

        """
        ...
    def get_emdb_latest_entries(self) -> List[Dict[str, Any]]:
        """
        Returns:
            List[Dict[str, Any]]: Successful Response

        """
        ...
    def get_discuss_top(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: Successful Response

        """
        ...
    def get_discuss_categories(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: Successful Response

        """
        ...
    def get_tutorials(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: Successful Response

        """
        ...
    def get_changelog(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: Successful Response

        """
        ...

class BenchmarksAPI(APINamespace):
    """
    Functions available in ``api.benchmarks``, e.g., ``api.benchmarks.get_reference_benchmarks(...)``
    """
    def get_reference_benchmarks(self) -> List[ReferencePerformanceBenchmark]:
        """
        Returns:
            List[ReferencePerformanceBenchmark]: Successful Response

        """
        ...
    def get_benchmark(self, project_uid: str, job_uid: str, benchmark_type: str, /) -> PerformanceBenchmark:
        """
        Args:
            project_uid (str):
            job_uid (str):
            benchmark_type (str):

        Returns:
            PerformanceBenchmark: Successful Response

        """
        ...

class DeveloperAPI(APINamespace):
    """
    Functions available in ``api.developer``, e.g., ``api.developer.get_developers(...)``
    """
    def get_developers(self) -> List[str]:
        """
        :meta private:

        Returns:
            List[str]: Successful Response

        """
        ...
    def reload(self) -> bool:
        """
        Restarts API service and scheduler.

        :meta private:

        Returns:
            bool: Successful Response

        """
        ...
    def save_job_registers(self, *, developer_name: Optional[str] = None) -> List[JobRegister]:
        """
        Re-saves the current job registers. Call this when restarting the api
        service without executing the /startup route, as we do during developer
        reloads.

        :meta private:

        Args:
            developer_name (str, optional): Defaults to None

        Returns:
            List[JobRegister]: Successful Response

        """
        ...

class APIClient:
    """
    Functions and namespaces available in top-level API object. e.g., ``api.read_root(...)``
    or ``api.config.get_file_browser_settings(...)``
    """

    config: ConfigAPI
    """``api.config`` functions"""
    instance: InstanceAPI
    """``api.instance`` functions"""
    cache: CacheAPI
    """``api.cache`` functions"""
    users: UsersAPI
    """``api.users`` functions"""
    resources: ResourcesAPI
    """``api.resources`` functions"""
    assets: AssetsAPI
    """``api.assets`` functions"""
    jobs: JobsAPI
    """``api.jobs`` functions"""
    workspaces: WorkspacesAPI
    """``api.workspaces`` functions"""
    sessions: SessionsAPI
    """``api.sessions`` functions"""
    exposures: ExposuresAPI
    """``api.exposures`` functions"""
    projects: ProjectsAPI
    """``api.projects`` functions"""
    tags: TagsAPI
    """``api.tags`` functions"""
    notifications: NotificationsAPI
    """``api.notifications`` functions"""
    blueprints: BlueprintsAPI
    """``api.blueprints`` functions"""
    workflows: WorkflowsAPI
    """``api.workflows`` functions"""
    external: ExternalAPI
    """``api.external`` functions"""
    benchmarks: BenchmarksAPI
    """``api.benchmarks`` functions"""
    developer: DeveloperAPI
    """``api.developer`` functions"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        auth: Optional[Auth] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 300,
    ) -> None:
        """
        Args:
            base_url (str, optional): Base URL for CryoSPARC API server. Defaults to None
            auth (Auth, optional): Auth token or email/password. Defaults to None
            headers (Dict[str, str], optional): Default HTTP headers to include in requests. Defaults to None
            timeout (float, optional): Request timeout in seconds. Defaults to 300

        """
        ...
    def __call__(self, *, auth: Optional[Auth] = None) -> None:
        """
        Args:
            auth (Auth, optional): Auth token or email/password. Password must be encoded as SHA256. Defaults to None

        """
        ...
    def read_root(self) -> Hello:
        """
        Returns:
            Hello: Successful Response

        """
        ...
    def health(self) -> str:
        """
        Returns:
            str: Successful Response

        """
        ...
    def login(
        self,
        *,
        expires_in: float = 1209600,
        username: str,
        password: str,
        grant_type: Optional[str] = None,
        scope: str = "",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> Token:
        """
        Login form. Note that plain-text passwords are not accepted; they must be
        hashed as SHA256.

        Args:
            expires_in (float, optional): Token expire time (in seconds). Can be up to 1 year.. Defaults to 1209600
            username (str):
            password (str):
            grant_type (str, optional): Defaults to None
            scope (str, optional): Defaults to ''
            client_id (str, optional): Defaults to None
            client_secret (str, optional): Defaults to None

        Returns:
            Token: Successful Response

        """
        ...
    def keycloak_login(self, *, keycloak_access_token: str) -> Token:
        """
        Args:
            keycloak_access_token (str):

        Returns:
            Token: Successful Response

        """
        ...
    def verify_app_session(self, body: AppSession) -> str:
        """
        Args:
            body (AppSession):

        Returns:
            str: Successful Response

        """
        ...
    def job_register(self) -> JobRegister:
        """
        Get a specification of available job types and their schemas.

        Returns:
            JobRegister: Successful Response

        """
        ...
    def start_and_migrate(self, *, license_id: str) -> None:
        """
        Start up CryoSPARC for the first time and perform database migrations

        Args:
            license_id (str):

        """
        ...
    def test(self, delay: float, /) -> str:
        """
        Sleep for the specified number of seconds and returns a value to indicate
        endpoint is working correctly.

        Args:
            delay (float):

        Returns:
            str: Successful Response

        """
        ...
