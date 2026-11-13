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
from typing import Any, Literal, TypeAlias

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
from .models.blueprint import Blueprint, BlueprintParameter
from .models.config import SystemInfo
from .models.datasource import Datasource
from .models.diagnostics import RuntimeDiagnostics
from .models.event import CheckpointEvent, Event, ImageEvent, InteractiveEvent, TextEvent
from .models.exposure import Exposure
from .models.external import ExternalOutputSpec
from .models.file_browser import BrowseFileResponse, FileBrowserSettings
from .models.instance import InstanceActivity
from .models.job import Job, JobStatus
from .models.job_export import (
    ExportOutputPreview,
    ExportOutputsSelection,
    ExportResultDefaults,
    ExportResultsSelection,
    TransferMethod,
)
from .models.job_register import JobRegister
from .models.job_spec import Category, InputSpec, OutputResult, OutputSpec
from .models.license import LicenseInstance, UpdateLicenseTag
from .models.notification import Notification
from .models.preview import DeleteProjectPreview, DeleteWorkspacePreview, ExportBundlePreview, SplitProjectPreview
from .models.project import GenerateIntermediateResultsSettings, Project
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
from .models.symlink import SymlinkInfo
from .models.tag import CreateTag, Tag, UpdateTag
from .models.user import User
from .models.workflow import ApplyWorkflowRequest, UpdateWorkflow, Workflow, WorkflowJob, WorkflowParameter
from .models.workspace import JobGroup, JobGroupCreate, JobGroupUpdate, Workspace
from .stream import Stream

Auth: TypeAlias = str | tuple[str, str]
"""
Type representing an auth token or email/password tuple.
"""

class APINamespace:
    def __init__(self, http_client: httpx.Client) -> None:
        """
        Args:
            http_client (httpx.Client): HTTP client object for making requests

        """

class ConfigAPI(APINamespace):
    """
    Functions available in ``api.config``, e.g., ``api.config.get_file_browser_settings(...)``
    """
    def get_file_browser_settings(self) -> FileBrowserSettings:
        """
        Get Instance file browser settings

        Returns:
            FileBrowserSettings: Successful Response

        """
    def set_file_browser_settings(self, body: FileBrowserSettings) -> FileBrowserSettings:
        """
        Update instance file browser settings

        Args:
            body (FileBrowserSettings):

        Returns:
            FileBrowserSettings: Successful Response

        """
    def get_export_result_defaults(self) -> ExportResultDefaults:
        """
        Get the instance-wide default policy for which export results get omitted

        Returns:
            ExportResultDefaults: Successful Response

        """
    def set_export_result_defaults(self, body: ExportResultDefaults) -> ExportResultDefaults:
        """
        Update the instance-wide default policy for which export results get omitted

        Args:
            body (ExportResultDefaults):

        Returns:
            ExportResultDefaults: Successful Response

        """
    def set_instance_banner(self, *, active: bool = False, title: str | None = None, body: str | None = None) -> Any:
        """
        Update the banner message shown on the home page

        Args:
            active (bool, optional): Defaults to False
            title (str, optional): Defaults to None
            body (str, optional): Defaults to None

        Returns:
            Any: Successful Response

        """
    def set_login_message(self, *, active: bool = False, title: str | None = None, body: str | None = None) -> Any:
        """
        Update the alert message shown in the app following user login

        Args:
            active (bool, optional): Defaults to False
            title (str, optional): Defaults to None
            body (str, optional): Defaults to None

        Returns:
            Any: Successful Response

        """
    def get_instance_uid(self) -> str:
        """
        Get this CryoSPARC instance's unique UID.

        Returns:
            str: Successful Response

        """
    def set_default_job_priority(self, value: int) -> Any:
        """
        Job priority

        Args:
            value (int):

        Returns:
            Any: Successful Response

        """
    def get_version(self) -> str:
        """
        Get the current CryoSPARC version (with patch suffix, if available)

        Returns:
            str: Successful Response

        """
    def get_system_info(self) -> SystemInfo:
        """
        System information related to the CryoSPARC application

        Returns:
            SystemInfo: Successful Response

        """
    def get(self, name: str, /, *, default: Any = "<<UNDEFINED>>") -> Any:
        """
        Get config collection entry value for the given variable name.
        Set ``default`` to return a default value instead of raising a 404 error if
        the variable is not found.

        Args:
            name (str):
            default (Any, optional): Defaults to '<<UNDEFINED>>'

        Returns:
            Any: Successful Response

        """
    def write(self, name: str, /, value: Any = None, *, set_on_insert_only: bool = False) -> Any:
        """
        Set config collection entry. Specify `set_on_insert_only` to prevent
        overwriting when the value already exists.

        Args:
            name (str):
            value (Any, optional): Defaults to None
            set_on_insert_only (bool, optional): Defaults to False

        Returns:
            Any: Value in the database

        """

class InstanceAPI(APINamespace):
    """
    Functions available in ``api.instance``, e.g., ``api.instance.get_update_tag(...)``
    """
    def get_update_tag(self) -> UpdateLicenseTag | None:
        """
        Get information about the latest CryoSPARC version update, if one is available.

        Returns:
            UpdateLicenseTag | None: Successful Response

        """
    def commercial_enabled(self) -> bool:
        """
        Checks if CryoSPARC a commercial license is enabled

        Returns:
            bool: Successful Response

        """
    def live_enabled(self) -> bool:
        """
        Check if CryoSPARC Live is enabled

        Returns:
            bool: Successful Response

        """
    def ecl_enabled(self) -> bool:
        """
        Check if embedded CryoSPARC Live is enabled

        Returns:
            bool: Successful Response

        """
    def get_license_usage(self) -> list[LicenseInstance]:
        """
        Get license usage information

        Returns:
            List[LicenseInstance]: Successful Response

        """
    def browse_files(self, *, abs_path_glob: str) -> BrowseFileResponse:
        """
        Request details for a set of files or directories available to CryoSPARC,
        given an absolute path or glob expression. If the given path is a folder,
        returns a list of files and subfolders in that folder.

        .. note::
            ``abs_path_glob`` may have shell variables in it (e.g., ``$HOME``,
            ``$SCRATCH``). Variables are expanded before globs.

        Args:
            abs_path_glob (str):

        Returns:
            BrowseFileResponse: List of available file details.

        """
    def get_service_log(
        self,
        service: LoggingService,
        /,
        *,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        max_lines: int | None = None,
    ) -> Any:
        """
        Get master service logs. Serves the whole log by default.

        Provide a maximum number of lines to return from the end of the log file.
        If no maximum is provided, return the entire log file.

        Provide a start and/or end time to filter logs to a specific time range.
        If the start and end times do not include a timezone, they will be assumed
        to be in server time.

        If only ``start`` is provided, reads from the first line after ``start``.
        If only ``end`` is provided, reads in reverse order from ``end``.

        Args:
            service (LoggingService):
            start (datetime.datetime, optional): Defaults to None
            end (datetime.datetime, optional): Defaults to None
            max_lines (int, optional): Defaults to None

        Returns:
            Any: A binary stream representing a TextStream class instance

        """
    def get_runtime_diagnostics(self) -> RuntimeDiagnostics:
        """
        Get runtime diagnostics for the CryoSPARC instance

        Returns:
            RuntimeDiagnostics: Successful Response

        """
    def audit_dump(self, *, timestamp: float | Literal["auto"] | None = None) -> str | None:
        """
        Generate an audit dump file containing all audit logs since the given timestamp.

        Args:
            timestamp (float | Literal['auto'], optional): Leave unspecified to dump all audit logs, set to "auto" to dump new logs since the last dump, or set to a UNIX timestamp to dump every log that occurred after it. Defaults to None

        Returns:
            str | None: Successful Response

        """
    def get_instance_activity(self, *, limit: int | None = 100) -> list[InstanceActivity]:
        """
        Args:
            limit (int, optional): Defaults to 100

        Returns:
            List[InstanceActivity]: Successful Response

        """
    def generate_new_uid(self, *, force_takeover_projects: bool = False) -> str:
        """
        Generate a new uid for the CryoSPARC instance.
        If ``force_takeover_projects`` is enabled, overwrites existing lockfiles.
        Otherwise, creates lockfiles in projects that don't already have one.

        Args:
            force_takeover_projects (bool, optional): Defaults to False

        Returns:
            str: New instance UID

        """
    def get_error_report(
        self,
        *,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        max_lines: int | None = None,
        offline: bool = False,
        skip_workers: bool = True,
    ) -> Stream:
        """
        Create an error report bundle for the app.

        :meta private:

        Args:
            start (datetime.datetime, optional): Defaults to None
            end (datetime.datetime, optional): Defaults to None
            max_lines (int, optional): Defaults to None
            offline (bool, optional): Defaults to False
            skip_workers (bool, optional): Defaults to True

        Returns:
            Stream: A binary stream representing a Stream class instance

        """

class CacheAPI(APINamespace):
    """
    Functions available in ``api.cache``, e.g., ``api.cache.get(...)``
    """
    def get(self, key: str, /, *, namespace: str | None = None) -> Any:
        """
        Get cached data. Returns None if the value is not set or expired

        Args:
            key (str):
            namespace (str, optional): Defaults to None

        Returns:
            Any: Successful Response

        """
    def set(self, key: str, /, value: Any = None, *, namespace: str | None = None, ttl: int = 60) -> None:
        """
        Set cache key to the given value, with a ttl (Time-to-Live) in seconds

        Args:
            key (str):
            value (Any, optional): Defaults to None
            namespace (str, optional): Defaults to None
            ttl (int, optional): Defaults to 60

        """

class UsersAPI(APINamespace):
    """
    Functions available in ``api.users``, e.g., ``api.users.admin_exists(...)``
    """
    def admin_exists(self) -> bool:
        """
        Return True if there exists at least one user with admin privileges,
        False otherwise

        Returns:
            bool: Successful Response

        """
    def find(self, *, role: Literal["admin", "user"] | None = None) -> list[User]:
        """
        List all users in the system, optionally filtering by role.
        Only admins may access this function.

        Args:
            role (Literal['admin', 'user'], optional): Defaults to None

        Returns:
            List[User]: Successful Response

        """
    def create(
        self,
        password: SHA256Password | None = None,
        *,
        email: str,
        username: str,
        first_name: str,
        last_name: str,
        role: Literal["user", "admin"] = "user",
    ) -> User:
        """
        Create a new CryoSPARC user account. Only authenticated admins may do this.
        If providing a password, first hash the password with SHA256.

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
    def count(self, *, role: Literal["admin", "user"] | None = None) -> int:
        """
        Counts the number of users in the system, optionally filtering by role

        Args:
            role (Literal['admin', 'user'], optional): Defaults to None

        Returns:
            int: Successful Response

        """
    def table(self) -> str:
        """
        Show a table of all CryoSPARC user accounts

        Returns:
            str: Successful Response

        """
    def find_with_file_browser_access(self, *, path: str) -> list[User]:
        """
        List all users that have file browser access to the given path

        Args:
            path (str):

        Returns:
            List[User]: Successful Response

        """
    def me(self) -> User:
        """
        Get the current authenticated user

        Returns:
            User: Successful Response

        """
    def find_one(self, user_id: str, /) -> User:
        """
        Find a user with a matching user ID or email

        Args:
            user_id (str): User ID or Email Address

        Returns:
            User: Successful Response

        """
    def update(
        self,
        user_id: str,
        /,
        *,
        email: str | None = None,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> User:
        """
        Update a user's general details. Parameters left empty will be left
        unchanged. Users can always change their own details, but only admins can
        change other users' details.

        Args:
            user_id (str): User ID or Email Address
            email (str, optional): Defaults to None
            username (str, optional): Defaults to None
            first_name (str, optional): Defaults to None
            last_name (str, optional): Defaults to None

        Returns:
            User: Successful Response

        """
    def delete(self, user_id: str, /) -> None:
        """
        Remove a user from the CryoSPARC. Note that projects created by the user are
        not deleted. Only authenticated admins may do this.

        Args:
            user_id (str): User ID or Email Address

        """
    def get_role(self, user_id: str, /) -> Literal["user", "admin"]:
        """
        Return "admin" if the user has admin privileges, "user" otherwise.

        Args:
            user_id (str): User ID or Email Address

        Returns:
            Literal['user', 'admin']: Successful Response

        """
    def request_reset_password(self, user_id: str, /) -> None:
        """
        Generate a password reset token for a user with the given email. The token
        will appear in the Admin > User Management interface.

        Args:
            user_id (str): User ID or Email Address

        """
    def register(self, user_id: str, /, body: SHA256Password, *, token: str) -> None:
        """
        Register user with a token (unauthenticated). Password must be hashed with SHA256.

        Args:
            user_id (str): User ID or Email Address
            body (SHA256Password):
            token (str):

        """
    def reset_password(self, user_id: str, /, body: SHA256Password, *, token: str) -> None:
        """
        Reset password function with a previously-issued reset token (unauthenticated).
        Password must be hashed with SHA256.

        Args:
            user_id (str): User ID or Email Address
            body (SHA256Password):
            token (str):

        """
    def set_role(self, user_id: str, /, role: Literal["user", "admin"]) -> User:
        """
        Change a user's role from between "user" and "admin". Only admins may do this.
        This revokes all access tokens for the target user.

        Args:
            user_id (str): User ID or Email Address
            role (Literal['user', 'admin']):

        Returns:
            User: Successful Response

        """
    def get_my_state_var(self, key: str, /) -> Any:
        """
        Retrieve a state variable value for the current user, such as
        "licenseAccepted" or "recentProjects".

        Args:
            key (str):

        Returns:
            Any: Successful Response

        """
    def get_state_var(self, user_id: str, key: str, /) -> Any:
        """
        Retrieve a user state variable value, such as "licenseAccepted" or "recentProjects".

        Args:
            user_id (str): User ID or Email Address
            key (str):

        Returns:
            Any: Successful Response

        """
    def set_state_var(self, user_id: str, key: str, /, value: Any) -> User:
        """
        Set a user state variable such as "licenseAccepted" or "recentProjects"

        Args:
            user_id (str): User ID or Email Address
            key (str):
            value (Any):

        Returns:
            User: Successful Response

        """
    def unset_state_var(self, user_id: str, key: str, /) -> User:
        """
        Delete a a user's state variable

        Args:
            user_id (str): User ID or Email Address
            key (str):

        Returns:
            User: Successful Response

        """
    def get_my_lanes(self) -> list[str]:
        """
        Get the lane names the current user has access to

        Returns:
            List[str]: Successful Response

        """
    def get_lanes(self, user_id: str, /) -> list[str]:
        """
        Get the lane names a user has access to

        Args:
            user_id (str): User ID or Email Address

        Returns:
            List[str]: Successful Response

        """
    def set_lanes(self, user_id: str, /, lanes: list[str]) -> User:
        """
        Restrict lanes the given user ID may to queue to.
        Only admins may access this function.

        Args:
            user_id (str): User ID or Email Address
            lanes (List[str]):

        Returns:
            User: Updated user

        """
    def get_my_file_browser_settings(self) -> FileBrowserSettings:
        """
        Get current user's file browser settings, used to determine what file paths
        the user has access to in the CryoSPARC UI.

        Returns:
            FileBrowserSettings: User file browser settings

        """
    def get_file_browser_settings(self, user_id: str, /) -> FileBrowserSettings:
        """
        Get a user's file browser settings, used to determine what file paths the
        user has access to in the CryoSPARC UI.

        Args:
            user_id (str): User ID or Email Address

        Returns:
            FileBrowserSettings: User file browser settings

        """
    def set_file_browser_settings(self, user_id: str, /, body: FileBrowserSettings) -> User:
        """
        Update a user's file browser settings. Only admins may access this function.

        Args:
            user_id (str): User ID or Email Address
            body (FileBrowserSettings):

        Returns:
            User: Successful Response

        """

class ResourcesAPI(APINamespace):
    """
    Functions available in ``api.resources``, e.g., ``api.resources.find_lanes(...)``
    """
    def find_lanes(self) -> list[SchedulerLane]:
        """
        Find registered scheduler lanes that jobs may be queued to.

        Returns:
            List[SchedulerLane]: List of lanes

        """
    def add_lane(self, body: SchedulerLane) -> SchedulerLane:
        """
        Add a new lane that jobs may be scheduled to.

        Args:
            body (SchedulerLane):

        Returns:
            SchedulerLane: Successful Response

        """
    def find_lane(self, name: str, /, *, type: Literal["node", "cluster"] | None = None) -> SchedulerLane:
        """
        Find a registered scheduler lane with the given name and optional type.

        Args:
            name (str):
            type (Literal['node', 'cluster'], optional): Defaults to None

        Returns:
            SchedulerLane: Successful Response

        """
    def remove_lane(self, name: str, /) -> None:
        """
        Remove the specified lane and any targets assigned to that lane. Once
        removed, jobs can no longer be scheduled to this lane.

        Args:
            name (str):

        """
    def reorder_lanes(self, order: list[int]) -> list[SchedulerLane]:
        """
        Reorder the lanes according to the given index list.
        Accepts partial reordering, i.e., the list of indices does not need to include all lanes.
        The lanes not included in the list will be appended to the end of the reordered list in their original order.

        Args:
            order (List[int]):

        Returns:
            List[SchedulerLane]: Successful Response

        """
    def find_targets(self, *, lane: str | None = None) -> list[SchedulerTarget]:
        """
        Find a list of connected worker node or cluster targets that jobs may be
        queued to.

        Args:
            lane (str, optional): Filter by lane name. Defaults to None

        Returns:
            List[SchedulerTarget]: List of targets

        """
    def find_nodes(self, *, lane: str | None = None) -> list[SchedulerTargetNode]:
        """
        Find a list of targets with type "node" that jobs may be scheduled to.
        These correspond to discrete worker hosts accessible over SSH.

        Args:
            lane (str, optional): Defaults to None

        Returns:
            List[SchedulerTargetNode]: List of targets with type 'node'

        """
    def add_node(self, body: SchedulerTargetNode, *, gpu: bool = True) -> SchedulerTargetNode:
        """
        Add a node or update an existing node. Updates an existing node if it
        has the same name. Attempts to connect to the node via SSH to run the
        ``cryosparcw connect`` command.

        Set ``gpu`` to False to skip GPU detection.

        Args:
            body (SchedulerTargetNode):
            gpu (bool, optional): Defaults to True

        Returns:
            SchedulerTargetNode: Successful Response

        """
    def find_clusters(self, *, lane: str | None = None) -> list[SchedulerTargetCluster]:
        """
        Find a list of targets with type "cluster" that that jobs may be scheduled to.
        These are multi-node clusters managed by workflow managers like SLURM or PBS
        and are accessible via submission script.

        Args:
            lane (str, optional): Defaults to None

        Returns:
            List[SchedulerTargetCluster]: List of targets with type 'cluster'

        """
    def add_cluster(self, body: SchedulerTargetCluster) -> SchedulerTargetCluster:
        """
        Add a cluster or update an existing cluster. Update an existing cluster if
        if has the same name.

        Args:
            body (SchedulerTargetCluster):

        Returns:
            SchedulerTargetCluster: Successful Response

        """
    def find_target_by_hostname(self, hostname: str, /) -> SchedulerTarget:
        """
        Find a node or cluster target with the given hostname.

        Args:
            hostname (str):

        Returns:
            SchedulerTarget: Successful Response

        """
    def find_target_by_name(self, name: str, /) -> SchedulerTarget:
        """
        Find a node or cluster target with the given name.

        Args:
            name (str):

        Returns:
            SchedulerTarget: Successful Response

        """
    def find_node(self, name: str, /) -> SchedulerTargetNode:
        """
        Find a node with the given name.

        Args:
            name (str):

        Returns:
            SchedulerTargetNode: Successful Response

        """
    def remove_node(self, name: str, /) -> None:
        """
        Remove a target worker node. Once removed, jobs can no longer be scheduled to this node.

        Args:
            name (str):

        """
    def find_cluster(self, name: str, /) -> SchedulerTargetCluster:
        """
        Find a cluster with the given name.

        Args:
            name (str):

        Returns:
            SchedulerTargetCluster: Successful Response

        """
    def remove_cluster(self, name: str, /) -> None:
        """
        Remove the specified cluster lane and target assigned to it. Once removed,
        jobs can no longer be scheduled to this cluster.

        Args:
            name (str):

        """
    def find_cluster_script(self, name: str, /) -> str:
        """
        Find the cluster submission script template for a cluster with the given name.

        Args:
            name (str):

        Returns:
            str: Successful Response

        """
    def find_cluster_template_vars(self, name: str, /) -> list[str]:
        """
        Compute and retrieve all variable names defined in cluster templates.

        Args:
            name (str):

        Returns:
            List[str]: Successful Response

        """
    def find_cluster_template_custom_vars(self, name: str, /) -> list[str]:
        """
        Compute and retrieve all custom variables names defined in cluster templates
        (i.e., all variables not in the internal list of known variable names).

        Args:
            name (str):

        Returns:
            List[str]: Successful Response

        """
    def update_node_lane(self, name: str, /, lane: str) -> SchedulerTargetNode:
        """
        Change the lane on the given target. Target type must match lane type. The
        lane will be created if it does not already exist.

        Args:
            name (str):
            lane (str):

        Returns:
            SchedulerTargetNode: Successful Response

        """
    def refresh_nodes(self) -> None:
        """
        Asynchronously access target worker nodes. Load latest CPU, RAM and GPU
        info.

        """
    def verify_cluster(self, name: str, /) -> str:
        """
        Ensure cluster has been properly configured by executing the info command.

        Args:
            name (str):

        Returns:
            str: Successful Response

        """
    def update_cluster_custom_vars(self, name: str, /, value: dict[str, str]) -> SchedulerTargetCluster:
        """
        Change the custom cluster variables on the given target (assumed to exist).

        Args:
            name (str):
            value (Dict[str, str]):

        Returns:
            SchedulerTargetCluster: Successful Response

        """
    def update_target_cache_path(self, name: str, /, value: str | None) -> SchedulerTarget:
        """
        Change the cache path on the given target (assumed to exist).

        Args:
            name (str):
            value (str | None):

        Returns:
            SchedulerTarget: Successful Response

        """

class AssetsAPI(APINamespace):
    """
    Functions available in ``api.assets``, e.g., ``api.assets.find(...)``
    """
    def find(self, *, project_uid: str | None = None, job_uid: str | None = None) -> list[GridFSFile]:
        """
        Find files available in the database associated with projects or jobs.

        Each entry in the resulting list is an object with an ``id`` key. Use ``id``
        to download the file.

        Args:
            project_uid (str, optional): Defaults to None
            job_uid (str, optional): Defaults to None

        Returns:
            List[GridFSFile]: Successful Response

        """
    def upload(
        self,
        project_uid: str,
        job_uid: str,
        /,
        stream: Stream,
        *,
        filename: str | None = None,
        format: Literal[
            "txt", "csv", "html", "json", "xml", "bild", "bld", "log", "pdf", "gif", "jpg", "jpeg", "png", "svg"
        ]
        | None = None,
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
    def download(self, id: str = "000000000000000000000000", /) -> Stream:
        """
        Download the asset with the given ID. When calling via HTTP, file contents
        will be in the response body.

        Args:
            id (str, optional): Defaults to '000000000000000000000000'

        Returns:
            Stream: A binary stream representing a Stream class instance

        """
    def find_one(self, id: str = "000000000000000000000000", /) -> GridFSFile:
        """
        Retrive the full details for an asset with the given ID.

        Args:
            id (str, optional): Defaults to '000000000000000000000000'

        Returns:
            GridFSFile: Successful Response

        """

class JobsAPI(APINamespace):
    """
    Functions available in ``api.jobs``, e.g., ``api.jobs.find(...)``
    """
    def find(
        self,
        *,
        id: list[str] | None = None,
        project_uid: list[str] | None = None,
        workspace_uid: list[str] | None = None,
        uid: list[str] | None = None,
        type: list[str] | None = None,
        status: list[JobStatus] | None = None,
        category: list[Category] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        queued_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        started_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        waiting_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        completed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        killed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        failed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        exported_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        deleted: bool | None = False,
        deleting: bool | None = None,
        import_status: str | None = None,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Job]:
        """
        List jobs that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
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
            deleting (bool, optional): Defaults to None
            import_status (str, optional): Defaults to None
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Job]: List of jobs matching supplied query

        """
    def delete_many(self, project_job_uids: list[tuple[str, str]]) -> None:
        """
        Delete the given jobs. Note that jobs in the following states cannot be deleted:

        - Job is active (running or waiting); please kill the job first
        - Job is marked as final
        - Job is an ancestor of a job marked as final
        - Job has connected child jobs that are running, waiting, completed, killed or failed;
          please clear or delete all connected jobs first

        Args:
            project_job_uids (List[Tuple[str, str]]):

        """
    def count(
        self,
        *,
        id: list[str] | None = None,
        project_uid: list[str] | None = None,
        workspace_uid: list[str] | None = None,
        uid: list[str] | None = None,
        type: list[str] | None = None,
        status: list[JobStatus] | None = None,
        category: list[Category] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        queued_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        started_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        waiting_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        completed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        killed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        failed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        exported_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        deleted: bool | None = False,
        deleting: bool | None = None,
        import_status: str | None = None,
    ) -> int:
        """
        Count jobs that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
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
            deleting (bool, optional): Defaults to None
            import_status (str, optional): Defaults to None

        Returns:
            int: Successful Response

        """
    def count_active(self) -> int:
        """
        Count number of active jobs.

        Returns:
            int: Successful Response

        """
    def clone_many(
        self,
        project_uid: str,
        /,
        job_uids: list[str],
        *,
        workspace_uid: str | None = None,
        new_workspace_title: str | None = None,
    ) -> list[Job]:
        """
        Clone the given list of jobs. If any target jobs are related, tries to
        re-create the input connections between the cloned jobs, while keeping
        connections to non-cloned jobs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uids (List[str]):
            workspace_uid (str, optional): Defaults to None
            new_workspace_title (str, optional): Defaults to None

        Returns:
            List[Job]: List of cloned jobs

        """
    def get_chain(self, project_uid: str, /, *, start_job_uid: str, end_job_uid: str) -> list[str]:
        """
        Find the chain of jobs between start job to end job. A job chain is the
        intersection of the start job's descendants and the end job's ancestors.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            start_job_uid (str):
            end_job_uid (str):

        Returns:
            List[str]: Successful Response

        """
    def clone_chain(
        self,
        project_uid: str,
        /,
        *,
        start_job_uid: str,
        end_job_uid: str,
        workspace_uid: str | None = None,
        new_workspace_title: str | None = None,
    ) -> list[Job]:
        """
        Clone jobs that directly descend from the specified start job up to the specified end job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            start_job_uid (str):
            end_job_uid (str):
            workspace_uid (str, optional): Defaults to None
            new_workspace_title (str, optional): Defaults to None

        Returns:
            List[Job]: Newly created cloned jobs

        """
    def get_final_results(self, project_uid: str, /) -> GetFinalResultsResponse:
        """
        Get all final results within a project, along with the ancestors and non-ancestors of those jobs.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            GetFinalResultsResponse: Successful Response

        """
    def find_one(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Get a job by its unique project and job ID.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
    def delete(self, project_uid: str, job_uid: str, /) -> None:
        """
        Delete a job and all associated events and results.

        A job cannot be deleted if it is in any of the following states:

        - Active (running or waiting); please kill the job first
        - Marked as final
        - An ancestor of a job marked as final
        - Has connected child jobs that are running, waiting, completed, killed or failed;
          please clear or delete all connected jobs first

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        """
    def get_directory(self, project_uid: str, job_uid: str, /) -> str:
        """
        Get a job's absolute directory path, e.g., ``"/path/to/project/{job_uid}"``.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            str: Successful Response

        """
    def get_log(self, project_uid: str, job_uid: str, /) -> str:
        """
        Get contents of a job.log file. Empty string if job.log does not exist.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            str: Successful Response

        """
    def get_log_path(self, project_uid: str, job_uid: str, /) -> str:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            str: Successful Response

        """
    def get_output_fields(
        self, project_uid: str, job_uid: str, output_name: str, /, dtype_params: dict[str, Any] = {}
    ) -> list[tuple[str, str]]:
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
    def create(
        self,
        project_uid: str,
        workspace_uid: str,
        /,
        params: dict[str, bool | int | float | str | None] = {},
        *,
        type: str,
        title: str = "",
        description: str = "",
    ) -> Job:
        """
        Add a new job with the given type to a project workspace.

        All available job types and associated metadata are available from the
        ``api.projects.get_job_register()`` function
        (``GET projects/{project_uid}:register`` endpoint).

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            params (Dict[str, Union[bool, int, float, str, str, None]], optional): Defaults to {}
            type (str): Type of job to create
            title (str, optional): Job title. Defaults to ''
            description (str, optional): Job Markdown description. Defaults to ''

        Returns:
            Job: Successful Response

        """
    def create_external_result(self, project_uid: str, workspace_uid: str, /, body: ExternalOutputSpec) -> Job:
        """
        Create an External result with the given specification. Returns an External
        Job with the given output ready to be saved. Used with cryosparc-tools.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            body (ExternalOutputSpec):

        Returns:
            Job: Successful Response

        """
    def get_status(self, project_uid: str, job_uid: str, /) -> JobStatus:
        """
        Get the status of a job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            JobStatus: Successful Response

        """
    def view(self, project_uid: str, workspace_uid: str, job_uid: str, /) -> Job:
        """
        Add a job to a user's recently viewed jobs list. Must specify the workspace
        from which the job was viewed.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
    def set_params(self, project_uid: str, job_uid: str, /, params: dict[str, Any]) -> Job:
        """
        Update job parameters with the given values. Only parameters that are valid
        for the job type are updated.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            params (Dict[str, Any]):

        Returns:
            Job: Successful Response

        """
    def clear_params(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Reset all job parameters to their default values.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
    def set_param(self, project_uid: str, job_uid: str, param: str, /, value: Any) -> Job:
        """
        Set a job parameter to the given value.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            param (str): Job parameter name
            value (Any):

        Returns:
            Job: Successful Response

        """
    def clear_param(self, project_uid: str, job_uid: str, param: str, /) -> Job:
        """
        Reset a job parameter to its default value.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            param (str): Job parameter name

        Returns:
            Job: Successful Response

        """
    def load_input(
        self,
        project_uid: str,
        job_uid: str,
        input_name: str,
        /,
        *,
        force_join: bool | Literal["auto"] = "auto",
        slots: Literal["default", "passthrough", "all"] | list[str] = "default",
    ) -> Dataset:
        """
        Load the dataset connected to a job input.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str): Job input name
            force_join (bool | Literal['auto'], optional): Defaults to 'auto'
            slots (Literal['default', 'passthrough', 'all'] | List[str], optional): Low-level results to load. Defaults to 'default'

        Returns:
            Dataset: A binary stream representing a Dataset class instance

        """
    def load_output(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        /,
        *,
        version: int | Literal["F"] = "F",
        slots: Literal["default", "passthrough", "all"] | list[str] = "default",
    ) -> Dataset:
        """
        Load the dataset for a job output.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): Job output name
            version (int | Literal['F'], optional): Set to F (default) to load the final version. Defaults to 'F'
            slots (Literal['default', 'passthrough', 'all'] | List[str], optional): Low-level results to load. Defaults to 'default'

        Returns:
            Dataset: A binary stream representing a Dataset class instance

        """
    def save_output(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        /,
        dataset: Dataset,
        *,
        filename: str | None = None,
        version: int = 0,
    ) -> Job:
        """
        Save job output dataset. Job must be running or waiting to accept outputs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): Job output name
            dataset (Dataset): A binary stream representing a Dataset class instance
            filename (str, optional): Filename override. Defaults to None
            version (int, optional): Intermediate output version. Defaults to 0

        Returns:
            Job: Successful Response

        """
    def connect(
        self, project_uid: str, job_uid: str, input_name: str, /, *, source_output_name: str, source_job_uid: str
    ) -> Job:
        """
        Connect the output of a parent (source) job to the input of a child (dest) job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str): Job input name
            source_output_name (str):
            source_job_uid (str):

        Returns:
            Job: Connected child job

        """
    def disconnect_all(self, project_uid: str, job_uid: str, input_name: str, /) -> Job:
        """
        Remove all connections on the given job input.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str): Job input name

        Returns:
            Job: Successful Response

        """
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
            input_name (str): Job input name
            connection_index (int):
            source_output_name (str):
            source_job_uid (str):

        Returns:
            Job: Successful Response

        """
    def disconnect(self, project_uid: str, job_uid: str, input_name: str, connection_index: int, /) -> Job:
        """
        Disconnect a job input.

        Set ``connection_index`` 0 to remove the first connection, 1 for the second, etc.
        Set ``connection_index`` to -1 to remove the last connection, -2 for the second-last, etc.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str): Job input name
            connection_index (int):

        Returns:
            Job: Successful Response

        """
    def find_output_result(self, project_uid: str, job_uid: str, output_name: str, result_name: str, /) -> OutputResult:
        """
        Get a job's low-level output result.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): Job output name
            result_name (str):

        Returns:
            OutputResult: Successful Response

        """
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
            output_name (str): Job output name
            result_name (str):

        """
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
        source_result_version: int | Literal["F"] = "F",
        source_job_uid: str,
    ) -> Job:
        """
        Connect or replace a low-level input result slot to an output result from
        another (source) job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str): Job input name
            connection_index (int):
            result_name (str):
            source_output_name (str):
            source_result_name (str):
            source_result_version (int | Literal['F'], optional): Defaults to 'F'
            source_job_uid (str):

        Returns:
            Job: Successful Response

        """
    def disconnect_result(
        self, project_uid: str, job_uid: str, input_name: str, connection_index: int, result_name: str, /
    ) -> Job:
        """
        Clear a job's low-level input result slot.

        Set ``connection_index`` 0 to remove the first connection, 1 for the second, etc.
        Set ``connection_index`` to -1 to remove the last connection, -2 for the second-last, etc.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str): Job input name
            connection_index (int):
            result_name (str):

        Returns:
            Job: Successful Response

        """
    def add_external_input(self, project_uid: str, job_uid: str, input_name: str, /, body: InputSpec) -> Job:
        """
        Add or replace an External Job input.

        This action is available while the job is building, running or waiting
        for results. Completed or failed External jobs cannot accept new inputs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            input_name (str): May only contain letters, numbers and underscores, and must start with a letter
            body (InputSpec):

        Returns:
            Job: Successful Response

        """
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
    def set_output_image(self, project_uid: str, job_uid: str, output_name: str, /, body: GridFSAsset) -> Job:
        """
        Set a custom image for an External job output.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): Job output name
            body (GridFSAsset):

        Returns:
            Job: Successful Response

        """
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
    def enqueue(
        self,
        project_uid: str,
        job_uid: str,
        /,
        *,
        lane: str | None = None,
        hostname: str | None = None,
        gpus: list[int] = [],
        no_check_inputs_ready: bool = False,
        oversubscribe_gpus: bool = True,
    ) -> Job:
        """
        Queue a job for execution. The job will launch when

        - all connected input jobs are completed (unless ``no_check_inputs_ready`` is True)
        - enough CPU/GPU/RAM is available on the lane or target
        - enough license tokens are available

        Some job types that don't require GPUs may be launched immediately by specifying
        neither lane nor target. Interactive jobs must be launched this way.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            lane (str, optional): Connected scheduler lane name to queue to. Defaults to None
            hostname (str, optional): Target name to queue to, cannot be specified with lane. Defaults to None
            gpus (List[int], optional): GPU number(s) the job should use, target must also be specified. Defaults to []
            no_check_inputs_ready (bool, optional): If enabled, launch job without waiting for connected input jobs to complete. Defaults to False
            oversubscribe_gpus (bool, optional): If enabled, launch job even if target GPU(s) are already in use by another job.. Defaults to True

        Returns:
            Job: Successful Response

        """
    def recalculate_size_async(self, project_uid: str, job_uid: str, /) -> None:
        """
        For a given job, find intermediate results and recalculate their total size.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        """
    def recalculate_project_intermediate_results_size(self, project_uid: str, /) -> None:
        """
        Recalculate intermediate result sizes for all jobs in a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
    def clear_intermediate_results(self, project_uid: str, job_uid: str, /, *, always_keep_final: bool = True) -> None:
        """
        Remove intermediate results from a job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            always_keep_final (bool, optional): Defaults to True

        """
    def export_output_results(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        /,
        results: ExportResultsSelection | list[str] | None = None,
    ) -> None:
        """
        Prepare a job's output for import to another project or instance.
        Will create a folder in the project directory's exports subfolder,
        then link the output's associated files there.

        Note that the returned .csg file's parent folder must be manually copied
        with symlinks resolved into the target project folder before importing.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): Job output name
            results (ExportResultsSelection | List[str], optional): Defaults to None

        """
    def export_job(
        self,
        project_uid: str,
        job_uid: str,
        /,
        outputs_selection: ExportOutputsSelection | TransferMethod | None = None,
        *,
        dest_path: str | None = None,
    ) -> None:
        """
        Start export for a job into the project's exports folder or a job-named subdirectory of the destination path.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            outputs_selection (ExportOutputsSelection | TransferMethod, optional): Defaults to None
            dest_path (str, optional): Absolute destination directory; the job is exported to a job-named subdirectory. Defaults to None

        """
    def copy_to_project(
        self,
        project_uid: str,
        job_uid: str,
        /,
        outputs_selection: ExportOutputsSelection | None = None,
        *,
        dest_project_uid: str,
        dest_workspace_uid: str,
    ) -> None:
        """
        Export the job into its own project's exports directory, then import that
        export directly into another project/workspace, as one operation.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            outputs_selection (ExportOutputsSelection, optional): Defaults to None
            dest_project_uid (str):
            dest_workspace_uid (str):

        """
    def copy_result_group_to_project(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        /,
        results: ExportResultsSelection | None = None,
        *,
        dest_project_uid: str,
        dest_workspace_uid: str,
        lane: str | None = None,
    ) -> None:
        """
        Export a job output's results into its own project's exports directory,
        then import that export directly as a result group into another
        project/workspace, as one operation.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): Job output name
            results (ExportResultsSelection, optional): Defaults to None
            dest_project_uid (str):
            dest_workspace_uid (str):
            lane (str, optional): Defaults to None

        """
    def export_output_preview(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        /,
        results: ExportResultsSelection | list[str] | None = None,
        *,
        include_size: bool = False,
        include_paths: bool = False,
    ) -> ExportOutputPreview:
        """
        Read-only preview of exporting a job's output's results. Set include_size to
        include on-disk sizes, and include_paths to include each result's
        resolved file paths.

        Reported sizes are an approximation of the referenced blob/movie/image files
        only; they do not include the size of the dataset file(s) written by the
        actual export (e.g. per-particle orientation/CTF metadata), so the true
        export size will typically be a little larger than this preview.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): Job output name
            results (ExportResultsSelection | List[str], optional): Defaults to None
            include_size (bool, optional): Defaults to False
            include_paths (bool, optional): Defaults to False

        Returns:
            ExportOutputPreview: Successful Response

        """
    def get_output_result_path(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        result_name: str,
        /,
        *,
        version: int | Literal["F"] = "F",
    ) -> str:
        """
        Get the absolute path for a job output's dataset or volume density.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            output_name (str): Job output name
            result_name (str):
            version (int | Literal['F'], optional): Defaults to 'F'

        Returns:
            str: Successful Response

        """
    def interactive_post(
        self, project_uid: str, job_uid: str, /, body: dict[str, Any], *, endpoint: str, timeout: int = 10
    ) -> Any:
        """
        Send a message to or perform an interactive action on a waiting interactive job.
        Possible actions and expected body depend on the job type.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            body (Dict[str, Any]):
            endpoint (str): Interactive endpoint to call, e.g., get_micrograph_data
            timeout (int, optional): Maximum time to wait for action to complete, in seconds. Defaults to 10

        Returns:
            Any: Successful Response

        """
    def mark_running(
        self, project_uid: str, job_uid: str, /, *, status: Literal["running", "waiting"] = "running"
    ) -> Job:
        """
        Indicate that an external job is running or waiting. This prepares the job
        for accepting results.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            status (Literal['running', 'waiting'], optional): Defaults to 'running'

        Returns:
            Job: Successful Response

        """
    def mark_completed(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Mark a killed or failed job, or an active external job, as completed.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
    def mark_failed(self, project_uid: str, job_uid: str, /, *, error: str | None = None) -> Job:
        """
        Manually mark a job as failed.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            error (str, optional): Defaults to None

        Returns:
            Job: Successful Response

        """
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
    def get_event_logs(
        self, project_uid: str, job_uid: str, /, *, checkpoint: int | None = None
    ) -> list[TextEvent | ImageEvent | InteractiveEvent | CheckpointEvent | Event]:
        """
        Get all event logs for a job.

        Note: this may return a large amount of data. Call repeatedly with an
        incrementing checkpoint to retrieve in batches.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            checkpoint (int, optional): Defaults to None

        Returns:
            List[Union[TextEvent, ImageEvent, InteractiveEvent, CheckpointEvent, Event]]: Successful Response

        """
    def add_image_log(
        self, project_uid: str, job_uid: str, /, images: list[GridFSAsset], *, text: str, flags: list[str] = ["plots"]
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
    def add_checkpoint(self, project_uid: str, job_uid: str, /, meta: dict[str, Any] = {}) -> CheckpointEvent:
        """
        Add a checkpoint the target job's event log.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            meta (Dict[str, Any], optional): Defaults to {}

        Returns:
            CheckpointEvent: Successful Response

        """
    def update_event_log(
        self,
        project_uid: str,
        job_uid: str,
        event_id: str = "000000000000000000000000",
        /,
        text: str | None = None,
        *,
        type: Literal["text", "warning", "error"] | None = None,
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
    def recalculate_size(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Recalculate the size of a given job's directory.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
    def clear(self, project_uid: str, job_uid: str, /, *, descendants: bool = False) -> Job:
        """
        Clear a job's outputs and events to get it back to building state.
        Active jobs must be killed before clearing.

        Retains input connections and parameter overrides.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            descendants (bool, optional): Enable to also clear all descendant jobs. Defaults to False

        Returns:
            Job: Successful Response

        """
    def clear_many(
        self,
        *,
        id: list[str] | None = None,
        project_uid: list[str] | None = None,
        workspace_uid: list[str] | None = None,
        uid: list[str] | None = None,
        type: list[str] | None = None,
        status: list[JobStatus] | None = None,
        category: list[Category] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        queued_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        started_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        waiting_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        completed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        killed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        failed_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        exported_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        deleted: bool | None = False,
        deleting: bool | None = None,
        import_status: str | None = None,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Job]:
        """
        Clear all jobs that match the given query.

        Args:
            id (List[str], optional): Defaults to None
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
            deleting (bool, optional): Defaults to None
            import_status (str, optional): Defaults to None
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Job]: Successful Response

        """
    def clone(self, project_uid: str, job_uid: str, /, *, workspace_uid: str | None = None) -> Job:
        """
        Clone a job, creating a new building job with the same inputs and
        parameters but no outputs.

        If ``workspace_uid`` is not specified, clones into the oldest workspace the
        original job is linked to.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str, optional): Target workspace to create the cloned job in. Defaults to None

        Returns:
            Job: Cloned job

        """
    def kill(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Kill a running job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
    def restart(self, project_uid: str, job_uid: str, /) -> None:
        """
        Kill and clear a completed or partially-completed job, queue it with the
        same scheduler settings as its previous run.

        Retains input connections and parameter overrides.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        """
    def set_final_result(self, project_uid: str, job_uid: str, /, *, is_final_result: bool) -> Job:
        """
        Mark or unmark a job as a final result. Final jobs and their connected
        ancestors cannot be cleared or deleted, and are protected during data cleanup.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            is_final_result (bool):

        Returns:
            Job: Successful Response

        """
    def set_title(self, project_uid: str, job_uid: str, /, *, title: str) -> Job:
        """
        Set job title.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            title (str): Job title

        Returns:
            Job: Successful Response

        """
    def set_description(self, project_uid: str, job_uid: str, /, description: str = "") -> Job:
        """
        Set job description. May include Markdown formatting.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            description (str, optional): Defaults to ''

        Returns:
            Job: Successful Response

        """
    def set_priority(self, project_uid: str, job_uid: str, /, *, priority: int) -> Job:
        """
        Set job priority. Once queued, higher priority jobs are scheduled before
        lower priority ones.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            priority (int): Larger number indicates higher priority

        Returns:
            Job: Successful Response

        """
    def set_cluster_custom_vars(self, project_uid: str, job_uid: str, /, cluster_custom_vars: dict[str, str]) -> Job:
        """
        Set cluster custom variables for job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            cluster_custom_vars (Dict[str, str]):

        Returns:
            Job: Successful Response

        """
    def get_active_licenses_count(self) -> int:
        """
        Get number of acquired licenses for running jobs

        Returns:
            int: Successful Response

        """
    def get_types(self) -> Any:
        """
        Get list of available job types

        Returns:
            Any: Successful Response

        """
    def get_categories(self) -> Any:
        """
        Get job types by category

        Returns:
            Any: Successful Response

        """
    def find_ancestor_uids(self, project_uid: str, job_uid: str, /, *, workspace_uid: str | None = None) -> list[str]:
        """
        Finds all ancestors of a single job and return a list of their UIDs

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str, optional): Defaults to None

        Returns:
            List[str]: Successful Response

        """
    def find_descendant_uids(self, project_uid: str, job_uid: str, /, *, workspace_uid: str | None = None) -> list[str]:
        """
        Find the list of all job UIDs that a job is an ancestor of based on its outputs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str, optional): Defaults to None

        Returns:
            List[str]: Successful Response

        """
    def link_to_workspace(self, project_uid: str, job_uid: str, workspace_uid: str, /) -> Job:
        """
        Link a job to an additional workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Job: Successful Response

        """
    def unlink_from_workspace(self, project_uid: str, job_uid: str, workspace_uid: str, /) -> Job:
        """
        Unlink a job from a workspace.

        Fails if the job is not linked to the given workspace, or if this is the
        only workspace the job is linked to.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Job: Successful Response

        """
    def move(self, project_uid: str, job_uid: str, /, *, from_workspace_uid: str, to_workspace_uid: str) -> Job:
        """
        Move a job from one workspace to another. Equivalent to linking to the new
        workspace and unlinking from the old one.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            from_workspace_uid (str):
            to_workspace_uid (str):

        Returns:
            Job: Successful Response

        """
    def get_symlinks(self, project_uid: str, job_uid: str, /) -> list[SymlinkInfo]:
        """
        Get all symbolic links in a job directory.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            List[SymlinkInfo]: Successful Response

        """
    def update_directory_symlinks(self, project_uid: str, job_uid: str, /, *, prefix_cut: str, prefix_new: str) -> int:
        """
        Rewrite symbolic link target paths in a job directory, changing target paths
        with prefix ``prefix_cut`` to instead have prefix ``prefix_new``.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            prefix_cut (str):
            prefix_new (str):

        Returns:
            int: Number of symlinks updated

        """
    def add_tag(self, project_uid: str, job_uid: str, tag_uid: str, /) -> Job:
        """
        Add a tag to a job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            tag_uid (str): Tag UID, e.g., "T1"

        Returns:
            Job: Successful Response

        """
    def remove_tag(self, project_uid: str, job_uid: str, tag_uid: str, /) -> Job:
        """
        Remove a tag from a job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            tag_uid (str): Tag UID, e.g., "T1"

        Returns:
            Job: Successful Response

        """
    def import_job(
        self, project_uid: str, workspace_uid: str, /, *, dereference_symlinks: bool = False, path: str
    ) -> Job:
        """
        Imports an exported job directory into the project. The directory may be
        located inside this project (e.g. its exports/ subfolder), or inside
        another project's directory that this instance/user is permitted to
        import from.

        Resulting job will not be ready to use until ``import_status`` is "complete".

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            dereference_symlinks (bool, optional): Defaults to False
            path (str):

        Returns:
            Job: Successful Response

        """
    def import_result_group(
        self,
        project_uid: str,
        workspace_uid: str,
        /,
        *,
        dereference_symlinks: bool = False,
        on_missing: Literal["drop_result", "import_anyway", "fail"] = "drop_result",
        path: str,
        lane: str | None = None,
    ) -> None:
        """
        Create and enqueue an Import Result Group job with the given path. The
        path may be located inside this project, or inside another project's
        directory that this instance/user is permitted to import from.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            dereference_symlinks (bool, optional): Defaults to False
            on_missing (Literal['drop_result', 'import_anyway', 'fail'], optional): Defaults to 'drop_result'
            path (str):
            lane (str, optional): Defaults to None

        """
    def star_job(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Stars a job for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
    def unstar_job(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Unstars a job for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Job: Successful Response

        """
    def apply_blueprint(self, project_uid: str, job_uid: str, /, *, blueprint_id: str) -> Job:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"
            blueprint_id (str):

        Returns:
            Job: Successful Response

        """
    def create_job_from_blueprint(self, project_uid: str, workspace_uid: str, /, *, blueprint_id: str) -> Job:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            blueprint_id (str):

        Returns:
            Job: Successful Response

        """

class WorkspacesAPI(APINamespace):
    """
    Functions available in ``api.workspaces``, e.g., ``api.workspaces.find(...)``
    """
    def find(
        self,
        *,
        id: list[str] | None = None,
        uid: list[str] | None = None,
        project_uid: list[str] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        deleted: bool | None = False,
        deleting: bool | None = None,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Workspace | Session]:
        """
        List workspaces that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            deleted (bool, optional): Defaults to False
            deleting (bool, optional): Defaults to None
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Union[Workspace, Session]]: Successful Response

        """
    def count(
        self,
        *,
        id: list[str] | None = None,
        uid: list[str] | None = None,
        project_uid: list[str] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        deleted: bool | None = False,
        deleting: bool | None = None,
    ) -> int:
        """
        Count workspaces that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            deleted (bool, optional): Defaults to False
            deleting (bool, optional): Defaults to None

        Returns:
            int: Successful Response

        """
    def preview_delete(self, project_uid: str, workspace_uid: str, /) -> DeleteWorkspacePreview:
        """
        Get a list of jobs that would be removed when the given workspace is deleted.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            DeleteWorkspacePreview: Successful Response

        """
    def find_one(self, project_uid: str, workspace_uid: str, /) -> Workspace | Session:
        """
        Find a workspace in a project by its unique ID.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Workspace | Session: Successful Response

        """
    def delete(self, project_uid: str, workspace_uid: str, /) -> None:
        """
        Initiate a workspace deletion. This deletes the workspace and any jobs
        exclusively in the workspace.

        Workspace jobs with the following conditions are never deleted:
        - Linked to other workspaces (unlinked instead)
        - Have "final" or "ancestor of final" status
        - Have descendants in other workspaces.

        If the workspace cannot be emptied due to these conditions, it is not
        marked as deleted.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        """
    def create(self, project_uid: str, /, *, title: str, description: str | None = None) -> Workspace:
        """
        Create a new empty workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            title (str):
            description (str, optional): Defaults to None

        Returns:
            Workspace: Successful Response

        """
    def set_title(self, project_uid: str, workspace_uid: str, /, *, title: str) -> Workspace | Session:
        """
        Set title of a workspace

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            title (str): Workspace title

        Returns:
            Workspace | Session: Successful Response

        """
    def set_description(self, project_uid: str, workspace_uid: str, /, description: str = "") -> Workspace | Session:
        """
        Set description of a workspace

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            description (str, optional): Defaults to ''

        Returns:
            Workspace | Session: Successful Response

        """
    def view(self, project_uid: str, workspace_uid: str, /) -> Workspace | Session:
        """
        Add a workspace to a user's recently viewed workspaces list.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Workspace | Session: Successful Response

        """
    def add_tag(self, project_uid: str, workspace_uid: str, tag_uid: str, /) -> Workspace | Session:
        """
        Tag the given workspace with the given tag.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            tag_uid (str): Tag UID, e.g., "T1"

        Returns:
            Workspace | Session: Successful Response

        """
    def remove_tag(self, project_uid: str, workspace_uid: str, tag_uid: str, /) -> Workspace | Session:
        """
        Removes a tag from a workspace.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            tag_uid (str): Tag UID, e.g., "T1"

        Returns:
            Workspace | Session: Successful Response

        """
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
    def find_workspace_ancestor_uids(
        self, project_uid: str, workspace_uid: str, /, *, job_uids: list[str]
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
    def find_workspace_descendant_uids(
        self, project_uid: str, workspace_uid: str, /, *, job_uids: list[str]
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
    def star_workspace(self, project_uid: str, workspace_uid: str, /) -> Workspace | Session:
        """
        Stars a workspace for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Workspace | Session: Successful Response

        """
    def unstar_workspace(self, project_uid: str, workspace_uid: str, /) -> Workspace | Session:
        """
        Unstars a workspace for a user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            Workspace | Session: Successful Response

        """
    def split(self, project_uid: str, workspace_uid: str, /, *, title: str | None = None, parent_dir: str) -> None:
        """
        Create a new project at target_dir containing the specified jobs and their
        external parent jobs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            title (str, optional): Defaults to None
            parent_dir (str):

        """
    def preview_split(self, project_uid: str, workspace_uid: str, /) -> SplitProjectPreview:
        """
        Preview which jobs, sessions, and workspaces would be included in a split call.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        Returns:
            SplitProjectPreview: Successful Response

        """

class SessionsAPI(APINamespace):
    """
    Functions available in ``api.sessions``, e.g., ``api.sessions.find(...)``
    """
    def find(
        self,
        *,
        id: list[str] | None = None,
        uid: list[str] | None = None,
        session_uid: list[str] | None = None,
        project_uid: list[str] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        cleared_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        status: list[SessionStatus] | None = None,
        deleted: bool | None = False,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Session]:
        """
        List sessions that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            session_uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            cleared_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            status (List[SessionStatus], optional): Defaults to None
            deleted (bool, optional): Defaults to False
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Session]: Successful Response

        """
    def count(
        self,
        *,
        id: list[str] | None = None,
        uid: list[str] | None = None,
        session_uid: list[str] | None = None,
        project_uid: list[str] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        cleared_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        status: list[SessionStatus] | None = None,
        deleted: bool | None = False,
    ) -> int:
        """
        Count sessions that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
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
    def find_one(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Find a session in a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def delete(self, project_uid: str, session_uid: str, /) -> None:
        """
        Start a background task to clear a session and mark as "deleted". Raise an
        error notification if the session contains jobs that cannot be deleted.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
    def create(self, project_uid: str, /, *, title: str, description: str | None = None) -> Session:
        """
        Create a new session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            title (str):
            description (str, optional): Defaults to None

        Returns:
            Session: Successful Response

        """
    def clone(
        self, project_uid: str, session_uid: str, /, *, title: str | None = None, description: str | None = None
    ) -> Session:
        """
        Clone an existing session, copying session configuration, parameters, and exposure groups.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            title (str, optional): Defaults to None
            description (str, optional): Defaults to None

        Returns:
            Session: Successful Response

        """
    def find_exposure_groups(self, project_uid: str, session_uid: str, /) -> list[ExposureGroup]:
        """
        Find all exposure groups in a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            List[ExposureGroup]: Successful Response

        """
    def create_exposure_group(self, project_uid: str, session_uid: str, /) -> ExposureGroup:
        """
        Create an exposure group in a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            ExposureGroup: Successful Response

        """
    def find_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> ExposureGroup:
        """
        Find an exposure group with a specific ID in a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_group_id (int):

        Returns:
            ExposureGroup: Successful Response

        """
    def update_exposure_group(
        self, project_uid: str, session_uid: str, exposure_group_id: int, /, body: ExposureGroupUpdate
    ) -> ExposureGroup:
        """
        Configure a session exposure group.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_group_id (int):
            body (ExposureGroupUpdate):

        Returns:
            ExposureGroup: Successful Response

        """
    def delete_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> Session:
        """
        Delete an exposure group from a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_group_id (int):

        Returns:
            Session: Successful Response

        """
    def finalize_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> ExposureGroup:
        """
        Finalize a session exposure group. If the session is running, CryoSPARC
        begins checking for new exposures in the group and processes them as they
        are found.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_group_id (int):

        Returns:
            ExposureGroup: Successful Response

        """
    def get_symlinks(self, project_uid: str, session_uid: str, /) -> list[SymlinkInfo]:
        """
        Get all symbolic links in the session directory.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            List[SymlinkInfo]: Successful Response

        """
    def update_directory_symlinks(
        self, project_uid: str, session_uid: str, /, *, prefix_cut: str, prefix_new: str
    ) -> int:
        """
        Rewrites all symbolic links in the session directory, modifying links prefixed with `prefix_cut` to instead be prefixed with `prefix_new`.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            prefix_cut (str):
            prefix_new (str):

        Returns:
            int: Number of symlinks updated

        """
    def start(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Build and start a CryoSPARC Live Session. Resources, parameters and exposure
        groups must already be configured.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def pause(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Gracefully stop and kill all preprocessing and streaming jobs associated
        with the session. Stop checking for and processing new exposures.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
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
        Configure auto-pause settings for a session, which come into effect once no
        more more new exposures are found for a session after some time
        (default 10 minutes).

        Set to "immediate" to pause as soon as all exposures are marked as ready,
        regardless of the status of other session jobs.

        Set to "graceful" to wait until reconstruction jobs (streaming 2D
        Classification and streaming 3D Refinement) enter "waiting" status.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            auto_pause (Literal['disabled', 'graceful', 'immediate']):
            auto_pause_after_idle_minutes (int, optional): Defaults to 10

        Returns:
            Session: Successful Response

        """
    def update_compute_configuration(
        self, project_uid: str, session_uid: str, /, body: LiveComputeResources
    ) -> LiveComputeResources:
        """
        Update compute resource configuration for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveComputeResources):

        Returns:
            LiveComputeResources: Successful Response

        """
    def add_tag(self, project_uid: str, session_uid: str, tag_uid: str, /) -> Session:
        """
        Add a tag to a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            tag_uid (str): Tag UID, e.g., "T1"

        Returns:
            Session: Successful Response

        """
    def remove_tag(self, project_uid: str, session_uid: str, tag_uid: str, /) -> Session:
        """
        Remove a tag from a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            tag_uid (str): Tag UID, e.g., "T1"

        Returns:
            Session: Successful Response

        """
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
        Update a session's parameters. Unless ``reprocess`` is ``False``, resets
        exposures to a previous processing stage corresponding to the new params.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LivePreprocessingParams):
            reprocess (bool, optional): Defaults to True
            priority (int, optional): Defaults to 1

        Returns:
            Session: Successful Response

        """
    def update_attribute_threshold(
        self,
        project_uid: str,
        session_uid: str,
        attribute: str,
        /,
        *,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> Session:
        """
        Update thresholds for a given session attribute.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            attribute (str):
            min_val (float, optional): Defaults to None
            max_val (float, optional): Defaults to None

        Returns:
            Session: Successful Response

        """
    def clear_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Delete all computed session data, including found and processed micrographs,
        2D classes and volumes. Retains associated processing jobs created while the
        session was running, though their outputs will no longer be usable.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def view(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Add the session to a user's recently viewed sessions list

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def setup_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Setup streaming 2D classification job for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
    def reinitialize_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Setup an existing streaming 2D classification job for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
    def enqueue_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueue streaming 2D Classification job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
    def stop_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stop streaming 2D Classification job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def clear_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clear streaming 2D Classification job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def update_phase2_class2D_params(self, project_uid: str, session_uid: str, /, body: LiveClass2DParams) -> Session:
        """
        Update streaming 2D Classification job params for session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveClass2DParams):

        Returns:
            Session: Successful Response

        """
    def toggle_class2d_template(self, project_uid: str, session_uid: str, template_idx: int, /) -> Session:
        """
        Toggle a specific 2D template for a session's streaming 2D classification job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_idx (int):

        Returns:
            Session: Successful Response

        """
    def toggle_all_class2d_templates(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Toggle the selection state of all templates for a session's streaming 2D classification job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def select_all_class2d_templates(
        self, project_uid: str, session_uid: str, direction: Literal["select", "deselect"], /
    ) -> Session:
        """
        Select or deselect all templates for a session's streaming 2D Classification job.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            direction (Literal['select', 'deselect']):

        Returns:
            Session: Successful Response

        """
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
        Select or deselect all templates inclusively above or below a specific template for a
        session's streaming 2D Classification. All templates within the threshold of
        the specified template index will be selected, and all others will be
        deselected.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_idx (int):
            dimension (Literal['num_particles_total', 'res_A', 'class_ess']):
            direction (Literal['above', 'below'], optional): Defaults to 'above'

        Returns:
            Session: Successful Response

        """
    def select_class2d_templates_with_thresholds(
        self, project_uid: str, session_uid: str, /, template_selection_thresholds: list[TemplateSelectionThreshold]
    ) -> Session:
        """
        Select templates for a session's streaming 2D Classification job based on the
        given thresholds.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_selection_thresholds (List[TemplateSelectionThreshold]):

        Returns:
            Session: Successful Response

        """
    def start_extract_manual(self, project_uid: str, session_uid: str, /) -> None:
        """
        Extract manually picked particles in a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
    def set_session_exposure_processing_priority(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        exposure_processing_priority: Literal["normal", "oldest", "latest", "alternate"],
    ) -> Session:
        """
        Set session exposure processing priority

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_processing_priority (Literal['normal', 'oldest', 'latest', 'alternate']):

        Returns:
            Session: Successful Response

        """
    def set_session_phase_one_wait_for_exposures(
        self, project_uid: str, session_uid: str, /, *, phase_one_wait_for_exposures: bool
    ) -> Session:
        """
        Set whether to wait until exposures are available before queuing the session's preprocessing worker jobs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            phase_one_wait_for_exposures (bool):

        Returns:
            Session: Successful Response

        """
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
        Update picking threshold values for a session

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
    def reset_attribute_threshold(self, project_uid: str, session_uid: str, attribute: str, /) -> Session:
        """
        Reset attribute threshold for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            attribute (str):

        Returns:
            Session: Successful Response

        """
    def reset_all_attribute_thresholds(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Reset all attribute thresholds for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def setup_template_creation_class2D(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        num_classes: int,
        picker_type: Literal["blob", "template", "manual"],
        num_mics: int,
        override_particle_diameter_A: float | None = None,
        uid_lte: int | None = None,
    ) -> Job:
        """
        Setup template creation 2D classification job for a session

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
    def set_template_creation_job(
        self, project_uid: str, session_uid: str, /, *, job_uid: str, template_creation_project_uid: str | None = None
    ) -> Session:
        """
        Set template creation 2D classification job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            job_uid (str):
            template_creation_project_uid (str, optional): Project where the template creation job is located, assumed to be in the same project as the session if not specified. Defaults to None

        Returns:
            Session: Successful Response

        """
    def clear_template_creation_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clear template creation 2D classification job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def toggle_picking_template(self, project_uid: str, session_uid: str, template_idx: int, /) -> Session:
        """
        Toggle a template for a session's template creation job at a specific index

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_idx (int):

        Returns:
            Session: Successful Response

        """
    def toggle_all_picking_templates(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Toggle the selection status of all templates for a session's template creation job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def select_all_picking_templates(
        self, project_uid: str, session_uid: str, direction: Literal["select", "deselect"], /
    ) -> Session:
        """
        Select or deselect all templates for a session's template creation job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            direction (Literal['select', 'deselect']):

        Returns:
            Session: Successful Response

        """
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
        Select or deselect all templates above or below a specific template for a
        session's template creation job. All templates within the threshold of the
        the specified template index will be selected, and all others will be
        deselected.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_idx (int):
            direction (Literal['above', 'below']):
            dimension (Literal['num_particles_total', 'res_A', 'class_ess']):

        Returns:
            Session: Successful Response

        """
    def select_picking_templates_with_thresholds(
        self, project_uid: str, session_uid: str, /, template_selection_thresholds: list[TemplateSelectionThreshold]
    ) -> Session:
        """
        Select all templates above or below an index for a session's template creation job

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            template_selection_thresholds (List[TemplateSelectionThreshold]):

        Returns:
            Session: Successful Response

        """
    def setup_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Setup Ab-Initio Reconstruction job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
    def set_phase2_abinit_job(self, project_uid: str, session_uid: str, /, *, job_uid: str) -> Session:
        """
        Set a Live Ab-Initio Reconstruction job for a session. May specify any job with volume outputs.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            job_uid (str):

        Returns:
            Session: Successful Response

        """
    def enqueue_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueue Ab-Initio Reconstruction job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
    def clear_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clear Ab-Initio Reconstruction job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def update_phase2_abinit_params(self, project_uid: str, session_uid: str, /, body: LiveAbinitParams) -> Session:
        """
        Update Ab-Initio Reconstruction parameters for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveAbinitParams):

        Returns:
            Session: Successful Response

        """
    def select_phase2_abinit_volume(self, project_uid: str, session_uid: str, /, *, volume_name: str) -> Session:
        """
        Select one of the computed or loaded volumes in a session's Ab-Initio
        Reconstruction stage, if multiple are available. The selected volume is
        used for streaming homogeneous refinement.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            volume_name (str): Volume output name from the session's Ab-Initio job, if multiple classes are available. e.g., "volume_class_0"

        Returns:
            Session: Successful Response

        """
    def stop_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stop the running Ab-Initio Reconstruction job for a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def clear_phase2_refine(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clear streaming Homogenous Refinement job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def setup_phase2_refine(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
    def enqueue_phase2_refine(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueue a streaming Homogenous Refinement job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Job: Successful Response

        """
    def stop_phase2_refine(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stop a streaming Homogenous Refinement job for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def update_phase2_refine_params(self, project_uid: str, session_uid: str, /, body: LiveRefineParams) -> Session:
        """
        Update streaming Homogenous Refinement job parameteres for a session

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveRefineParams):

        Returns:
            Session: Successful Response

        """
    def create_and_enqueue_export_particles(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        picker_type: Literal["blob", "template", "manual"] | None = None,
        num_mics: int | None = None,
        uid_lte: int | None = None,
        test_only: bool = False,
    ) -> Job:
        """
        Create and queue a session Live Particle Export job. Job will run on master.

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
    def create_and_enqueue_export_exposures(
        self, project_uid: str, session_uid: str, /, *, export_ignored: bool = False
    ) -> Job:
        """
        Create and queue a session Live Exposure Export job. Job will run on master.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            export_ignored (bool, optional): Defaults to False

        Returns:
            Job: Enqueued Live Exposure Export job

        """
    def mark_session_completed(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Mark the session as completed

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def get_configuration_profiles(self) -> list[SessionConfigProfile]:
        """
        Get all session configuration profiles

        Returns:
            List[SessionConfigProfile]: Successful Response

        """
    def create_configuration_profile(self, body: SessionConfigProfileBody) -> SessionConfigProfile:
        """
        Create a session configuration profile

        Args:
            body (SessionConfigProfileBody):

        Returns:
            SessionConfigProfile: Successful Response

        """
    def apply_configuration_profile(self, project_uid: str, session_uid: str, /, *, configuration_id: str) -> Session:
        """
        Apply a configuration profile to a session, overwriting its resources, parameters, and exposure groups

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            configuration_id (str):

        Returns:
            Session: Successful Response

        """
    def update_configuration_profile(
        self, configuration_id: str, /, body: SessionConfigProfileBody
    ) -> SessionConfigProfile:
        """
        Update a session configuration profile

        Args:
            configuration_id (str):
            body (SessionConfigProfileBody):

        Returns:
            SessionConfigProfile: Successful Response

        """
    def delete_configuration_profile(self, configuration_id: str, /) -> None:
        """
        Delete a session configuration profile

        Args:
            configuration_id (str):

        """
    def compact_session(self, project_uid: str, session_uid: str, /) -> None:
        """
        Compact a session by clearing exposure processing results, including
        motion-corrected micrographs, extracted particle stacks, thumbnails and
        preview data. Retains original particle picks for re-extraction if the
        session is restored. The session can be restored as long as the original
        movies are not deleted.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
    def restore_session(self, project_uid: str, session_uid: str, /, body: LiveComputeResources) -> None:
        """
        Restore exposures of a compacted session. Starts Live preprocessing
        worker(s) on the specified lane. Each exposure is motion-corrected, the CTF
        is re-estimated, and particles are re-extracted using the original picks.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            body (LiveComputeResources):

        """
    def get_session_base_params(self) -> Any:
        """
        Get base session parameter details.

        Returns:
            Any: Successful Response

        """
    def star_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Star a session for the authenticated user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def unstar_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Unstar a session for the authenticated user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            Session: Successful Response

        """
    def export_session(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        export_movies: bool = False,
        export_ignored_exposures: bool = False,
        picker_type: Literal["blob", "template", "manual"] | None = None,
    ) -> None:
        """
        Write session results, including all motion-corrected exposures and
        extracted particles, to the project exports directory.

        The resulting folder contains .csg files. Copy the folder to another
        CryoSPARC project (while resolving symlinks) and import .csg files with the
        Import Result Group job.

        Example copy command::

            mkdir -p /path/to/projects/cs-project-b/imports/
            cp -rL /path/to/projects/cs-project-a/exports/S1 /path/to/projects/cs-project-b/imports/

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            export_movies (bool, optional): Include original movies. Defaults to False
            export_ignored_exposures (bool, optional): Include ignored exposures. Defaults to False
            picker_type (Literal['blob', 'template', 'manual'], optional): Include particles from this picker type instead of session selection. Defaults to None

        """
    def refresh_size(self, project_uid: str, session_uid: str, /) -> None:
        """
        Update session data size and statistics.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """

class ExposuresAPI(APINamespace):
    """
    Functions available in ``api.exposures``, e.g., ``api.exposures.find(...)``
    """
    def find(
        self,
        *,
        id: list[str] | None = None,
        uid: list[str] | None = None,
        session_uid: list[str] | None = None,
        project_uid: list[str] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        stage: list[
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
        | None = None,
        deleted: bool | None = False,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Exposure]:
        """
        List Live session exposures that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            session_uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            stage (List[Literal['go_to_found', 'found', 'check', 'motion', 'ctf', 'thumbs', 'pick', 'extract', 'extract_manual', 'ready', 'restoring', 'restoring_motion', 'restoring_thumbs', 'restoring_ctf', 'restoring_extract', 'restoring_extract_manual', 'compacted']], optional): Defaults to None
            deleted (bool, optional): Defaults to False
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Exposure]: Successful Response

        """
    def count(
        self,
        *,
        id: list[str] | None = None,
        uid: list[str] | None = None,
        session_uid: list[str] | None = None,
        project_uid: list[str] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        stage: list[
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
        | None = None,
        deleted: bool | None = False,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> int:
        """
        Count Live session exposures that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            session_uid (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            stage (List[Literal['go_to_found', 'found', 'check', 'motion', 'ctf', 'thumbs', 'pick', 'extract', 'extract_manual', 'ready', 'restoring', 'restoring_motion', 'restoring_thumbs', 'restoring_ctf', 'restoring_extract', 'restoring_extract_manual', 'compacted']], optional): Defaults to None
            deleted (bool, optional): Defaults to False
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            int: Successful Response

        """
    def find_one(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Find an exposure within a session by its numeric ID.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Successful Response

        """
    def reset_manual_reject_exposures(self, project_uid: str, session_uid: str, /) -> list[Exposure]:
        """
        Reset manual rejection status on all exposures in a session.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        Returns:
            List[Exposure]: List of exposures reset

        """
    def reset_all_exposures(self, project_uid: str, session_uid: str, /) -> None:
        """
        Reset all exposures in a session to initial state.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
    def reset_failed_exposures(self, project_uid: str, session_uid: str, /) -> None:
        """
        Reset all failed exposures in a session to initial state.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"

        """
    def reset_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Reset exposure to initial state.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Exposure that has been reset

        """
    def manual_reject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Manually reject exposure.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Rejected exposure

        """
    def manual_unreject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Manually unreject exposure.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Unrejected exposure

        """
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
    def toggle_manual_reject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Toggle manual rejection state on exposure.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):

        Returns:
            Exposure: Toggled exposure

        """
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
        Set a test exposure for the session, reprocess with the given parameters.
        If the session already has a test exposure, it is reset to the "ctf" stage
        and its test status is removed.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):
            body (LivePreprocessingParams):
            picker_type (Literal['blob', 'template']): Automatic picker types available in Live.

        Returns:
            Exposure: Reprocessed exposure

        """
    def add_manual_pick(
        self, project_uid: str, session_uid: str, exposure_uid: int, /, *, x_frac: float, y_frac: float
    ) -> Exposure:
        """
        Add a manual particle pick to an exposure.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):
            x_frac (float):
            y_frac (float):

        Returns:
            Exposure: Exposure with added manual picks

        """
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
        Remove manual particle pick from an exposure

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
    def get_individual_picks(
        self,
        project_uid: str,
        session_uid: str,
        exposure_uid: int,
        picker_type: Literal["blob", "template", "manual"],
        /,
    ) -> list[list[float]]:
        """
        Get list of particle picks for an exposure. Each item in the list is a list
        of four floating point numbers which represent the following:

        - x fractional coordinate of the pick
        - y fractional coordinate of the pick
        - Pick's Noise Cross-Correlation (NCC) score, if applicable, otherwise 0
        - Pick's Power score, if applicable, otherwise 0

        Args:
            project_uid (str): Project UID, e.g., "P3"
            session_uid (str): Session UID, e.g., "S3"
            exposure_uid (int):
            picker_type (Literal['blob', 'template', 'manual']):

        Returns:
            List[List[float]]: Successful Response

        """

class ProjectsAPI(APINamespace):
    """
    Functions available in ``api.projects``, e.g., ``api.projects.check_directory(...)``
    """
    def check_directory(self, *, path: str) -> str:
        """
        Check if a candidate project directory exists, and if it is readable and writeable.

        Args:
            path (str):

        Returns:
            str: Successful Response

        """
    def get_title_slug(self, *, title: str) -> str:
        """
        Get a URL-friendly version of a project title.

        Args:
            title (str):

        Returns:
            str: Successful Response

        """
    def find(
        self,
        *,
        id: list[str] | None = None,
        uid: list[str] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        project_dir: str | None = None,
        owner_user_id: str | None = None,
        users_with_access: list[str] | None = None,
        archived: bool | None = False,
        detached: bool | None = False,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Project]:
        """
        List projects available to the current user that match the supplied filters
        (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            project_dir (str, optional): Defaults to None
            owner_user_id (str, optional): Defaults to None
            users_with_access (List[str], optional): Defaults to None
            archived (bool, optional): Defaults to False
            detached (bool, optional): Defaults to False
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Project]: Successful Response

        """
    def create(self, *, title: str, description: str | None = None, parent_dir: str) -> Project:
        """
        Start a new empty project. Creates new subfolder in the parent directory
        with a generated name based on the provided title.

        Args:
            title (str):
            description (str, optional): Defaults to None
            parent_dir (str):

        Returns:
            Project: Successful Response

        """
    def count(
        self,
        *,
        project_dir: str | None = None,
        owner_user_id: str | None = None,
        archived: bool | None = False,
        detached: bool | None = False,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
        id: list[str] | None = None,
        uid: list[str] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        users_with_access: list[str] | None = None,
    ) -> int:
        """
        Count projects available to the current user that match the supplied filters
        (all if no filters specified).

        Args:
            project_dir (str, optional): Defaults to None
            owner_user_id (str, optional): Defaults to None
            archived (bool, optional): Defaults to False
            detached (bool, optional): Defaults to False
            order (Literal[-1, 1], optional): Defaults to 1
            after (str, optional): Defaults to None
            limit (int, optional): Defaults to 100
            id (List[str], optional): Defaults to None
            uid (List[str], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            users_with_access (List[str], optional): Defaults to None

        Returns:
            int: Successful Response

        """
    def set_title(self, project_uid: str, /, *, title: str) -> Project:
        """
        Set project title.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            title (str): Project title

        Returns:
            Project: Successful Response

        """
    def set_description(self, project_uid: str, /, description: str = "") -> Project:
        """
        Set project description. May include Markdown formatting.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            description (str, optional): Defaults to ''

        Returns:
            Project: Successful Response

        """
    def view(self, project_uid: str, /) -> Project:
        """
        Add a project to a user's recently viewed projects list.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
    def mkdir(self, project_uid: str, /, *, parents: bool = False, exist_ok: bool = False, path: str = "") -> str:
        """
        Create a subfolder in a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            parents (bool, optional): Create missing parent folders. Defaults to False
            exist_ok (bool, optional): No error if folder already exists. Defaults to False
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            str: Successful Response

        """
    def cp(self, project_uid: str, /, *, source: str, path: str = "") -> str:
        """
        Copy a file or folder to a project directory. Returns the absolute path of
        the copied file or folder. May only copy files within the project directory
        or from paths the current user is authorized to access by an administrator.

        Raises an error if the source is not readable or the destination is
        not writable.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            source (str): Source path, either absolute or relative to project directory
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            str: Successful Response

        """
    def symlink(self, project_uid: str, /, *, source: str, path: str = "") -> str:
        """
        Create a symbolic link in a project directory. May only create links to
        files or folders the user is authorized to access by an administrator.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            source (str): Source path, either absolute or relative to project directory
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            str: Successful Response

        """
    def preview_import_job_directory(self, project_uid: str, /, *, source: str) -> ExportBundlePreview:
        """
        Preview how much data would be transferred if the source
        were copied into this project

        Args:
            project_uid (str): Project UID, e.g., "P3"
            source (str): Source path, either absolute or relative to project directory

        Returns:
            ExportBundlePreview: Successful Response

        """
    def upload_file(self, project_uid: str, /, stream: Stream, *, overwrite: bool = False, path: str = "") -> str:
        """
        Upload a file to the project directory.

        Path may be specified as a filename, a relative path inside the project
        directory, or a full absolute path. When calling via HTTP, provide the
        contents of the file in the request body.

        Returns the uploaded absolute path.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            stream (Stream): A binary stream representing a Stream class instance
            overwrite (bool, optional): Defaults to False
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            str: Successful Response

        """
    def download_file(self, project_uid: str, /, *, path: str = "") -> Stream:
        """
        Download a file from the project directory.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            Stream: A binary stream representing a Stream class instance

        """
    def ls(self, project_uid: str, /, *, recursive: bool = False, path: str = "") -> list[str]:
        """
        List files in the project directory.

        Note that enabling ``recursive`` includes *both* subdirectories and their
        files in the list.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            recursive (bool, optional): Include files in all subfolders. Defaults to False
            path (str, optional): Relative path or absolute path within project directory. Defaults to ''

        Returns:
            List[str]: Successful Response

        """
    def get_job_register(self, project_uid: str, /) -> JobRegister:
        """
        Get the job register model for the project. The same for every project.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            JobRegister: Successful Response

        """
    def preview_delete(self, project_uid: str, /) -> DeleteProjectPreview:
        """
        Get the workspaces and jobs that would be affected when the project is deleted.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            DeleteProjectPreview: Successful Response

        """
    def find_one(self, project_uid: str, /) -> Project:
        """
        Get a project by its unique ID.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
    def delete(self, project_uid: str, /) -> None:
        """
        Start project deletion task. Will delete the project, its full directory,
        all associated workspaces, sessions, jobs and results. Kills any running
        jobs or active sessions in the project before deleting.

        The directory for an archived or detached project will not be deleted,
        but the project and all associated jobs will be fully removed from the
        web UI.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
    def get_directory(self, project_uid: str, /) -> str:
        """
        Get the project's absolute directory path with all environment variables resolved.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            str: Successful Response

        """
    def get_owner_id(self, project_uid: str, /) -> str:
        """
        Get user account ID for the owner of a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            str: Successful Response

        """
    def set_owner(self, project_uid: str, user_id: str, /) -> Project:
        """
        Set owner of the project to the user

        Args:
            project_uid (str): Project UID, e.g., "P3"
            user_id (str): User ID or email

        Returns:
            Project: Successful Response

        """
    def add_user_access(self, project_uid: str, user_id: str, /) -> Project:
        """
        Grant access to another user to view and edit the project.
        May only be called by project owners and administrators.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            user_id (str): User ID or email

        Returns:
            Project: Successful Response

        """
    def remove_user_access(self, project_uid: str, user_id: str, /) -> Project:
        """
        Remove a user's access from a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            user_id (str): User ID or email

        Returns:
            Project: Successful Response

        """
    def refresh_size(self, project_uid: str, /) -> None:
        """
        Start project size recalculation asynchronously.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
    def get_symlinks(self, project_uid: str, /) -> list[SymlinkInfo]:
        """
        Get all symbolic links in the project directory.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            List[SymlinkInfo]: Successful Response

        """
    def update_directory_symlinks(self, project_uid: str, /, *, prefix_cut: str, prefix_new: str) -> int:
        """
        Rewrite symbolic link target paths in a project directory, changing target
        paths with prefix ``prefix_cut`` to instead have prefix ``prefix_new``.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            prefix_cut (str):
            prefix_new (str):

        Returns:
            int: Number of symlinks updated

        """
    def set_default_param(self, project_uid: str, name: str, /, value: bool | float | str) -> Project:
        """
        Set a default value for a job parameter for use in a project. All new
        jobs created in the project will be initialized with this default value
        for the parameter.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            name (str):
            value (bool | int | float | str):

        Returns:
            Project: Successful Response

        """
    def clear_default_param(self, project_uid: str, name: str, /) -> Project:
        """
        Clear the per-project default value for a job parameter.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            name (str):

        Returns:
            Project: Successful Response

        """
    def claim_instance_ownership(self, project_uid: str, /, *, force: bool = False) -> None:
        """
        Claim ownership of an existing project with a missing lock file.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            force (bool, optional): Defaults to False

        """
    def claim_all_instance_ownership(self, *, force: bool = False) -> None:
        """
        Claim ownership of all projects in the instance. Use when upgrading
        from an older CryoSPARC version that did not support project locks.

        Args:
            force (bool, optional): Defaults to False

        """
    def archive(self, project_uid: str, /) -> None:
        """
        Archive this project. Archived projects are hidden in the web UI. They
        cannot be modified and their jobs cannot run. Once archived, an admin may
        safely move the project directory to a long-term storage location.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
    def unarchive(self, project_uid: str, /, *, path: str) -> Project:
        """
        Revert an archive operation.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            path (str):

        Returns:
            Project: Successful Response

        """
    def detach(self, project_uid: str, /) -> None:
        """
        Detach this project from this CryoSPARC instance, removing its lock file.
        Detached projects are hidden in the web UI and may be attached to other instances.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        """
    def attach(self, *, project_owner_id: str | None = None, path: str) -> Project:
        """
        Attach an existing project directory to this instance.

        The directory may not already be attached to any other CryoSPARC instance.
        A lock file will be created in the project directory to prevent it from
        being attached to multiple instances at the same time.

        Project will not be available to modify until the attach process completes,
        which may take some time depending on the size of the project.

        Once attach completes, the project will be visible and modifiable in the
        web UI. The project is assigned a new unique ID upon attach.

        Args:
            project_owner_id (str, optional): Assign project to user with this ID or email. Defaults to None
            path (str):

        Returns:
            Project: Successful Response

        """
    def accept(self, project_uid: str, /, *, path: str | None = None) -> Project:
        """
        Accept a failed attach for this project. Some job or workspace data may
        be inaccessible or lost after a failed attach, so use with caution.

        Accepting a failed project will not fix any underlying issues with the
        project directory, such as missing files or corrupted data. Please
        ensure that the project is valid and complete before accepting. Instead
        of accepting a failed attach, consider investigating why the attach failed,
        then detach the project, fix the issue and re-attach.

        Optionally provide a new project directory, if the project was moved there.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            path (str, optional): Defaults to None

        Returns:
            Project: Successful Response

        """
    def move(self, project_uid: str, /, *, path: str) -> None:
        """
        Move the project directory to a new location on the file system.

        Provide either the new folder name or the full new directory path.
        If the given path is a folder that already exists, the project directory
        will be moved inside it with the same name.

        Returns immediately, the move operation is performed in the background.

        May take a long time when moving projects between file systems.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            path (str):

        """
    def get_next_exposure_group_id(self, project_uid: str, /) -> int:
        """
        Get next exposure group ID

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            int: Successful Response

        """
    def cleanup_data(
        self,
        project_uid: str,
        /,
        *,
        workspace_uid: str | None = None,
        delete_non_final: bool = False,
        delete_statuses: list[JobStatus] = [],
        clear_non_final: bool = False,
        clear_categories: list[Category] = [],
        clear_types: list[str] = [],
        clear_statuses: list[JobStatus] = [],
        clear_preprocessing: bool = False,
        clear_intermediate_results: bool = False,
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
            clear_intermediate_results (bool, optional): Defaults to False

        """
    def add_tag(self, project_uid: str, tag_uid: str, /) -> Project:
        """
        Add a tag to a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            tag_uid (str): Tag UID, e.g., "T1"

        Returns:
            Project: Successful Response

        """
    def remove_tag(self, project_uid: str, tag_uid: str, /) -> Project:
        """
        Remove a tag from a project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            tag_uid (str): Tag UID, e.g., "T1"

        Returns:
            Project: Successful Response

        """
    def get_generate_intermediate_results_settings(self, project_uid: str, /) -> GenerateIntermediateResultsSettings:
        """
        Get generate intermediate result settings.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            GenerateIntermediateResultsSettings: Successful Response

        """
    def set_generate_intermediate_results_settings(
        self, project_uid: str, /, body: GenerateIntermediateResultsSettings
    ) -> Project:
        """
        Set settings for intermediate result generation.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            body (GenerateIntermediateResultsSettings):

        Returns:
            Project: Successful Response

        """
    def get_export_result_defaults(self, project_uid: str, /) -> ExportResultDefaults:
        """
        Get the project-level defaults for which export results get omitted.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            ExportResultDefaults: Successful Response

        """
    def set_export_result_defaults(self, project_uid: str, /, body: ExportResultDefaults) -> Project:
        """
        Set the project-level defaults for which export results get omitted.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            body (ExportResultDefaults):

        Returns:
            Project: Successful Response

        """
    def clear_intermediate_results(self, project_uid: str, /, *, always_keep_final: bool = True) -> None:
        """
        Remove intermediate results from the project.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            always_keep_final (bool, optional): Defaults to True

        """
    def get_generate_intermediate_results_job_types(self) -> list[str]:
        """
        Get intermediate result job types

        Returns:
            List[str]: Successful Response

        """
    def star_project(self, project_uid: str, /) -> Project:
        """
        Star a project for the authenticated user

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
    def unstar_project(self, project_uid: str, /) -> Project:
        """
        Unstar a project for the authenticated user

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """
    def reset_autodump(self, project_uid: str, /) -> Project:
        """
        Clear project directory write failures. After, CryoSPARC's scheduler will
        attempt to write modified jobs and workspaces to the project directory that
        previously could not be saved.

        Args:
            project_uid (str): Project UID, e.g., "P3"

        Returns:
            Project: Successful Response

        """

class TagsAPI(APINamespace):
    """
    Functions available in ``api.tags``, e.g., ``api.tags.find(...)``
    """
    def find(
        self,
        *,
        id: list[str] | None = None,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = None,
        created_by_user_id: str | None = None,
        type: list[Literal["general", "project", "workspace", "session", "job"]] | None = None,
        uid: str | None = None,
    ) -> list[Tag]:
        """
        List tags that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Limit number of results to return, maximum limit is 1000. Defaults to None
            created_by_user_id (str, optional): Defaults to None
            type (List[Literal['general', 'project', 'workspace', 'session', 'job']], optional): Defaults to None
            uid (str, optional): Defaults to None

        Returns:
            List[Tag]: Successful Response

        """
    def create(
        self,
        body: CreateTag,
        *,
        type: Literal["general", "project", "workspace", "session", "job"],
        created_by_workflow: str | None = None,
    ) -> Tag:
        """
        Create a new tag

        Args:
            body (CreateTag):
            type (Literal['general', 'project', 'workspace', 'session', 'job']):
            created_by_workflow (str, optional): Defaults to None

        Returns:
            Tag: Successful Response

        """
    def update(self, tag_uid: str, /, body: UpdateTag) -> Tag:
        """
        Update tag title, colour and/or description

        Args:
            tag_uid (str): Tag UID, e.g., "T1"
            body (UpdateTag):

        Returns:
            Tag: Successful Response

        """
    def delete(self, tag_uid: str, /) -> None:
        """
        Delete a tag

        Args:
            tag_uid (str): Tag UID, e.g., "T1"

        """
    def get_tags_by_type(self) -> dict[str, list[Tag]]:
        """
        Get all tags, organized by type

        Returns:
            Dict[str, List[Tag]]: Successful Response

        """
    def get_tag_count_by_type(self) -> dict[str, int]:
        """
        Get a count of all tags by type

        Returns:
            Dict[str, int]: Successful Response

        """

class NotificationsAPI(APINamespace):
    """
    Functions available in ``api.notifications``, e.g., ``api.notifications.find(...)``
    """
    def find(
        self,
        *,
        id: list[str] | None = None,
        project_uid: list[str] | None = None,
        job_uid: list[str] | None = None,
        active: bool | None = None,
        status: list[Literal["success", "primary", "warning", "danger"]] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Notification]:
        """
        List notifications that match the supplied query.

        Args:
            id (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            job_uid (List[str], optional): Defaults to None
            active (bool, optional): Defaults to None
            status (List[Literal['success', 'primary', 'warning', 'danger']], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Notification]: Successful Response

        """
    def count(
        self,
        *,
        id: list[str] | None = None,
        project_uid: list[str] | None = None,
        job_uid: list[str] | None = None,
        active: bool | None = None,
        status: list[Literal["success", "primary", "warning", "danger"]] | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
    ) -> int:
        """
        Count all notifications that match the supplied query.

        Args:
            id (List[str], optional): Defaults to None
            project_uid (List[str], optional): Defaults to None
            job_uid (List[str], optional): Defaults to None
            active (bool, optional): Defaults to None
            status (List[Literal['success', 'primary', 'warning', 'danger']], optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None

        Returns:
            int: Successful Response

        """
    def find_by_id(self, notification_id: str, /) -> Notification:
        """
        Finds a notification by its ID

        Args:
            notification_id (str):

        Returns:
            Notification: Successful Response

        """
    def deactivate_notification(self, notification_id: str, /) -> Notification:
        """
        Deactivates a notification

        Args:
            notification_id (str):

        Returns:
            Notification: Successful Response

        """

class BlueprintsAPI(APINamespace):
    """
    Functions available in ``api.blueprints``, e.g., ``api.blueprints.find(...)``
    """
    def find(
        self,
        *,
        id: list[str] | None = None,
        job_type: list[str] | None = None,
        created_by: str | None = None,
        pinned: bool | None = None,
        imported: bool | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Blueprint]:
        """
        List blueprints that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            job_type (List[str], optional): Defaults to None
            created_by (str, optional): Defaults to None
            pinned (bool, optional): Defaults to None
            imported (bool, optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Blueprint]: Successful Response

        """
    def create(self, body: Blueprint) -> Blueprint:
        """
        Create or import a blueprint for a job.

        Args:
            body (Blueprint):

        Returns:
            Blueprint: Successful Response

        """
    def draft_blueprint_from_job(self, *, project_uid: str, job_uid: str) -> Blueprint:
        """
        Create a blueprint from a job without saving it

        Args:
            project_uid (str): Project UID, e.g., "P3"
            job_uid (str): Job UID, e.g., "J3"

        Returns:
            Blueprint: Successful Response

        """
    def get_draft(self, job_type: str, /) -> Blueprint:
        """
        Args:
            job_type (str):

        Returns:
            Blueprint: Successful Response

        """
    def find_one(self, blueprint_id: str, /) -> Blueprint:
        """
        Get a blueprint by its id

        Args:
            blueprint_id (str):

        Returns:
            Blueprint: Successful Response

        """
    def edit(
        self,
        blueprint_id: str,
        /,
        parameters: dict[str, BlueprintParameter] | None = None,
        *,
        title: str | None = None,
        description: str | None = None,
        apply_title: bool | None = None,
        apply_description: bool | None = None,
        pinned: bool | None = None,
    ) -> Blueprint:
        """
        Edit an existing blueprint

        Args:
            blueprint_id (str):
            parameters (Dict[str, BlueprintParameter], optional): Defaults to None
            title (str, optional): Defaults to None
            description (str, optional): Defaults to None
            apply_title (bool, optional): Defaults to None
            apply_description (bool, optional): Defaults to None
            pinned (bool, optional): Defaults to None

        Returns:
            Blueprint: Successful Response

        """
    def delete(self, blueprint_id: str, /) -> None:
        """
        Delete a blueprint.

        Args:
            blueprint_id (str):

        """

class WorkflowsAPI(APINamespace):
    """
    Functions available in ``api.workflows``, e.g., ``api.workflows.find(...)``
    """
    def find(
        self,
        *,
        id: list[str] | None = None,
        title: str | None = None,
        category: list[str] | None = None,
        created_by: list[str] | None = None,
        pinned: bool | None = None,
        created_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        updated_at: tuple[datetime.datetime, datetime.datetime] | None = None,
        order: Literal[-1, 1] = 1,
        after: str | None = None,
        limit: int | None = 100,
    ) -> list[Workflow]:
        """
        List workflows that match the supplied filters (all if no filters specified).

        Args:
            id (List[str], optional): Defaults to None
            title (str, optional): Defaults to None
            category (List[str], optional): Defaults to None
            created_by (List[str], optional): Defaults to None
            pinned (bool, optional): Defaults to None
            created_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            updated_at (Tuple[datetime.datetime, datetime.datetime], optional): Defaults to None
            order (Literal[-1, 1], optional): 1 for ascending order, -1 for descending order. Defaults to 1
            after (str, optional): Cursor for pagination; only return results with id greater than (if order=1) or less than (if order=-1) this value. Defaults to None
            limit (int, optional): Defaults to 100

        Returns:
            List[Workflow]: Successful Response

        """
    def create(self, body: Workflow, *, forked: bool = False, rebuilt: bool = False) -> Workflow:
        """
        Create a workflow

        Args:
            body (Workflow):
            forked (bool, optional): Defaults to False
            rebuilt (bool, optional): Defaults to False

        Returns:
            Workflow: Successful Response

        """
    def find_one(self, workflow_id: str, /) -> Workflow:
        """
        Get a workflow by its id

        Args:
            workflow_id (str):

        Returns:
            Workflow: Successful Response

        """
    def edit(self, workflow_id: str, /, body: UpdateWorkflow) -> Workflow:
        """
        Update a workflow.

        Args:
            workflow_id (str):
            body (UpdateWorkflow):

        Returns:
            Workflow: Successful Response

        """
    def delete(self, workflow_id: str, /) -> None:
        """
        Delete a workflow

        Args:
            workflow_id (str):

        """
    def set_job_param(
        self, workflow_id: str, workflow_juid: str, param: str, /, body: WorkflowParameter
    ) -> WorkflowJob:
        """
        Set a parameter on a job in a workflow

        Args:
            workflow_id (str):
            workflow_juid (str):
            param (str):
            body (WorkflowParameter):

        Returns:
            WorkflowJob: Successful Response

        """
    def apply(self, body: ApplyWorkflowRequest, *, project_uid: str, workspace_uid: str) -> None:
        """
        Apply a workflow to a workspace

        Args:
            body (ApplyWorkflowRequest):
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"

        """

class ExternalAPI(APINamespace):
    """
    Functions available in ``api.external``, e.g., ``api.external.get_empiar_latest_entries(...)``
    """
    def get_empiar_latest_entries(self) -> dict[str, Any]:
        """
        Get information for the 6 most-recently submitted EMPIAR datasets.

        Returns:
            Dict[str, Any]: Successful Response

        """
    def get_emdb_latest_entries(self) -> list[dict[str, Any]]:
        """
        Get information for the 6 most-recently released EMDB entries.

        Returns:
            List[Dict[str, Any]]: Successful Response

        """
    def get_discuss_top(self) -> dict[str, Any]:
        """
        Get top weekly topics in the CryoSPARC discussion forum.

        Returns:
            Dict[str, Any]: Successful Response

        """
    def get_discuss_categories(self) -> dict[str, Any]:
        """
        Get categories available in the CryoSPARC discussion forum.

        Returns:
            Dict[str, Any]: Successful Response

        """
    def get_tutorials(self) -> dict[str, Any]:
        """
        Get information about available CryoSPARC tutorials.

        Returns:
            Dict[str, Any]: Successful Response

        """
    def get_changelog(self) -> dict[str, Any]:
        """
        Get the latest CryoSPARC changelog.

        Returns:
            Dict[str, Any]: Successful Response

        """

class BenchmarksAPI(APINamespace):
    """
    Functions available in ``api.benchmarks``, e.g., ``api.benchmarks.get_reference_benchmarks(...)``
    """
    def get_reference_benchmarks(self) -> list[ReferencePerformanceBenchmark]:
        """
        Get reference benchmarks for comparison against newly-measured benchmarks.

        Returns:
            List[ReferencePerformanceBenchmark]: Successful Response

        """
    def get_benchmark(self, project_uid: str, job_uid: str, benchmark_type: str, /) -> PerformanceBenchmark:
        """
        Get a performance benchmark measured by the given benchmarking job.

        Args:
            project_uid (str):
            job_uid (str):
            benchmark_type (str):

        Returns:
            PerformanceBenchmark: Successful Response

        """

class JobgroupsAPI(APINamespace):
    """
    Functions available in ``api.job_groups``, e.g., ``api.job_groups.get(...)``
    """
    def get(self, project_uid: str, workspace_uid: str, group_id: int, /) -> JobGroup:
        """
        Get a job group by ID.

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            group_id (int):

        Returns:
            JobGroup: Successful Response

        """
    def edit(self, project_uid: str, workspace_uid: str, group_id: int, /, body: JobGroupUpdate) -> Workspace:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            group_id (int):
            body (JobGroupUpdate):

        Returns:
            Workspace: Successful Response

        """
    def delete(self, project_uid: str, workspace_uid: str, group_id: int, /) -> Workspace:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            group_id (int):

        Returns:
            Workspace: Successful Response

        """
    def create(self, project_uid: str, workspace_uid: str, /, body: JobGroupCreate) -> JobGroup:
        """
        Create a job group

        Args:
            project_uid (str): Project UID, e.g., "P3"
            workspace_uid (str): Workspace UID, e.g., "W3"
            body (JobGroupCreate):

        Returns:
            JobGroup: Successful Response

        """

class DatasourcesAPI(APINamespace):
    """
    Functions available in ``api.datasources``, e.g., ``api.datasources.set_name(...)``
    """
    def set_name(self, project_uid: str, datasource_id: str, /, name: str) -> Datasource:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            datasource_id (str):
            name (str):

        Returns:
            Datasource: Successful Response

        """
    def set_path(self, project_uid: str, datasource_id: str, /, path: str) -> Datasource:
        """
        Args:
            project_uid (str): Project UID, e.g., "P3"
            datasource_id (str):
            path (str):

        Returns:
            Datasource: Successful Response

        """

class DeveloperAPI(APINamespace):
    """
    Functions available in ``api.developer``, e.g., ``api.developer.get_developers(...)``
    """
    def get_developers(self) -> list[str]:
        """
        :meta private:

        Returns:
            List[str]: Successful Response

        """
    def reload(self) -> bool:
        """
        Restarts API service and scheduler.

        :meta private:

        Returns:
            bool: Successful Response

        """
    def save_job_registers(self, *, developer_name: str | None = None) -> list[JobRegister]:
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
    job_groups: JobgroupsAPI
    """``api.job_groups`` functions"""
    datasources: DatasourcesAPI
    """``api.datasources`` functions"""
    developer: DeveloperAPI
    """``api.developer`` functions"""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        auth: Auth | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 300,
    ) -> None:
        """
        Args:
            base_url (str, optional): Base URL for CryoSPARC API server. Defaults to None
            auth (Auth, optional): Auth token or email/password. Defaults to None
            headers (Dict[str, str], optional): Default HTTP headers to include in requests. Defaults to None
            timeout (float, optional): Request timeout in seconds. Defaults to 300

        """
    def __call__(self, *, auth: Auth | None = None) -> None:
        """
        Args:
            auth (Auth, optional): Auth token or email/password. Password must be encoded as SHA256. Defaults to None

        """
    def read_root(self) -> Hello:
        """
        Get basic information about the API, including running CryoSPARC version.

        Returns:
            Hello: Successful Response

        """
    def health(self) -> str:
        """
        Health check endpoint to verify that the API is running and responsive.

        Returns:
            str: Successful Response

        """
    def login(
        self,
        *,
        expires_in: float = 1209600,
        username: str,
        password: str,
        grant_type: str | None = None,
        scope: str = "",
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> Token:
        """
        Login form. Note that plain-text passwords are not accepted; they must be
        hashed as SHA256.

        Args:
            expires_in (float, optional): Token expiration time (in seconds). Can be up to 1 year. Defaults to 1209600
            username (str):
            password (str):
            grant_type (str, optional): Defaults to None
            scope (str, optional): Defaults to ''
            client_id (str, optional): Defaults to None
            client_secret (str, optional): Defaults to None

        Returns:
            Token: Successful Response

        """
    def keycloak_login(self, *, keycloak_access_token: str) -> Token:
        """
        Login from app using a Keycloak access token. Only available if Keycloak
        integration is enabled.

        Additional documentation coming soon.

        :meta private:

        Args:
            keycloak_access_token (str):

        Returns:
            Token: Successful Response

        """
    def verify_app_session(self, body: AppSession) -> str:
        """
        Internal function to verify an app session and return a JWT access token for the user.

        :meta private:

        Args:
            body (AppSession):

        Returns:
            str: Successful Response

        """
    def job_register(self) -> JobRegister:
        """
        Information and metadata about job types available in this instance.

        Returns:
            JobRegister: Successful Response

        """
    def start_and_migrate(self, *, license_id: str) -> None:
        """
        Start up CryoSPARC for the first time after installation or after an upgrade.
        Triggers any migrations, refreshes connected targets and updates on-disk
        project files to the latest schema.

        Args:
            license_id (str):

        """
    def test(self, delay: float, /) -> str:
        """
        Sleep for the specified number of seconds and returns a value to indicate
        endpoint is working correctly.

        Args:
            delay (float):

        Returns:
            str: Successful Response

        """
