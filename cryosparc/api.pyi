# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_client.py

import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .dataset import Dataset
from .models.api_request import AppSession, SHA256Password
from .models.api_response import (
    BrowseFileResponse,
    DeleteProjectPreview,
    DeleteWorkspacePreview,
    GetFinalResultsResponse,
    Hello,
    WorkspaceAncestorUidsResponse,
    WorkspaceDescendantUidsResponse,
)
from .models.asset import GridFSAsset, GridFSFile
from .models.auth import Token
from .models.diagnostics import RuntimeDiagnostics
from .models.event import CheckpointEvent, Event, ImageEvent, InteractiveEvent, TextEvent
from .models.exposure import Exposure
from .models.external import ExternalOutputSpec
from .models.job import Job, JobStatus
from .models.job_register import JobRegister
from .models.job_spec import Category, InputSpec, OutputResult, OutputSpec
from .models.license import LicenseInstance, UpdateTag
from .models.notification import Notification
from .models.project import GenerateIntermediateResultsSettings, Project, ProjectSymlink
from .models.scheduler_lane import SchedulerLane
from .models.scheduler_target import Cluster, Node, SchedulerTarget
from .models.service import LoggingService, ServiceLogLevel
from .models.session import (
    DataManagementStats,
    ExposureGroup,
    ExposureGroupUpdate,
    LiveComputeResources,
    Session,
    TemplateSelectionThreshold,
)
from .models.session_config_profile import SessionConfigProfile, SessionConfigProfileBody
from .models.session_params import LiveAbinitParams, LiveClass2DParams, LivePreprocessingParams, LiveRefineParams
from .models.session_spec import SessionStatus
from .models.tag import Tag
from .models.user import User
from .models.workspace import Workspace
from .stream import Stream

Auth = Union[str, Tuple[str, str]]
"""
Auth token or email/password.
"""

class APINamespace:
    def __init__(self, http_client: Any = None) -> None: ...

class ConfigNamespace(APINamespace):
    """
    Methods available in api.config, e.g., api.config.get_instance_uid(...)
    """
    def get_instance_uid(self) -> str:
        """
        Gets this CryoSPARC instance's unique UID.
        """
        ...
    def generate_new_instance_uid(self, *, force_takeover_projects: bool = False) -> str:
        """
        Generates a new uid for the CryoSPARC instance
        If force_takeover_projects is True, overwrites existing lockfiles,
        otherwise if force_takeover_projects is False, only creates lockfile in projects that don't already have one
        """
        ...
    def set_default_job_priority(self, value: int) -> Any:
        """
        Job priority
        """
        ...
    def get_version(self) -> str:
        """
        Gets the current CryoSPARC version (with patch suffix, if available)
        """
        ...
    def get_system_info(self) -> Dict[str, Any]:
        """
        System information related to the CryoSPARC application
        """
        ...
    def get(self, name: str, /, *, default: Any = "<<UNDEFINED>>") -> Any:
        """
        Gets config collection entry value for the given variable name.
        """
        ...
    def write(self, name: str, /, value: Any = ..., *, set_on_insert_only: bool = False) -> Any:
        """
        Sets config collection entry. Specify `set_on_insert_only` to prevent
        overwriting when the value already exists.
        """
        ...

class InstanceNamespace(APINamespace):
    """
    Methods available in api.instance, e.g., api.instance.get_update_tag(...)
    """
    def get_update_tag(self) -> UpdateTag | None:
        """
        Gets information about updating to the next CryoSPARC version, if one is available.
        """
        ...
    def live_enabled(self) -> bool:
        """
        Checks if CryoSPARC Live is enabled
        """
        ...
    def ecl_enabled(self) -> bool:
        """
        Checks if embedded CryoSPARC Live is enabled
        """
        ...
    def link_log(
        self,
        type: str,
        /,
        data: Any = ...,
        *,
        user_id: Optional[str] = ...,
        project_uid: Optional[str] = ...,
        job_uid: Optional[str] = ...,
        job_type: Optional[str] = ...,
    ) -> None: ...
    def get_license_usage(self) -> List[LicenseInstance]: ...
    def browse_files(self, *, abs_path_glob: str) -> BrowseFileResponse:
        """
        Backend for the file browser in the cryosparc UI.
        .. note::
                abs_path_glob could have shell vars in it (i.e. $HOME, $SCRATCH)
                0. expand vars
                1. if abs path is already a dir: just list the dir
                2. else: expand the glob
                3. if the glob returns empty: return empty
        """
        ...
    def get_service_log(
        self,
        service: LoggingService,
        /,
        *,
        days: int = 7,
        date: Optional[str] = ...,
        log_name: str = "",
        func_name: str = "",
        level: Optional[ServiceLogLevel] = ...,
        max_lines: Optional[int] = ...,
    ) -> str:
        """
        Gets cryosparc service logs, filterable by date, name, function, and level
        """
        ...
    def get_runtime_diagnostics(self) -> RuntimeDiagnostics:
        """
        Gets runtime diagnostics for the CryoSPARC instance
        """
        ...

class CacheNamespace(APINamespace):
    """
    Methods available in api.cache, e.g., api.cache.get(...)
    """
    def get(self, key: str, /, *, namespace: Optional[str] = ...) -> Any:
        """
        Returns None if the value is not set or expired
        """
        ...
    def set(self, key: str, /, value: Any = ..., *, namespace: Optional[str] = ..., ttl: int = 60) -> None:
        """
        Sets key to the given value, with a ttl (Time-to-Live) in seconds
        """
        ...

class UsersNamespace(APINamespace):
    """
    Methods available in api.users, e.g., api.users.admin_exists(...)
    """
    def admin_exists(self) -> bool:
        """
        Returns True if there exists at least one user with admin privileges, False
        otherwise
        """
        ...
    def count(self, *, role: Optional[Literal["user", "admin"]] = ...) -> int: ...
    def table(self) -> str:
        """
        Show a table of all CryoSPARC user accounts
        """
        ...
    def me(self) -> User:
        """
        Returns the current user
        """
        ...
    def find_one(self, user_id: str, /) -> User:
        """
        Finds a user with a matching user ID or email
        """
        ...
    def update(
        self,
        user_id: str,
        /,
        *,
        email: Optional[str] = ...,
        username: Optional[str] = ...,
        first_name: Optional[str] = ...,
        last_name: Optional[str] = ...,
    ) -> User:
        """
        Updates a user's general details. other params will only be set if they are
        not empty.
        """
        ...
    def delete(self, user_id: str, /) -> None:
        """
        Removes a user from the CryoSPARC. Only authenticated admins may do this.
        """
        ...
    def get_role(self, user_id: str, /) -> Literal["user", "admin"]:
        """
        Returns "admin" if the user has admin privileges, "user" otherwise.
        """
        ...
    def create(
        self,
        password: Optional[SHA256Password] = ...,
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
        """
        ...
    def request_reset_password(self, user_id: str, /) -> None:
        """
        Generates a password reset token for a user with the given email. The token
        will appear in the Admin > User Management interface.
        """
        ...
    def register(self, user_id: str, /, body: SHA256Password, *, token: str) -> None:
        """
        Registers user with a token (unauthenticated).
        """
        ...
    def reset_password(self, user_id: str, /, body: SHA256Password, *, token: str) -> None:
        """
        Resets password function with a token (unauthenticated). password is expected
        as a sha256 hash.
        """
        ...
    def set_role(self, user_id: str, /, role: Literal["user", "admin"]) -> User:
        """
        Changes a user's from between "user" and "admin". Only admins may do this.
        This revokes all access tokens for the given used ID.
        """
        ...
    def get_my_state_var(self, key: str, /) -> Any:
        """
        Retrieves a user's state variable such as "licenseAccepted" or
        "recentProjects"
        """
        ...
    def set_allowed_prefix_dir(self, user_id: str, /, allowed_prefix: str) -> User:
        """
        Sets directories that users are allowed to query from the file browser.
        ``allowed_prefix`` is the path of the directory the user can query inside.
        (must start with "/", and must be an absolute path)
        Returns True if successful
        """
        ...
    def get_state_var(self, user_id: str, key: str, /) -> Any:
        """
        Retrieves a given user's state variable such as "licenseAccepted" or
        "recentProjects"
        """
        ...
    def set_state_var(self, user_id: str, key: str, /, value: Any) -> User:
        """
        Sets a property of the user's state
        """
        ...
    def unset_state_var(self, user_id: str, key: str, /) -> User:
        """
        Deletes a property of the user's state
        """
        ...
    def get_lanes(self, user_id: str, /) -> List[str]:
        """
        Gets the lanes a user has access to
        """
        ...
    def set_lanes(self, user_id: str, /, lanes: List[str]) -> User:
        """
        Restrict lanes the given user ID may to queue to. Only admins and account
        owners may access this function.
        """
        ...

class ResourcesNamespace(APINamespace):
    """
    Methods available in api.resources, e.g., api.resources.find_lanes(...)
    """
    def find_lanes(self) -> List[SchedulerLane]:
        """
        Finds lanes that are registered with the master scheduler.
        """
        ...
    def add_lane(self, body: SchedulerLane) -> SchedulerLane:
        """
        Adds a new lane to the master scheduler.
        """
        ...
    def find_lane(self, name: str, /, *, type: Literal["node", "cluster", None] = ...) -> SchedulerLane:
        """
        Finds a lane registered to the master scheduler with a given name and optional type.
        """
        ...
    def remove_lane(self, name: str, /) -> None:
        """
        Removes the specified lane and any targets assigned under the lane in the
        master scheduler.
        """
        ...
    def find_targets(self, *, lane: Optional[str] = ...) -> List[SchedulerTarget]:
        """
        Finds a list of targets that are registered with the master scheduler.
        """
        ...
    def find_nodes(self, *, lane: Optional[str] = ...) -> List[SchedulerTarget[Node]]:
        """
        Finds a list of targets with type "node" that are registered with the master scheduler.
        These correspond to discrete worker hostname accessible over SSH.
        """
        ...
    def add_node(self, body: SchedulerTarget[Node]) -> SchedulerTarget[Node]:
        """
        Adds a node or updates an existing node. Updates existing node if they share
        share the same name.
        """
        ...
    def find_clusters(self, *, lane: Optional[str] = ...) -> List[SchedulerTarget[Cluster]]:
        """
        Finds a list of targets with type "cluster" that are registered with the master scheduler.
        These are multi-node clusters managed by workflow managers like SLURM or PBS and are accessible via submission script.
        """
        ...
    def add_cluster(self, body: SchedulerTarget[Cluster]) -> SchedulerTarget[Cluster]:
        """
        Adds a cluster or updates an existing cluster. Updates existing cluster if
        they share share the same name.
        """
        ...
    def find_target_by_hostname(self, hostname: str, /) -> SchedulerTarget:
        """
        Finds a target with a given hostname.
        """
        ...
    def find_target_by_name(self, name: str, /) -> SchedulerTarget:
        """
        Finds a target with a given name.
        """
        ...
    def find_node(self, name: str, /) -> SchedulerTarget[Node]:
        """
        Finds a node with a given name.
        """
        ...
    def remove_node(self, name: str, /) -> None:
        """
        Removes a target worker node from the master scheduler
        """
        ...
    def find_cluster(self, name: str, /) -> SchedulerTarget[Cluster]:
        """
        Finds a cluster with a given name.
        """
        ...
    def remove_cluster(self, name: str, /) -> None:
        """
        Removes the specified cluster/lane and any targets assigned under the lane
        in the master scheduler

        Note: This will remove any worker node associated with the specified cluster/lane.
        """
        ...
    def find_cluster_script(self, name: str, /) -> str:
        """
        Finds the cluster script for a cluster with a given name.
        """
        ...
    def find_cluster_template_vars(self, name: str, /) -> List[str]:
        """
        Computes and retrieves all variable names defined in cluster templates.
        """
        ...
    def find_cluster_template_custom_vars(self, name: str, /) -> List[str]:
        """
        Computes and retrieves all custom variables names defined in cluster templates
        (i.e., all variables not in the internal list of known variable names).
        """
        ...
    def update_node_lane(self, name: str, /, lane: str) -> SchedulerTarget[Node]:
        """
        Changes the lane on the given target (assumed to exist). Target type must
        match lane type.
        """
        ...
    def refresh_nodes(self) -> Any:
        """
        Asynchronously access target worker nodes. Load latest CPU, RAM and GPU
        info.
        """
        ...
    def verify_cluster(self, name: str, /) -> str:
        """
        Ensures cluster has been properly configured by executing a generic 'info'
        command
        """
        ...
    def update_cluster_custom_vars(self, name: str, /, value: Dict[str, str]) -> SchedulerTarget[Cluster]:
        """
        Changes the custom cluster variables on the given target (assumed to exist)
        """
        ...
    def update_target_cache_path(self, name: str, /, value: Optional[str]) -> SchedulerTarget:
        """
        Changes the cache path on the given target (assumed to exist)
        """
        ...

class AssetsNamespace(APINamespace):
    """
    Methods available in api.assets, e.g., api.assets.find(...)
    """
    def find(self, *, project_uid: Optional[str] = ..., job_uid: Optional[str] = ...) -> List[GridFSFile]:
        """
        List assets associated with projects or jobs on the given instance.
        Typically returns files creating during job runs, including plots and metadata.
        """
        ...
    def upload(
        self,
        project_uid: str,
        job_uid: str,
        /,
        stream: Stream,
        *,
        filename: Optional[str] = ...,
        format: Union[
            Literal["txt", "csv", "html", "json", "xml", "bild", "bld", "log"],
            Literal["pdf", "gif", "jpg", "jpeg", "png", "svg"],
            None,
        ] = ...,
    ) -> GridFSAsset:
        """
        Upload a new asset associated with the given project/job. When calling
        via HTTP, provide the contents of the file in the request body. At least
        one of filename or format must be provided.
        """
        ...
    def download(self, id: str = "000000000000000000000000", /) -> Stream:
        """
        Download the asset with the given ID. When calling via HTTP, file contents
        will be in the response body.
        """
        ...
    def find_one(self, id: str = "000000000000000000000000", /) -> GridFSFile:
        """
        Retrive the full details for an asset with the given ID.
        """
        ...

class JobsNamespace(APINamespace):
    """
    Methods available in api.jobs, e.g., api.jobs.find(...)
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        project_uid: Optional[List[str]] = ...,
        workspace_uid: Optional[List[str]] = ...,
        uid: Optional[List[str]] = ...,
        type: Optional[List[str]] = ...,
        status: Optional[List[JobStatus]] = ...,
        category: Optional[List[Category]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        waiting_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        failed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        deleted: Optional[bool] = False,
    ) -> List[Job]:
        """
        Finds all jobs that match the supplied query
        """
        ...
    def delete_many(self, project_job_uids: List[Tuple[str, str]], *, force: bool = False) -> Any:
        """
        Deletes the given jobs. Ignores protected jobs if `force` is `True`.
        """
        ...
    def count(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        project_uid: Optional[List[str]] = ...,
        workspace_uid: Optional[List[str]] = ...,
        uid: Optional[List[str]] = ...,
        type: Optional[List[str]] = ...,
        status: Optional[List[JobStatus]] = ...,
        category: Optional[List[Category]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        waiting_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        failed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        deleted: Optional[bool] = False,
    ) -> int:
        """
        Counts number of jobs that match the supplied query.
        """
        ...
    def get_active_count(self) -> int:
        """
        Counts number of active jobs.
        """
        ...
    def clone_many(
        self,
        project_uid: str,
        /,
        job_uids: List[str],
        *,
        workspace_uid: Optional[str] = ...,
        new_workspace_title: Optional[str] = ...,
    ) -> List[Job]:
        """
        Clones the given list of jobs. If any jobs are related, it will try to
        re-create the input connections between the cloned jobs (but maintain the
        same connections to jobs that were not cloned)
        """
        ...
    def get_chain(self, project_uid: str, /, *, start_job_uid: str, end_job_uid: str) -> List[str]:
        """
        Finds the chain of jobs between start job to end job.
        A job chain is the intersection of the start job's descendants and the end job's
        ancestors.
        """
        ...
    def clone_chain(
        self,
        project_uid: str,
        /,
        *,
        start_job_uid: str,
        end_job_uid: str,
        workspace_uid: Optional[str] = ...,
        new_workspace_title: Optional[str] = ...,
    ) -> List[Job]:
        """
        Clones jobs that directly descend from the start job UID up to the end job UID.
        """
        ...
    def get_final_results(self, project_uid: str, /) -> GetFinalResultsResponse:
        """
        Gets all final results within a project, along with the ancestors and non-ancestors of those jobs.
        """
        ...
    def find_one(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Finds the job.
        """
        ...
    def delete(self, project_uid: str, job_uid: str, /, *, force: bool = False) -> Any:
        """
        Deletes a job. Will kill (if running) and clearing the job before deleting.
        """
        ...
    def get_directory(self, project_uid: str, job_uid: str, /) -> str:
        """
        Gets the job directory for a given job.
        """
        ...
    def get_log(self, project_uid: str, job_uid: str, /) -> str:
        """
        Returns contents of the job.log file. Returns empty string if job.log does not exist.
        """
        ...
    def get_log_path(self, project_uid: str, job_uid: str, /) -> str: ...
    def get_output_fields(
        self, project_uid: str, job_uid: str, output_name: str, /, dtype_params: Dict[str, Any] = {}
    ) -> List[Tuple[str, str]]:
        """
        Expected dataset column definitions for given job output, excluding passthroughs.
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
        created_by_job_uid: Optional[str] = ...,
        enable_bench: bool = False,
    ) -> Job:
        """
        Creates a new job with the given type in the project/workspace

        To see all available job types and their parameters, see the `GET projects/{project_uid}:register` endpoint
        """
        ...
    def create_external_result(self, project_uid: str, workspace_uid: str, /, body: ExternalOutputSpec) -> Job:
        """
        Create an external result with the given specification. Returns an external
        job with the given output ready to be saved. Used with cryosparc-tools
        """
        ...
    def get_status(self, project_uid: str, job_uid: str, /) -> JobStatus:
        """
        Gets the status of a job.
        """
        ...
    def view(self, project_uid: str, workspace_uid: str, job_uid: str, /) -> Job:
        """
        Adds a project, workspace and job uid to a user's recently viewed jobs list
        """
        ...
    def set_param(self, project_uid: str, job_uid: str, param: str, /, *, value: Any) -> Job:
        """
        Sets the given job parameter to the value
        """
        ...
    def clear_param(self, project_uid: str, job_uid: str, param: str, /) -> Job:
        """
        Resets the given parameter to its default value.
        """
        ...
    def load_input(
        self,
        project_uid: str,
        job_uid: str,
        input_name: str,
        /,
        *,
        force_join: Union[bool, str] = "auto",
        slots: Union[Literal["default", "passthrough", "all"], List[str]] = "default",
    ) -> Dataset:
        """
        Load job input dataset. Raises exception if no inputs are connected.
        """
        ...
    def load_output(
        self,
        project_uid: str,
        job_uid: str,
        output_name: str,
        /,
        *,
        version: Union[int, str] = "F",
        slots: Union[Literal["default", "passthrough", "all"], List[str]] = "default",
    ) -> Dataset:
        """
        Load job output dataset. Raises exception if output is empty or does not exists.
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
        filename: Optional[str] = ...,
        version: int = 0,
    ) -> Job:
        """
        Save job output dataset. Job must be running or waiting.
        """
        ...
    def connect(
        self, project_uid: str, job_uid: str, input_name: str, /, *, source_job_uid: str, source_output_name: str
    ) -> Job:
        """
        Connects the input slot on the child job to the output group on the
        parent job.
        """
        ...
    def disconnect_all(self, project_uid: str, job_uid: str, input_name: str, /) -> Job: ...
    def disconnect(self, project_uid: str, job_uid: str, input_name: str, connection_index: int, /) -> Job:
        """
        Removes connected inputs on the given input.

        Optionally specify an index to disconnect a specific connection.

        Optionally provide specific results to disconnect from matching connections (other results will be preserved).
        """
        ...
    def find_output_result(self, project_uid: str, job_uid: str, output_name: str, result_name: str, /) -> OutputResult:
        """
        Get a job's low-level output result.
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
        source_job_uid: str,
        source_output_name: str,
        source_result_name: str,
    ) -> Job:
        """
        Adds or replaces a result within an input connection with the given output result from a different job.
        """
        ...
    def disconnect_result(
        self, project_uid: str, job_uid: str, input_name: str, connection_index: int, result_name: str, /
    ) -> Job:
        """
        Removes an output result connected within the given input connection.
        """
        ...
    def add_external_input(self, project_uid: str, job_uid: str, input_name: str, /, body: InputSpec) -> Job:
        """
        Add or replace an external job's input. This action is available while the
        job is building, running or waiting for results.
        """
        ...
    def add_external_output(self, project_uid: str, job_uid: str, output_name: str, /, body: OutputSpec) -> Job:
        """
        Add or replace an external job's output. This action is available while the
        job is building, running or waiting for results.
        """
        ...
    def enqueue(
        self,
        project_uid: str,
        job_uid: str,
        /,
        *,
        lane: Optional[str] = ...,
        hostname: Optional[str] = ...,
        gpus: List[int] = [],
        no_check_inputs_ready: bool = False,
    ) -> Job:
        """
        Adds the job to the queue for the given worker lane (default lane if not specified)
        """
        ...
    def recalculate_intermediate_results_size(self, project_uid: str, job_uid: str, /) -> Any:
        """
        For a job, find intermediate results and recalculate their total size.
        """
        ...
    def recalculate_project_intermediate_results_size(self, project_uid: str, /) -> Any:
        """
        Recaclulates intermediate result sizes for all jobs in a project.
        """
        ...
    def clear_intermediate_results(self, project_uid: str, job_uid: str, /, *, always_keep_final: bool = True) -> Any:
        """
        Removes intermediate results from the job
        """
        ...
    def export_output_results(
        self, project_uid: str, job_uid: str, output_name: str, /, result_names: Optional[List[str]] = ...
    ) -> Any:
        """
        Prepares a job's output for import to another project or instance. Creates a folder in the project directory → exports subfolder,
        then links the output's associated files there..
        Note that the returned .csg file's parent folder must be manually copied with symlinks resolved into the target project folder before importing.
        """
        ...
    def export_job(self, project_uid: str, job_uid: str, /) -> Any:
        """
        Start export for the job into the project's exports directory
        """
        ...
    def get_output_result_path(
        self, project_uid: str, job_uid: str, output_name: str, result_name: str, /, *, version: Union[int, str] = "F"
    ) -> str:
        """
        Get the absolute path for a job output's dataset or volume density.
        """
        ...
    def interactive_post(
        self, project_uid: str, job_uid: str, /, body: Dict[str, Any], *, endpoint: str, timeout: int = 10
    ) -> Any:
        """
        Sends a message to an interactive job.
        """
        ...
    def mark_running(
        self, project_uid: str, job_uid: str, /, *, status: Literal["running", "waiting"] = "running"
    ) -> Job:
        """
        Indicate that an external job is running or waiting.
        """
        ...
    def mark_completed(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Mark a killed or failed job, or an active external job, as completed.
        """
        ...
    def mark_failed(self, project_uid: str, job_uid: str, /, *, error: Optional[str] = ...) -> Job:
        """
        Manually mark a job as failed.
        """
        ...
    def add_event_log(
        self, project_uid: str, job_uid: str, /, text: str, *, type: Literal["text", "warning", "error"] = "text"
    ) -> TextEvent:
        """
        Add the message to the target job's event log.
        """
        ...
    def get_event_logs(
        self, project_uid: str, job_uid: str, /, *, checkpoint: Optional[int] = ...
    ) -> List[Union[Event, CheckpointEvent, TextEvent, ImageEvent, InteractiveEvent]]:
        """
        Gets all event logs for a job.

        Note: this may return a lot of documents.
        """
        ...
    def add_image_log(
        self, project_uid: str, job_uid: str, /, images: List[GridFSAsset], *, text: str, flags: List[str] = ["plots"]
    ) -> ImageEvent:
        """
        Add an image or figure to the target job's event log.
        """
        ...
    def add_checkpoint(self, project_uid: str, job_uid: str, /, meta: Dict[str, Any]) -> CheckpointEvent:
        """
        Add a checkpoint the target job's event log.
        """
        ...
    def recalculate_size(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Recalculates the size of a given job's directory.
        """
        ...
    def clear(self, project_uid: str, job_uid: str, /, *, force: bool = False) -> Job:
        """
        Clears a job to get it back to building state (do not clear params or inputs).
        """
        ...
    def clear_many(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        project_uid: Optional[List[str]] = ...,
        workspace_uid: Optional[List[str]] = ...,
        uid: Optional[List[str]] = ...,
        type: Optional[List[str]] = ...,
        status: Optional[List[JobStatus]] = ...,
        category: Optional[List[Category]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        waiting_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        failed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        deleted: Optional[bool] = False,
    ) -> List[Job]:
        """
        Clears all jobs that matches the query.
        """
        ...
    def clone(
        self,
        project_uid: str,
        job_uid: str,
        /,
        *,
        workspace_uid: Optional[str] = ...,
        created_by_job_uid: Optional[str] = ...,
    ) -> Job:
        """
        Creates a new job as a clone of the provided job.
        """
        ...
    def kill(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Kills a running job
        """
        ...
    def set_final_result(self, project_uid: str, job_uid: str, /, *, is_final_result: bool) -> Job:
        """
        Marks a job as a final result. A job marked as a final result and its ancestor jobs are protected during data cleanup.
        """
        ...
    def set_title(self, project_uid: str, job_uid: str, /, *, title: str) -> Job:
        """
        Sets job title.
        """
        ...
    def set_description(self, project_uid: str, job_uid: str, /, description: str) -> Job:
        """
        Sets job description.
        """
        ...
    def set_priority(self, project_uid: str, job_uid: str, /, *, priority: int) -> Job:
        """
        Sets job priority
        """
        ...
    def set_cluster_custom_vars(self, project_uid: str, job_uid: str, /, cluster_custom_vars: Dict[str, Any]) -> Job:
        """
        Sets cluster custom variables for job
        """
        ...
    def get_active_licenses_count(self) -> int:
        """
        Gets number of acquired licenses for running jobs
        """
        ...
    def get_types(self) -> Any:
        """
        Gets list of available job types
        """
        ...
    def get_categories(self) -> Any:
        """
        Gets job types by category
        """
        ...
    def find_ancestor_uids(self, project_uid: str, job_uid: str, /, *, workspace_uid: Optional[str] = ...) -> List[str]:
        """
        Finds all ancestors of a single job and return a list of their UIDs
        """
        ...
    def find_descendant_uids(
        self, project_uid: str, job_uid: str, /, *, workspace_uid: Optional[str] = ...
    ) -> List[str]:
        """
        Find the list of all job UIDs that this job is an ancestor of based
        on its outputs.
        """
        ...
    def link_to_workspace(self, project_uid: str, job_uid: str, workspace_uid: str, /) -> Job:
        """
        Adds a job to a workspace.
        """
        ...
    def unlink_from_workspace(self, project_uid: str, job_uid: str, workspace_uid: str, /) -> Job:
        """
        Removes a job from a workspace.
        """
        ...
    def move(self, project_uid: str, job_uid: str, /, *, from_workspace_uid: str, to_workspace_uid: str) -> Job:
        """
        Moves a job from one workspace to another.
        """
        ...
    def update_directory_symlinks(self, project_uid: str, job_uid: str, /, *, prefix_cut: str, prefix_new: str) -> int:
        """
        Rewrites all symbolic links in the job directory, modifying links prefixed with `prefix_cut` to instead be prefixed with `prefix_new`.
        """
        ...
    def add_tag(self, project_uid: str, job_uid: str, tag_uid: str, /) -> Job:
        """
        Tags a job with the given tag.
        """
        ...
    def remove_tag(self, project_uid: str, job_uid: str, tag_uid: str, /) -> Job:
        """
        Removes the given tag a job.
        """
        ...
    def import_job(self, project_uid: str, workspace_uid: str, /, *, path: str = "") -> Any:
        """
        Imports the exported job directory into the project. Exported job
        directory must be copied to the target project directory with all its symbolic links resolved.
        By convention, the exported job directory should be located in the project directory → exports subfolder
        """
        ...
    def import_result_group(
        self, project_uid: str, workspace_uid: str, /, *, lane: Optional[str] = ..., path: str = ""
    ) -> Job:
        """
        Creates and enqueues an Import Result Group job with the given path
        """
        ...
    def star_job(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Stars a job for a user
        """
        ...
    def unstar_job(self, project_uid: str, job_uid: str, /) -> Job:
        """
        Unstars a job for a user
        """
        ...

class WorkspacesNamespace(APINamespace):
    """
    Methods available in api.workspaces, e.g., api.workspaces.find(...)
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        uid: Optional[List[str]] = ...,
        project_uid: Optional[List[str]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        deleted: Optional[bool] = False,
    ) -> List[Workspace]:
        """
        List all workspaces. Specify a filter to list all workspaces in a specific
        project.

        Examples:

            >>> api.workspaces.find(project_uid="P1")
        """
        ...
    def count(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        uid: Optional[List[str]] = ...,
        project_uid: Optional[List[str]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        deleted: Optional[bool] = False,
    ) -> int:
        """
        Count all workspaces. Use a query to count workspaces in a specific project.
        """
        ...
    def preview_delete(self, project_uid: str, workspace_uid: str, /) -> DeleteWorkspacePreview:
        """
        Get a list of jobs that would be removed when the given workspace is deleted.
        """
        ...
    def find_one(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Find a specific workspace in a project
        """
        ...
    def delete(self, project_uid: str, workspace_uid: str, /) -> Any:
        """
        Marks the workspace as "deleted". Deletes jobs that are only linked to this workspace
        and no other workspace.
        """
        ...
    def create(
        self,
        project_uid: str,
        /,
        *,
        title: str,
        description: Optional[str] = ...,
        created_by_job_uid: Optional[str] = ...,
    ) -> Workspace:
        """
        Create a new workspace
        """
        ...
    def set_title(self, project_uid: str, workspace_uid: str, /, *, title: str) -> Workspace:
        """
        Set title of a workspace
        """
        ...
    def set_description(self, project_uid: str, workspace_uid: str, /, description: str) -> Workspace:
        """
        Set description of a workspace
        """
        ...
    def view(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Adds a workspace uid to a user's recently viewed workspaces list.
        """
        ...
    def add_tag(self, project_uid: str, workspace_uid: str, tag_uid: str, /) -> Workspace:
        """
        Tag the given workspace with the given tag.
        """
        ...
    def remove_tag(self, project_uid: str, workspace_uid: str, tag_uid: str, /) -> Workspace:
        """
        Removes a tag from a workspace.
        """
        ...
    def clear_intermediate_results(
        self, project_uid: str, workspace_uid: str, /, *, always_keep_final: bool = False
    ) -> Any:
        """
        Remove intermediate results from a workspace.
        """
        ...
    def find_workspace_ancestor_uids(
        self, project_uid: str, workspace_uid: str, /, *, job_uids: List[str]
    ) -> WorkspaceAncestorUidsResponse:
        """
        Finds ancestors of jobs in the workspace
        """
        ...
    def find_workspace_descendant_uids(
        self, project_uid: str, workspace_uid: str, /, *, job_uids: List[str]
    ) -> WorkspaceDescendantUidsResponse:
        """
        Finds descendants of jobs in the workspace
        """
        ...
    def star_workspace(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Stars a workspace for a user
        """
        ...
    def unstar_workspace(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Unstars a workspace for a user
        """
        ...

class SessionsNamespace(APINamespace):
    """
    Methods available in api.sessions, e.g., api.sessions.find(...)
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        uid: Optional[List[str]] = ...,
        session_uid: Optional[List[str]] = ...,
        project_uid: Optional[List[str]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        cleared_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        status: Optional[List[SessionStatus]] = ...,
        deleted: Optional[bool] = False,
    ) -> List[Session]:
        """
        Lists all sessions (optionally, in a project)
        """
        ...
    def count(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        uid: Optional[List[str]] = ...,
        session_uid: Optional[List[str]] = ...,
        project_uid: Optional[List[str]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        cleared_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        status: Optional[List[SessionStatus]] = ...,
        deleted: Optional[bool] = False,
    ) -> int:
        """
        Counts all sessions in a project
        """
        ...
    def find_one(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Finds a session
        """
        ...
    def delete(self, project_uid: str, session_uid: str, /) -> Any:
        """
        Sets the session document as "deleted"
        Will throw an error if any undeleted jobs exist within the session.
        """
        ...
    def create(
        self,
        project_uid: str,
        /,
        *,
        title: str,
        description: Optional[str] = ...,
        created_by_job_uid: Optional[str] = ...,
    ) -> Session:
        """
        Creates a new session
        """
        ...
    def find_exposure_groups(self, project_uid: str, session_uid: str, /) -> List[ExposureGroup]:
        """
        Finds all exposure groups in a session.
        """
        ...
    def create_exposure_group(self, project_uid: str, session_uid: str, /) -> ExposureGroup:
        """
        Creates an exposure group for a session.
        """
        ...
    def find_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> ExposureGroup:
        """
        Finds an exposure group with a specific id for a session.
        """
        ...
    def update_exposure_group(
        self, project_uid: str, session_uid: str, exposure_group_id: int, /, body: ExposureGroupUpdate
    ) -> ExposureGroup:
        """
        Updates properties of an exposure group.
        """
        ...
    def delete_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> Session:
        """
        Deletes an exposure group from a session.
        """
        ...
    def finalize_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> ExposureGroup:
        """
        Finalizes an exposure group.
        """
        ...
    def start(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Builds and starts a CryoSPARC Live Session. Builds file engines based on file
        engine parameters in the session doc, builds phase one workers based on lane
        parameters in the session doc.
        """
        ...
    def pause(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Pauses a CryoSPARC Live Session. Gracefully stops and kills all phase one workers, file engines and phase two jobs
        """
        ...
    def update_compute_configuration(
        self, project_uid: str, session_uid: str, /, body: LiveComputeResources
    ) -> LiveComputeResources:
        """
        Updates compute configuration for a session.
        """
        ...
    def add_tag(self, project_uid: str, session_uid: str, tag_uid: str, /) -> Session:
        """
        Tags a session with the given tag.
        """
        ...
    def remove_tag(self, project_uid: str, session_uid: str, tag_uid: str, /) -> Session:
        """
        Removes the given tag from a session.
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
        """
        ...
    def update_session_picker(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        activate_picker_type: Literal["blob", "template", "deep"],
        use_thresholds: bool = True,
    ) -> Session:
        """
        Updates a session's picker.
        """
        ...
    def update_attribute_threshold(
        self,
        project_uid: str,
        session_uid: str,
        attribute: str,
        /,
        *,
        min_val: Optional[float] = ...,
        max_val: Optional[float] = ...,
    ) -> Session:
        """
        Updates thresholds for a given attribute.
        """
        ...
    def clear_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Deletes all file engine documents (removing all previously known files and
        max timestamps), all Phase 1 Worker jobs and all associated
        exposure documents.
        """
        ...
    def view(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Adds a project, workspace and job uid to a user's recently viewed sessions list
        """
        ...
    def setup_phase2_class2D(self, project_uid: str, session_uid: str, /, *, force_restart: bool = True) -> Job:
        """
        Setup streaming 2D classification job for a session.
        """
        ...
    def enqueue_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueues streaming 2D Classification job for a session
        """
        ...
    def stop_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stops streaming 2D Classification job for a session
        """
        ...
    def clear_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clears streaming 2D Classification job for a session
        """
        ...
    def update_phase2_class2D_params(self, project_uid: str, session_uid: str, /, body: LiveClass2DParams) -> Session:
        """
        Updates streaming 2D Classification job params for session
        """
        ...
    def toggle_class2d_template(self, project_uid: str, session_uid: str, template_idx: int, /) -> Session:
        """
        Inverts selected template for the streaming 2D Classification job of a job
        """
        ...
    def toggle_all_class2d_templates(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Inverts all templates for a session's streaming 2D classification job
        """
        ...
    def select_all_class2d_templates(
        self, project_uid: str, session_uid: str, direction: Literal["select", "deselect"], /
    ) -> Session:
        """
        Sets all templates in the session's streaming 2D Classification job
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
        """
        ...
    def select_class2d_templates_with_thresholds(
        self, project_uid: str, session_uid: str, /, template_selection_thresholds: List[TemplateSelectionThreshold]
    ) -> Session:
        """
        Selects all templates above or below an index in a template creation job for a session
        """
        ...
    def start_extract_manual(self, project_uid: str, session_uid: str, /) -> None:
        """
        Extracts manual picks from a session
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
        """
        ...
    def reset_attribute_threshold(self, project_uid: str, session_uid: str, attribute: str, /) -> Session:
        """
        Resets attribute threshold for a session
        """
        ...
    def reset_all_attribute_thresholds(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Resets all attribute thresholds for a session
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
        override_particle_diameter_A: Optional[float] = ...,
        uid_lte: Optional[int] = ...,
    ) -> Job:
        """
        Setup template creation class2D job for a session
        """
        ...
    def set_template_creation_job(
        self, project_uid: str, session_uid: str, /, *, job_uid: str, template_creation_project_uid: Optional[str] = ...
    ) -> Session:
        """
        Set template creation class2D job for a session
        """
        ...
    def clear_template_creation_class2D(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clears template creation class2D job for a session
        """
        ...
    def toggle_picking_template(self, project_uid: str, session_uid: str, template_idx: int, /) -> Session:
        """
        Toggles template for template creation job at a specific index for a session's template creation job
        """
        ...
    def toggle_all_picking_templates(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Toggles templates for all templates for a session's template creation job
        """
        ...
    def select_all_picking_templates(
        self, project_uid: str, session_uid: str, direction: Literal["select", "deselect"], /
    ) -> Session:
        """
        Selects or deselects all templates for a template creation job in a session
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
        """
        ...
    def select_picking_templates_with_thresholds(
        self, project_uid: str, session_uid: str, /, template_selection_thresholds: List[TemplateSelectionThreshold]
    ) -> Session:
        """
        Selects all templates above or below an index in a template creation job for a session
        """
        ...
    def setup_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Setup Ab-Initio Reconstruction job for a session
        """
        ...
    def set_phase2_abinit_job(self, project_uid: str, session_uid: str, /, *, job_uid: str) -> Session:
        """
        Sets a Live Ab-Initio Reconstruction job for the session. May specify any job with volume outputs.
        """
        ...
    def enqueue_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueues Ab-Initio Reconstruction job for a session
        """
        ...
    def clear_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clears Ab-Initio Reconstruction job for a session
        """
        ...
    def update_phase2_abinit_params(self, project_uid: str, session_uid: str, /, body: LiveAbinitParams) -> Session:
        """
        Updates Ab-Initio Reconstruction parameters for the session
        """
        ...
    def select_phase2_abinit_volume(self, project_uid: str, session_uid: str, /, *, volume_name: str) -> Session:
        """
        Selects a volume for an Ab-Initio Reconstruction job in a session
        """
        ...
    def stop_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stops an Ab-Initio Reconstruction job for a session.
        """
        ...
    def clear_phase2_refine(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Clears streaming Homogenous Refinement job for a session
        """
        ...
    def setup_phase2_refine(self, project_uid: str, session_uid: str, /) -> Job: ...
    def enqueue_phase2_refine(self, project_uid: str, session_uid: str, /) -> Job:
        """
        Enqueues a streaming Homogenous Refinement job for a session
        """
        ...
    def stop_phase2_refine(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stops a streaming Homogenous Refinement job for a session
        """
        ...
    def update_phase2_refine_params(self, project_uid: str, session_uid: str, /, body: LiveRefineParams) -> Session:
        """
        Updates parameters for a streaming Homogenous Refinement job for a session
        """
        ...
    def create_and_enqueue_dump_particles(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        picker_type: Optional[Literal["blob", "template", "manual"]] = ...,
        num_mics: Optional[int] = ...,
        uid_lte: Optional[int] = ...,
        test_only: bool = False,
    ) -> Job:
        """
        Creates and enqueues a dump particles job for a session
        """
        ...
    def create_and_enqueue_dump_exposures(
        self, project_uid: str, session_uid: str, /, *, export_ignored: bool = False
    ) -> Job:
        """
        Creates and enqueues a dump exposures job for a session
        """
        ...
    def get_data_management_stats(self, project_uid: str, /) -> Dict[str, DataManagementStats]:
        """
        Gets the data management stats of all sessions in a project.
        """
        ...
    def mark_session_completed(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Marks the session as completed
        """
        ...
    def change_session_data_management_state(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        datatype: Literal["micrographs", "raw", "particles", "metadata", "thumbnails"],
        status: Literal["active", "archiving", "archived", "deleted", "deleting", "missing", "calculating"],
    ) -> Session:
        """
        Updates data management status of a session's datatype
        """
        ...
    def update_session_datatype_sizes(self, project_uid: str, session_uid: str, /) -> int:
        """
        Updates the session's data_management information with the current size of each datatype.
        """
        ...
    def get_datatype_size(
        self,
        project_uid: str,
        session_uid: str,
        datatype: Literal["micrographs", "raw", "particles", "metadata", "thumbnails"],
        /,
    ) -> int:
        """
        Gets the total size of a datatype inside a session in bytes.
        """
        ...
    def delete_live_datatype(
        self,
        project_uid: str,
        session_uid: str,
        datatype: Literal["micrographs", "raw", "particles", "metadata", "thumbnails"],
        /,
    ) -> Job | None:
        """
        Deletes a specific datatype inside a session.
        """
        ...
    def update_all_sessions_datatype_sizes(self, project_uid: str, /) -> None:
        """
        Asynchronously updates the datatype sizes of all sessions within a project
        """
        ...
    def get_datatype_file_paths(
        self,
        project_uid: str,
        session_uid: str,
        datatype: Literal["micrographs", "raw", "particles", "metadata", "thumbnails"],
        /,
    ) -> List[str]:
        """
        Gets all the file paths associated with a specific datatype inside a session as a list
        """
        ...
    def get_configuration_profiles(self) -> List[SessionConfigProfile]:
        """
        Gets all session configuration profiles
        """
        ...
    def create_configuration_profile(self, body: SessionConfigProfileBody) -> SessionConfigProfile:
        """
        Creates a session configuration profile
        """
        ...
    def apply_configuration_profile(self, project_uid: str, session_uid: str, /, *, configuration_id: str) -> Session:
        """
        Applies a configuration profile to a session, overwriting its resources, parameters, and exposure group
        """
        ...
    def update_configuration_profile(
        self, configuration_id: str, /, body: SessionConfigProfileBody
    ) -> SessionConfigProfile:
        """
        Updates a configuration profile
        """
        ...
    def delete_configuration_profile(self, configuration_id: str, /) -> None:
        """
        Deletes a configuration profile
        """
        ...
    def compact_session(self, project_uid: str, session_uid: str, /) -> Any:
        """
        Compacts a session by clearing each exposure group and its related files for each exposure in the session.
        Also clears gridfs data.
        """
        ...
    def restore_session(self, project_uid: str, session_uid: str, /, body: LiveComputeResources) -> Any:
        """
        Restores exposures of a compacted session. Starts phase 1 worker(s) on the specified lane to restore each exposure by re-processing starting from motion correction, skipping the
        picking stage.
        """
        ...
    def get_session_base_params(self) -> Any:
        """
        Gets base session parameters
        """
        ...
    def star_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stars a session for a user
        """
        ...
    def unstar_session(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Stars a session for a user
        """
        ...

class ExposuresNamespace(APINamespace):
    """
    Methods available in api.exposures, e.g., api.exposures.find(...)
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        uid: Optional[List[str]] = ...,
        session_uid: Optional[List[str]] = ...,
        project_uid: Optional[List[str]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
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
        ] = ...,
        deleted: Optional[bool] = False,
    ) -> List[Exposure]: ...
    def reset_manual_reject_exposures(self, project_uid: str, session_uid: str, /) -> List[Exposure]:
        """
        Resets manual rejection status on all exposures in a session.
        """
        ...
    def reset_all_exposures(self, project_uid: str, session_uid: str, /) -> None:
        """
        Resets all exposures in a session to initial state.
        """
        ...
    def reset_failed_exposures(self, project_uid: str, session_uid: str, /) -> None:
        """
        Resets all failed exposures in a session to initial state.
        """
        ...
    def reset_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Resets exposure to intial state.
        """
        ...
    def manual_reject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Manually rejects exposure.
        """
        ...
    def manual_unreject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Manually unrejects exposure.
        """
        ...
    def toggle_manual_reject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure:
        """
        Toggles manual rejection state on exposure.
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
        """
        ...
    def add_manual_pick(
        self, project_uid: str, session_uid: str, exposure_uid: int, /, *, x_frac: float, y_frac: float
    ) -> Exposure:
        """
        Adds a manual pick to an exposure.
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
        """
        ...

class ProjectsNamespace(APINamespace):
    """
    Methods available in api.projects, e.g., api.projects.check_directory(...)
    """
    def check_directory(self, *, path: str) -> str:
        """
        Checks if a candidate project directory exists, and if it is readable and writeable.
        """
        ...
    def get_title_slug(self, *, title: str) -> str:
        """
        Returns a slugified version of a project title.
        """
        ...
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        uid: Optional[List[str]] = ...,
        project_dir: Optional[str] = ...,
        owner_user_id: Optional[str] = ...,
        deleted: Optional[bool] = False,
        archived: Optional[bool] = ...,
        detached: Optional[bool] = ...,
        hidden: Optional[bool] = ...,
    ) -> List[Project]:
        """
        Finds projects matching the filter.
        """
        ...
    def create(self, *, title: str, description: Optional[str] = ..., parent_dir: str) -> Project:
        """
        Creates a new project, project directory and creates a new document in
        the project collection
        """
        ...
    def count(
        self,
        uid: Optional[List[str]] = ...,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        project_dir: Optional[str] = ...,
        owner_user_id: Optional[str] = ...,
        deleted: Optional[bool] = False,
        archived: Optional[bool] = ...,
        detached: Optional[bool] = ...,
        hidden: Optional[bool] = ...,
    ) -> int:
        """
        Counts the number of projects matching the filter.
        """
        ...
    def set_title(self, project_uid: str, /, *, title: str) -> Project:
        """
        Sets the title of a project.
        """
        ...
    def set_description(self, project_uid: str, /, description: str) -> Project:
        """
        Sets the description of a project.
        """
        ...
    def view(self, project_uid: str, /) -> Project:
        """
        Adds a project uid to a user's recently viewed projects list.
        """
        ...
    def mkdir(self, project_uid: str, /, *, parents: bool = False, exist_ok: bool = False, path: str = "") -> str:
        """
        Create a directory in the project directory at the given path.
        """
        ...
    def cp(self, project_uid: str, /, *, source: str, path: str = "") -> str:
        """
        Copy the source file or directory to the project directory at the given
        path. Returns the absolute path of the copied file.
        """
        ...
    def symlink(self, project_uid: str, /, *, source: str, path: str = "") -> str:
        """
        Create a symlink from the source path in the project directory at the given path.
        """
        ...
    def upload_file(self, project_uid: str, /, stream: Stream, *, overwrite: bool = False, path: str = "") -> str:
        """
        Upload a file to the project directory at the given path. Returns absolute
        path of the uploaded file.

        Path may be specified as a filename, a relative path inside the project
        directory, or a full absolute path.
        """
        ...
    def download_file(self, project_uid: str, /, *, path: str = "") -> Stream:
        """
        Download a file from the project directory at the given path.
        """
        ...
    def ls(self, project_uid: str, /, *, recursive: bool = False, path: str = "") -> List[str]:
        """
        List files in the project directory. Note that enabling recursive will
        include parent directories in the result.
        """
        ...
    def get_job_register(self, project_uid: str, /) -> JobRegister:
        """
        Gets the job register model for the project. The same for every project.
        """
        ...
    def preview_delete(self, project_uid: str, /) -> DeleteProjectPreview:
        """
        Retrieves the workspaces and jobs that would be affected when the project is deleted.
        """
        ...
    def find_one(self, project_uid: str, /) -> Project:
        """
        Finds a project by its UID
        """
        ...
    def delete(self, project_uid: str, /) -> Any:
        """
        Starts project deletion task. Will delete the project, its full directory, and all associated workspaces, sessions, jobs and results.
        """
        ...
    def get_directory(self, project_uid: str, /) -> str:
        """
        Gets the project's absolute directory with all environment variables in the
        path resolved
        """
        ...
    def get_owner_id(self, project_uid: str, /) -> str:
        """
        Get user account ID for the owner of a project.
        """
        ...
    def set_owner(self, project_uid: str, user_id: str, /) -> Project:
        """
        Sets owner of the project to the user
        """
        ...
    def add_user_access(self, project_uid: str, user_id: str, /) -> Project:
        """
        Grants access to another user to view and edit the project.
        May only be called by project owners and administrators.
        """
        ...
    def remove_user_access(self, project_uid: str, user_id: str, /) -> Project:
        """
        Removes a user's access from a project.
        """
        ...
    def refresh_size(self, project_uid: str, /) -> Any:
        """
        Starts project size recalculation asynchronously.
        """
        ...
    def get_symlinks(self, project_uid: str, /) -> List[ProjectSymlink]:
        """
        Gets all symbolic links in the project directory
        """
        ...
    def set_default_param(self, project_uid: str, name: str, /, value: Union[bool, int, float, str]) -> Project:
        """
        Sets a default value for a parameter name globally for the project
        """
        ...
    def clear_default_param(self, project_uid: str, name: str, /) -> Project:
        """
        Clears the per-project default value for a parameter name.
        """
        ...
    def claim_instance_ownership(self, project_uid: str, /, *, force: bool = False) -> None: ...
    def claim_all_instance_ownership(self, *, force: bool = False) -> None:
        """
        Claims ownership of all projects in instance. Call when upgrading from an older CryoSPARC version that did not support project locks.
        """
        ...
    def archive(self, project_uid: str, /) -> Project:
        """
        Archives a project. This means that the project can no longer be modified
        and jobs cannot be created or run. Once archived, the project directory may
        be safely moved to long-term storage.
        """
        ...
    def unarchive(self, project_uid: str, /, *, path: str) -> Project:
        """
        Reverses archive operation.
        """
        ...
    def detach(self, project_uid: str, /) -> Project:
        """
        Detaches a project, removing its lockfile. This hides the project from the interface and allows other
        instances to attach and run this project.
        """
        ...
    def attach(self, *, path: str) -> Project:
        """
        Attaches a project directory at a specified path and writes a new
        lockfile. Must be run on a project directory without a lockfile.
        """
        ...
    def move(self, project_uid: str, /, *, path: str) -> Project:
        """
        Renames the project directory for a project. Provide either the new
        directory name or the full new directory path.
        """
        ...
    def get_next_exposure_group_id(self, project_uid: str, /) -> int:
        """
        Gets next exposure group ID
        """
        ...
    def cleanup_data(
        self,
        project_uid: str,
        /,
        *,
        workspace_uid: Optional[str] = ...,
        delete_non_final: bool = False,
        delete_statuses: List[JobStatus] = [],
        clear_non_final: bool = False,
        clear_categories: List[Category] = [],
        clear_types: List[str] = [],
        clear_statuses: List[JobStatus] = [],
    ) -> Any:
        """
        Cleanup project or workspace data, clearing/deleting jobs based on final result status, sections, types, or job status
        """
        ...
    def add_tag(self, project_uid: str, tag_uid: str, /) -> Project:
        """
        Tags a project with the given tag.
        """
        ...
    def remove_tag(self, project_uid: str, tag_uid: str, /) -> Project:
        """
        Removes the given tag from a project.
        """
        ...
    def get_generate_intermediate_results_settings(self, project_uid: str, /) -> GenerateIntermediateResultsSettings:
        """
        Gets generate intermediate result settings.
        """
        ...
    def set_generate_intermediate_results_settings(
        self, project_uid: str, /, body: GenerateIntermediateResultsSettings
    ) -> Project:
        """
        Sets settings for intermediate result generation.
        """
        ...
    def clear_intermediate_results(self, project_uid: str, /, *, always_keep_final: bool = True) -> Any:
        """
        Removes intermediate results from the project.
        """
        ...
    def get_generate_intermediate_results_job_types(self) -> List[str]:
        """
        Gets intermediate result job types
        """
        ...
    def star_project(self, project_uid: str, /) -> Project:
        """
        Stars a project for a user
        """
        ...
    def unstar_project(self, project_uid: str, /) -> Project:
        """
        Unstars a project for a user
        """
        ...
    def reset_autodump(self, project_uid: str, /) -> Project:
        """
        Clear project directory write failures. After calling this endpoint,
        CryoSPARC's scheduler will attempt to write modified jobs and workspaces to
        the project directory that previously could not be saved.
        """
        ...

class TagsNamespace(APINamespace):
    """
    Methods available in api.tags, e.g., api.tags.find(...)
    """
    def find(
        self,
        *,
        order: int = 1,
        after: Optional[str] = ...,
        limit: int = 100,
        created_by_user_id: Optional[str] = ...,
        type: Optional[List[Literal["general", "project", "workspace", "session", "job"]]] = ...,
        uid: Optional[str] = ...,
    ) -> List[Tag]:
        """
        Finds tags that match the given query.
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
        ] = ...,
        description: Optional[str] = ...,
        created_by_workflow: Optional[str] = ...,
        title: Optional[str],
    ) -> Tag:
        """
        Creates a new tag
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
        ] = ...,
        description: Optional[str] = ...,
        title: Optional[str],
    ) -> Tag:
        """
        Updates the title, colour and/or description of the given tag UID
        """
        ...
    def delete(self, tag_uid: str, /) -> None:
        """
        Deletes a given tag
        """
        ...
    def get_tags_by_type(self) -> Dict[str, List[Tag]]:
        """
        Gets all tags as a dictionary, where the types are the keys
        """
        ...
    def get_tag_count_by_type(self) -> Dict[str, int]:
        """
        Gets a count of all tags by type
        """
        ...

class NotificationsNamespace(APINamespace):
    """
    Methods available in api.notifications, e.g., api.notifications.deactivate_notification(...)
    """
    def deactivate_notification(self, notification_id: str, /) -> Notification:
        """
        Deactivates a notification
        """
        ...

class BlueprintsNamespace(APINamespace):
    """
    Methods available in api.blueprints, e.g., api.blueprints.create_blueprint(...)
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
        """
        ...
    def edit_blueprint(
        self, blueprint_id: str, /, schema: Dict[str, Any], *, project_uid: str, job_uid: str, job_type: str
    ) -> None:
        """
        For cryosparc app only
        """
        ...
    def delete_blueprint(self, blueprint_id: str, /, *, job_type: str) -> None:
        """
        For cryosparc app only
        """
        ...
    def apply_blueprint(
        self, blueprint_id: str, /, schema: Dict[str, Any], *, project_uid: str, job_uid: str, job_type: str
    ) -> None:
        """
        For cryosparc app only
        """
        ...

class WorkflowsNamespace(APINamespace):
    """
    Methods available in api.workflows, e.g., api.workflows.create_workflow(...)
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
        """
        ...
    def edit_workflow(self, workflow_id: str, /, schema: Dict[str, Any]) -> None:
        """
        For cryosparc app only
        """
        ...
    def delete_workflow(self, workflow_id: str, /) -> None:
        """
        For cryosparc app only
        """
        ...
    def apply_workflow(self, workflow_id: str, /, schema: Dict[str, Any]) -> None:
        """
        For cryosparc app only
        """
        ...

class ExternalNamespace(APINamespace):
    """
    Methods available in api.external, e.g., api.external.get_empiar_latest_entries(...)
    """
    def get_empiar_latest_entries(self) -> Dict[str, Any]: ...
    def get_emdb_latest_entries(self) -> List[Dict[str, Any]]: ...
    def get_discuss_top(self) -> Dict[str, Any]: ...
    def get_discuss_categories(self) -> Dict[str, Any]: ...
    def get_tutorials(self) -> Dict[str, Any]: ...
    def get_changelog(self) -> Dict[str, Any]: ...

class DeveloperNamespace(APINamespace):
    """
    Methods available in api.developer, e.g., api.developer.get_developers(...)
    """
    def get_developers(self) -> List[str]: ...
    def reload(self) -> bool:
        """
        Restarts API service and scheduler.
        """
        ...
    def save_job_registers(self, *, developer_name: Optional[str] = ...) -> List[JobRegister]:
        """
        Re-saves the current job registers. Call this when restarting the api
        service without executing the /startup route, as we do during developer
        reloads.
        """
        ...

class APIClient:
    """
    Top-level API client class. e.g., ``api.read_root(...)``
    or ``api.config.get_instance_uid(...)``
    """

    config: ConfigNamespace
    instance: InstanceNamespace
    cache: CacheNamespace
    users: UsersNamespace
    resources: ResourcesNamespace
    assets: AssetsNamespace
    jobs: JobsNamespace
    workspaces: WorkspacesNamespace
    sessions: SessionsNamespace
    exposures: ExposuresNamespace
    projects: ProjectsNamespace
    tags: TagsNamespace
    notifications: NotificationsNamespace
    blueprints: BlueprintsNamespace
    workflows: WorkflowsNamespace
    external: ExternalNamespace
    developer: DeveloperNamespace

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        auth: Union[str, tuple[str, str], None] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = ...,
    ) -> None: ...
    def __call__(self, *, auth: Union[str, tuple[str, str], None] = None) -> Any: ...
    def read_root(self) -> Hello: ...
    def health(self) -> str: ...
    def login(
        self,
        *,
        expires_in: float = 1209600,
        username: str,
        password: str,
        grant_type: Optional[str] = ...,
        scope: str = "",
        client_id: Optional[str] = ...,
        client_secret: Optional[str] = ...,
    ) -> Token:
        """
        Login form. Note that plain-text passwords are not accepted; they must be
        hashed as SHA256.
        """
        ...
    def keycloak_login(self, *, keycloak_access_token: str) -> Token: ...
    def verify_app_session(self, body: AppSession) -> str: ...
    def job_register(self) -> JobRegister:
        """
        Get a specification of available job types and their schemas.
        """
        ...
    def start_and_migrate(self, *, license_id: str) -> Any:
        """
        Start up CryoSPARC for the first time and perform database migrations
        """
        ...
    def test(self, delay: float, /) -> str:
        """
        Sleep for the specified number of seconds and returns a value to indicate
        endpoint is working correctly.
        """
        ...
