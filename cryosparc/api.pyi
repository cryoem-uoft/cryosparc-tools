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
from .models.auth import Token
from .models.diagnostics import RuntimeDiagnostics
from .models.event import CheckpointEvent, Event, ImageEvent, InteractiveEvent, TextEvent
from .models.exposure import Exposure
from .models.job import Job, JobStatus
from .models.job_register import JobRegister
from .models.job_spec import Category, OutputResult
from .models.license import LicenseInstance, UpdateTag
from .models.notification import Notification
from .models.project import GenerateIntermediateResultsSettings, Project, ProjectSymlink
from .models.scheduler_lane import SchedulerLane
from .models.scheduler_target import Cluster, Node, SchedulerTarget
from .models.service import LoggingService, ServiceLogLevel
from .models.session import DataManagementStats, ExposureGroup, ExposureGroupUpdate, LiveComputeResources, Session
from .models.session_config_profile import SessionConfigProfile, SessionConfigProfileBody
from .models.session_params import LiveAbinitParams, LiveClass2DParams, LivePreprocessingParams, LiveRefineParams
from .models.tag import Tag
from .models.user import User
from .models.workspace import Workspace

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
    def get_instance_uid(self) -> str: ...
    def generate_new_instance_uid(self, *, force_takeover_projects: bool = False) -> str:
        """
        Generates a new uid for the CryoSPARC instance
        If force_takeover_projects is True, overwrites existing lockfiles,
        otherwise if force_takeover_projects is False, only creates lockfile in projects that don't already have one
        """
        ...
    def set_default_job_priority(self, value: int) -> Any: ...
    def get_version(self) -> str:
        """
        Get the current CryoSPARC version (with patch suffix, if available)
        """
        ...
    def get_system_info(self) -> dict:
        """
        System information related to the CryoSPARC application
        """
        ...
    def get(self, name: str, /, *, default: Any = "<<UNDEFINED>>") -> Any:
        """
        Get config collection entry value for the given variable name.
        """
        ...
    def write(self, name: str, /, value: Any = ..., *, set_on_insert_only: bool = False) -> Any:
        """
        Set config collection entry. Specify `set_on_insert_only` to prevent
        overwriting when the value already exists. Returns the value in the
        database.
        """
        ...

class InstanceNamespace(APINamespace):
    """
    Methods available in api.instance, e.g., api.instance.get_update_tag(...)
    """
    def get_update_tag(self) -> UpdateTag | None: ...
    def live_enabled(self) -> bool: ...
    def ecl_enabled(self) -> bool: ...
    def link_log(
        self,
        type: str,
        /,
        data: Any = {},
        *,
        user_id: Optional[str] = ...,
        project_uid: Optional[str] = ...,
        job_uid: Optional[str] = ...,
        job_type: Optional[str] = ...,
    ) -> None: ...
    def get_license_usage(self) -> List[LicenseInstance]: ...
    def browse_files(self, *, abs_path_glob: str) -> BrowseFileResponse:
        """
        Backend for the file browser in the cryosparc UI. Returns a list of files
        for the UI to display.

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
        Get cryosparc service logs, filterable by date, name, function, and level
        """
        ...
    def get_runtime_diagnostics(self) -> RuntimeDiagnostics:
        """
        Get runtime diagnostics for the CryoSPARC instance
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
    def set(self, key: str, /, value: Any = None, *, namespace: Optional[str] = ..., ttl: int = 60) -> None:
        """
        Set the given key to the given value, with a ttl (Time-to-Live) in seconds
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
    def me(self) -> User: ...
    def find_one(self, user_id: str, /) -> User: ...
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
        Remove a user from the CryoSPARC. Only authenticated admins may do this.
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
        Create a new CryoSPARC user account. Specify ``created_by_user_id`` as the
        id of user who is creating the new user.

        The password is expected as a SHA256 hash.
        """
        ...
    def request_reset_password(self, user_id: str, /) -> None:
        """
        Generate a password reset token for a user with the given email. The token
        will appear in the Admin > User Management interface.
        """
        ...
    def register(self, user_id: str, /, body: SHA256Password, *, token: str) -> None:
        """
        Register user with a token (unauthenticated).
        """
        ...
    def reset_password(self, user_id: str, /, body: SHA256Password, *, token: str) -> None:
        """
        Reset password function with a token (unauthenticated). password is expected
        as a sha256 hash.
        """
        ...
    def set_role(self, user_id: str, /, role: Literal["user", "admin"]) -> User:
        """
        Change a user's from between "user" and "admin". Only admins may do this.
        This revokes all access tokens for the given used ID.
        """
        ...
    def get_my_state_var(self, key: str, /) -> Any:
        """
        Retrieve a user's state variable such as "licenseAccepted" or
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
        Retrieve a given user's state variable such as "licenseAccepted" or
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
        Get the lanes a user has access to
        """
        ...
    def set_lanes(self, user_id: str, /, lanes: List[str]) -> User: ...

class ResourcesNamespace(APINamespace):
    """
    Methods available in api.resources, e.g., api.resources.find_lanes(...)
    """
    def find_lanes(self) -> List[SchedulerLane]: ...
    def add_lane(self, body: SchedulerLane) -> SchedulerLane: ...
    def find_lane(self, name: str, /, *, type: Literal["node", "cluster", None] = ...) -> SchedulerLane: ...
    def remove_lane(self, name: str, /) -> None: ...
    def find_targets(self, *, lane: Optional[str] = ...) -> List[SchedulerTarget]: ...
    def find_nodes(self, *, lane: Optional[str] = ...) -> List[SchedulerTarget[Node]]: ...
    def add_node(self, body: SchedulerTarget[Node]) -> SchedulerTarget[Node]:
        """
        Add a node or update an existing node. Updates existing node if they share
        share the same name.
        """
        ...
    def find_clusters(self, *, lane: Optional[str] = ...) -> List[SchedulerTarget[Cluster]]: ...
    def add_cluster(self, body: SchedulerTarget[Cluster]) -> SchedulerTarget[Cluster]:
        """
        Add a cluster or update an existing cluster. Updates existing cluster if
        they share share the same name.
        """
        ...
    def find_target_by_hostname(self, hostname: str, /) -> SchedulerTarget: ...
    def find_target_by_name(self, name: str, /) -> SchedulerTarget: ...
    def find_node(self, name: str, /) -> SchedulerTarget[Node]: ...
    def remove_node(self, name: str, /) -> None: ...
    def find_cluster(self, name: str, /) -> SchedulerTarget[Cluster]: ...
    def remove_cluster(self, name: str, /) -> None: ...
    def find_cluster_script(self, name: str, /) -> str: ...
    def find_cluster_template_vars(self, name: str, /) -> List[str]:
        """
        Compute and retrieve all variable names defined in cluster templates.
        """
        ...
    def find_cluster_template_custom_vars(self, name: str, /) -> List[str]:
        """
        Compute and retrieve all custom variables names defined in cluster templates
        (i.e., all variables not in the internal list of known variable names).
        """
        ...
    def update_node_lane(self, name: str, /, lane: str) -> SchedulerTarget[Node]: ...
    def refresh_nodes(self) -> Any:
        """
        Asynchronously access target worker nodes. Load latest CPU, RAM and GPU
        info.
        """
        ...
    def verify_cluster(self, name: str, /) -> str: ...
    def update_cluster_custom_vars(self, name: str, /, value: Dict[str, str]) -> SchedulerTarget[Cluster]: ...
    def update_target_cache_path(self, name: str, /, value: Optional[str]) -> SchedulerTarget: ...

class JobsNamespace(APINamespace):
    """
    Methods available in api.jobs, e.g., api.jobs.find(...)
    """
    def find(
        self,
        *,
        sort: str = "created_at",
        order: Literal[1, -1] = 1,
        project_uid: Optional[List[str]] = ...,
        workspace_uid: Optional[List[str]] = ...,
        uid: Optional[List[str]] = ...,
        type: Optional[List[str]] = ...,
        status: Optional[List[JobStatus]] = ...,
        category: Optional[List[Category]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        deleted: Optional[bool] = False,
    ) -> List[Job]: ...
    def delete_many(self, project_job_uids: List[Tuple[str, str]], *, force: bool = False) -> List[Job]: ...
    def count(
        self,
        *,
        project_uid: Optional[List[str]] = ...,
        workspace_uid: Optional[List[str]] = ...,
        uid: Optional[List[str]] = ...,
        type: Optional[List[str]] = ...,
        status: Optional[List[JobStatus]] = ...,
        category: Optional[List[Category]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        deleted: Optional[bool] = False,
    ) -> int: ...
    def get_active_count(self) -> int: ...
    def find_in_project(
        self, project_uid: str, /, *, sort: str = "created_at", order: Literal[1, -1] = 1
    ) -> List[Job]: ...
    def clone_many(
        self,
        project_uid: str,
        /,
        job_uids: List[str],
        *,
        workspace_uid: Optional[str] = ...,
        new_workspace_title: Optional[str] = ...,
    ) -> List[Job]: ...
    def get_chain(self, project_uid: str, /, *, start_job_uid: str, end_job_uid: str) -> List[str]: ...
    def clone_chain(
        self,
        project_uid: str,
        /,
        *,
        start_job_uid: str,
        end_job_uid: str,
        workspace_uid: Optional[str] = ...,
        new_workspace_title: Optional[str] = ...,
    ) -> List[Job]: ...
    def find_in_workspace(
        self, project_uid: str, workspace_uid: str, /, *, sort: str = "created_at", order: Literal[1, -1] = 1
    ) -> List[Job]: ...
    def create(
        self,
        project_uid: str,
        workspace_uid: str,
        /,
        params: Optional[Dict[str, Union[bool, int, float, str, str, None]]] = ...,
        *,
        type: str,
        title: str = "",
        description: str = "Enter a description.",
        created_by_job_uid: Optional[str] = ...,
        enable_bench: bool = False,
    ) -> Job: ...
    def get_final_results(self, project_uid: str, /) -> GetFinalResultsResponse: ...
    def find_one(self, project_uid: str, job_uid: str, /) -> Job: ...
    def delete(self, project_uid: str, job_uid: str, /, *, force: bool = False) -> Job: ...
    def get_directory(self, project_uid: str, job_uid: str, /) -> str: ...
    def get_log(self, project_uid: str, job_uid: str, /) -> str: ...
    def get_status(self, project_uid: str, job_uid: str, /) -> JobStatus: ...
    def view(self, project_uid: str, workspace_uid: str, job_uid: str, /) -> Job:
        """
        Adds a project, workspace and job id to a user's "recentJobs" (recently
        viewed workspaces) state key
        """
        ...
    def set_param(self, project_uid: str, job_uid: str, param: str, /, *, value: Any) -> Job: ...
    def clear_param(self, project_uid: str, job_uid: str, param: str, /) -> Job: ...
    def connect(
        self, project_uid: str, job_uid: str, input_name: str, /, *, source_job_uid: str, source_output_name: str
    ) -> Job: ...
    def disconnect(self, project_uid: str, job_uid: str, input_name: str, connection_index: int, /) -> Job: ...
    def find_output_result(
        self, project_uid: str, job_uid: str, output_name: str, result_name: str, /
    ) -> OutputResult: ...
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
    ) -> Job: ...
    def disconnect_result(
        self, project_uid: str, job_uid: str, input_name: str, connection_index: int, result_name: str, /
    ) -> Job: ...
    def load_input(
        self,
        project_uid: str,
        job_uid: str,
        input_name: str,
        /,
        *,
        slots: Union[Literal["default", "passthrough", "all"], List[str]] = "default",
        force_join: bool = False,
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
        slots: Union[Literal["default", "passthrough", "all"], List[str]] = "default",
        version: Union[int, str] = "F",
    ) -> Dataset:
        """
        Load job output dataset. Raises exception if output is empty or does not exists.
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
    ) -> Job: ...
    def recalculate_intermediate_results_size(self, project_uid: str, job_uid: str, /) -> Job: ...
    def recalculate_project_intermediate_results_size(self, project_uid: str, /) -> List[Job]: ...
    def clear_intermediate_results(
        self, project_uid: str, job_uid: str, /, *, always_keep_final: bool = True
    ) -> Job: ...
    def export_output_results(
        self, project_uid: str, job_uid: str, output_name: str, /, result_names: Optional[List[str]] = ...
    ) -> str: ...
    def export(self, project_uid: str, job_uid: str, /) -> Job: ...
    def get_output_result_path(
        self, project_uid: str, job_uid: str, output_name: str, result_name: str, /, *, version: Union[int, str] = "F"
    ) -> str: ...
    def interactive_post(
        self, project_uid: str, job_uid: str, /, body: dict, *, endpoint: str, timeout: int = 10
    ) -> Any: ...
    def mark_completed(self, project_uid: str, job_uid: str, /) -> Job: ...
    def add_event_log(
        self, project_uid: str, job_uid: str, /, message: str, *, type: Literal["text", "warning", "error"] = "text"
    ) -> TextEvent: ...
    def get_event_logs(
        self, project_uid: str, job_uid: str, /
    ) -> List[Union[Event, CheckpointEvent, TextEvent, ImageEvent, InteractiveEvent]]: ...
    def recalculate_size(self, project_uid: str, job_uid: str, /) -> Job: ...
    def clear(self, project_uid: str, job_uid: str, /, *, force: bool = False) -> Job: ...
    def clear_many(
        self,
        *,
        project_uid: Optional[List[str]] = ...,
        workspace_uid: Optional[List[str]] = ...,
        uid: Optional[List[str]] = ...,
        type: Optional[List[str]] = ...,
        status: Optional[List[JobStatus]] = ...,
        category: Optional[List[Category]] = ...,
        created_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        updated_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        queued_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        completed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        killed_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        started_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        exported_at: Optional[Tuple[datetime.datetime, datetime.datetime]] = ...,
        deleted: Optional[bool] = False,
    ) -> List[Job]: ...
    def clone(
        self,
        project_uid: str,
        job_uid: str,
        /,
        *,
        workspace_uid: Optional[str] = ...,
        created_by_job_uid: Optional[str] = ...,
    ) -> Job: ...
    def kill(self, project_uid: str, job_uid: str, /) -> Job: ...
    def set_final_result(self, project_uid: str, job_uid: str, /, *, is_final_result: bool) -> Job:
        """
        Sets job final result flag and updates flags for all jobs in the project
        """
        ...
    def set_title(self, project_uid: str, job_uid: str, /, *, title: str) -> Job: ...
    def set_description(self, project_uid: str, job_uid: str, /, description: str) -> Job: ...
    def set_priority(self, project_uid: str, job_uid: str, /, *, priority: int) -> Job: ...
    def set_cluster_custom_vars(self, project_uid: str, job_uid: str, /, cluster_custom_vars: dict) -> Job: ...
    def get_active_licenses_count(self) -> int: ...
    def get_types(self) -> Any: ...
    def get_categories(self) -> Any: ...
    def find_ancestor_uids(
        self, project_uid: str, job_uid: str, /, *, workspace_uid: Optional[str] = ...
    ) -> List[str]: ...
    def find_descendant_uids(
        self, project_uid: str, job_uid: str, /, *, workspace_uid: Optional[str] = ...
    ) -> List[str]: ...
    def link_to_workspace(self, project_uid: str, job_uid: str, workspace_uid: str, /) -> Job: ...
    def unlink_from_workspace(self, project_uid: str, job_uid: str, workspace_uid: str, /) -> Job: ...
    def move(self, project_uid: str, job_uid: str, /, *, from_workspace_uid: str, to_workspace_uid: str) -> Job: ...
    def update_directory_symlinks(
        self, project_uid: str, job_uid: str, /, *, prefix_cut: str, prefix_new: str
    ) -> int: ...
    def add_tag(self, project_uid: str, job_uid: str, tag_uid: str, /) -> None: ...
    def remove_tag(self, project_uid: str, job_uid: str, tag_uid: str, /) -> None: ...
    def import_job(self, project_uid: str, workspace_uid: str, /, *, exported_job_dir_abs: str) -> Job: ...
    def import_result_group(
        self, project_uid: str, workspace_uid: str, /, *, csg_path: str, lane: Optional[str] = ...
    ) -> Job: ...

class WorkspacesNamespace(APINamespace):
    """
    Methods available in api.workspaces, e.g., api.workspaces.find(...)
    """
    def find(
        self,
        *,
        sort: str = "created_at",
        order: Literal[1, -1] = 1,
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
    def find_in_project(
        self, project_uid: str, /, *, sort: str = "created_at", order: Literal[1, -1] = 1
    ) -> List[Workspace]:
        """
        List all workspaces in a project with an optional filter.
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
    def preview_delete(self, project_uid: str, workspace_uid: str, /) -> DeleteWorkspacePreview: ...
    def find_one(self, project_uid: str, workspace_uid: str, /) -> Workspace:
        """
        Find a specific workspace in a project
        """
        ...
    def delete(self, project_uid: str, workspace_uid: str, /) -> None: ...
    def set_title(self, project_uid: str, workspace_uid: str, /, *, title: str) -> Workspace: ...
    def set_description(self, project_uid: str, workspace_uid: str, /, description: str) -> Workspace: ...
    def view(self, project_uid: str, workspace_uid: str, /) -> Workspace: ...
    def delete_async(self, project_uid: str, workspace_uid: str, /) -> Any: ...
    def add_tag(self, project_uid: str, workspace_uid: str, tag_uid: str, /) -> None: ...
    def remove_tag(self, project_uid: str, workspace_uid: str, tag_uid: str, /) -> None: ...
    def clear_intermediate_results(
        self, project_uid: str, workspace_uid: str, /, *, always_keep_final: bool = False
    ) -> Workspace: ...
    def find_workspace_ancestor_uids(
        self, project_uid: str, workspace_uid: str, /, job_uids: List[str]
    ) -> WorkspaceAncestorUidsResponse: ...
    def find_workspace_descendant_uids(
        self, project_uid: str, workspace_uid: str, /, job_uids: List[str]
    ) -> WorkspaceDescendantUidsResponse: ...

class SessionsNamespace(APINamespace):
    """
    Methods available in api.sessions, e.g., api.sessions.find(...)
    """
    def find(self, *, project_uid: Optional[str] = ...) -> List[Session]:
        """
        List all sessions (optionally, in a project)
        """
        ...
    def count(self, *, project_uid: Optional[str]) -> int:
        """
        Count all sessions in a project
        """
        ...
    def find_one(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Find a session
        """
        ...
    def delete(self, project_uid: str, session_uid: str, /) -> None: ...
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
        Create a new session
        """
        ...
    def find_exposure_groups(self, project_uid: str, session_uid: str, /) -> List[ExposureGroup]: ...
    def create_exposure_group(self, project_uid: str, session_uid: str, /) -> ExposureGroup: ...
    def find_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> ExposureGroup: ...
    def update_exposure_group(
        self, project_uid: str, session_uid: str, exposure_group_id: int, /, body: ExposureGroupUpdate
    ) -> ExposureGroup: ...
    def delete_exposure_group(self, project_uid: str, session_uid: str, exposure_group_id: int, /) -> Session: ...
    def finalize_exposure_group(
        self, project_uid: str, session_uid: str, exposure_group_id: int, /
    ) -> ExposureGroup: ...
    def start(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Build and start a CryoSPARC Live Session. Builds file engines based on file
        engine parameters in the session doc, builds phase one workers based on lane
        parameters in the session doc.
        """
        ...
    def pause(self, project_uid: str, session_uid: str, /) -> Session: ...
    def update_compute_configuration(
        self, project_uid: str, session_uid: str, /, body: LiveComputeResources
    ) -> LiveComputeResources: ...
    def add_tag(self, project_uid: str, session_uid: str, tag_uid: str, /) -> None: ...
    def remove_tag(self, project_uid: str, session_uid: str, tag_uid: str, /) -> None: ...
    def update_session_params(
        self,
        project_uid: str,
        session_uid: str,
        /,
        body: LivePreprocessingParams,
        *,
        reprocess: bool = True,
        priority: int = 1,
    ) -> Session: ...
    def update_session_picker(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        activate_picker_type: Literal["blob", "template", "deep"],
        use_thresholds: bool = True,
    ) -> Session: ...
    def update_attribute_threshold(
        self,
        project_uid: str,
        session_uid: str,
        attribute: str,
        /,
        *,
        min_val: Optional[float] = ...,
        max_val: Optional[float] = ...,
    ) -> Session: ...
    def clear_session(self, project_uid: str, session_uid: str, /) -> Session: ...
    def view(self, project_uid: str, session_uid: str, /) -> Session:
        """
        Adds a project, workspace and job id to a user's "recentJobs" (recently
        viewed workspaces) state key
        """
        ...
    def setup_phase2_class2D(self, project_uid: str, session_uid: str, /, *, force_restart: bool = True) -> Job: ...
    def enqueue_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Job: ...
    def stop_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session: ...
    def clear_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session: ...
    def update_phase2_class2D_params(
        self, project_uid: str, session_uid: str, /, body: LiveClass2DParams
    ) -> Session: ...
    def invert_template_phase2_class2D(self, project_uid: str, session_uid: str, template_idx: int, /) -> Session: ...
    def invert_all_templates_phase2_class2D(self, project_uid: str, session_uid: str, /) -> Session: ...
    def set_all_templates_phase2_class2D(
        self, project_uid: str, session_uid: str, direction: Literal["select", "deselect"], /
    ) -> Session: ...
    def select_direction_template_phase2_class2D(
        self,
        project_uid: str,
        session_uid: str,
        template_idx: int,
        /,
        *,
        dimension: str,
        direction: Literal["above", "below"] = "above",
    ) -> Session: ...
    def start_extract_manual(self, project_uid: str, session_uid: str, /) -> None: ...
    def set_session_exposure_processing_priority(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        exposure_processing_priority: Literal["normal", "oldest", "latest", "alternate"],
    ) -> Session: ...
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
    ) -> Session: ...
    def reset_attribute_threshold(self, project_uid: str, session_uid: str, attribute: str, /) -> Session: ...
    def reset_all_attribute_thresholds(self, project_uid: str, session_uid: str, /) -> Session: ...
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
    ) -> Session: ...
    def set_template_creation_job(self, project_uid: str, session_uid: str, /, *, job_uid: str) -> Session: ...
    def enqueue_template_creation_class2D(self, project_uid: str, session_uid: str, /) -> Job: ...
    def clear_template_creation_class2D(self, project_uid: str, session_uid: str, /) -> Session: ...
    def toggle_template_creation_template(
        self, project_uid: str, session_uid: str, template_idx: int, /
    ) -> Session: ...
    def toggle_template_creation_all_templates(self, project_uid: str, session_uid: str, /) -> Session: ...
    def select_template_creation_all(
        self, project_uid: str, session_uid: str, direction: Literal["select", "deselect"], /
    ) -> Session: ...
    def select_template_creation_in_direction(
        self, project_uid: str, session_uid: str, template_idx: int, direction: Literal["above", "below"], /
    ) -> Session: ...
    def setup_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Job: ...
    def set_phase2_abinit_job(self, project_uid: str, session_uid: str, /, *, job_uid: str) -> Session: ...
    def enqueue_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Job: ...
    def clear_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Session: ...
    def update_phase2_abinit_params(self, project_uid: str, session_uid: str, /, body: LiveAbinitParams) -> Session: ...
    def select_phase2_abinit_volume(self, project_uid: str, session_uid: str, /, *, volume_name: str) -> Session: ...
    def stop_phase2_abinit(self, project_uid: str, session_uid: str, /) -> Session: ...
    def clear_phase2_refine(self, project_uid: str, session_uid: str, /) -> Session: ...
    def setup_phase2_refine(self, project_uid: str, session_uid: str, /) -> Job: ...
    def enqueue_phase2_refine(self, project_uid: str, session_uid: str, /) -> Job: ...
    def stop_phase2_refine(self, project_uid: str, session_uid: str, /) -> Session: ...
    def update_phase2_refine_params(self, project_uid: str, session_uid: str, /, body: LiveRefineParams) -> Session: ...
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
    ) -> Job: ...
    def create_and_enqueue_dump_exposures(
        self, project_uid: str, session_uid: str, /, *, export_ignored: bool = False
    ) -> Job: ...
    def get_data_management_stats(self, project_uid: str, /) -> Dict[str, DataManagementStats]: ...
    def mark_session_completed(self, project_uid: str, session_uid: str, /) -> Session: ...
    def change_session_data_management_state(
        self,
        project_uid: str,
        session_uid: str,
        /,
        *,
        datatype: Literal["micrographs", "raw", "particles", "metadata", "thumbnails"],
        status: Literal["active", "archiving", "archived", "deleted", "deleting", "missing", "calculating"],
    ) -> Session: ...
    def update_session_datatype_sizes(self, project_uid: str, session_uid: str, /) -> int: ...
    def get_datatype_size(
        self,
        project_uid: str,
        session_uid: str,
        datatype: Literal["micrographs", "raw", "particles", "metadata", "thumbnails"],
        /,
    ) -> int: ...
    def delete_live_datatype(
        self,
        project_uid: str,
        session_uid: str,
        datatype: Literal["micrographs", "raw", "particles", "metadata", "thumbnails"],
        /,
    ) -> Job | None: ...
    def update_all_sessions_datatype_sizes(self, project_uid: str, /) -> None: ...
    def get_datatype_file_paths(
        self,
        project_uid: str,
        session_uid: str,
        datatype: Literal["micrographs", "raw", "particles", "metadata", "thumbnails"],
        /,
    ) -> List[str]: ...
    def get_configuration_profiles(self) -> List[SessionConfigProfile]: ...
    def create_configuration_profile(self, body: SessionConfigProfileBody) -> SessionConfigProfile: ...
    def apply_configuration_profile(
        self, project_uid: str, session_uid: str, /, *, configuration_id: str
    ) -> Session: ...
    def update_configuration_profile(
        self, configuration_id: str, /, body: SessionConfigProfileBody
    ) -> SessionConfigProfile: ...
    def delete_configuration_profile(self, configuration_id: str, /) -> None: ...
    def compact_session(self, project_uid: str, session_uid: str, /) -> Any: ...
    def restore_session(self, project_uid: str, session_uid: str, /, body: LiveComputeResources) -> Any: ...
    def get_session_base_params(self) -> Any: ...

class ProjectsNamespace(APINamespace):
    """
    Methods available in api.projects, e.g., api.projects.check_directory(...)
    """
    def check_directory(self, *, path: str) -> str: ...
    def get_title_slug(self, *, title: str) -> str: ...
    def find(
        self,
        *,
        sort: str = "created_at",
        order: Literal[1, -1] = 1,
        uid: Optional[List[str]] = ...,
        project_dir: Optional[str] = ...,
        owner_user_id: Optional[str] = ...,
        deleted: Optional[bool] = False,
        archived: Optional[bool] = ...,
        detached: Optional[bool] = ...,
        hidden: Optional[bool] = ...,
    ) -> List[Project]: ...
    def create(self, *, title: str, description: Optional[str] = ..., parent_dir: str) -> Project: ...
    def count(
        self,
        *,
        uid: Optional[List[str]] = ...,
        project_dir: Optional[str] = ...,
        owner_user_id: Optional[str] = ...,
        deleted: Optional[bool] = False,
        archived: Optional[bool] = ...,
        detached: Optional[bool] = ...,
        hidden: Optional[bool] = ...,
    ) -> int: ...
    def set_title(self, project_uid: str, /, *, title: str) -> Project: ...
    def set_description(self, project_uid: str, /, description: str) -> Project: ...
    def view(self, project_uid: str, /) -> Project: ...
    def get_job_register(self, project_uid: str, /) -> JobRegister: ...
    def preview_delete(self, project_uid: str, /) -> DeleteProjectPreview: ...
    def find_one(self, project_uid: str, /) -> Project: ...
    def delete(self, project_uid: str, /) -> None: ...
    def delete_async(self, project_uid: str, /) -> Any: ...
    def get_directory(self, project_uid: str, /) -> str: ...
    def get_owner_id(self, project_uid: str, /) -> str: ...
    def set_owner(self, project_uid: str, user_id: str, /) -> Project: ...
    def add_user_access(self, project_uid: str, user_id: str, /) -> Project: ...
    def remove_user_access(self, project_uid: str, user_id: str, /) -> Project: ...
    def refresh_size(self, project_uid: str, /) -> Project: ...
    def refresh_size_async(self, project_uid: str, /) -> Any: ...
    def get_symlinks(self, project_uid: str, /) -> List[ProjectSymlink]: ...
    def set_default_param(self, project_uid: str, name: str, /, value: Union[bool, int, float, str]) -> Project: ...
    def clear_default_param(self, project_uid: str, name: str, /) -> Project: ...
    def claim_instance_ownership(self, project_uid: str, /, *, force: bool = False) -> None: ...
    def claim_all_instance_ownership(self, *, force: bool = False) -> None: ...
    def archive(self, project_uid: str, /) -> Project: ...
    def unarchive(self, project_uid: str, /, *, path: str) -> Project: ...
    def detach(self, project_uid: str, /) -> Project: ...
    def attach(self, *, path: str) -> Project: ...
    def move(self, project_uid: str, /, *, path: str) -> Project:
        """
        Rename the project directory for the given project. Provide either the new
        directory name or the full new directory path.
        """
        ...
    def get_next_exposure_group_id(self, project_uid: str, /) -> int: ...
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
    ) -> int: ...
    def add_tag(self, project_uid: str, tag_uid: str, /) -> None: ...
    def remove_tag(self, project_uid: str, tag_uid: str, /) -> None: ...
    def get_generate_intermediate_results_settings(
        self, project_uid: str, /
    ) -> GenerateIntermediateResultsSettings: ...
    def set_generate_intermediate_results_settings(
        self, project_uid: str, /, body: GenerateIntermediateResultsSettings
    ) -> Project: ...
    def clear_intermediate_results(self, project_uid: str, /, *, always_keep_final: bool = True) -> Project: ...
    def get_generate_intermediate_results_job_types(self) -> List[str]: ...

class ExposuresNamespace(APINamespace):
    """
    Methods available in api.exposures, e.g., api.exposures.reset_manual_reject_exposures(...)
    """
    def reset_manual_reject_exposures(self, project_uid: str, session_uid: str, /) -> List[Exposure]: ...
    def reset_all_exposures(self, project_uid: str, session_uid: str, /) -> None: ...
    def reset_failed_exposures(self, project_uid: str, session_uid: str, /) -> None: ...
    def reset_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure: ...
    def manual_reject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure: ...
    def manual_unreject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure: ...
    def toggle_manual_reject_exposure(self, project_uid: str, session_uid: str, exposure_uid: int, /) -> Exposure: ...
    def reprocess_single_exposure(
        self,
        project_uid: str,
        session_uid: str,
        exposure_uid: int,
        /,
        body: LivePreprocessingParams,
        *,
        picker_type: Literal["blob", "template", "deep"],
    ) -> Exposure: ...
    def add_manual_pick(
        self, project_uid: str, session_uid: str, exposure_uid: int, /, *, x_frac: float, y_frac: float
    ) -> Exposure: ...
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
    ) -> Exposure: ...
    def get_individual_picks(
        self,
        project_uid: str,
        session_uid: str,
        exposure_uid: int,
        picker_type: Literal["blob", "template", "manual"],
        /,
    ) -> List[List[float]]: ...

class TagsNamespace(APINamespace):
    """
    Methods available in api.tags, e.g., api.tags.find(...)
    """
    def find(
        self,
        *,
        sort: str = "created_at",
        order: Literal["1", "-1"] = "1",
        created_by_user_id: Optional[str] = ...,
        type: Optional[List[Literal["general", "project", "workspace", "session", "job"]]] = ...,
        uid: Optional[str] = ...,
    ) -> List[Tag]: ...
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
    ) -> Tag: ...
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
    ) -> Tag: ...
    def delete(self, tag_uid: str, /) -> None: ...
    def get_tags_by_type(self) -> Dict[str, List[Tag]]: ...
    def get_tag_count_by_type(self) -> Dict[str, int]: ...

class NotificationsNamespace(APINamespace):
    """
    Methods available in api.notifications, e.g., api.notifications.deactivate_notification(...)
    """
    def deactivate_notification(self, notification_id: str, /) -> Notification: ...

class BlueprintsNamespace(APINamespace):
    """
    Methods available in api.blueprints, e.g., api.blueprints.create_blueprint(...)
    """
    def create_blueprint(
        self, schema: dict, *, blueprint_id: str, imported: bool, project_uid: str, job_uid: str, job_type: str
    ) -> None:
        """
        For cryosparc app only
        """
        ...
    def edit_blueprint(
        self, blueprint_id: str, /, schema: dict, *, project_uid: str, job_uid: str, job_type: str
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
        self, blueprint_id: str, /, schema: dict, *, project_uid: str, job_uid: str, job_type: str
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
        self, schema: dict, *, workflow_id: str, forked: bool = False, imported: bool = False, rebuilt: bool = False
    ) -> None:
        """
        For cryosparc app only
        """
        ...
    def edit_workflow(self, workflow_id: str, /, schema: dict) -> None:
        """
        For cryosparc app only
        """
        ...
    def delete_workflow(self, workflow_id: str, /) -> None:
        """
        For cryosparc app only
        """
        ...
    def apply_workflow(self, workflow_id: str, /, schema: dict) -> None:
        """
        For cryosparc app only
        """
        ...

class ExternalNamespace(APINamespace):
    """
    Methods available in api.external, e.g., api.external.get_empiar_latest_entries(...)
    """
    def get_empiar_latest_entries(self) -> dict: ...
    def get_emdb_latest_entries(self) -> List[dict]: ...
    def get_discuss_top(self) -> dict: ...
    def get_discuss_categories(self) -> dict: ...
    def get_tutorials(self) -> dict: ...
    def get_changelog(self) -> dict: ...

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
    def save_job_registers(self) -> List[JobRegister]:
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
    jobs: JobsNamespace
    workspaces: WorkspacesNamespace
    sessions: SessionsNamespace
    projects: ProjectsNamespace
    exposures: ExposuresNamespace
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
