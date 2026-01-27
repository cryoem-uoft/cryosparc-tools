# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobGroup(BaseModel):
    """
    User-defined group of jobs within a workspace.
    """

    id: int
    """
    """
    jobs: List[str]
    """
    """
    title: Optional[str] = None
    """
    """
    description: Optional[str] = None
    """
    """
    color: Optional[str] = None
    """
    """


class WorkspaceStats(BaseModel):
    """
    Statistics about the contents of a workspace.
    """

    updated_at: Optional[datetime.datetime] = None
    """
    """
    job_count: int = 0
    """
    """
    job_sections: Dict[str, int] = {}
    """
    Number of jobs per section.
    """
    job_status: Dict[str, int] = {}
    """
    Number of jobs per status.
    """
    job_types: Dict[str, int] = {}
    """
    Number of jobs per type.
    """


class WorkspaceLastAccessed(BaseModel):
    """
    Record for when a user last accessed a workspace.
    """

    name: str
    """
    Username that last accessed the workspace.
    """
    accessed_at: datetime.datetime
    """
    Timestamp of the last access.
    """


class Workspace(BaseModel):
    """
    A workspace within a project, which contains jobs and related data. Jobs
    are stored in a flat structure within the project and linked to one or more
    workspaces.
    """

    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was last modified.
    """
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was first created. Imported objects such as projects
    and jobs will retain the created time from their original CryoSPARC instance.
    """
    dumped_at: Optional[datetime.datetime] = None
    """
    When the model was last dumped to disk
    """
    last_dumped_version: Optional[str] = None
    """
    The version of CryoSPARC last dumped at
    """
    autodump: bool = True
    """
    Whether the model was updated recently and must be dumped

    :meta private:
    """
    uid: str
    """
    Workspace unique ID, e.g., 'W1'.
    """
    project_uid: str
    """
    Project unique ID for , e.g., 'P1'.
    """
    workspace_type: Literal["base", "live"] = "base"
    """
    Workspace type, either 'base' or 'live'.
    """
    created_by_user_id: Optional[str] = None
    """
    User ID that created the workspace.
    """
    title: Optional[str] = None
    """
    """
    description: Optional[str] = None
    """
    """
    created_by_job_uid: Optional[str] = None
    """
    """
    tags: List[str] = []
    """
    """
    starred_by: List[str] = []
    """
    List of user IDs who have starred the workspace.
    """
    deleted: bool = False
    """
    """
    deleting: bool = False
    """
    """
    last_accessed: Optional[WorkspaceLastAccessed] = None
    """
    """
    workspace_stats: WorkspaceStats = WorkspaceStats()
    """
    """
    imported_at: Optional[datetime.datetime] = None
    """
    """
    groups: List[JobGroup] = []
    """
    """
    uid_num: int
    """
    Numeric part of the workspace UID.
    """
    project_uid_num: int
    """
    Numeric part of the project UID.
    """

    model_config = ConfigDict(extra="allow")
    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None: ...
        def __getattr__(self, key: str) -> Any: ...
