# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobGroup(BaseModel):
    id: int
    jobs: List[str]
    title: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None


class WorkspaceStats(BaseModel):
    updated_at: Optional[datetime.datetime] = None
    job_count: int = 0
    job_sections: Dict[str, int] = {}
    job_status: Dict[str, int] = {}
    job_types: Dict[str, int] = {}


class WorkspaceLastAccessed(BaseModel):
    name: str
    accessed_at: datetime.datetime


class Workspace(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
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
    """
    uid: str
    project_uid: str
    created_by_user_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    created_by_job_uid: Optional[str] = None
    tags: List[str] = []
    starred_by: List[str] = []
    deleted: bool = False
    deleting: bool = False
    last_accessed: Optional[WorkspaceLastAccessed] = None
    workspace_stats: WorkspaceStats = WorkspaceStats()
    imported_at: Optional[datetime.datetime] = None
    workspace_type: Literal["base", "live"] = "base"
    groups: List[JobGroup] = []
    uid_num: int
    project_uid_num: int

    model_config = ConfigDict(extra="allow")
    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None: ...
        def __getattr__(self, key: str) -> Any: ...
