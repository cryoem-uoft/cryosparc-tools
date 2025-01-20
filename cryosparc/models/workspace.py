# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    dumped_at: Optional[datetime.datetime] = None
    last_dumped_version: Optional[str] = None
    autodump: bool = True
    uid: str
    project_uid: str
    created_by_user_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    created_by_job_uid: Optional[str] = None
    tags: List[str] = []
    starred_by: List[str] = []
    deleted: bool = False
    last_accessed: Optional[WorkspaceLastAccessed] = None
    workspace_stats: WorkspaceStats = WorkspaceStats()
    notes: str = ""
    notes_lock: Optional[str] = None
    imported_at: Optional[datetime.datetime] = None
    workspace_type: Literal["base", "live"] = "base"

    model_config = ConfigDict(extra="allow")
    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None: ...
        def __getattr__(self, key: str) -> Any: ...
