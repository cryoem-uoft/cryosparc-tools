# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Literal

from pydantic import BaseModel, Field


class JobGroupMeta(BaseModel):
    """
    Meta information a user-defined group of jobs within a workspace.
    """

    title: str | None = None
    """
    """
    description: str | None = None
    """
    """
    color: str | None = None
    """
    """


class JobGroup(BaseModel):
    title: str | None = None
    """
    """
    description: str | None = None
    """
    """
    color: str | None = None
    """
    """
    jobs: list[str]
    """
    """
    id: int
    """
    """


class JobGroupCreate(BaseModel):
    """
    Create a user-defined group of jobs within a workspace.
    """

    title: str | None = None
    """
    """
    description: str | None = None
    """
    """
    color: str | None = None
    """
    """
    jobs: list[str]
    """
    """


class JobGroupUpdate(BaseModel):
    """
    Update a user-defined group of jobs within a workspace.
    """

    title: str | None = None
    """
    """
    description: str | None = None
    """
    """
    color: str | None = None
    """
    """
    jobs: list[str] | None = None
    """
    """


class WorkspaceStats(BaseModel):
    """
    Statistics about the contents of a workspace.
    """

    updated_at: datetime.datetime | None = None
    """
    """
    job_count: int = 0
    """
    """
    job_sections: dict[str, int] = {}
    """
    Number of jobs per section.
    """
    job_status: dict[str, int] = {}
    """
    Number of jobs per status.
    """
    job_types: dict[str, int] = {}
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
    dumped_at: datetime.datetime | None = None
    """
    When the model was last dumped to disk
    """
    last_dumped_version: str | None = None
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
    workspace_type: Literal["base"]
    """
    Workspace type, always 'base'.
    """
    schema_version: int = 1
    """
    Workspace schema version for determining exported workspace and session
    compatibility between CryoSPARC versions, similar to the ``schema_version``
    present in projects.
    """
    created_by_user_id: str | None = None
    """
    User ID that created the workspace.
    """
    title: str | None = None
    """
    """
    description: str | None = None
    """
    """
    created_by_job_uid: str | None = None
    """
    """
    tags: list[str] = []
    """
    """
    starred_by: list[str] = []
    """
    List of user IDs who have starred the workspace.
    """
    deleted: bool = False
    """
    """
    deleting: bool = False
    """
    """
    last_accessed: WorkspaceLastAccessed | None = None
    """
    """
    workspace_stats: WorkspaceStats = WorkspaceStats()
    """
    """
    imported_at: datetime.datetime | None = None
    """
    """
    groups: list[JobGroup] = []
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
