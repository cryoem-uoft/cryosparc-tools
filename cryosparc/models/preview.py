# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List, Literal, Optional

from pydantic import BaseModel


class DeleteJobPreview(BaseModel):
    project_uid: str
    uid: str
    workspace_uids: List[str]
    status: str
    title: str
    type: str


class DeleteProjectWorkspacePreview(BaseModel):
    project_uid: str
    uid: str
    title: Optional[str]


class DeleteProjectPreview(BaseModel):
    jobs: List[DeleteJobPreview]
    workspaces: List[DeleteProjectWorkspacePreview]


class KeepJobPreview(BaseModel):
    """
    Job from a delete request that cannot be deleted and the reasons for it.
    """

    project_uid: str
    uid: str
    workspace_uids: List[str]
    status: str
    title: str
    type: str
    reason: Literal["final", "descendants"]
    descendant_job_uids: List[str] = []
    descendant_workspace_uids: List[str] = []


class DeleteWorkspacePreview(BaseModel):
    """
    Preview of a workspace delete operation, including jobs that will be deleted
    or unlinked and those that will be kept. If there any kept jobs, the
    workspace will not be marked as deleted.
    """

    delete: List[DeleteJobPreview]
    unlink: List[DeleteJobPreview]
    keep: List[KeepJobPreview]
    jobs: List[DeleteJobPreview]
