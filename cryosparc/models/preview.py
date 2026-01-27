# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List, Literal, Optional

from pydantic import BaseModel


class DeleteJobPreview(BaseModel):
    """
    Job from a delete request that will be deleted or unlinked.
    """

    project_uid: str
    """
    """
    uid: str
    """
    """
    workspace_uids: List[str]
    """
    """
    status: str
    """
    """
    title: str
    """
    """
    type: str
    """
    """


class DeleteProjectWorkspacePreview(BaseModel):
    """
    Preview of a workspace within a project to be deleted.
    """

    project_uid: str
    """
    """
    uid: str
    """
    """
    title: Optional[str]
    """
    """


class DeleteProjectPreview(BaseModel):
    """
    Preview of a project delete operation, including jobs and workspaces that will be deleted.
    """

    jobs: List[DeleteJobPreview]
    """
    """
    workspaces: List[DeleteProjectWorkspacePreview]
    """
    """


class KeepJobPreview(BaseModel):
    """
    Job from a delete request that cannot be deleted and the reasons for it.
    """

    project_uid: str
    """
    """
    uid: str
    """
    """
    workspace_uids: List[str]
    """
    """
    status: str
    """
    """
    title: str
    """
    """
    type: str
    """
    """
    reason: Literal["final", "descendants"]
    """
    Reason for not deleting this job, either "final" because it or one if its
    descendants is marked a final, or "descendants" because it has children that
    are not or cannot be deleted.
    """
    descendant_job_uids: List[str] = []
    """
    This job's descendant UIDs that cannot be deleted.
    """
    descendant_workspace_uids: List[str] = []
    """
    Workspaces that contain this job's descendants that cannot be deleted.
    """


class DeleteWorkspacePreview(BaseModel):
    """
    Preview of a workspace delete operation, including jobs that will be deleted
    or unlinked and those that will be kept. If there any kept jobs, the
    workspace will not be marked as deleted.
    """

    delete: List[DeleteJobPreview]
    """
    Jobs to be deleted.
    """
    unlink: List[DeleteJobPreview]
    """
    Jobs to be unlinked.
    """
    keep: List[KeepJobPreview]
    """
    Jobs that cannot be deleted, either because they are final or have descendants.
    """
    jobs: List[DeleteJobPreview]
    """
    """
