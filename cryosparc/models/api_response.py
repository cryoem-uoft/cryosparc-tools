# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List, Optional

from pydantic import BaseModel


class BrowseFile(BaseModel):
    file_name: str
    base_path: str
    is_hidden: bool
    path_abs: str
    is_link: bool = False
    mtime: Optional[float] = None
    size: Optional[int] = None
    type: Optional[str] = None
    link_path: Optional[str] = None
    errmesg: Optional[str] = None


class BrowseFileResponse(BaseModel):
    back_path: str
    files: List[BrowseFile]
    type: str


class DeleteProjectJobPreview(BaseModel):
    project_uid: str
    uid: str
    workspace_uids: List[str]
    status: str


class DeleteProjectWorkspacePreview(BaseModel):
    project_uid: str
    uid: str
    title: Optional[str]


class DeleteProjectPreview(BaseModel):
    jobs: List[DeleteProjectJobPreview]
    workspaces: List[DeleteProjectWorkspacePreview]


class DeleteWorkspacePreview(BaseModel):
    jobs_in_one_workspace: List[str]
    jobs_in_multiple_workspaces: List[str]


class GetFinalResultsResponse(BaseModel):
    final_results: List[str]
    ancestors_of_final_results: List[str]
    non_ancestors_of_final_results: List[str]


class Hello(BaseModel):
    name: str = "CryoSPARC"
    version: str
    service: str


class WorkspaceAncestorUidsResponse(BaseModel):
    ancestors: List[str]
    non_ancestors: List[str]


class WorkspaceDescendantUidsResponse(BaseModel):
    descendants: List[str]
    non_descendants: List[str]
