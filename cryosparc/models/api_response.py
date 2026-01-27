# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List

from pydantic import BaseModel


class GetFinalResultsResponse(BaseModel):
    """
    Information about jobs marked as final in a project.
    """

    final_results: List[str]
    """
    Job uids of jobs marked as final result.
    """
    ancestors_of_final_results: List[str]
    """
    Job uids of ancestors of final result jobs.
    """
    non_ancestors_of_final_results: List[str]
    """
    All other job uids in the project that are neither marked as final nor ancestors of final result jobs.
    """


class Hello(BaseModel):
    """
    Response model for root endpoint
    """

    name: str = "CryoSPARC"
    """
    """
    version: str
    """
    Running CryoSPARC version. Includes patch, if installed
    """
    service: str
    """
    API service name
    """


class WorkspaceAncestorUidsResponse(BaseModel):
    """
    Listings of ancestor and non-ancestor jobs in a given workspace.
    """

    ancestors: List[str]
    """
    Jobs in the workspace that are ancestors of a given set of jobs.
    """
    non_ancestors: List[str]
    """
    Jobs in the workspace that are not ancestors of a given set of jobs.
    """


class WorkspaceDescendantUidsResponse(BaseModel):
    """
    Listings of descendant and non-descendant jobs in a given workspace.
    """

    descendants: List[str]
    """
    Jobs in the workspace that are descendants of a given set of jobs.
    """
    non_descendants: List[str]
    """
    Jobs in the workspace that are not descendants of a given set of jobs.
    """
