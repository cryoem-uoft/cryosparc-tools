# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import List

from pydantic import BaseModel


class GetFinalResultsResponse(BaseModel):
    final_results: List[str]
    """
    Job uids of jobs marked as final result
    """
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
