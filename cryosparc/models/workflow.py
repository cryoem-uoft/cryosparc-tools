# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .job_spec import JobSpec
from .workspace import JobGroupMeta


class AppliedWorkflowTag(BaseModel):
    description: Optional[str] = None
    """
    """
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
    ] = None
    """
    """
    title: Optional[str] = None
    """
    """
    uid: Optional[str] = None
    """
    """


class WorkflowParentGroup(BaseModel):
    name: str
    """
    """
    connections: List[str] = []
    """
    """
    required: List[str] = []
    """
    """


class WorkflowParent(BaseModel):
    groups: Dict[str, WorkflowParentGroup] = {}
    """
    """
    jobType: str
    """
    """


class WorkflowParameter(BaseModel):
    value: Union[str, int, float, bool, None] = None
    """
    """
    locked: bool = False
    """
    """
    visible: bool = True
    """
    """
    flagged: bool = False
    """
    """
    notes: str = ""
    """
    """


class WorkflowJob(BaseModel):
    jobType: str
    """
    """
    title: str = ""
    """
    """
    description: str = ""
    """
    """
    parameters: Dict[str, WorkflowParameter] = {}
    """
    """
    spec: JobSpec
    """
    """
    individualResults: List[List[str]] = []
    """
    Array representing result overrides for the app, computed from the spec. This consists of a list of string pairs, with the first string pair representing
    the source result and the second the destination result, in the format of
    ["{result.job_uid}.{result.output}.{result.result}", "{uid}.{input_name}.{conn_idx}.{result.name}",]
    """
    groups: List[List[str]]
    """
    """


class Workflow(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
    updatedAt: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    """
    createdAt: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    """
    title: str = ""
    """
    """
    description: str = ""
    """
    """
    category: str = ""
    """
    """
    workflowVersion: str = "1.0.0"
    """
    """
    csVersion: str
    """
    """
    createdBy: Optional[str] = None
    """
    """
    updatedBy: Optional[str] = None
    """
    """
    pinned: bool = False
    """
    """
    imported: bool = False
    """
    """
    parents: Dict[str, WorkflowParent] = {}
    """
    Dictionary of parent jobs for the workflow, where the key is PXX
    """
    jobs: Dict[str, WorkflowJob] = {}
    """
    Dictionary of jobs for the workflow, where the key is JXX
    """
    reference: bool = False
    """
    """


class ApplyWorkflowRequest(BaseModel):
    """
    Request body for the workflow apply endpoint.
    """

    workflow: Workflow
    """
    """
    parent_juids: List[str] = []
    """
    """
    lane: Optional[str] = None
    """
    """
    tag: Optional[AppliedWorkflowTag] = None
    """
    """
    group: Optional[JobGroupMeta] = None
    """
    """


class UpdateWorkflow(BaseModel):
    title: Optional[str] = None
    """
    """
    description: Optional[str] = None
    """
    """
    category: Optional[str] = None
    """
    """
    pinned: Optional[bool] = None
    """
    """
    parents: Optional[Dict[str, WorkflowParent]] = None
    """
    """
    jobs: Optional[Dict[str, WorkflowJob]] = None
    """
    """
