# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .job_spec import JobSpec
from .workspace import JobGroupMeta


class AppliedWorkflowTag(BaseModel):
    description: str | None = None
    """
    """
    colour: (
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
        | None
    ) = None
    """
    """
    title: str | None = None
    """
    """
    uid: str | None = None
    """
    """


class WorkflowParentGroup(BaseModel):
    name: str
    """
    """
    connections: list[str] = []
    """
    """
    required: list[str] = []
    """
    """


class WorkflowParent(BaseModel):
    groups: dict[str, WorkflowParentGroup] = {}
    """
    """
    jobType: str
    """
    """


class WorkflowParameter(BaseModel):
    value: str | int | float | bool | None = None
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
    parameters: dict[str, WorkflowParameter] = {}
    """
    """
    spec: JobSpec
    """
    """
    individualResults: list[list[str]] = []
    """
    Array representing result overrides for the app, computed from the spec. This consists of a list of string pairs, with the first string pair representing
    the source result and the second the destination result, in the format of
    ["{result.job_uid}.{result.output}.{result.result}", "{uid}.{input_name}.{conn_idx}.{result.name}",]
    """
    groups: list[list[str]]
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
    createdBy: str | None = None
    """
    """
    updatedBy: str | None = None
    """
    """
    pinned: bool = False
    """
    """
    imported: bool = False
    """
    """
    parents: dict[str, WorkflowParent] = {}
    """
    Dictionary of parent jobs for the workflow, where the key is PXX
    """
    jobs: dict[str, WorkflowJob] = {}
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
    parent_juids: list[str] = []
    """
    """
    lane: str | None = None
    """
    """
    tag: AppliedWorkflowTag | None = None
    """
    """
    group: JobGroupMeta | None = None
    """
    """
    strict: bool = False
    """
    """


class UpdateWorkflow(BaseModel):
    title: str | None = None
    """
    """
    description: str | None = None
    """
    """
    category: str | None = None
    """
    """
    pinned: bool | None = None
    """
    """
    parents: dict[str, WorkflowParent] | None = None
    """
    """
    jobs: dict[str, WorkflowJob] | None = None
    """
    """
