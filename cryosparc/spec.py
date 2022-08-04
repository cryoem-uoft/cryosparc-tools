from typing import List, Optional, Tuple, Union
from typing_extensions import Literal, TypedDict


Datatype = Literal["exposure", "particle", "template", "volume", "mask"]


class Datafield(TypedDict):
    """Definition of a prefix field within a CS file."""

    dtype: str
    """Datatype-specific string from common.py. e.g., movie_blob, ctf,
    alignments2D."""

    prefix: str
    """where to find that fild in a corresponding .cs file e.g.,
    alignments_class_1"""

    required: bool
    """whether this field must necessarily exist in a corresponding
    input/output"""


class InputSlot(TypedDict):
    type: str
    name: str
    title: str
    description: str
    optional: bool


class ConnectionSlot(TypedDict):
    slot_name: Optional[str]  # passthrough slots have slot_name None
    job_uid: str
    group_name: str  # e.g., particles
    result_name: str  # e.g., blob
    result_type: str  # e.g., particle.blob
    version: Union[int, Literal["F"]]


class Connection(TypedDict):
    job_uid: str
    group_name: str
    slots: List[ConnectionSlot]


class InputSlotGroup(TypedDict):
    type: Datatype
    name: str
    title: str
    description: str
    count_min: int
    count_max: Optional[int]
    repeat_allowed: bool
    slots: List[InputSlot]
    connections: List[Connection]


class OutputResultGroupContains(TypedDict):
    uid: str
    type: str  # e.g., particle.alignments3D
    group_name: str  # e.g., particles
    name: str  # e.g., alignments_class_1
    passthrough: bool


class OutputResultGroup(TypedDict):
    uid: str
    type: Datatype
    name: str
    title: str
    description: str
    contains: List[OutputResultGroupContains]
    passthrough: Union[str, Literal[False]]
    num_items: int
    summary: dict


class OutputResult(TypedDict):
    uid: str
    type: str
    name: str
    group_name: str
    title: str
    description: str
    versions: List[int]
    metafiles: List[str]
    min_fields: List[Tuple[str, str]]
    num_items: int
    passthrough: bool


class JobDocument(TypedDict):
    _id: str
    uid: str
    uid_num: int
    project_uid: str
    project_uid_num: int
    job_type: str
    title: str
    description: str
    status: str
    created_at: str
    created_by_user_id: Optional[str]
    deleted: bool
    parents: List[str]
    children: List[str]
    input_slot_groups: List[InputSlotGroup]
    output_result_groups: List[OutputResultGroup]
    output_results: List[OutputResult]
    workspace_uids: List[str]


class WorkspaceDocument(TypedDict):
    _id: str
    uid: str
    uid_num: int
    project_uid: str
    project_uid_num: int
    created_at: str
    created_by_user_id: str
    deleted: bool
    title: str
    description: Optional[str]
    workspace_type: Literal["base", "live"]
