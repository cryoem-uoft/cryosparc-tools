from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, TypedDict


Datatype = Literal["exposure", "particle", "template", "volume", "mask"]

# Valid plot file types
TextFormat = Literal["txt", "csv", "json", "xml"]
"""
Supported job stream log asset file text formats
"""

ImageFormat = Literal["pdf", "gif", "jpg", "jpeg", "png", "svg"]
"""
Supported job stream log asset file image formats
"""

AssetFormat = Union[TextFormat, ImageFormat]
"""
Supported job stream log asset file formats
"""

TextContentType = Literal[
    "text/plain",
    "text/csv",
    "application/json",
    "application/xml",
]
"""
Supported job stream log text asset MIME types
"""

ImageContentType = Literal[
    "application/pdf",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
]
"""
Supported job image asset MIME types
"""

AssetContentType = Union[TextContentType, ImageContentType]
"""
Supported job asset MIME types
"""

TEXT_CONTENT_TYPES: Dict[TextFormat, TextContentType] = {
    "txt": "text/plain",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
}

IMAGE_CONTENT_TYPES: Dict[ImageFormat, ImageContentType] = {
    "pdf": "application/pdf",
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "svg": "image/svg+xml",
}

ASSET_CONTENT_TYPES: Dict[AssetFormat, AssetContentType] = {**TEXT_CONTENT_TYPES, **IMAGE_CONTENT_TYPES}  # type: ignore


class AssetDetails(TypedDict):
    """
    Dictinary result of job asset files query.
    """

    _id: str
    """Document ID"""

    filename: str
    """File name"""

    contentType: AssetContentType
    """Asset content type, e.g., "image/png" """

    uploadDate: str  # ISO formatted
    """ISO 8601-formatted asset upload date"""

    length: int  # in bytes
    """Size of file in bytes"""

    chunkSize: int  # in bytes
    """File chunk size in bytes"""

    md5: str
    """MD5 hash of asset"""

    project_uid: str
    """Associated project UID"""

    job_uid: str  # also used for Session UID
    """Associated job or session UID"""


class EventLogAsset(TypedDict):
    """
    Dictionary item in a job event log's ``imgfiles`` property (in the ``events``
    collection)
    """

    fileid: str
    """Reference to file ``_id`` property in GridFS collection"""

    filename: str
    """File name"""

    filetype: AssetContentType
    """File content type, e.g., "image/png" """


class Datafield(TypedDict):
    """Definition of a prefix field within a CS file."""

    dtype: str
    """Datatype-specific string from common.py. e.g., "movie_blob", "ctf",
    "alignments2D"."""

    prefix: str
    """where to find field in an associated ``.cs`` file. e.g.,
    "alignments_class_1" """

    required: bool
    """whether this field must necessarily exist in a corresponding
    input/output"""


class InputSlot(TypedDict):
    """
    Entry in Job document's input_slot_groups.slots property
    """

    type: Datatype
    """Cryo-EM native data type, e.g., "exposure", "particle" or "volume" """

    name: str
    """Input slot name, e.g., "movie_blob" or "location" """

    title: str
    """Human-readable input slot title"""

    description: str
    """Human-readable description"""

    optional: bool
    """If True, input is not required for the job"""


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
