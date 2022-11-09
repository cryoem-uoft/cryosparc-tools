"""
Unless otherwise noted, classes defined here are ``TypedDict`` instances which
whose attributes may be accessed with dictionary syntax.

Examples:

    Accessing job document details

    >>> cs = CryoSPARC()
    >>> job = cs.find_job("P3", "J42")
    >>> print(job.doc["output_results"][0]["metafiles"])
    [
      "J118/J118_000_particles.cs",
      "J118/J118_001_particles.cs",
      "J118/J118_002_particles.cs",
      "J118/J118_003_particles.cs"
    ]
"""
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, TypedDict


Datatype = Literal["exposure", "particle", "template", "volume", "mask"]

# Valid plot file types
TextFormat = Literal["txt", "csv", "json", "xml"]
"""
Supported job stream log asset file text formats.
"""

ImageFormat = Literal["pdf", "gif", "jpg", "jpeg", "png", "svg"]
"""
Supported job stream log asset file image formats.
"""

AssetFormat = Union[TextFormat, ImageFormat]
"""
Supported job stream log asset file formats.
"""

TextContentType = Literal[
    "text/plain",
    "text/csv",
    "application/json",
    "application/xml",
]
"""
Supported job stream log text asset MIME types.
"""

ImageContentType = Literal[
    "application/pdf",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
]
"""
Supported job image asset MIME types.
"""

AssetContentType = Union[TextContentType, ImageContentType]
"""
Supported job asset MIME types.
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
    Result of job asset files query. Keys may be accessed with dictionary
    syntax.

    Examples:

        >>> print(details['filename'])
        image.png
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
    Dictionary item in a job event log's ``imgfiles`` property (in the
    ``events`` collection). Keys may be accessed with dictionary key syntax.

    Examples:

        >>> print(asset['filename'])
        image.png
    """

    fileid: str
    """Reference to file ``_id`` property in GridFS collection"""

    filename: str
    """File name"""

    filetype: AssetContentType
    """File content type, e.g., "image/png" """


class Datafield(TypedDict):
    """
    Definition of a prefix field within a CS file.

    Examples:

        >>> field = Datafield(dtype='alignments3D', prefix='alignments_class_0', required=False)
        >>> print(field['dtype'])
        alignments3D
    """

    dtype: str
    """Datatype-specific string from based on entry in
    ``cryosparc_compute/jobs/common.py``. e.g., "movie_blob", "ctf",
    "alignments2D"."""

    prefix: str
    """where to find field in an associated ``.cs`` file. e.g.,
    "alignments_class_1" """

    required: bool
    """whether this field must necessarily exist in a corresponding
    input/output"""


class InputSlot(TypedDict):
    """
    Dictionary entry in Job document's ``input_slot_groups.slots`` property.
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
    """
    Slots specified entry in a Job document's ``input_slot_groups[].connections[].slots`` list.
    """

    slot_name: Optional[str]
    """Passthrough slots have ``slot_name`` set to ``None``."""

    job_uid: str
    """Parent job UID source of this input slot connection."""

    group_name: str
    """Name of output group in parent job. e.g., "particles" """

    result_name: str
    """Name of output slot in parent job, e.g., "blob" """

    result_type: str
    """Type of result slot based on entry in ``cryosparc_compute/jobs/common.py``, e.g., "particle.blob" """

    version: Union[int, Literal["F"]]
    """Version number or specifier to use. Usually "F" """


class Connection(TypedDict):
    """
    Connection element specified in a Job document's ``input_slot_groups[].connections`` list.
    """

    job_uid: str
    """Parent job UID source of main input group connection."""

    group_name: str
    """Name of output group in parent job. e.g., "particles" """

    slots: List[ConnectionSlot]
    """List of connection specifiers for each slot"""


class InputSlotGroup(TypedDict):
    """Element specified in a Job document's ``input_slot_groups`` list."""

    type: Datatype
    """Possible Cryo-EM data type for this group, e.g., "particle"."""

    name: str
    """Input group name, e.g., "particles"."""

    title: str
    """Human-readable input group title."""

    description: str
    """Human-readable input group description."""

    count_min: int
    """Minimum required output groups that may be connected to this input slot."""

    count_max: Optional[int]
    """Maximum allowed output groups that may be connected to this input slot. Infinity if not specified."""

    repeat_allowed: bool
    """If True, the same output group may be connected twice."""

    slots: List[InputSlot]
    """List of slot definitions in the input group."""

    connections: List[Connection]
    """Connected output for this input group."""


class OutputResultGroupContains(TypedDict):
    """
    Elements of a Job document's ``output_result_groups[].contains`` list.
    """

    uid: str
    """Result unique ID, e.g., "J42-R1"."""

    type: str
    """Result type based on entry in ``cryosparc_compute/jobs/common.py``, e.g., "particle.alignments3D"."""

    name: str
    """Name of output result (a.k.a. slot), e.g., "alignments_class_1"."""

    group_name: str
    """Name of output group, e.g., "particles"."""

    passthrough: bool
    """If True, this result is passed through as-is from an associated input."""


class OutputResultGroup(TypedDict):
    """
    Elements of a Job document's ``output_result_groups`` list.
    """

    uid: str
    """Ouptut group unique ID, e.g., "J42-G1"."""

    type: Datatype
    """Possible Cryo-EM data type for this group, e.g., "particle"."""

    name: str
    """Output group name, e.g., "particles_selected" """

    title: str
    """Human-readable output group title."""

    description: str
    """Human-readable output group description."""

    contains: List[OutputResultGroupContains]
    """List of specific results (a.k.a. slots) in this output group."""

    passthrough: Union[str, Literal[False]]
    """Either ``False`` if this is a newly-created output or the name of an
    input group used to forward passthrough slots for this result group."""

    num_items: int
    """Number of rows in the dataset for this result group populated by jobs when they run."""

    summary: dict
    """Context-specific details about this result populated by jobs when they run."""


class OutputResult(TypedDict):
    """
    Detailed schema and metadata for a Job document's ``output_results`` list.
    Similar to a flattened ``output_result_groups[].contains`` but with more
    details.
    """

    uid: str
    """Result unique ID, e.g., "J42-R1"."""

    type: str
    """Result type based on entry in ``cryosparc_compute/jobs/common.py``, e.g., "particle.alignments3D"."""

    name: str
    """Name of output result (a.k.a. slot), e.g., "alignments_class_1"."""

    group_name: str
    """Name of output group, e.g., "particles"."""

    title: str
    """Human-readable output result title."""

    description: str
    """Human-readable output result description."""

    versions: List[int]
    """List of available intermediate result version numbers."""

    metafiles: List[str]
    """List of available intermediate result files (same size as ``versions``)."""

    min_fields: List[Tuple[str, str]]
    """Minimum included dataset field definitions in this result."""

    num_items: int
    """Number of rows in the dataset for this result populated by jobs when they run."""

    passthrough: bool
    """If True, this result is passed through as-is from an associated input."""


class JobDocument(TypedDict):
    """
    Specification for a Job document from the MongoDB database.
    """

    _id: str
    """MongoDB ID"""

    uid: str
    """Job unique ID, e.g., "J42"."""
    uid_num: int
    """Job number, e.g., 42."""

    project_uid: str
    """Project unique ID, e.g., "P3"."""

    project_uid_num: int
    """Project number, e.g., 3."""

    job_type: str
    """Job type identifier, e.g., "class2d"."""

    title: str
    """Human-readable job title."""

    description: str
    """Human-readable job markdown description."""

    status: str
    """Job scheduling status, e.g., "building", "queued", "running"."""

    created_at: str
    """Job creation date in ISO 8601 format."""

    created_by_user_id: Optional[str]
    """Object ID of user account that created this job."""

    deleted: bool
    """True if the job has been marked as deleted."""

    parents: List[str]
    """List of parent jobs UIDs based on input connections."""

    children: List[str]
    """List of child job UIDs based on output connections."""

    input_slot_groups: List[InputSlotGroup]
    """Input group specifications, including schema and connection information."""

    output_result_groups: List[OutputResultGroup]
    """Output group specifications."""

    output_results: List[OutputResult]
    """Aggregated output results specification (similar to
    ``output_result_groups`` with additional field information)."""

    workspace_uids: List[str]
    """List of workspace UIDs this job belongs to."""


class WorkspaceDocument(TypedDict):
    """
    Specification for a Workspace document from the MongoDB database.
    Live-related fields are not yet included.
    """

    _id: str
    """MongoDB ID"""

    uid: str
    """Workspace unique ID, e.g., "W1"."""

    uid_num: int
    """Workspace number, e.g., 1."""

    project_uid: str
    """Project unique ID, e.g., "P3"."""

    project_uid_num: int
    """Project number, e.g., 3."""

    created_at: str
    """Workspace creation date in ISO 8601 format."""

    created_by_user_id: str
    """Object ID of user account that created this workspace."""

    deleted: bool
    """True if the workspace has been marked as deleted."""

    title: str
    """Human-readable workspace title."""

    description: Optional[str]
    """Human-readable workspace markdown description."""

    workspace_type: Literal["base", "live"]
    """Either "live" or "base". """
