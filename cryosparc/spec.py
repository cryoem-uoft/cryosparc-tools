"""
Type specifications for CryoSPARC database entities.

Unless otherwise noted, classes defined here represent dictionary instances
whose attributes may be accessed with dictionary key syntax.

Examples:

    Accessing job document details

    >>> cs = CryoSPARC()
    >>> job = cs.find_job("P3", "J118")
    >>> job.doc["output_results"][0]["metafiles"]
    [
      "J118/J118_000_particles.cs",
      "J118/J118_001_particles.cs",
      "J118/J118_002_particles.cs",
      "J118/J118_003_particles.cs"
    ]
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from typing_extensions import Literal, TypedDict

# Database document
D = TypeVar("D", bound=TypedDict)


Datatype = Literal[
    "exposure", "particle", "template", "volume", "mask", "ml_model", "symmetry_candidate", "flex_mesh", "flex_model"
]
"""Supported data types for job inputs and outputs."""


JobStatus = Literal[
    "building",
    "queued",
    "launched",
    "started",
    "running",
    "waiting",
    "completed",
    "killed",
    "failed",
]
"""
Possible job status values.
"""

# Valid plot file types
TextFormat = Literal["txt", "csv", "html", "json", "xml"]
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
    "text/html",
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
    "html": "text/html",
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
ASSET_EXTENSIONS = {v: k for k, v in ASSET_CONTENT_TYPES.items()}


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
    input/output. Assumed to be ``True`` if not specified"""


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


class BaseParam(TypedDict):
    """
    Base parameter specification.
    """

    value: bool
    """Base parameter value. Should not be changed."""

    title: str
    """Human-readable parameter title."""

    desc: str
    """Human-readable parameter description."""

    order: int
    """Parameter order in the builder list."""

    section: str
    """Parameter section identifier."""

    advanced: bool
    """True if this is an advanced parameter (hidden unlesss the "Advanced"
    checkbox is enabled in the Job Builder"."""

    hidden: bool
    """If True, this parameter is always hidden from the interface."""


class Param(BaseParam):
    """
    Specifies possible values for type property. Inherits from
    BaseParam_.

    .. _BaseParam:
        #cryosparc.spec.BaseParam
    """

    type: Literal["number", "string", "boolean"]
    """Possible Parameter type."""


class EnumParam(BaseParam):
    """
    Additional Param keys available for enum params. Inherits from BaseParam_.

    .. _BaseParam:
        #cryosparc.spec.BaseParam
    """

    type: Literal["enum"]
    """Possible Parameter type."""

    enum_keys: List[str]
    """Possible enum names for display for selection. Parameter must be set to
    one of these values."""

    enum_dict: Dict[str, Any]
    """Map from enum key names to their equivalent values."""


class PathParam(BaseParam):
    """
    Additional Param key available for path params. Inherits Inherits from
    BaseParam_.

    .. _BaseParam:
        #cryosparc.spec.BaseParam
    """

    type: Literal["path"]

    path_dir_allowed: bool
    """If True, directories may be specified."""

    path_file_allowed: bool
    """If True, files may be specified."""

    path_glob_allowed: bool
    """If True, a wildcard string that refers to many files may be specified.."""


class ParamSpec(TypedDict):
    """Param specification. Dictionary with single ``"value"`` key."""

    value: Any
    """Value of param."""


class ProjectLastAccessed(TypedDict, total=False):
    """
    Details on when a project was last accessed.
    """

    name: str
    """User account name that accessed this project."""

    accessed_at: str
    """Last access date in ISO 8601 format."""


class ProjectDocument(TypedDict):
    """
    Specification for a project document in the MongoDB database.
    """

    _id: str
    """MongoDB ID"""

    uid: str
    """Project unique ID, e.g., "J42"."""

    uid_num: int
    """Project number, e.g., 42."""

    title: str
    """Human-readable Project title."""

    description: str
    """Human-readable project markdown description."""

    project_dir: str
    """Project directory on disk. May include unresolved shell variables."""

    project_params_pdef: dict
    """Project-level job parameter default definitions."""

    owner_user_id: str
    """Object ID of user account that created this project."""

    created_at: str
    """Project creation date in ISO 8601 format."""

    deleted: bool
    """Whether this project has been deleted from the interface."""

    users_with_access: List[str]
    """Object IDs of user accounts that may access this project."""

    size: int
    """Computed size of project on disk."""

    last_accessed: ProjectLastAccessed
    """Details about when the project was last accessed by a user account."""

    archived: bool
    """Whether this project has been marked as archived from the inteface."""

    detached: bool
    """Whether this project is detached."""

    hidden: bool
    """Whether this project is hidden."""

    project_stats: dict
    """Computed project statistics."""

    generate_intermediate_results: bool
    """Whether intermediate results should be generated on this project."""


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

    status: JobStatus
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

    params_base: Dict[str, Union[Param, EnumParam, PathParam]]
    """Job param specification and their base values. Each key represents a
    parameter name."""

    params_spec: Dict[str, ParamSpec]
    """User-specified parameter values. Each key is a parameter value. Not all
    keys from ``params_base`` are included here, only ones that were explicitly
    set."""

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


class ResourceSlots(TypedDict):
    """
    Listings of available resources on a worker node that may be allocated for
    scheduling.
    """

    CPU: List[int]
    """List of available CPU core indices."""
    GPU: List[int]
    """List of available GPU indices."""
    RAM: List[int]
    """List of available 8GB slots."""


class FixedResourceSlots(TypedDict):
    """
    Available resource slots that only indicate presence, not the amount that
    may be allocated. (i.e., "SSD is available or not available")
    """

    SSD: bool
    """Whether this target thas an SSD"""


class Gpu(TypedDict):
    """
    GPU details for a target.
    """

    id: int
    """Index of GPU. Generally based on which PCI slot the GPU occupies."""
    name: str
    """Identifiable model name for this GPU, e.g.,"GeForce RTX 3090"."""
    mem: int
    """Amount of memory available on this GPU, in bytes."""


class SchedulerLane(TypedDict):
    """
    Description for a CryoSPARC scheduler lane.
    """

    name: str
    """Identifier for this lane."""
    type: Literal["node", "cluster"]
    """What kind of lane this is based on how on what kind of target(s) it contains."""
    title: str
    """Human-readable lane title."""
    desc: str
    """Human-readable lane description."""


class BaseSchedulerTarget(TypedDict):
    """
    Properties shared by both node and cluster scheduler targets.
    """

    lane: str
    """Lane name this target belongs to."""

    name: str
    """Identifier for this target."""

    title: str
    """Human-readable title for this target."""

    desc: Optional[str]
    """Human-readable description for this target."""

    hostname: str
    """Network machine hostname (same as name for for clusters)."""

    worker_bin_path: str
    """Path to cryosparc_worker/bin/cryosparcw executable."""

    cache_path: Optional[str]
    """Path the SSD cache scratch directory, if applicable."""

    cache_reserve_mb: int  # 10G default
    """Ensure at least this much space is free on the SSD scratch drive before
    caching."""

    cache_quota_mb: int
    """Do not cache more than this amoun on the SSD scrath drive.."""


class SchedulerTargetNode(BaseSchedulerTarget):
    """
    node-type scheduler target that does not include GPUs. Inherits from
    BaseSchedulerTarget_.

    .. _BaseSchedulerTarget:
        #cryosparc.spec.BaseSchedulerTarget
    """

    type: Literal["node"]
    """Node scheduler targets have type "node"."""

    ssh_str: str
    """Shell command used to access this node, e.g., ``ssh cryosparcuser@worker``."""

    resource_slots: ResourceSlots
    """Available compute resources."""

    resource_fixed: FixedResourceSlots
    """Available fixed resources."""

    monitor_port: Optional[int]
    """Not used."""


class SchedulerTargetGpuNode(SchedulerTargetNode):
    """
    node-type scheduler target that includes GPUs. Inherits from
    BaseSchedulerTarget_ and SchedulerTargetNode_.

    .. _BaseSchedulerTarget:
        #cryosparc.spec.BaseSchedulerTarget
    .. _SchedulerTargetNode:
        #cryosparc.spec.SchedulerTargetNode
    """

    gpus: List[Gpu]
    """Details about GPUs available on this node."""


class SchedulerTargetCluster(BaseSchedulerTarget):
    """
    Cluster-type scheduler targets. Inherits from BaseSchedulerTarget_.

    .. _BaseSchedulerTarget:
        #cryosparc.spec.BaseSchedulerTarget
    """

    type: Literal["cluster"]
    """Cluster scheduler targets have type "cluster"."""

    script_tpl: str
    """Full cluster submission script Jinja template."""

    send_cmd_tpl: str
    """Template command to access the cluster and running commands."""

    qsub_cmd_tpl: str
    """Template command to submit jobs to the cluster."""

    qstat_cmd_tpl: str
    """Template command to check the cluster job by its ID."""

    qdel_cmd_tpl: str
    """Template command to delete cluster jobs."""

    qinfo_cmd_tpl: str
    """Template command to check cluster queue info."""


SchedulerTarget = Union[SchedulerTargetNode, SchedulerTargetGpuNode, SchedulerTargetCluster]
"""
Scheduler target details.
"""


class JobSection(TypedDict):
    """
    Specification of available job types of a certain category.

    Example:

        >>> {
        ...     "name": "refinement",
        ...     "title": "3D Refinement",
        ...     "description: "...",
        ...     "contains" : [
        ...         "homo_refine",
        ...         "hetero_refine",
        ...         "nonuniform_refine",
        ...         "homo_reconstruct"
        ...     ]
        ... }
    """

    name: str
    """Section identifier."""
    title: str
    """Human-readable section title."""
    description: str
    """Human-readable section description."""
    contains: List[str]
    """Job type identifiers contained by this section."""


class MongoController(ABC, Generic[D]):
    """
    Abstract base class for Project, Workspace, Job classes and any other types
    that have underlying Mongo database documents.

    Generic type argument D is a typed dictionary definition for a Mongo
    document.

    :meta private:
    """

    _doc: Optional[D] = None

    @property
    def doc(self) -> D:
        if not self._doc:
            self.refresh()
        assert self._doc, "Could not refresh database document"
        return self._doc

    @abstractmethod
    def refresh(self):
        # Must be implemented in subclasses
        return self
