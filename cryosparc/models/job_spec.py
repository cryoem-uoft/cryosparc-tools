# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, RootModel

from .resource import ResourceSpec

BuilderTag = Literal[
    "new", "interactive", "gpuEnabled", "multiGpu", "utility", "import", "live", "benchmark", "wrapper"
]
"""
Visual indicators for jobs in the builder.
"""

Category = Literal[
    "import",
    "motion_correction",
    "ctf_estimation",
    "exposure_curation",
    "particle_picking",
    "extraction",
    "deep_picker",
    "particle_curation",
    "reconstruction",
    "refinement",
    "ctf_refinement",
    "variability",
    "flexibility",
    "postprocessing",
    "local_refinement",
    "helix",
    "utilities",
    "simulations",
    "live",
    "instance_testing",
    "workflows",
]
"""
Section under which to group a job in the builder.
"""


class InputResult(BaseModel):
    """
    Concrete job input result connection to an output result.
    """

    name: Optional[str] = None
    """
    Input slot name. Passthrough slots have ``name`` set to ``None``.
    """
    dtype: str
    """
    Datatype-specific string from data_registry.py. e.g., stat_blob, ctf,
    alignments2D.
    """
    job_uid: str
    """
    Parent job UID source of this input slot connection.
    """
    output: str
    """
    Name of output in parent job. e.g., "particles"
    """
    result: str
    """
    Name of output result slot in parent job, e.g., "blob". Usually the same
    as "name" but may differ if there are multiple outputs of the same type
    """
    version: Union[int, Literal["F"]] = "F"
    """
    Version number or specifier to use. Usually "F"
    """


class Connection(BaseModel):
    """
    Job input connection details.
    """

    job_uid: str
    """
    Connected parent output job uid.
    """
    output: str
    """
    Name of output on connected parent output job.
    """
    results: List[InputResult] = []
    """
    Specific results from parent job. Some slots may have a different job UID.
    """


class OutputSlot(BaseModel):
    """
    Specification of an output slot in the job configuration. Each output includes one or more.
    """

    name: str
    """
    where to find field in a corresponding .cs file e.g.,
    background_blob
    """
    dtype: str
    """
    Datatype-specific string from data_registry.py. e.g., stat_blob, ctf,
    alignments2D.
    """


class OutputSpec(BaseModel):
    """
    Used for outputs with some generated data based on data forwarded from
    input inheritance
    """

    type: Literal[
        "exposure",
        "particle",
        "template",
        "volume",
        "volume_multi",
        "mask",
        "live",
        "ml_model",
        "symmetry_candidate",
        "flex_mesh",
        "flex_model",
        "hyperparameter",
        "denoise_model",
        "annotation_model",
    ]
    """
    Cryo-EM native data type, e.g., "exposure", "particle" or "volume"
    """
    title: str
    """
    Human-readable title
    """
    description: str = ""
    """
    Detailed description.
    """
    slots: List[Union[OutputSlot, str]] = []
    """
    Expected input/output result slots.

    "str" is a shortcut for ``Slot(dtype="str", "prefix="str")``

    For input specs:
    "str" is a shortcut for ``InputSlot(dtype="str", "prefix="str", required=True)``
    "?str" is a shortcut for ``InputSlot(dtype="str", "prefix="str", required=False)``
    """
    passthrough: Optional[str] = None
    """
    Associated passthrough input name
    """


class OutputRef(BaseModel):
    """
    Minimal name reference to a specific job output
    """

    job_uid: str
    """
    Connected parent output job uid.
    """
    output: str
    """
    Name of output on connected parent output job.
    """


class InputSlot(BaseModel):
    """
    Specification of an input slot in the job configuration. Each input includes one or more.
    """

    name: str
    """
    where to find field in a corresponding .cs file e.g.,
    background_blob
    """
    dtype: str
    """
    Datatype-specific string from data_registry.py. e.g., stat_blob, ctf,
    alignments2D.
    """
    required: bool = False
    """
    whether this field must necessarily exist in acorresponding
    input/output
    """


class Input(BaseModel):
    """
    Job input connection details.
    """

    type: Literal[
        "exposure",
        "particle",
        "template",
        "volume",
        "volume_multi",
        "mask",
        "live",
        "ml_model",
        "symmetry_candidate",
        "flex_mesh",
        "flex_model",
        "hyperparameter",
        "denoise_model",
        "annotation_model",
    ]
    """
    Cryo-EM native data type, e.g., "exposure", "particle" or "volume"
    """
    title: str
    """
    Human-readable title
    """
    description: str = ""
    """
    Detailed description.
    """
    slots: List[InputSlot] = []
    """
    Expected low-level input definitions
    """
    count_min: int = 0
    """
    Minimum number of connections to this input.
    """
    count_max: Union[int, Literal["inf"]] = "inf"
    """
    Maximum number of connections supported for this input. Should be any
    integer >= 0 and <= 500. Inputs with a ``count_max`` set to ``"inf"`` also
    support a maximum of 500 connections.
    """
    repeat_allowed: bool = False
    """
    Whether repeated connections to the same output allowed for this input.
    """
    connections: List[Connection] = []
    """
    Connected output details
    """


class InputSpec(BaseModel):
    """
    Input specification. Used to define the expected connections to a job input.
    """

    type: Literal[
        "exposure",
        "particle",
        "template",
        "volume",
        "volume_multi",
        "mask",
        "live",
        "ml_model",
        "symmetry_candidate",
        "flex_mesh",
        "flex_model",
        "hyperparameter",
        "denoise_model",
        "annotation_model",
    ]
    """
    Cryo-EM native data type, e.g., "exposure", "particle" or "volume"
    """
    title: str
    """
    Human-readable title
    """
    description: str = ""
    """
    Detailed description.
    """
    slots: List[Union[InputSlot, str]] = []
    """
    Expected input/output result slots.

    "str" is a shortcut for ``Slot(dtype="str", "prefix="str")``

    For input specs:
    "str" is a shortcut for ``InputSlot(dtype="str", "prefix="str", required=True)``
    "?str" is a shortcut for ``InputSlot(dtype="str", "prefix="str", required=False)``
    """
    count_min: int = 0
    """
    Minimum number of connections to this input.
    """
    count_max: Union[int, Literal["inf"]] = "inf"
    """
    Maximum number of connections supported for this input. Should be any
    integer >= 0 and <= 500. Inputs with a ``count_max`` set to ``"inf"`` also
    support a maximum of 500 connections.
    """
    repeat_allowed: bool = False
    """
    Whether repeated connections to the same output allowed for this input.
    """


class InputSpecs(RootModel):
    """
    Dictionary of input specifications, where each key is the input name.
    """

    root: Dict[str, InputSpec] = {}
    """
    Dictionary of input specifications, where each key is the input name.
    """


class Inputs(RootModel):
    """
    Dictionary of job input connection details, where each key is the input name.
    """

    root: Dict[str, Input] = {}
    """
    Dictionary of job input connection details, where each key is the input name.
    """


class Params(BaseModel):
    """
    Job parameter specifications. See API function projects.get_job_register
    for allowed parameters based on job spec type.
    """

    model_config = ConfigDict(extra="allow")
    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None: ...
        def __getattr__(self, key: str) -> Any: ...


class OutputResult(BaseModel):
    """
    Concrete job output.
    """

    name: str
    """
    where to find field in a corresponding .cs file e.g.,
    background_blob
    """
    dtype: str
    """
    Datatype-specific string from data_registry.py. e.g., stat_blob, ctf,
    alignments2D.
    """
    versions: List[int] = []
    """
    List of available intermediate result version numbers.
    """
    metafiles: List[str] = []
    """
    List of available intermediate result files (same size as ``versions``).
    """
    num_items: List[int] = []
    """
    Number of rows in each metafile
    """
    passthrough: bool = False
    """
    If True, this result is passed through as-is from an associated input.
    """


class Output(BaseModel):
    """
    Job output details. Includes saved dataset paths and summary statistics.
    """

    type: Literal[
        "exposure",
        "particle",
        "template",
        "volume",
        "volume_multi",
        "mask",
        "live",
        "ml_model",
        "symmetry_candidate",
        "flex_mesh",
        "flex_model",
        "hyperparameter",
        "denoise_model",
        "annotation_model",
    ]
    """
    Cryo-EM native data type, e.g., "exposure", "particle" or "volume"
    """
    title: str
    """
    Human-readable title
    """
    description: str = ""
    """
    Detailed description.
    """
    slots: List[OutputSlot] = []
    """
    Low-level output result definitions.
    """
    passthrough: Optional[str] = None
    """
    Associated passthrough input name
    """
    results: List[OutputResult] = []
    """
    All individial outputs based on available output slots
    """
    num_items: int = 0
    """
    Number of items in final version file
    """
    image: Optional[str] = None
    """
    Asset ID of thumbnail for this output
    """
    summary: Dict[str, Any] = {}
    """
    Result dataset summary data
    """
    latest_summary_stats: Dict[str, Any] = {}
    """
    Additional context-specific summary statistics
    """


class Outputs(RootModel):
    """
    Dictionary of job output details, where each key is the output name.
    """

    root: Dict[str, Output] = {}
    """
    Dictionary of job output details, where each key is the output name.
    """


class JobSpec(BaseModel):
    """
    Job's unique specification details. Defines the parameters, inputs, outputs
    and required resources. Contents vary by job type, configured parameters and
    connected inputs.
    """

    type: str
    """
    Job type key, e.g., "import_particles" or "class_2D_new"
    """
    params: Params = Params()
    """
    Parameters for job, attributes vary per job type.

    NOTE: After changing a job parameter, the spec may need to be refreshed.
    Instead of directly modifying this field with job.params.foo = ... ,
    use job.set_param("foo", ...) instead.
    """
    inputs: Inputs = Inputs()
    """
    Connected inputs
    """
    outputs: Outputs = Outputs()
    """
    Available outputs. Empty when job has not run.
    """
    ui_tile_width: int
    """
    Number of horizontal tiles this job should take in the UI.
    """
    ui_tile_height: int
    """
    Number of vertical tiles this job should take in the UI.
    """
    resource_spec: ResourceSpec
    """
    Compute resource requirements for this job.
    """


class JobBuildError(BaseModel):
    type: Literal[
        "no_such_attribute",
        "json_invalid",
        "json_type",
        "needs_python_object",
        "recursion_loop",
        "missing",
        "frozen_field",
        "frozen_instance",
        "extra_forbidden",
        "invalid_key",
        "get_attribute_error",
        "model_type",
        "model_attributes_type",
        "dataclass_type",
        "dataclass_exact_type",
        "none_required",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "multiple_of",
        "finite_number",
        "too_short",
        "too_long",
        "iterable_type",
        "iteration_error",
        "string_type",
        "string_sub_type",
        "string_unicode",
        "string_too_short",
        "string_too_long",
        "string_pattern_mismatch",
        "enum",
        "dict_type",
        "mapping_type",
        "list_type",
        "tuple_type",
        "set_type",
        "set_item_not_hashable",
        "bool_type",
        "bool_parsing",
        "int_type",
        "int_parsing",
        "int_parsing_size",
        "int_from_float",
        "float_type",
        "float_parsing",
        "bytes_type",
        "bytes_too_short",
        "bytes_too_long",
        "bytes_invalid_encoding",
        "value_error",
        "assertion_error",
        "literal_error",
        "date_type",
        "date_parsing",
        "date_from_datetime_parsing",
        "date_from_datetime_inexact",
        "date_past",
        "date_future",
        "time_type",
        "time_parsing",
        "datetime_type",
        "datetime_parsing",
        "datetime_object_invalid",
        "datetime_from_date_parsing",
        "datetime_past",
        "datetime_future",
        "timezone_naive",
        "timezone_aware",
        "timezone_offset",
        "time_delta_type",
        "time_delta_parsing",
        "frozen_set_type",
        "is_instance_of",
        "is_subclass_of",
        "callable_type",
        "union_tag_invalid",
        "union_tag_not_found",
        "arguments_type",
        "missing_argument",
        "unexpected_keyword_argument",
        "missing_keyword_only_argument",
        "unexpected_positional_argument",
        "missing_positional_only_argument",
        "multiple_argument_values",
        "url_type",
        "url_parsing",
        "url_syntax_violation",
        "url_too_long",
        "url_scheme",
        "uuid_type",
        "uuid_parsing",
        "uuid_version",
        "decimal_type",
        "decimal_parsing",
        "decimal_max_digits",
        "decimal_max_places",
        "decimal_whole_digits",
        "complex_type",
        "complex_str_parsing",
    ]
    """
    values based on https://docs.pydantic.dev/latest/errors/validation_errors
    """
    loc: List[Union[str, int]]
    """
    path to the invalid property
    """
    input: Any
    """
    value of the invalid property - must be serializable
    """
    ctx: Dict[str, Any] = {}
    """
    error context for pydantic
    """
    input_type: str
    """
    """


Stability = Literal["develop", "beta", "stable", "legacy", "obsolete"]
"""
Lifecycle/development stage for a job. Jobs will change stabilities as they
are continually developed or replaced with other jobs.
"""


class OutputSpecs(RootModel):
    """
    Dictionary of output specifications, where each key is the output name.
    """

    root: Dict[str, OutputSpec] = {}
    """
    Dictionary of output specifications, where each key is the output name.
    """


class JobRegisterError(BaseModel):
    """
    Error that occurs when loading a developer job register.
    """

    type: str
    """
    """
    message: str
    """
    """
    traceback: str
    """
    """
