# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, RootModel

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
    dtype: str
    job_uid: str
    output: str
    result: str
    version: Union[int, str] = "F"


class Connection(BaseModel):
    """
    Job input connection details.
    """

    job_uid: str
    output: str
    results: List[InputResult] = []


class OutputSlot(BaseModel):
    """
    Specification of an output slot in the job configuration. Part of a group
    """

    name: str
    dtype: str


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
    title: str
    description: str = ""
    slots: List[Union[OutputSlot, str]] = []
    passthrough: Optional[str] = None


class OutputRef(BaseModel):
    """
    Minimal name reference to a specific job output
    """

    job_uid: str
    output: str


class InputSlot(BaseModel):
    """
    Specification of an input slot in the job configuration. Part of a group.
    """

    name: str
    dtype: str
    required: bool = False


class Input(BaseModel):
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
    title: str
    description: str = ""
    slots: List[InputSlot] = []
    count_min: int = 0
    count_max: Union[int, str] = "inf"
    repeat_allowed: bool = False
    connections: List[Connection] = []


class InputSpec(BaseModel):
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
    title: str
    description: str = ""
    slots: List[Union[InputSlot, str]] = []
    count_min: int = 0
    count_max: Union[int, str] = "inf"
    repeat_allowed: bool = False


class InputSpecs(RootModel):
    root: Dict[str, InputSpec] = {}


class Inputs(RootModel):
    root: Dict[str, Input] = {}


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
    dtype: str
    versions: List[int] = []
    metafiles: List[str] = []
    num_items: List[int] = []
    passthrough: bool = False


class Output(BaseModel):
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
    title: str
    description: str = ""
    slots: List[OutputSlot] = []
    passthrough: Optional[str] = None
    results: List[OutputResult] = []
    num_items: int = 0
    image: Optional[str] = None
    summary: Dict[str, Any] = {}
    latest_summary_stats: Dict[str, Any] = {}


class Outputs(RootModel):
    root: Dict[str, Output] = {}


class ResourceSpec(BaseModel):
    cpu: int = 1
    gpu: int = 0
    ram: int = 1
    ssd: bool = False


class JobSpec(BaseModel):
    type: str
    params: Params = Params()
    inputs: Inputs = Inputs()
    outputs: Outputs = Outputs()
    ui_tile_width: int
    ui_tile_height: int
    resource_spec: ResourceSpec


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
    loc: List[Union[str, int]]
    input: Any
    ctx: Dict[str, Any] = {}
    input_type: str


Stability = Literal["develop", "beta", "stable", "legacy", "obsolete"]
"""
Lifecycle/development stage for a job. Jobs will change stabilities as they
are continually developed or replaced with other jobs.
"""


class OutputSpecs(RootModel):
    root: Dict[str, OutputSpec] = {}


class JobRegisterError(BaseModel):
    """
    Error that occurs when loading a developer job register.
    """

    type: str
    message: str
    traceback: str
