# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, RootModel

BuilderTag = Literal[
    "new",
    "beta",
    "legacy",
    "interactive",
    "gpuEnabled",
    "multiGpu",
    "utility",
    "import",
    "live",
    "benchmark",
    "wrapper",
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
    job_uid: str
    output: str
    results: List[InputResult] = []


class InputSlot(BaseModel):
    """
    Specification of an input slot in the job configuration. Part of a group.
    """

    name: str
    dtype: str
    required: bool = False


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
    root: Dict[str, List[Connection]] = {}


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
    results: List[OutputResult] = []
    num_items: int = 0
    image: Optional[str] = None
    summary: dict = {}
    latest_summary_stats: dict = {}


class Outputs(RootModel):
    root: Dict[str, Output] = {}


class JobSpec(BaseModel):
    type: str
    params: Params
    inputs: Inputs = Inputs()
    outputs: Outputs = Outputs()


Stability = Literal["develop", "beta", "stable", "legacy", "obsolete"]
"""
Lifecycle/development stage for a job. Jobs will change stabilities as they
are continually developed or replaced with other jobs.
"""


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
    passthrough_exclude_slots: List[str] = []


class OutputSpecs(RootModel):
    root: Dict[str, OutputSpec] = {}


class JobRegisterError(BaseModel):
    """
    Error that occurs when loading a developer job register.
    """

    type: str
    message: str
    traceback: str


class ResourceSpec(BaseModel):
    cpu: int = 1
    gpu: int = 0
    ram: int = 1
    ssd: bool = False