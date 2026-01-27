# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .job_spec import BuilderTag, Category, InputSpecs, JobRegisterError, OutputSpecs, Stability
from .params import ParamSection
from .when import When


class JobRegisterParamTypeSpec(BaseModel):
    type: Literal["string", "integer", "number", "boolean", "null", "array"]
    """
    Basic type of the parameter.
    """
    format: Optional[Literal["path", "resources"]] = None
    """
    Expected format or subtype of the parameter, if applicable.
    """


class JobRegisterParamSpec(BaseModel):
    """
    Full specification for a job parameter.
    """

    type: Optional[Literal["string", "integer", "number", "boolean", "null", "array"]] = None
    """
    Basic type of the parameter. Will be empty if 'anyOf' is used.
    """
    anyOf: List[JobRegisterParamTypeSpec] = []
    """
    List of possible types for union or nullable parameters.
    """
    items: Optional[JobRegisterParamTypeSpec] = None
    """
    Specification for items in an 'array' type parameter.
    """
    format: Optional[Literal["path", "resources"]] = None
    """
    Expected format or subtype of the parameter, if applicable.
    """
    default: Union[str, int, float, str, List[str], None] = None
    """
    Default value for the parameter.
    """
    title: Optional[str]
    """
    Human-readable title of the parameter.
    """
    description: Optional[str] = None
    """
    Detailed description of the parameter.
    """
    examples: Optional[List[Any]] = None
    """
    Example values for the parameter.
    """
    gt: Optional[float] = None
    """
    Greater than constraint for numeric parameters.
    """
    ge: Optional[float] = None
    """
    Greater than or equal to constraint for numeric parameters.
    """
    lt: Optional[float] = None
    """
    Less than constraint for numeric parameters.
    """
    le: Optional[float] = None
    """
    Less than or equal to constraint for numeric parameters.
    """
    multiple_of: Optional[float] = None
    """
    Multiple of constraint for numeric parameters.
    """
    enum: Optional[List[Union[str, int, float, str, List[str]]]] = None
    """
    List of allowed values for the parameter, if applicable.
    """
    labels: Optional[List[str]] = None
    """
    Human-readable labels for each value in enum.
    """
    pattern: Optional[str] = None
    """
    Regex pattern that string parameters must match.
    """
    hidden: Union[bool, When] = False
    """
    Whether the parameter is hidden in the UI.
    """
    advanced: bool = False
    """
    Whether the parameter is considered advanced.
    """
    section: Optional[ParamSection] = None
    """
    Section the parameter belongs to in the UI.
    """
    allowed: List[Literal["dir", "file", "glob"]] = []
    """
    Allowed path types for path parameters that have type 'string' and format 'path'.
    """
    validate_path: bool = True
    """
    Whether to validate that paths parameter values exist and are the right allowed type before launching a job.
    """
    legacy_name: Optional[str] = None
    """
    Legacy name of the parameter in previous versions of CryoSPARC, if applicable.
    """
    required_param: bool = False
    """
    Whether the parameter must be set to launch the job.
    """
    ecl_visible: bool = False
    """
    :meta private:
    """


class JobRegisterJobSpec(BaseModel):
    """
    Full specification for a job type in the job register.
    """

    type: str
    """
    Unique identifier for the job type, e.g., 'import_movies'.
    """
    title: str
    """
    Human-readable title of the job type.
    """
    shorttitle: str
    """
    Short title of the job type.
    """
    description: str
    """
    Detailed description of the job type.
    """
    stability: Stability
    """
    Stability level of the job type. 'obsolete' jobs cannot be created or run.
    """
    category: Category
    """
    Category of the job type for grouping in the UI.
    """
    tags: List[BuilderTag] = []
    """
    Job Builder tags associated with the job type.
    """
    hidden: bool = False
    """
    Whether the job type is hidden in the UI.
    """
    interactive: bool = False
    """
    Whether this is an interactive job. Jobs of this type can only run directly on master.
    """
    lightweight: bool = False
    """
    Whether this is a lightweight job. Lightweight jobs never require GPUs
    and can run on either master or workers.
    """
    inputs: InputSpecs
    """
    Input specifications for the job type.
    """
    outputs: OutputSpecs
    """
    Output specifications for the job type.
    """
    params: Dict[str, JobRegisterParamSpec]
    """
    Parameter specifications for the job type.
    """


class JobRegister(BaseModel):
    """
    A registry of all available job types and their specifications.
    """

    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was last modified.
    """
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was first created. Imported objects such as projects
    and jobs will retain the created time from their original CryoSPARC instance.
    """
    specs: List[JobRegisterJobSpec] = []
    """
    List of job type specifications in the registry.
    """
    error: Optional[JobRegisterError] = None
    """
    Error information if there was an issue generating the registry.
    """
    username: Optional[str] = None
    """
    :meta private:
    """
    categories: Dict[str, str]
    """
    Category values to titles map which also indicates order in which jobs
    should be grouped.
    """
