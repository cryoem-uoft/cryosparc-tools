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
    format: Optional[Literal["path", "resources"]] = None


class JobRegisterParamSpec(BaseModel):
    type: Optional[Literal["string", "integer", "number", "boolean", "null", "array"]] = None
    anyOf: List[JobRegisterParamTypeSpec] = []
    items: Optional[JobRegisterParamTypeSpec] = None
    format: Optional[Literal["path", "resources"]] = None
    default: Union[str, int, float, str, List[str], None] = None
    title: Optional[str]
    description: Optional[str] = None
    examples: Optional[List[Any]] = None
    gt: Optional[float] = None
    ge: Optional[float] = None
    lt: Optional[float] = None
    le: Optional[float] = None
    multiple_of: Optional[float] = None
    enum: Optional[List[Union[str, int, float, str, List[str]]]] = None
    labels: Optional[List[str]] = None
    pattern: Optional[str] = None
    hidden: Union[bool, When] = False
    advanced: bool = False
    section: Optional[ParamSection] = None
    allowed: List[Literal["dir", "file", "glob"]] = []
    validate_path: bool = True
    legacy_name: Optional[str] = None
    ecl_visible: bool = False
    required_param: bool = False


class JobRegisterJobSpec(BaseModel):
    type: str
    title: str
    shorttitle: str
    description: str
    stability: Stability
    category: Category
    tags: List[BuilderTag] = []
    hidden: bool = False
    interactive: bool = False
    lightweight: bool = False
    inputs: InputSpecs
    outputs: OutputSpecs
    params: Dict[str, JobRegisterParamSpec]


class JobRegister(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    specs: List[JobRegisterJobSpec] = []
    error: Optional[JobRegisterError] = None
    username: Optional[str] = None
    categories: Dict[str, str]
