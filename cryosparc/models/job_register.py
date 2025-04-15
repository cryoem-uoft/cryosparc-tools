# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .job_spec import BuilderTag, Category, InputSpecs, JobRegisterError, OutputSpecs, Stability


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
    params: Dict[str, Any]


class JobRegister(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    specs: List[JobRegisterJobSpec] = []
    error: Optional[JobRegisterError] = None
    username: Optional[str] = None
    categories: Dict[str, str]
