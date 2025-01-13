# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Optional

from pydantic import BaseModel

from .job_spec import OutputRef, OutputSpec


class ExternalOutputSpec(BaseModel):
    """
    Specification for an external job with a single output.
    """

    name: str
    spec: OutputSpec
    connection: Optional[OutputRef] = None
