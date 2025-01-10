# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Optional

from pydantic import BaseModel

from .job_spec import OutputRef, OutputSpec


class ExternalOutputSpec(BaseModel):
    """
    Specification used to create an external job with an output defined by
    the spec.
    """

    name: str
    spec: OutputSpec
    connection: Optional[OutputRef] = None
