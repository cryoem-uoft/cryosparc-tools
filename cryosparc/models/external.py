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
    """
    Name of output to create on external job.
    """
    spec: OutputSpec
    """
    Specification of output.
    """
    connection: Optional[OutputRef] = None
    """
    Optional source job UID and output to connect and passthrough unmodified
    results to the output. Must have the same type as in the output spec.
    Input name to create and connect should be defined in ``spec.passthrough``,
    otherwise will default to ``connection.output``.
    """
