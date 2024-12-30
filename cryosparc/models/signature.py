# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class ImportSignature(BaseModel):
    """
    Binary signatures of imported paths used in import jobs and sessions.
    Meant to analyze unique imports across projects, but currently unused.
    """

    count: int = 0
    signatures: str = ""
