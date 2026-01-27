# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class ImportSignature(BaseModel):
    """
    Binary signatures of imported paths used in import jobs and sessions,
    to track re-importing the same data
    """

    count: int = 0
    """
    Number of imported paths
    """
    signatures: str = ""
    """
    Binary signatures of imported paths
    """
