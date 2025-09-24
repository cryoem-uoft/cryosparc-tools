# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Literal, Optional

from pydantic import BaseModel


class SchedulerLane(BaseModel):
    name: str
    """
    Identifier for this lane.
    """
    type: Literal["node", "cluster"]
    """
    What kind of lane this is based on how on what kind of target(s) it contains.
    """
    title: str
    """
    Human-readable lane title.
    """
    desc: Optional[str] = None
    """
    Optional lane description
    """
