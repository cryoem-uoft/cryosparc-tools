# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Literal, Optional

from pydantic import BaseModel


class SchedulerLane(BaseModel):
    name: str
    type: Literal["node", "cluster"]
    title: str
    desc: Optional[str] = None
