# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from .session import ExposureGroup, LiveComputeResources
from .session_params import LivePreprocessingParams


class SessionConfigProfile(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    title: str
    created_by_user_id: str
    last_applied_at: Optional[datetime.datetime] = None
    compute_resources: Optional[LiveComputeResources] = None
    exp_groups: List[ExposureGroup] = []
    session_params: dict = {}


class SessionConfigProfileBody(BaseModel):
    title: str
    compute_resources: Optional[LiveComputeResources] = None
    exp_groups: List[ExposureGroup] = []
    session_params: Optional[LivePreprocessingParams] = None
