# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .session import ExposureGroup, LiveComputeResources
from .session_params import LivePreprocessingParams


class RunConfiguration(BaseModel):
    exposure_processing_priority: Literal["normal", "oldest", "latest", "alternate"] = "normal"
    phase_one_wait_for_exposures: bool = False
    auto_pause: Literal["disabled", "graceful", "immediate"] = "disabled"
    auto_pause_after_idle_minutes: int = 10


class SessionConfigProfile(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was last modified.
    """
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was first created. Imported objects such as projects
    and jobs will retain the created time from their original CryoSPARC instance.
    """
    title: str
    created_by_user_id: str
    last_applied_at: Optional[datetime.datetime] = None
    compute_resources: Optional[LiveComputeResources] = None
    exp_groups: List[ExposureGroup] = []
    session_params: Dict[str, Any] = {}
    run_configuration: Optional[RunConfiguration] = None


class SessionConfigProfileBody(BaseModel):
    title: str
    compute_resources: Optional[LiveComputeResources] = None
    exp_groups: List[ExposureGroup] = []
    session_params: Optional[LivePreprocessingParams] = None
    run_configuration: Optional[RunConfiguration] = None
