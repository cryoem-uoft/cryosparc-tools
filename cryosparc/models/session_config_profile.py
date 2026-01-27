# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .session import ExposureGroup, LiveComputeResources
from .session_params import LivePreprocessingParams


class RunConfiguration(BaseModel):
    """
    Runtime configuration for a session.
    """

    exposure_processing_priority: Literal["normal", "oldest", "latest", "alternate"] = "normal"
    """
    """
    phase_one_wait_for_exposures: bool = False
    """
    Whether to wait for exposures to become available before launching the Live workers.
    """
    auto_pause: Literal["disabled", "graceful", "immediate"] = "disabled"
    """
    Auto-pause configuration for the session.
    """
    auto_pause_after_idle_minutes: int = 10
    """
    Number of idle minutes before auto-pausing the session. Only used if
    ``auto_pause`` is set to ``"enabled"``.
    """


class SessionConfigProfile(BaseModel):
    """
    A profile for session configuration that can be reused across multiple sessions.
    """

    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
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
    """
    """
    created_by_user_id: str
    """
    """
    last_applied_at: Optional[datetime.datetime] = None
    """
    """
    compute_resources: Optional[LiveComputeResources] = None
    """
    """
    exp_groups: List[ExposureGroup] = []
    """
    """
    session_params: Dict[str, Any] = {}
    """
    """
    run_configuration: Optional[RunConfiguration] = None
    """
    """


class SessionConfigProfileBody(BaseModel):
    """
    Required fields to create or update a session configuration profile.
    """

    title: str
    """
    """
    compute_resources: Optional[LiveComputeResources] = None
    """
    """
    exp_groups: List[ExposureGroup] = []
    """
    """
    session_params: Optional[LivePreprocessingParams] = None
    """
    """
    run_configuration: Optional[RunConfiguration] = None
    """
    """
