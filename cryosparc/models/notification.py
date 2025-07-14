# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .job import JobStatus


class Notification(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    project_uid: Optional[str] = None
    job_uid: Optional[str] = None
    message: str
    progress_pct: Optional[float] = None
    active: bool = True
    status: Optional[Literal["success", "primary", "warning", "danger"]] = "success"
    icon: str = "flag"
    hide: bool = False
    job_status: Optional[JobStatus] = None
    ttl_seconds: Optional[int] = 7
