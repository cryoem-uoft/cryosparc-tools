# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .instance import InstanceInformation
from .scheduler_target import Cluster, Node


class SchedulerTargetInformation(BaseModel):
    cache_path: Optional[str] = None
    """
    Path the SSD cache scratch directory, if applicable.
    """
    cache_reserve_mb: Optional[int] = None
    """
    Ensure at least this much space is free on the SSD scratch drive before
    caching.
    """
    cache_quota_mb: Optional[int] = None
    """
    Do not cache more than this amount on the SSD scrath drive..
    """
    lane: str
    """
    Lane name this target belongs to.
    """
    name: str
    """
    Identifier for this target.
    """
    title: str
    """
    Human-readable title for this target.
    """
    desc: Optional[str] = None
    """
    Human-readable description for this target.
    """
    hostname: str
    """
    Network machine hostname (same as name for for clusters).
    """
    worker_bin_path: str
    """
    Path to cryosparc_worker/bin/cryosparcw executable.
    """
    config: Union[Node, Cluster]
    instance_information: Optional[InstanceInformation] = None


class RuntimeDiagnostics(BaseModel):
    cryosparc_version: str
    cryosparc_patch: str
    instance_information: InstanceInformation
    scheduler_targets: List[SchedulerTargetInformation]
    db_stats: Dict[str, Any]
    date_generated: datetime.datetime
