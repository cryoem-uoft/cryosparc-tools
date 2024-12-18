# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import List, Optional, Union

from pydantic import BaseModel

from .instance import InstanceInformation
from .scheduler_target import Cluster, Node


class SchedulerTargetInformation(BaseModel):
    cache_path: Optional[str] = None
    cache_reserve_mb: Optional[int] = None
    cache_quota_mb: Optional[int] = None
    lane: str
    name: str
    title: str
    desc: Optional[str] = None
    hostname: str
    worker_bin_path: str
    config: Union[Node, Cluster]
    instance_information: Optional[InstanceInformation] = None


class RuntimeDiagnostics(BaseModel):
    cryosparc_version: str
    cryosparc_patch: str
    instance_information: InstanceInformation
    scheduler_targets: List[SchedulerTargetInformation]
    db_stats: dict
    date_generated: datetime.datetime
