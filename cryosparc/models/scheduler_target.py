# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from .gpu import Gpu


class ResourceSlots(BaseModel):
    """
    Listings of available resources on a worker node that may be allocated for
    scheduling.
    """

    CPU: List[int] = []
    GPU: List[int] = []
    RAM: List[int] = []


class FixedResourceSlots(BaseModel):
    """
    Available resource slots that only indicate presence, not the amount that
    may be allocated. (i.e., "SSD is available or not available")
    """

    SSD: bool = False


class Node(BaseModel):
    """
    Node-type scheduler target that may include GPUs
    """

    type: str
    ssh_str: str
    resource_slots: ResourceSlots = ResourceSlots()
    resource_fixed: FixedResourceSlots = FixedResourceSlots()
    monitor_port: Optional[int] = None
    gpus: Optional[List[Gpu]] = None


class Cluster(BaseModel):
    """
    Cluster-type scheduler targets details
    """

    send_cmd_tpl: str = "{{ command }}"
    qsub_cmd_tpl: str = "qsub {{ script_path_abs }}"
    qstat_cmd_tpl: str = "qstat -as {{ cluster_job_id }}"
    qdel_cmd_tpl: str = "qdel {{ cluster_job_id }}"
    qinfo_cmd_tpl: str = "qstat -q"
    type: str
    script_tpl: str = ""
    custom_vars: Dict[str, str] = {}
    tpl_vars: List[str]
    custom_var_names: List[str]


class SchedulerTarget(BaseModel):
    """
    Details and configuration for a node or cluster target.
    """

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


class SchedulerTarget_Cluster_(BaseModel):
    cache_path: Optional[str] = None
    cache_reserve_mb: Optional[int] = None
    cache_quota_mb: Optional[int] = None
    lane: str
    name: str
    title: str
    desc: Optional[str] = None
    hostname: str
    worker_bin_path: str
    config: Cluster


class SchedulerTarget_Node_(BaseModel):
    cache_path: Optional[str] = None
    cache_reserve_mb: Optional[int] = None
    cache_quota_mb: Optional[int] = None
    lane: str
    name: str
    title: str
    desc: Optional[str] = None
    hostname: str
    worker_bin_path: str
    config: Node
