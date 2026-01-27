# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from .gpu import Gpu
from .resource import FixedResourceSlots, ResourceSlots


class Node(BaseModel):
    """
    Node-type scheduler target that may include GPUs
    """

    type: Literal["node"]
    """
    Node scheduler targets have type "node".
    """
    ssh_str: str
    """
    Shell command used to access this node, e.g., ``ssh cryosparcuser@worker``.
    """
    resource_slots: ResourceSlots = ResourceSlots()
    """
    Available compute resources.
    """
    resource_fixed: FixedResourceSlots = FixedResourceSlots()
    """
    Available fixed resources.
    """
    monitor_port: Optional[int] = None
    """
    Not used.
    """
    gpus: Optional[List[Gpu]] = None
    """
    Details for all GPUs available on this node, including those excluded from ``resource_slots``.
    """


class Cluster(BaseModel):
    """
    Cluster-type scheduler targets details
    """

    send_cmd_tpl: str = "{{ command }}"
    """
    Template command to access the cluster and running commands.
    """
    qsub_cmd_tpl: str = "qsub {{ script_path_abs }}"
    """
    Template command to submit jobs to the cluster.
    """
    qstat_cmd_tpl: str = "qstat -as {{ cluster_job_id }}"
    """
    Template command to check the cluster job by its ID.
    """
    qdel_cmd_tpl: str = "qdel {{ cluster_job_id }}"
    """
    Template command to delete cluster jobs.
    """
    qinfo_cmd_tpl: str = "qstat -q"
    """
    Template command to check cluster queue info.
    """
    type: Literal["cluster"]
    """
    Cluster scheduler targets have type "cluster".
    """
    script_tpl: str = ""
    """
    Full cluster submission script Jinja template.
    """
    custom_vars: Dict[str, str] = {}
    """
    Custom variable values
    """
    tpl_vars: List[str]
    """
    List of template variable names in a cluster target
    """
    custom_var_names: List[str]
    """
    Computed list if custom variable names defined in the template
    """


class SchedulerTarget(BaseModel):
    """
    Details and configuration for a node or cluster target.
    """

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
    """
    Target configuration details.
    """


class SchedulerTargetCluster(BaseModel):
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
    config: Cluster
    """
    Target configuration details.
    """


class SchedulerTargetNode(BaseModel):
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
    config: Node
    """
    Target configuration details.
    """
