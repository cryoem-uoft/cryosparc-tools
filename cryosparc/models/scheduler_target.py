# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Literal

from pydantic import BaseModel

from .gpu import Gpu
from .resource import FixedResourceSlots, ResourceSlots
from .version import Arch


class AWSBatchNodeType(BaseModel):
    name: str
    """
    """
    cpus: int
    """
    """
    gpus: int
    """
    """
    ram_gb: int
    """
    """
    ssd_gb: int
    """
    """


class AWSBatchJobQueue(BaseModel):
    arn: str
    """
    """
    node_types: list[AWSBatchNodeType]
    """
    """
    worker_bin_path: str
    """
    Path to cryosparc_worker/bin/cryosparcw executable for this job queue.
    """


class AWSBatch(BaseModel):
    """
    AWS Batch-type scheduler target details
    """

    type: Literal["awsbatch"] = "awsbatch"
    """
    """
    job_definition_arn: str
    """
    ARN of the AWS Batch job definition to use when submitting jobs.
    """
    job_queues: list[AWSBatchJobQueue]
    """
    List of AWS Batch job queues to submit to, with details on the node types available in each queue.
    """
    region_name: str = "us-east-1"
    """
    AWS region name where the Batch job queues are located.
    """
    gpu_concurrency_limit: int | None = None
    """
    If set, limit concurrent GPU usage to this amount.
    """


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
    monitor_port: int | None = None
    """
    Not used.
    """
    gpus: list[Gpu] | None = None
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
    custom_vars: dict[str, str] = {}
    """
    Custom variable values
    """
    gpu_concurrency_limit: int | None = None
    """
    If set, limit concurrent GPU usage to this amount.
    """
    tpl_vars: list[str]
    """
    List of template variable names in a cluster target
    """
    custom_var_names: list[str]
    """
    Computed list if custom variable names defined in the template
    """


class SchedulerTarget(BaseModel):
    """
    Details and configuration for a node or cluster target.
    """

    cache_path: str | None = None
    """
    Path the SSD cache scratch directory, if applicable.
    """
    cache_reserve_mb: int | None = None
    """
    Ensure at least this much space is free on the SSD scratch drive before
    caching.
    """
    cache_quota_mb: int | None = None
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
    desc: str | None = None
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
    config: Node | Cluster | AWSBatch
    """
    Target configuration details.
    """
    arch: Arch = "x86_64"
    """
    Architecture of cryoSPARC installed on this target
    """


class SchedulerTargetCluster(BaseModel):
    cache_path: str | None = None
    """
    Path the SSD cache scratch directory, if applicable.
    """
    cache_reserve_mb: int | None = None
    """
    Ensure at least this much space is free on the SSD scratch drive before
    caching.
    """
    cache_quota_mb: int | None = None
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
    desc: str | None = None
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
    arch: Arch = "x86_64"
    """
    Architecture of cryoSPARC installed on this target
    """


class SchedulerTargetNode(BaseModel):
    cache_path: str | None = None
    """
    Path the SSD cache scratch directory, if applicable.
    """
    cache_reserve_mb: int | None = None
    """
    Ensure at least this much space is free on the SSD scratch drive before
    caching.
    """
    cache_quota_mb: int | None = None
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
    desc: str | None = None
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
    arch: Arch = "x86_64"
    """
    Architecture of cryoSPARC installed on this target
    """
