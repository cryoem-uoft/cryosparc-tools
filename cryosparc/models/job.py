# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .instance import InstanceInformation
from .job_spec import JobBuildError, JobSpec
from .scheduler_target import FixedResourceSlots, ResourceSlots, SchedulerTarget


class AllocatedResources(BaseModel):
    lane: Optional[str] = None
    lane_type: Optional[str] = None
    hostname: str
    target: Optional[SchedulerTarget] = None
    slots: ResourceSlots = ResourceSlots()
    fixed: FixedResourceSlots = FixedResourceSlots()
    licenses_acquired: int = 0


JobStatus = Literal["building", "queued", "launched", "started", "running", "waiting", "completed", "killed", "failed"]


class JobLastAccessed(BaseModel):
    name: str
    accessed_at: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)


class RunError(BaseModel):
    message: str
    warning: bool = False


class UiTileImage(BaseModel):
    name: str
    fileid: str
    num_rows: Optional[int] = None
    num_cols: Optional[int] = None


class JobWorkflowInfo(BaseModel):
    id: str
    jobId: str
    run: int


class Job(BaseModel):
    """
    Specification for a Job document from the MongoDB database.
    """

    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    dumped_at: Optional[datetime.datetime] = None
    last_dumped_version: Optional[str] = None
    autodump: bool = True
    uid: str
    project_uid: str
    workspace_uids: List[str]
    spec: JobSpec
    job_dir: str
    job_dir_size: int = 0
    job_dir_size_last_updated: Optional[datetime.datetime] = None
    run_as_user: Optional[str] = None
    title: str = ""
    description: str = ""
    status: JobStatus = "building"
    created_by_user_id: Optional[str] = None
    created_by_job_uid: Optional[str] = None
    cloned_from: Optional[str] = None
    queued_at: Optional[datetime.datetime] = None
    started_at: Optional[datetime.datetime] = None
    launched_at: Optional[datetime.datetime] = None
    running_at: Optional[datetime.datetime] = None
    waiting_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    killed_at: Optional[datetime.datetime] = None
    failed_at: Optional[datetime.datetime] = None
    heartbeat_at: Optional[datetime.datetime] = None
    tokens_acquired_at: Optional[datetime.datetime] = None
    tokens_requested_at: Optional[datetime.datetime] = None
    last_scheduled_at: Optional[datetime.datetime] = None
    last_accessed: Optional[JobLastAccessed] = None
    has_error: bool = False
    has_warning: bool = False
    version_created: Optional[str] = None
    version: Optional[str] = None
    priority: int = 0
    deleted: bool = False
    deleting: bool = False
    parents: List[str] = []
    children: List[str] = []
    resources_allocated: Optional[AllocatedResources] = None
    queued_by_user_id: Optional[str] = None
    queued_to_lane: Optional[str] = None
    queued_to_hostname: Optional[str] = None
    queued_to_gpu: Optional[List[int]] = None
    queue_index: Optional[int] = None
    queue_status: Optional[
        Literal[
            "waiting_inputs",
            "project_paused",
            "actively_queued",
            "launched",
            "waiting_maintenance",
            "waiting_licenses",
            "waiting_resources",
        ]
    ] = None
    queue_message: Optional[str] = None
    queued_job_hash: Optional[int] = None
    num_tokens: int = 0
    job_sig: Optional[str] = None
    errors_run: List[RunError] = []
    interactive_port: Optional[int] = None
    PID_monitor: Optional[int] = None
    PID_main: Optional[int] = None
    PID_workers: List[int] = []
    cluster_job_id: Optional[str] = None
    cluster_job_status: Optional[str] = None
    cluster_job_status_code: Optional[str] = None
    cluster_job_monitor_event_id: Optional[str] = None
    cluster_job_monitor_retries: int = 0
    cluster_job_monitor_last_run_at: Optional[datetime.datetime] = None
    cluster_job_submission_script: Optional[str] = None
    cluster_job_custom_vars: Dict[str, str] = {}
    ui_tile_images: List[UiTileImage] = []
    is_experiment: bool = False
    enable_bench: bool = False
    bench: dict = {}
    bench_timings: Dict[str, List[Tuple[datetime.datetime, Optional[datetime.datetime]]]] = {}
    completed_count: int = 0
    instance_information: InstanceInformation = InstanceInformation()
    generate_intermediate_results: bool = False
    last_intermediate_data_cleared_at: Optional[datetime.datetime] = None
    last_intermediate_data_cleared_amount: int = 0
    intermediate_results_size_bytes: int = 0
    intermediate_results_size_last_updated: Optional[datetime.datetime] = None
    is_final_result: bool = False
    is_ancestor_of_final_result: bool = False
    no_check_inputs_ready: bool = False
    ui_layouts: Optional[dict] = None
    progress: List[dict] = []
    last_exported_at: Optional[datetime.datetime] = None
    last_exported_location: Optional[str] = None
    last_exported_version: Optional[str] = None
    tags: List[str] = []
    workflow: Optional[JobWorkflowInfo] = None
    imported_at: Optional[datetime.datetime] = None
    deleted_at: Optional[datetime.datetime] = None
    import_status: Optional[Literal["importing", "complete", "failed"]] = None
    starred_by: List[str] = []
    uid_num: int
    project_uid_num: int
    build_errors: List[JobBuildError]
    status_num: int
